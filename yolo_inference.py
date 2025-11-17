"""
YOLO-based Ball Detection for Tennis Analysis
ROI + Global Fallback for robust tracking
"""
import numpy as np
from ultralytics import YOLO
import torch
import cv2


class BallDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Fix for PyTorch 2.6 weights_only issue
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        
        try:
            self.model = YOLO(model_path)
        finally:
            torch.load = original_load
            
        self.prev_center = None
        self.ball_class_id = 32  # COCO sports ball class
        self.use_color_filter = False  # Soft penalty, not hard reject
        
        print(f"✅ YOLO Ball Detector initialized (model: {model_path})")
        print(f"   Filters: size=0.002%-1.0%, aspect<3.0, ROI=80px → global fallback")

    def detect(self, frame, prev_center=None, missing_frames=0):
        """
        Detect tennis ball with ROI + global fallback + aggressive re-acquisition
        
        Args:
            frame: Input BGR frame
            prev_center: (x, y) from previous frame for ROI optimization
            missing_frames: How many frames since last detection (for re-acquisition mode)
            
        Returns:
            dict with 'center', 'conf', 'bbox' or None
        """
        h, w = frame.shape[:2]
        
        # -------- ADAPTIVE RE-ACQUISITION MODE (MUCH FASTER) --------
        is_tracking = prev_center is not None  # Already tracking
        in_reacquisition = missing_frames > 5   # from 10 → trigger earlier
        in_desperate_mode = missing_frames > 12  # from 20 → recover faster
        
        if in_desperate_mode:
            # DESPERATE MODE: Ultra-low confidence, full frame, no filters
            det = self._run_yolo_and_pick_best(frame, (0, 0), conf=0.05, skip_filters=True)
        elif in_reacquisition:
            # RE-ACQUISITION MODE: Lower confidence, full frame
            det = self._run_yolo_and_pick_best(frame, (0, 0), conf=0.06, skip_filters=False)  # LOWER from 0.08
        else:
            # NORMAL MODE: ROI first, then full frame fallback
            use_roi = prev_center is not None
            if use_roi:
                cx, cy = int(prev_center[0]), int(prev_center[1])
                search_radius = 150  # INCREASED from 80 to 150 - track faster balls better
                x1 = max(0, cx - search_radius)
                y1 = max(0, cy - search_radius)
                x2 = min(w, cx + search_radius)
                y2 = min(h, cy + search_radius)
                
                # SAFETY CHECK: Ensure ROI has valid dimensions
                if x2 <= x1 or y2 <= y1 or x2 - x1 < 10 or y2 - y1 < 10:
                    # Invalid ROI, use full frame instead
                    roi = frame
                    roi_offset = (0, 0)
                else:
                    roi = frame[y1:y2, x1:x2].copy()
                    roi_offset = (x1, y1)
            else:
                roi = frame
                roi_offset = (0, 0)
            
            # Run YOLO on ROI with LOWER confidence when actively tracking
            track_conf = 0.08 if is_tracking else 0.15  # MUCH lower threshold when tracking
            det = self._run_yolo_and_pick_best(roi, roi_offset, conf=track_conf, skip_filters=False)
            
            # If ROI failed, run YOLO on full frame to re-acquire (also with lower conf when tracking)
            if det is None and use_roi:
                det = self._run_yolo_and_pick_best(frame, (0, 0), conf=track_conf, skip_filters=False)
        
        if det is not None:
            self.prev_center = det["center"]
        
        return det

    def _run_yolo_and_pick_best(self, img, offset, conf=0.15, skip_filters=False):
        """
        Helper: run YOLO on img and return best ball candidate dict or None
        
        Args:
            img: Image to search (ROI or full frame)
            offset: (x, y) offset to convert local coords to frame coords
            conf: Confidence threshold for YOLO
            skip_filters: If True, skip size/aspect filters (desperate mode)
            
        Returns:
            dict with 'center', 'conf', 'bbox' or None
        """
        results = self.model(img, conf=conf, verbose=False)
        
        best = None
        best_score = 0
        
        frame_h, frame_w = img.shape[:2]
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Filter by class (sports ball)
                cls_id = int(box.cls[0])
                if cls_id != self.ball_class_id:
                    continue
                
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                score = float(box.conf[0])
                
                w = x2 - x1
                h = y2 - y1
                area = w * h
                aspect = max(w, h) / max(1e-3, min(w, h))
                
                # -------- MORE FORGIVING FILTERS (skip in desperate mode) --------
                if not skip_filters:
                    rel_area = area / (frame_h * frame_w + 1e-6)
                    
                    # Accept tinier balls (far away/top court)
                    if rel_area < 0.000003:   # from 0.00002 → much smaller allowed
                        continue
                    
                    # Allow slightly bigger close-up balls
                    if rel_area > 0.05:       # from 0.01 → much larger allowed
                        continue
                    
                    # Motion blur allowance (elongated ball shapes)
                    if aspect > 6.0:          # from 3.0 → more tolerant
                        continue
                
                # Optional color filter (soft penalty, not rejection)
                if self.use_color_filter:
                    if not self._looks_like_tennis_ball(img, int(x1), int(y1), int(x2), int(y2)):
                        score *= 0.7  # Downweight, don't reject
                
                # Track best candidate
                if score > best_score:
                    cx = offset[0] + (x1 + x2) / 2
                    cy = offset[1] + (y1 + y2) / 2
                    best_score = score
                    best = {
                        "center": (float(cx), float(cy)),
                        "bbox": (float(offset[0]+x1), float(offset[1]+y1),
                                float(offset[0]+x2), float(offset[1]+y2)),
                        "conf": score,
                    }
        
        return best
    
    def _looks_like_tennis_ball(self, img, x1, y1, x2, y2):
        """
        Check if bbox region looks like a tennis ball (yellow-green)
        Soft check, returns True if plausible
        """
        # Ensure valid bbox
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            return False
        
        # Convert to HSV
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        
        # Mean color
        mean_h, mean_s, mean_v = hsv[:, :, 0].mean(), hsv[:, :, 1].mean(), hsv[:, :, 2].mean()
        
        # Yellow-green tennis ball hue range (20-80 in OpenCV HSV)
        # But be forgiving
        if 15 < mean_h < 85 and mean_s > 60 and mean_v > 60:
            return True
        
        return False
