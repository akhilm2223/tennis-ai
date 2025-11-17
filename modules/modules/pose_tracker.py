"""
MediaPipe Pose Tracker for Tennis Players
Detects and tracks complete player skeletons (33 keypoints)
ENHANCED: Supports 4-player detection for doubles matches with bounding boxes
"""
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import mediapipe as mp

PoseLm = mp.solutions.pose.PoseLandmark


@dataclass
class PoseDet:
    """
    Complete pose detection for a single player
    
    Attributes:
        landmarks: List of 33 landmarks (x, y, z, visibility)
        centroid: Player center (x, y) based on shoulders + hips
        score: Average visibility/confidence score
        bbox: Bounding box (x1, y1, x2, y2) for this player
    """
    landmarks: List[Tuple[float, float, float, float]]
    centroid: Tuple[float, float]
    score: float
    bbox: Optional[Tuple[float, float, float, float]] = None


class PoseTracker:
    """
    Multi-player pose tracker using MediaPipe Pose
    Supports both 2-player and 4-player (doubles) detection with bounding boxes
    """
    
    def __init__(self, w: int, h: int, min_conf: float = 0.3, smooth: int = 20, num_players: int = 2):
        """
        Initialize pose tracker with improved stability
        
        Args:
            w: Frame width
            h: Frame height
            min_conf: Minimum detection/tracking confidence (lower = more sticky)
            smooth: Number of frames for centroid smoothing (higher = more stable)
            num_players: Number of players to track (2 or 4 for doubles)
        """
        self.w, self.h = w, h
        self.min_conf = min_conf
        self.num_players = num_players
        
        # Landmark history for temporal smoothing (reduce jitter)
        self.landmark_history = {i: deque(maxlen=5) for i in range(num_players)}
        
        # Initialize YOLO person detector for multi-player detection
        if num_players == 4:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            print(f"✅ YOLO person detector loaded for {num_players}-player tracking")
        else:
            self.yolo_model = None
        
        # Initialize MediaPipe Pose instances
        # For 4 players, we'll use 4 separate pose instances for stability
        self.pose_instances = []
        for i in range(max(num_players, 2)):  # At least 2 instances
            pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # 0=lite, 1=full, 2=heavy
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.3,  # Lower for initial detection
                min_tracking_confidence=0.3     # Lower to keep tracking
            )
            self.pose_instances.append(pose)
        
        # Legacy 2-player references (for backwards compatibility)
        self.pose_L = self.pose_instances[0]
        self.pose_R = self.pose_instances[1]
        
        # Smoothing history for centroids (one per player)
        self.smoothing_history = {i: deque(maxlen=smooth) for i in range(num_players)}
        # Legacy references
        self.histL = self.smoothing_history[0]
        self.histR = self.smoothing_history[1]
        
        # Store last valid pose for continuity (one per player)
        self.last_poses = {i: None for i in range(num_players)}
        self.missing_frames = {i: 0 for i in range(num_players)}
        self.last_centroids = {i: None for i in range(num_players)}  # For ID persistence
        self.last_bboxes = {i: None for i in range(num_players)}  # Store last bounding boxes
        self.player_locked = {i: False for i in range(num_players)}  # Lock status
        # Legacy references
        self.last_pose_L = None
        self.last_pose_R = None
        self.missing_frames_L = 0
        self.missing_frames_R = 0
        self.max_missing = 150  # Keep last pose for up to 150 frames (~5 seconds at 30fps) - SUPER STICKY
        
        # First frame detection flags
        self.is_first_detection = True
        self.frame_count = 0
        self.player_L_detected = False
        self.player_R_detected = False
        self.players_detected = {i: False for i in range(num_players)}
        
        # Fallback full-frame detection
        self.fallback_enabled = True
        self.last_fallback_frame = 0
        self.fallback_interval = 15  # Run fallback every 15 frames if needed
        
        # Debug visualization
        self.debug_zones = False  # Set to True to see zone split lines
        self.debug_boxes = True   # Show bounding boxes for 4-player mode
        
        print(f"✅ PoseTracker initialized ({w}x{h}, conf={min_conf}, smooth={smooth})")
        print(f"   Mode: {num_players}-player tracking")
        print(f"   Enhanced stability: Lower thresholds, better smoothing, instant detection")
        if num_players == 2:
            print(f"   Looking for 2 players: NEAR camera (bottom) = RED, FAR from camera (top) = BLUE")
        else:
            print(f"   Looking for {num_players} players: DOUBLES MODE with bounding boxes")
    
    def _landmarks_px(self, res, x_offset: int, width: int) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Convert normalized landmarks to pixel coordinates (legacy left/right split)
        
        Args:
            res: MediaPipe pose results
            x_offset: X offset for this half of the frame
            width: Width of this half
            
        Returns:
            List of (x, y, z, visibility) tuples or None
        """
        if not res or not res.pose_landmarks:
            return None
        
        pts = []
        for lm in res.pose_landmarks.landmark:
            # Convert normalized coords to pixels
            x_px = lm.x * width + x_offset
            y_px = lm.y * self.h
            z_depth = lm.z  # Relative depth
            visibility = lm.visibility
            pts.append((x_px, y_px, z_depth, visibility))
        
        return pts
    
    def _landmarks_px_depth(self, res, y_offset: int, zone_height: int) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Convert normalized landmarks to pixel coordinates (depth-based split)
        Used for near/far player detection based on court position
        
        Args:
            res: MediaPipe pose results
            y_offset: Y offset for this zone of the frame
            zone_height: Height of this zone
            
        Returns:
            List of (x, y, z, visibility) tuples or None
        """
        if not res or not res.pose_landmarks:
            return None
        
        pts = []
        for lm in res.pose_landmarks.landmark:
            # Convert normalized coords to pixels
            x_px = lm.x * self.w  # Full width
            y_px = lm.y * zone_height + y_offset  # Adjusted for zone
            z_depth = lm.z  # Relative depth
            visibility = lm.visibility
            pts.append((x_px, y_px, z_depth, visibility))
        
        return pts
    
    def _centroid(self, lms: List[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float]]:
        """
        Calculate player centroid from shoulders + hips
        
        Args:
            lms: Landmarks list
            
        Returns:
            (x, y) centroid or None if insufficient landmarks
        """
        # Use shoulders and hips for stable centroid
        idx = [
            PoseLm.LEFT_SHOULDER.value,
            PoseLm.RIGHT_SHOULDER.value,
            PoseLm.LEFT_HIP.value,
            PoseLm.RIGHT_HIP.value
        ]
        
        # Only use visible landmarks
        pts = [(lms[i][0], lms[i][1]) for i in idx if lms[i][3] > 0.5]
        
        if len(pts) < 2:
            return None
        
        c = np.mean(np.array(pts, dtype=np.float32), axis=0)
        return (float(c[0]), float(c[1]))
    
    def _calculate_pose_size(self, lms: List[Tuple[float, float, float, float]]) -> float:
        """
        Calculate the size of the detected pose (for filtering small/far people)
        
        Args:
            lms: Landmarks list
            
        Returns:
            Size score (larger = closer/bigger person)
        """
        # Use key body points to estimate size
        visible_points = [(lms[i][0], lms[i][1]) for i in range(len(lms)) if lms[i][3] > 0.5]
        
        if len(visible_points) < 4:
            return 0.0
        
        # Calculate bounding box of all visible points
        points = np.array(visible_points)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        
        # Area of bounding box
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        
        return float(area)
    
    def _is_valid_player_position(self, lms: List[Tuple[float, float, float, float]]) -> bool:
        """
        Check if pose is in valid player position (on court, not in audience)
        
        Args:
            lms: Landmarks list
            
        Returns:
            True if position looks like a player on court
        """
        # Get centroid
        centroid = self._centroid(lms)
        if centroid is None:
            return False
        
        cx, cy = centroid
        
        # Filter out people in top 20% of frame (likely audience/background)
        if cy < self.h * 0.2:
            return False
        
        # Filter out people in bottom 5% (likely scoreboard/ads)
        if cy > self.h * 0.95:
            return False
        
        # Check if person has reasonable visibility
        # RELAXED for far player: lower threshold and fewer required landmarks
        visible_count = sum(1 for lm in lms if lm[3] > 0.2)  # Lowered from 0.5
        if visible_count < 5:  # Reduced from 10 - far players have low visibility
            return False
        
        return True
    
    def detect_four(self, bgr: np.ndarray) -> Dict[int, Optional[PoseDet]]:
        """
        Detect poses for 4 players in doubles match with bounding boxes
        Uses YOLO for person detection, then MediaPipe Pose for each player
        
        Args:
            bgr: Input frame (BGR format)
            
        Returns:
            Dictionary mapping player_id (0-3) to PoseDet or None
            Players are sorted by Y position: 0,1 = far (top), 2,3 = near (bottom)
        """
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        self.frame_count += 1
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Step 1: Detect all people using YOLO
        results = self.yolo_model(bgr, conf=0.15, classes=[0], verbose=False)  # class 0 = person, lower conf
        
        person_boxes = []
        raw_count = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                raw_count += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # Calculate box area and center
                width = x2 - x1
                height = y2 - y1
                area = width * height
                center_y = (y1 + y2) / 2
                center_x = (x1 + x2) / 2
                
                # Filter: Must be reasonable size (actual person, not tiny detection)
                min_area = (self.w * self.h) * 0.001  # At least 0.1% of frame (very lenient)
                max_area = (self.w * self.h) * 0.8   # Not more than 80% of frame
                if area < min_area or area > max_area:
                    if self.frame_count <= 5:
                        print(f"      Filtered by size: area={area:.0f} ({area/(self.w*self.h)*100:.2f}%), bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")
                    continue
                
                # Filter: Must be in bottom 90% of frame (not audience)
                if center_y < self.h * 0.10:
                    if self.frame_count <= 5:
                        print(f"      Filtered by Y-pos: center_y={int(center_y)} < {self.h*0.10:.0f}")
                    continue
                
                # Filter: Reasonable aspect ratio for standing person (height > width usually)
                aspect_ratio = height / width if width > 0 else 0
                if aspect_ratio < 0.4 or aspect_ratio > 7.0:  # Very lenient
                    if self.frame_count <= 5:
                        print(f"      Filtered by aspect ratio: {aspect_ratio:.2f}")
                    continue
                
                person_boxes.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'conf': conf,
                    'area': area,
                    'center': (center_x, center_y),
                    'center_y': center_y,
                    'center_x': center_x,
                    'width': width,
                    'height': height
                })
        
        if self.frame_count <= 5:
            print(f"   🔍 YOLO raw detections: {raw_count}, after filtering: {len(person_boxes)}")
        
        # Step 1.5: Remove duplicate/overlapping detections using NMS-like approach
        def boxes_overlap(box1, box2, threshold=0.4):
            """Check if two boxes overlap significantly"""
            x1_min, y1_min, x1_max, y1_max = box1['bbox']
            x2_min, y2_min, x2_max, y2_max = box2['bbox']
            
            # Calculate intersection
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            intersection = x_overlap * y_overlap
            
            # Calculate union
            area1 = box1['area']
            area2 = box2['area']
            union = area1 + area2 - intersection
            
            # IoU (Intersection over Union)
            iou = intersection / union if union > 0 else 0
            return iou > threshold
        
        # Remove overlapping boxes (keep the one with higher confidence)
        filtered_boxes = []
        person_boxes.sort(key=lambda x: x['conf'], reverse=True)  # Sort by confidence
        
        for box in person_boxes:
            is_duplicate = False
            for existing_box in filtered_boxes:
                if boxes_overlap(box, existing_box, threshold=0.5):  # Only remove if >50% overlap
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_boxes.append(box)
        
        person_boxes = filtered_boxes
        
        # Step 2: Filter to players ON the tennis court using position heuristics
        # Tennis court is in the center of the frame, players should be:
        # - In the middle 60% horizontally (not at edges where audience sits)
        # - In the bottom 90% vertically (include far players at top of court)
        # - Reasonable aspect ratio (standing people)
        
        court_players = []
        for box in person_boxes:
            cx, cy = box['center']
            ar = box['height'] / box['width'] if box['width'] > 0 else 0
            
            # Horizontal filter: court is in middle 60% of frame
            x_margin = self.w * 0.20  # 20% margin on each side (stricter)
            if cx < x_margin or cx > self.w - x_margin:
                if self.frame_count <= 5:
                    print(f"      Filtered by X: center_x={int(cx)} (margin: {int(x_margin)})")
                continue
            
            # Vertical filter: court is in bottom 90%, exclude top 10% (audience/scoreboard)
            y_threshold = self.h * 0.10
            if cy < y_threshold:
                if self.frame_count <= 5:
                    print(f"      Filtered by Y: center_y={int(cy)} < {int(y_threshold)}")
                continue
            
            # Aspect ratio: players should be taller than wide (standing)
            if ar < 0.8:  # Too wide = probably not a standing person
                if self.frame_count <= 5:
                    print(f"      Filtered by AR: {ar:.2f} < 0.8 (too wide)")
                continue
            
            # Add to court players
            court_players.append(box)
        
        # If we found fewer than 4 court players, fall back to all detected players
        if len(court_players) < 4:
            court_players = person_boxes
            if self.frame_count <= 5:
                print(f"   ⚠️  Court filter found {len(court_players)} players, using all {len(person_boxes)} detections")
        else:
            if self.frame_count <= 5:
                print(f"   ✅ Court filter: {len(person_boxes)} people → {len(court_players)} on-court players")
        
        # Sort by confidence * area (bigger and more confident = more likely a player)
        court_players.sort(key=lambda x: x['conf'] * x['area'], reverse=True)
        top_4_boxes = court_players[:4]
        
        # ===== SUPER-STICKY TRACKING: IoU-based matching with lock-in =====
        # Once a player ID is assigned, it NEVER changes unless truly lost
        if self.frame_count > 1 and any(bbox is not None for bbox in self.last_bboxes.values()):
            from scipy.optimize import linear_sum_assignment
            
            def calculate_iou(box1, box2):
                """Calculate Intersection over Union between two boxes"""
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2
                
                # Intersection
                x_left = max(x1_min, x2_min)
                y_top = max(y1_min, y2_min)
                x_right = min(x1_max, x2_max)
                y_bottom = min(y1_max, y2_max)
                
                if x_right < x_left or y_bottom < y_top:
                    return 0.0
                
                intersection = (x_right - x_left) * (y_bottom - y_top)
                area1 = (x1_max - x1_min) * (y1_max - y1_min)
                area2 = (x2_max - x2_min) * (y2_max - y2_min)
                union = area1 + area2 - intersection
                
                return intersection / union if union > 0 else 0.0
            
            # Get valid last boxes and centroids
            valid_last = [(pid, bbox, c) for pid, (bbox, c) in 
                         enumerate(zip(self.last_bboxes.values(), self.last_centroids.values())) 
                         if bbox is not None and c is not None]
            
            if len(valid_last) > 0 and len(top_4_boxes) > 0:
                # Build SUPER-STICKY cost matrix using IoU + distance
                cost_matrix = np.zeros((len(valid_last), len(top_4_boxes)))
                
                for i, (pid, last_bbox, last_c) in enumerate(valid_last):
                    for j, box in enumerate(top_4_boxes):
                        curr_bbox = box['bbox']
                        curr_c = box['center']
                        
                        # Calculate IoU (higher = better match)
                        iou = calculate_iou(last_bbox, curr_bbox)
                        
                        # Calculate centroid distance
                        dist = np.sqrt((last_c[0] - curr_c[0])**2 + (last_c[1] - curr_c[1])**2)
                        
                        # SUPER-STICKY: If player is locked, heavily favor same assignment
                        if self.player_locked.get(pid, False):
                            # For locked players, use IoU as primary metric
                            # Low cost = high IoU (want to minimize cost)
                            cost = (1.0 - iou) * 100  # IoU from 0-1, cost from 100-0
                            # Add small distance penalty
                            cost += dist * 0.1
                        else:
                            # For unlocked players, balance IoU and distance
                            cost = (1.0 - iou) * 50 + dist * 0.5
                        
                        cost_matrix[i, j] = cost
                
                # Solve assignment problem
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Apply assignments with STRICT rules
                matched_boxes = [None] * 4
                used_boxes = set()
                
                for i, j in zip(row_ind, col_ind):
                    pid = valid_last[i][0]
                    curr_bbox = top_4_boxes[j]['bbox']
                    last_bbox = valid_last[i][1]
                    
                    # Calculate IoU for this match
                    iou = calculate_iou(last_bbox, curr_bbox)
                    
                    # SUPER-STICKY rules:
                    # 1. If locked: accept ANY match with IoU > 0.1 (even small overlap = same player)
                    # 2. If unlocked: accept if IoU > 0.3 OR distance < 100px
                    if self.player_locked.get(pid, False):
                        accept = iou > 0.1  # Very lenient for locked players
                    else:
                        dist = cost_matrix[i, j]
                        accept = iou > 0.3 or dist < 100
                    
                    if accept:
                        matched_boxes[pid] = top_4_boxes[j]
                        used_boxes.add(j)
                        # LOCK this player after first successful match
                        self.player_locked[pid] = True
                        
                        if self.frame_count <= 5:
                            print(f"      🔒 P{pid+1} LOCKED ← IoU={iou:.2f}, cost={cost_matrix[i,j]:.1f}")
                
                # Assign unmatched boxes to empty slots (new players)
                unmatched = [box for idx, box in enumerate(top_4_boxes) if idx not in used_boxes]
                for i in range(4):
                    if matched_boxes[i] is None and len(unmatched) > 0:
                        matched_boxes[i] = unmatched.pop(0)
                        # New player - will be locked on next frame
                
                # Update top_4_boxes
                top_4_boxes = [b for b in matched_boxes if b is not None]
                
                if self.frame_count % 100 == 0:
                    locked_count = sum(1 for v in self.player_locked.values() if v)
                    print(f"   🔗 Tracking: {len(top_4_boxes)}/4 detected, {locked_count}/4 locked")
        
        # First frame: sort by Y position to establish initial order
        if self.frame_count <= 1:
            top_4_boxes.sort(key=lambda x: x['center_y'])
        
        if self.frame_count % 100 == 0:
            print(f"   🎾 Frame {self.frame_count}: Found {len(person_boxes)} valid people, tracking {len(top_4_boxes)}/4")
        
        # Step 3: Run MediaPipe Pose on each bounding box
        player_detections = {}
        
        for player_id, person_box in enumerate(top_4_boxes):
            bbox = person_box['bbox']
            x1, y1, x2, y2 = bbox
            
            # Adaptive padding: less for large players (avoid overlap), more for small/far players
            w = x2 - x1
            h = y2 - y1
            
            # Check for potential overlap with other boxes
            min_distance_to_other = float('inf')
            for other_id, other_box in enumerate(top_4_boxes):
                if other_id == player_id:
                    continue
                other_bbox = other_box['bbox']
                ox1, oy1, ox2, oy2 = other_bbox
                # Distance between box centers
                cx1 = (x1 + x2) / 2
                cy1 = (y1 + y2) / 2
                cx2 = (ox1 + ox2) / 2
                cy2 = (oy1 + oy2) / 2
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                min_distance_to_other = min(min_distance_to_other, dist)
            
            # Dynamic padding based on size and proximity
            if min_distance_to_other < 100:  # Very close to another player
                padding = 0.03  # Minimal padding to prevent overlap
            elif h > self.h * 0.4:  # Large player
                padding = 0.05
            elif h < 150:  # Small/far player
                padding = 0.18  # More padding for better detection
            else:
                padding = 0.12  # Medium padding
            
            x1 = max(0, x1 - w * padding)
            y1 = max(0, y1 - h * padding)
            x2 = min(self.w, x2 + w * padding)
            y2 = min(self.h, y2 + h * padding)
            
            # Crop region for this player
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            player_crop = rgb[y1:y2, x1:x2]
            
            if player_crop.size == 0:
                continue
            
            # MULTI-SCALE DETECTION for small/far players
            # MediaPipe needs ~200px minimum, but far players may be only ~60px
            crop_height = y2 - y1
            
            if crop_height < 150:  # Small player - try multiple scales
                scales = [1.0, 1.5, 2.0]  # 100%, 150%, 200% zoom
                best_result = None
                best_visibility = 0
                
                pose_instance = self.pose_instances[player_id % len(self.pose_instances)]
                
                for scale in scales:
                    if scale == 1.0:
                        scaled_crop = player_crop
                    else:
                        # Zoom in by upscaling
                        new_h = int(crop_height * scale)
                        new_w = int((x2 - x1) * scale)
                        zoomed = cv2.resize(player_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        # Center crop back to original size
                        y_start = (new_h - (y2 - y1)) // 2
                        x_start = (new_w - (x2 - x1)) // 2
                        scaled_crop = zoomed[y_start:y_start+(y2-y1), x_start:x_start+(x2-x1)]
                    
                    res = pose_instance.process(scaled_crop)
                    if res and res.pose_landmarks:
                        avg_vis = float(np.mean([lm.visibility for lm in res.pose_landmarks.landmark]))
                        if avg_vis > best_visibility:
                            best_result = res
                            best_visibility = avg_vis
                
                result = best_result
                if self.frame_count <= 5 and result:
                    print(f"      Player {player_id+1}: Multi-scale detection, visibility={best_visibility:.2f}")
            else:
                # Normal sized player - single scale detection
                pose_instance = self.pose_instances[player_id % len(self.pose_instances)]
                result = pose_instance.process(player_crop)
            
            if result and result.pose_landmarks:
                # Convert landmarks to global frame coordinates
                # CRITICAL: Ensure landmarks stay within bounding box
                lms = []
                for lm in result.pose_landmarks.landmark:
                    # Normalized coords in crop -> pixel coords in crop -> global frame
                    x_crop = lm.x * (x2 - x1)
                    y_crop = lm.y * (y2 - y1)
                    x_global = x_crop + x1
                    y_global = y_crop + y1
                    
                    # CLAMP to bounding box to prevent drift outside rectangle
                    x_global = np.clip(x_global, x1, x2)
                    y_global = np.clip(y_global, y1, y2)
                    
                    lms.append((x_global, y_global, lm.z, lm.visibility))
                
                # Check if this is a valid pose detection
                avg_visibility = np.mean([lm[3] for lm in lms])
                visible_landmarks = sum(1 for lm in lms if lm[3] > 0.2)
                
                # Only accept if reasonable visibility (lenient for far players)
                if avg_visibility < 0.1 or visible_landmarks < 5:
                    # Poor detection - use last known pose if available
                    self.missing_frames[player_id] += 1
                    if self.last_poses[player_id] and self.missing_frames[player_id] < self.max_missing:
                        player_detections[player_id] = self.last_poses[player_id]
                    else:
                        player_detections[player_id] = None
                    continue
                
                # ===== LANDMARK SMOOTHING: Blend with previous frames to reduce jitter =====
                # Store current landmarks in history
                self.landmark_history[player_id].append(lms)
                
                # If we have history, apply temporal smoothing
                if len(self.landmark_history[player_id]) > 1:
                    smoothed_lms = []
                    history_list = list(self.landmark_history[player_id])
                    
                    # For each landmark, average across recent frames
                    for lm_idx in range(33):  # 33 landmarks in MediaPipe Pose
                        # Collect this landmark from recent history
                        lm_history = [history_list[frame_idx][lm_idx] for frame_idx in range(len(history_list))]
                        
                        # Weighted average: more recent = higher weight
                        weights = np.linspace(0.4, 1.0, len(lm_history))
                        weights = weights / weights.sum()
                        
                        x_smooth = np.average([lm[0] for lm in lm_history], weights=weights)
                        y_smooth = np.average([lm[1] for lm in lm_history], weights=weights)
                        z_smooth = np.average([lm[2] for lm in lm_history], weights=weights)
                        v_smooth = np.max([lm[3] for lm in lm_history])  # Keep best visibility
                        
                        smoothed_lms.append((float(x_smooth), float(y_smooth), float(z_smooth), float(v_smooth)))
                    
                    lms = smoothed_lms
                
                # Calculate centroid
                centroid = self._centroid(lms)
                if centroid is None:
                    # Use bbox center as fallback
                    centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                else:
                    # CLAMP centroid to bounding box to prevent drift
                    centroid = (
                        np.clip(centroid[0], x1, x2),
                        np.clip(centroid[1], y1, y2)
                    )
                
                # Smooth centroid with extra weight on previous position
                hist = self.smoothing_history[player_id]
                hist.append(centroid)
                if len(hist) > 1:
                    # Very aggressive smoothing for stability (40% → 100% weight)
                    weights = np.linspace(0.2, 1.0, len(hist))
                    weights = weights / weights.sum()
                    centroid_smooth = tuple(np.average(hist, axis=0, weights=weights))
                else:
                    centroid_smooth = centroid
                
                # Calculate score
                score = float(np.mean([p[3] for p in lms]))
                
                # Create PoseDet with bbox
                pose_det = PoseDet(
                    landmarks=lms,
                    centroid=centroid_smooth,
                    score=score,
                    bbox=(float(x1), float(y1), float(x2), float(y2))
                )
                
                player_detections[player_id] = pose_det
                self.last_poses[player_id] = pose_det
                self.last_centroids[player_id] = centroid_smooth  # Store for tracking
                self.last_bboxes[player_id] = (float(x1), float(y1), float(x2), float(y2))  # Store bbox for IoU matching
                self.missing_frames[player_id] = 0
                
                # Track first detection
                if not self.players_detected[player_id]:
                    self.players_detected[player_id] = True
                    print(f"   ✅ Player {player_id+1} detected! (pos: {int(centroid_smooth[0])},{int(centroid_smooth[1])})")
            else:
                # No pose detected - use last known pose if available
                self.missing_frames[player_id] += 1
                if self.last_poses[player_id] and self.missing_frames[player_id] < self.max_missing:
                    player_detections[player_id] = self.last_poses[player_id]
                    if self.frame_count % 100 == 0:
                        print(f"   🔒 Player {player_id+1} using cached pose (missing: {self.missing_frames[player_id]})")
                else:
                    player_detections[player_id] = None
        
        # Fill missing players with last known poses (STICKY TRACKING)
        for i in range(4):
            if i not in player_detections:
                # Try to use last known pose
                self.missing_frames[i] = self.missing_frames.get(i, 0) + 1
                if i in self.last_poses and self.last_poses[i] and self.missing_frames[i] < self.max_missing:
                    player_detections[i] = self.last_poses[i]
                    if self.frame_count % 100 == 0:
                        print(f"   ⚠️ Player {i+1} not found in YOLO, using cached pose (missing: {self.missing_frames[i]})")
                else:
                    player_detections[i] = None
        
        # Debug: Show status every 100 frames
        if self.frame_count % 100 == 0:
            detected_count = sum(1 for p in player_detections.values() if p is not None)
            print(f"   Frame {self.frame_count}: Tracking {detected_count}/4 players")
        
        return player_detections
    
    def detect_two(self, bgr: np.ndarray) -> Tuple[Optional[PoseDet], Optional[PoseDet]]:
        """
        Detect poses for two players with instant first-frame detection
        Players separated by court position (near vs far from camera)
        
        Args:
            bgr: Input frame (BGR format)
            
        Returns:
            (player_near, player_far) tuple of PoseDet or None
            player_near = bottom of court (close to camera) = Player 1 (Red)
            player_far = top of court (far from camera/net) = Player 2 (Blue)
        """
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        self.frame_count += 1
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Dynamic zone split: 55/45 biased toward bottom
        # Near player (bottom) is larger, so give more area
        mid_height = int(self.h * 0.55)  # 55% to bottom, 45% to top
        
        # Process FULL WIDTH for each depth zone
        # Bottom half = near player (closer to camera)
        # Top half = far player (at net, far from camera)
        near_zone = rgb[mid_height:, :]  # Bottom half - near player
        far_zone = rgb[:mid_height, :]   # Top half - far player
        
        # For first frames, process multiple times for guaranteed detection
        if self.frame_count <= 5:
            # Process multiple times to ensure both players detected
            for _ in range(2):
                _ = self.pose_L.process(near_zone)
                _ = self.pose_R.process(far_zone)
        
        # NEAR PLAYER: Standard processing (already large enough)
        result_L = self.pose_L.process(near_zone)  # Near player (bottom)
        
        # FAR PLAYER: Multi-scale detection to handle small distant player
        # MediaPipe Pose needs ~200px minimum, but far player may be only ~100px
        # Solution: Try multiple zoom levels and pick best detection
        far_results = []
        scales = [1.0, 1.3, 1.6]  # 100%, 130%, 160% zoom levels
        
        for scale in scales:
            # Resize the far zone by scale factor
            h_far, w_far, _ = far_zone.shape
            new_h, new_w = int(h_far * scale), int(w_far * scale)
            
            if scale == 1.0:
                # No zoom, use original
                zoomed = far_zone
            else:
                # Zoom in by upscaling
                zoomed = cv2.resize(far_zone, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                # Center-crop back to original size
                y0 = (new_h - h_far) // 2
                x0 = (new_w - w_far) // 2
                zoomed = zoomed[y0:y0+h_far, x0:x0+w_far]
            
            # Process this scale
            res = self.pose_R.process(zoomed)
            if res and res.pose_landmarks:
                # Calculate average visibility score for this detection
                avg_visibility = float(np.mean([lm.visibility for lm in res.pose_landmarks.landmark]))
                far_results.append((res, avg_visibility, scale))
        
        # Pick the detection with highest visibility (best quality)
        if far_results:
            # Sort by visibility score, pick best
            far_results.sort(key=lambda x: x[1], reverse=True)
            result_R, best_vis, best_scale = far_results[0]
            
            # Log when using zoomed detection
            if self.frame_count % 100 == 0 and best_scale > 1.0:
                print(f"   🔍 Far player using {best_scale}x zoom (visibility: {best_vis:.2f})")
        else:
            # Fallback: use original scale if no detections at any scale
            result_R = self.pose_R.process(far_zone)
        
        # Debug warning: Far player lost for extended period
        # DISABLED: We now use CSRT tracker for Player 2, not MediaPipe
        # if self.missing_frames_R > 50:
        #     print(f"⚠️  Player 2 (FAR) lost for {self.missing_frames_R} frames — activating full-frame fallback")
        
        # Track which players are detected
        if result_L.pose_landmarks and not self.player_L_detected:
            self.player_L_detected = True
            print("   🔴 Player 1 (RED) detected - NEAR camera (bottom court)!")
        
        if result_R.pose_landmarks and not self.player_R_detected:
            self.player_R_detected = True
            print("   🔵 Player 2 (BLUE) detected - FAR from camera (top/net)!")
        
        # Mark first detection as complete after both players detected
        if self.is_first_detection and self.player_L_detected and self.player_R_detected:
            self.is_first_detection = False
            print("   ✅ BOTH PLAYERS DETECTED! Near(Red) + Far(Blue) locked!")
        
        # FALLBACK: Full-frame detection if either player missing for 10+ frames
        use_fallback = False
        fallback_result = None
        if self.fallback_enabled and (self.missing_frames_L > 10 or self.missing_frames_R > 10):
            # Only run fallback every N frames (not every frame for performance)
            if self.frame_count - self.last_fallback_frame >= self.fallback_interval:
                use_fallback = True
                self.last_fallback_frame = self.frame_count
                # Run full-frame detection
                import mediapipe as mp
                pose_full = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                fallback_result = pose_full.process(rgb)
                pose_full.close()
        
        def make_pose_det(res, y_offset: int, zone_height: int, hist: deque, 
                         last_pose: Optional[PoseDet], missing_count: int, 
                         is_far_player: bool = False) -> Tuple[Optional[PoseDet], int]:
            """Helper to create PoseDet from results with filtering and fallback"""
            # Convert landmarks to pixels (now using full width, different y offset)
            lms = self._landmarks_px_depth(res, y_offset, zone_height)
            
            # POSE BLENDING: If far player with low visibility, blend with last pose for stability
            if is_far_player and last_pose and lms:
                avg_vis = np.mean([p[3] for p in lms])
                if avg_vis < 0.3:  # Low confidence detection
                    # Blend: 70% last pose + 30% new detection for smooth continuity
                    blended = []
                    for i in range(min(len(lms), len(last_pose.landmarks))):
                        lx, ly, lz, lv = lms[i]
                        px, py, pz, pv = last_pose.landmarks[i]
                        x = 0.7 * px + 0.3 * lx
                        y = 0.7 * py + 0.3 * ly
                        z = 0.7 * pz + 0.3 * lz
                        v = max(lv, pv * 0.8)  # Keep better visibility
                        blended.append((x, y, z, v))
                    lms = blended
                    if self.frame_count % 100 == 0:
                        print(f"   🔄 Far player: Low confidence ({avg_vis:.2f}), blending with last pose")
            
            if lms is None:
                # No detection - use last pose if available
                missing_count += 1
                if last_pose and missing_count < self.max_missing:
                    # Keep returning last valid pose
                    return last_pose, missing_count
                else:
                    # Too many missing frames, give up
                    return None, missing_count
            
            # ===== FILTERING: Only accept valid players =====
            
            # For first frames, be VERY lenient to ensure BOTH players detected
            use_lenient_filters = self.is_first_detection or self.frame_count <= 10
            
            # Filter 1: Check if position is valid (not audience/background)
            if not self._is_valid_player_position(lms):
                # On first frames, be more forgiving
                if not use_lenient_filters:
                    missing_count += 1
                    if last_pose and missing_count < self.max_missing:
                        return last_pose, missing_count
                    return None, missing_count
            
            # Filter 2: Check if pose is large enough (actual player, not distant person)
            pose_size = self._calculate_pose_size(lms)
            # VERY low threshold for first frames to catch BOTH players
            if use_lenient_filters:
                min_size_threshold = 0.003  # 0.3% - very lenient
            else:
                # Far player (top) needs lower threshold since they're smaller/further
                if is_far_player:
                    min_size_threshold = 0.0008  # 0.08% - ULTRA lenient for far player (reduced from 0.002)
                else:
                    min_size_threshold = 0.008  # 0.8% - normal for near player
            
            min_size = (self.w * zone_height) * min_size_threshold
            if pose_size < min_size:
                # During lenient phase, be very forgiving
                if not use_lenient_filters:
                    missing_count += 1
                    if last_pose and missing_count < self.max_missing:
                        return last_pose, missing_count
                    return None, missing_count
                # During lenient phase, only reject if REALLY tiny
                if pose_size < min_size * 0.5:
                    missing_count += 1
                    if last_pose and missing_count < self.max_missing:
                        return last_pose, missing_count
                    return None, missing_count
            
            # ===== Passed all filters - this is a valid player =====
            
            # Got valid detection - reset missing counter
            missing_count = 0
            
            # Calculate centroid
            c = self._centroid(lms)
            if c is None:
                # Bad centroid - try to use last pose
                if last_pose:
                    return last_pose, missing_count
                return None, missing_count
            
            # Smooth centroid using history
            hist.append(c)
            
            # Use weighted average favoring recent frames
            if len(hist) > 1:
                weights = np.linspace(0.5, 1.0, len(hist))
                weights = weights / weights.sum()
                c_smooth = tuple(np.average(hist, axis=0, weights=weights))
            else:
                c_smooth = c
            
            # EXTRA CENTROID DAMPING for far player (prevent flicker)
            # Heavily weight previous position to avoid jitter when confidence is low
            if is_far_player and last_pose:
                c_smooth = (
                    0.8 * last_pose.centroid[0] + 0.2 * c_smooth[0],
                    0.8 * last_pose.centroid[1] + 0.2 * c_smooth[1],
                )
            
            # Calculate average confidence
            score = float(np.mean([p[3] for p in lms]))
            
            # Calculate bounding box from visible landmarks
            # Use lower threshold for far player (they have lower visibility)
            vis_threshold = 0.2 if is_far_player else 0.5
            visible_points = [(lm[0], lm[1]) for lm in lms if lm[3] > vis_threshold]
            bbox = None
            if len(visible_points) >= 4:
                points = np.array(visible_points)
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
            
            return PoseDet(lms, c_smooth, score, bbox), missing_count
        
        # Create pose detections for both players with continuity
        # Near player (bottom half of frame - closer to camera)
        near_height = self.h - mid_height
        pose_near, self.missing_frames_L = make_pose_det(
            result_L, mid_height, near_height, self.histL, self.last_pose_L, self.missing_frames_L,
            is_far_player=False
        )
        
        # Far player (top half of frame - at net, far from camera)
        far_height = mid_height
        pose_far, self.missing_frames_R = make_pose_det(
            result_R, 0, far_height, self.histR, self.last_pose_R, self.missing_frames_R,
            is_far_player=True
        )
        
        # Process fallback result if we ran it
        if use_fallback and fallback_result and fallback_result.pose_landmarks:
            # Extract all landmarks from full-frame detection
            full_lms = []
            for lm in fallback_result.pose_landmarks.landmark:
                x_px = lm.x * self.w
                y_px = lm.y * self.h
                z_depth = lm.z
                visibility = lm.visibility
                full_lms.append((x_px, y_px, z_depth, visibility))
            
            # Calculate centroid Y position to determine which player this is
            centroid = self._centroid(full_lms)
            if centroid:
                cy = centroid[1]
                # If centroid is in bottom half and near player is missing, assign to near
                if cy > mid_height and pose_near is None:
                    pose_near = PoseDet(full_lms, centroid, float(np.mean([p[3] for p in full_lms])))
                    self.missing_frames_L = 0
                    print(f"   🔄 Fallback: Player 1 (NEAR) recovered via full-frame detection!")
                # If centroid is in top half and far player is missing, assign to far
                elif cy <= mid_height and pose_far is None:
                    pose_far = PoseDet(full_lms, centroid, float(np.mean([p[3] for p in full_lms])))
                    self.missing_frames_R = 0
                    print(f"   🔄 Fallback: Player 2 (FAR) recovered via full-frame detection!")
        
        # FINAL PERSISTENCE: If Player 2 (far) still None, use last known pose
        # This prevents blue skeleton flicker between frames
        if pose_far is None and self.last_pose_R is not None:
            if self.missing_frames_R < self.max_missing:
                pose_far = self.last_pose_R
                if self.frame_count % 100 == 0:
                    print(f"   🔒 Player 2 (FAR) persisting last known pose (missing: {self.missing_frames_R})")
        
        # Store last valid poses - ALWAYS keep last good detection
        if pose_near is not None:
            self.last_pose_L = pose_near
        if pose_far is not None:
            self.last_pose_R = pose_far
        
        # Debug: Show status every 100 frames
        if self.frame_count % 100 == 0:
            status_near = "✅" if pose_near else "❌"
            status_far = "✅" if pose_far else "❌"
            print(f"   Frame {self.frame_count}: Near(Red)={status_near} Far(Blue)={status_far} " +
                  f"[Missing: L={self.missing_frames_L}, R={self.missing_frames_R}]")
        
        # Optional: Draw zone split line for debugging
        if self.debug_zones:
            cv2.line(bgr, (0, mid_height), (self.w, mid_height), (0, 0, 255), 2)
            cv2.putText(bgr, "FAR ZONE (45%)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(bgr, "NEAR ZONE (55%)", (10, mid_height + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return pose_near, pose_far
    
    def get_keypoint(self, pose_det: Optional[PoseDet], keypoint_id: int) -> Optional[Tuple[float, float]]:
        """
        Get specific keypoint from pose detection
        
        Args:
            pose_det: PoseDet object
            keypoint_id: MediaPipe landmark ID (0-32)
            
        Returns:
            (x, y) coordinates or None
        """
        if pose_det is None or keypoint_id >= len(pose_det.landmarks):
            return None
        
        lm = pose_det.landmarks[keypoint_id]
        if lm[3] < 0.5:  # Check visibility
            return None
        
        return (lm[0], lm[1])
    
    def get_wrist_position(self, pose_det: Optional[PoseDet], side: str = 'right') -> Optional[Tuple[float, float]]:
        """
        Get wrist position (useful for racket tracking)
        
        Args:
            pose_det: PoseDet object
            side: 'left' or 'right'
            
        Returns:
            (x, y) wrist coordinates or None
        """
        if side == 'right':
            keypoint = PoseLm.RIGHT_WRIST.value
        else:
            keypoint = PoseLm.LEFT_WRIST.value
        
        return self.get_keypoint(pose_det, keypoint)
    
    def close(self):
        """Release MediaPipe resources"""
        for pose_instance in self.pose_instances:
            try:
                pose_instance.close()
            except:
                pass
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except:
            pass

