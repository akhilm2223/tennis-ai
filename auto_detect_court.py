"""
Automatic Tennis Court Detection with Stable Line Tracking
Supports both traditional detection and ML-based keypoint detection
"""
import cv2
import numpy as np
import os


class CourtDetector:
    """Tennis court detector with homography and stable line tracking"""
    
    def __init__(self):
        self.corners = None
        self.homography_matrix = None
        self.court_lines = None
        self.stable_lines = None  # Cached stable lines
        self.line_history = []  # History for smoothing
        self.max_history = 10
        self.corner_history = []  # History for corner smoothing
        self.max_corner_history = 5
        self.keypoints = None  # ML-detected keypoints (14 points)
        self.detection_method = None  # 'color', 'lines', 'ml', or 'default'
    
    def is_calibrated(self):
        return self.homography_matrix is not None
    
    def update_lines(self, new_lines):
        """Update court lines with temporal smoothing"""
        if new_lines is None or len(new_lines) == 0:
            return self.stable_lines
        
        # Add to history
        self.line_history.append(new_lines)
        if len(self.line_history) > self.max_history:
            self.line_history.pop(0)
        
        # Average lines across history for stability
        if len(self.line_history) >= 3:
            # Use median of recent detections
            all_lines = np.concatenate(self.line_history, axis=0)
            self.stable_lines = all_lines[:20]  # Keep top 20 most stable
        else:
            self.stable_lines = new_lines
        
        return self.stable_lines
    
    def update_corners(self, new_corners):
        """Update court corners with temporal smoothing"""
        if new_corners is None:
            return self.corners
        
        # Add to history
        self.corner_history.append(new_corners)
        if len(self.corner_history) > self.max_corner_history:
            self.corner_history.pop(0)
        
        # Average corners for stability
        if len(self.corner_history) >= 2:
            avg_corners = np.mean(self.corner_history, axis=0)
            self.corners = avg_corners.astype(np.float32)
        else:
            self.corners = new_corners
        
        return self.corners
    
    def detect_frame(self, frame):
        """
        Detect court in current frame and update tracking
        Returns True if court detected successfully
        """
        h, w = frame.shape[:2]
        
        # Enhanced court detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Color-based court detection
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        court_mask = cv2.bitwise_or(green_mask, blue_mask)
        
        # Find court contour
        contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > (w * h * 0.2):
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = box.astype(np.int32)
                
                # Sort corners
                box = sorted(box, key=lambda p: (p[1], p[0]))
                top_pts = sorted(box[:2], key=lambda p: p[0])
                bottom_pts = sorted(box[2:], key=lambda p: p[0])
                new_corners = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)
                
                # Update with smoothing
                self.update_corners(new_corners)
                
                # Update homography
                court_width = 400
                court_height = 800
                dst_corners = np.array([
                    [0, 0],
                    [court_width, 0],
                    [court_width, court_height],
                    [0, court_height]
                ], dtype=np.float32)
                
                self.homography_matrix = cv2.getPerspectiveTransform(self.corners, dst_corners)
                return True
        
        # Line-based detection fallback
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                                minLineLength=150, maxLineGap=20)
        
        if lines is not None and len(lines) > 6:
            self.update_lines(lines)
            return True
        
        return False


def detect_court_automatic(frame, use_ml_detector=False, ml_model_path=None):
    """
    Automatically detect tennis court in frame with improved stability
    
    Args:
        frame: Input frame
        use_ml_detector: Whether to use ML-based court line detector
        ml_model_path: Path to trained model weights (required if use_ml_detector=True)
        
    Returns:
        CourtDetector object
    """
    detector = CourtDetector()
    
    # Method 0: ML-based keypoint detection (if enabled)
    if use_ml_detector and ml_model_path:
        try:
            from trackers.court_line_detector import CourtLineDetector
            
            ml_detector = CourtLineDetector(ml_model_path)
            keypoints = ml_detector.predict(frame)
            
            # Extract corners from keypoints (first 4 points)
            corners = ml_detector.get_court_corners(keypoints)
            
            if len(corners) >= 4:
                detector.corners = corners
                detector.keypoints = keypoints
                detector.detection_method = 'ml'
                
                # Create homography
                court_width = 400
                court_height = 800
                dst_corners = np.array([
                    [0, 0],
                    [court_width, 0],
                    [court_width, court_height],
                    [0, court_height]
                ], dtype=np.float32)
                
                detector.homography_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
                print("✅ Court detected using ML keypoint detection!")
                return detector
        except Exception as e:
            print(f"⚠ ML detection failed: {e}")
            print("   Falling back to traditional detection...")
    
    detector = CourtDetector()
    
    h, w = frame.shape[:2]
    
    # Enhanced court detection using multiple methods
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Color-based court detection (green/blue courts)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Green court mask
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Blue court mask (hard courts)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Combine masks
    court_mask = cv2.bitwise_or(green_mask, blue_mask)
    
    # Find largest contour (likely the court)
    contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # If court area is significant (>20% of frame)
        if area > (w * h * 0.2):
            # Get bounding rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            
            # Sort corners: top-left, top-right, bottom-right, bottom-left
            box = sorted(box, key=lambda p: (p[1], p[0]))
            top_pts = sorted(box[:2], key=lambda p: p[0])
            bottom_pts = sorted(box[2:], key=lambda p: p[0])
            corners = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)
            
            detector.corners = corners
            
            # Create homography to standard court
            # Standard court dimensions: 23.77m x 10.97m
            court_width = 400
            court_height = 800
            dst_corners = np.array([
                [0, 0],
                [court_width, 0],
                [court_width, court_height],
                [0, court_height]
            ], dtype=np.float32)
            
            detector.homography_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
            detector.detection_method = 'color'
            print("✅ Court detected using color analysis!")
            return detector
    
    # Method 2: Line-based detection (fallback)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhanced edge detection
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
    
    # Detect lines with stricter parameters for stability
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                            minLineLength=150, maxLineGap=20)
    
    if lines is not None and len(lines) > 6:
        # Filter horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Horizontal lines (close to 0 or 180 degrees)
            if angle < 15 or angle > 165:
                horizontal_lines.append(line)
            # Vertical lines (close to 90 degrees)
            elif 75 < angle < 105:
                vertical_lines.append(line)
        
        # Need at least 3 horizontal and 2 vertical lines for a court
        if len(horizontal_lines) >= 3 and len(vertical_lines) >= 2:
            detector.court_lines = lines
            detector.stable_lines = lines
            
            # Estimate corners from line intersections
            # Simplified: use frame boundaries with margins
            margin_x = w * 0.15
            margin_y_top = h * 0.1
            margin_y_bottom = h * 0.85
            
            detector.corners = np.array([
                [margin_x, margin_y_top],
                [w - margin_x, margin_y_top],
                [w - margin_x, margin_y_bottom],
                [margin_x, margin_y_bottom]
            ], dtype=np.float32)
            
            # Create homography
            court_width = 400
            court_height = 800
            dst_corners = np.array([
                [0, 0],
                [court_width, 0],
                [court_width, court_height],
                [0, court_height]
            ], dtype=np.float32)
            
            detector.homography_matrix = cv2.getPerspectiveTransform(detector.corners, dst_corners)
            detector.detection_method = 'lines'
            print("✅ Court detected using line analysis!")
            return detector
    
    # Method 3: Default fallback (use frame with margins)
    print("⚠ Automatic detection uncertain - using default court region")
    margin_x = w * 0.15
    margin_y_top = h * 0.15
    margin_y_bottom = h * 0.85
    
    detector.corners = np.array([
        [margin_x, margin_y_top],
        [w - margin_x, margin_y_top],
        [w - margin_x, margin_y_bottom],
        [margin_x, margin_y_bottom]
    ], dtype=np.float32)
    
    # Create homography
    court_width = 400
    court_height = 800
    dst_corners = np.array([
        [0, 0],
        [court_width, 0],
        [court_width, court_height],
        [0, court_height]
    ], dtype=np.float32)
    
    detector.homography_matrix = cv2.getPerspectiveTransform(detector.corners, dst_corners)
    detector.detection_method = 'default'
    
    return detector


def draw_court_lines(frame, court_detector, color=(0, 255, 0), thickness=2, show_keypoints=False):
    """Draw stable court lines overlay"""
    # If ML keypoints are available, draw only the keypoints (no lines)
    if court_detector.keypoints is not None:
        from trackers.court_line_detector import CourtLineDetector
        
        # Create temporary detector just for drawing
        temp_detector = type('obj', (object,), {})()
        frame = CourtLineDetector.draw_keypoints(temp_detector, frame, court_detector.keypoints, 
                                                 color=color, radius=5, show_labels=False)
        return frame
    
    if court_detector.corners is not None:
        # Draw court boundary (most stable visualization)
        corners = court_detector.corners.astype(np.int32)
        
        # Draw court perimeter
        cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=thickness)
        
        # Draw center line (horizontal)
        h = frame.shape[0]
        mid_y = int((corners[0][1] + corners[2][1]) / 2)
        cv2.line(frame, 
                (int(corners[0][0]), mid_y), 
                (int(corners[1][0]), mid_y), 
                color, thickness)
        
        # Draw service lines (approximate)
        third_y1 = int(corners[0][1] + (corners[2][1] - corners[0][1]) * 0.33)
        third_y2 = int(corners[0][1] + (corners[2][1] - corners[0][1]) * 0.67)
        
        cv2.line(frame, 
                (int(corners[0][0]), third_y1), 
                (int(corners[1][0]), third_y1), 
                color, max(1, thickness-1))
        
        cv2.line(frame, 
                (int(corners[0][0]), third_y2), 
                (int(corners[1][0]), third_y2), 
                color, max(1, thickness-1))
    
    return frame


def draw_court_corners(frame, court_detector, color=(0, 0, 255), radius=8):
    """Draw court corners"""
    if court_detector.corners is not None:
        for corner in court_detector.corners:
            cv2.circle(frame, (int(corner[0]), int(corner[1])), radius, color, -1)
    return frame


def draw_court_status(frame, court_detector, show_status=True):
    """Draw court calibration status"""
    if show_status:
        status = "Court: Calibrated" if court_detector.is_calibrated() else "Court: Not Calibrated"
        color = (0, 255, 0) if court_detector.is_calibrated() else (0, 0, 255)
        cv2.putText(frame, status, (frame.shape[1] - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame
