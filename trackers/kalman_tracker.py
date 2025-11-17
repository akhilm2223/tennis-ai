"""
Kalman Filter implementation for ball tracking smoothing
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from constants.config import KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE, MAX_MISSING_FRAMES


class BallKalmanTracker:
    """Kalman Filter tracker for smoothing ball trajectory"""
    
    def __init__(self):
        # State: [x, y, vx, vy] - position and velocity
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise (lower = trust measurements more)
        self.kf.R *= KALMAN_MEASUREMENT_NOISE * 0.5  # Trust detections more
        
        # Process noise (higher = allow more dynamic movement)
        self.kf.Q *= KALMAN_PROCESS_NOISE * 2.0  # Allow faster ball movement
        
        # Initial state covariance
        self.kf.P *= 100
        
        self.initialized = False
        self.missing_frames = 0
        self.last_prediction = None
    
    def update(self, detection):
        """
        Update tracker with new detection
        
        Args:
            detection: (x, y) tuple or None if no detection
            
        Returns:
            (x, y) predicted position
        """
        if detection is None:
            # No detection - predict only
            if self.initialized:
                self.kf.predict()
                self.missing_frames += 1
                
                if self.missing_frames > MAX_MISSING_FRAMES:
                    self.initialized = False
                    return None
                
                self.last_prediction = (self.kf.x[0], self.kf.x[1])
                return self.last_prediction
            else:
                return None
        
        # We have a detection
        x, y = detection
        
        if not self.initialized:
            # Initialize filter with first detection
            self.kf.x = np.array([x, y, 0, 0])
            self.initialized = True
            self.missing_frames = 0
            self.last_prediction = (x, y)
            return (x, y)
        
        # Predict and update
        self.kf.predict()
        self.kf.update(np.array([x, y]))
        
        self.missing_frames = 0
        self.last_prediction = (self.kf.x[0], self.kf.x[1])
        return self.last_prediction
    
    def reset(self):
        """Reset the tracker"""
        self.initialized = False
        self.missing_frames = 0
        self.last_prediction = None


class PlayerTracker:
    """Simple player tracker based on position consistency"""
    
    def __init__(self):
        self.players = {}  # {player_id: {'bbox': [x1,y1,x2,y2], 'center': (cx, cy)}}
        self.next_id = 0
    
    def update(self, detections):
        """
        Update player tracking
        
        Args:
            detections: List of bboxes [(x1, y1, x2, y2), ...]
            
        Returns:
            Dict {player_id: bbox}
        """
        if len(detections) == 0:
            return {}
        
        # Calculate centers
        centers = []
        for bbox in detections:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))
        
        # Simple assignment: sort by x-coordinate (left to right)
        # This assumes players stay relatively in position
        sorted_detections = sorted(zip(detections, centers), key=lambda x: x[1][0])
        
        result = {}
        for i, (bbox, center) in enumerate(sorted_detections[:2]):  # Keep max 2 players
            result[i] = {
                'bbox': bbox,
                'center': center
            }
        
        return result

