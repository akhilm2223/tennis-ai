"""
Tennis Analysis Utility Functions
"""
import cv2
import numpy as np
from collections import deque


class SpeedTracker:
    """Track and smooth speed calculations"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.positions = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.frame_count = 0
    
    def update(self, position, fps, pixel_to_meter=0.05):
        """
        Update speed calculation
        
        Args:
            position: (x, y) tuple or None
            fps: Frames per second
            pixel_to_meter: Conversion factor
            
        Returns:
            Speed in km/h
        """
        self.frame_count += 1
        
        if position is None:
            return 0.0
        
        self.positions.append(position)
        self.timestamps.append(self.frame_count / fps)
        
        if len(self.positions) < 2:
            return 0.0
        
        # Calculate speed from recent positions
        pos1 = self.positions[0]
        pos2 = self.positions[-1]
        time1 = self.timestamps[0]
        time2 = self.timestamps[-1]
        
        # Distance in pixels
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance_pixels = np.sqrt(dx**2 + dy**2)
        
        # Convert to meters
        distance_meters = distance_pixels * pixel_to_meter
        
        # Time difference
        time_diff = time2 - time1
        
        if time_diff == 0:
            return 0.0
        
        # Speed in m/s
        speed_ms = distance_meters / time_diff
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        return speed_kmh


def draw_circle(frame, center, radius=5, color=(0, 255, 0), thickness=-1):
    """Draw a circle on frame"""
    if center is not None:
        cv2.circle(frame, (int(center[0]), int(center[1])), radius, color, thickness)


def draw_trail(frame, trail, color=(255, 255, 0), max_length=30, thickness=2):
    """Draw a trail of points"""
    if len(trail) < 2:
        return
    
    points = list(trail)[-max_length:]
    
    for i in range(len(points) - 1):
        if points[i] is None or points[i+1] is None:
            continue
        
        # Fade effect
        alpha = (i + 1) / len(points)
        pt1 = (int(points[i][0]), int(points[i][1]))
        pt2 = (int(points[i+1][0]), int(points[i+1][1]))
        
        cv2.line(frame, pt1, pt2, color, thickness)


def draw_stats_panel(frame, stats, x=10, y=30, font_scale=0.6, thickness=2):
    """
    Draw statistics panel on frame
    
    Args:
        frame: Image frame
        stats: Dictionary of stat_name: value
        x, y: Top-left position
    """
    line_height = 25
    current_y = y
    
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (x, current_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        current_y += line_height
