"""
ML-based Tennis Court Line Detector using ResNet50
Detects 14 primary keypoints + 5 calculated midpoints = 19 total keypoints
"""
import torch
from torchvision import models, transforms
import cv2
import numpy as np


def keypoints_to_idx(keypoints, idx):
    """
    Extract (x, y) coordinates for a specific keypoint index
    
    Args:
        keypoints: 1-D array of [x1, y1, x2, y2, ...]
        idx: Keypoint index (0-based)
    
    Returns:
        (x, y) tuple
    """
    return (keypoints[idx * 2], keypoints[idx * 2 + 1])


def midpoint(point1, point2):
    """
    Calculate midpoint between two points
    
    Args:
        point1: (x, y) tuple
        point2: (x, y) tuple
    
    Returns:
        (x, y) midpoint
    """
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)


class CourtLineDetector:
    """
    Tennis court line detector using trained ResNet50 model
    Predicts 14 primary keypoints + calculates 5 midpoints
    """
    
    def __init__(self, model_path):
        """
        Initialize court line detector
        
        Args:
            model_path: Path to trained model weights (.pth file)
        """
        self.model = models.resnet50(weights=None)
        # Modify fc layer to output 14 keypoints (28 values for x,y coordinates)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        
        # Load saved weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            self.model.eval()
            print(f"✅ Court line detector model loaded from: {model_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load court line detector model: {e}")
            self.model = None
        
        # Image preprocessing transforms (ResNet50 standard)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Converts to (C, H, W) format
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """
        Predict court keypoints using the loaded model
        
        Args:
            image: Input frame (BGR format)
        
        Returns:
            1-D numpy array: [x1, y1, x2, y2, ...] coordinates of all 19 keypoints
            Returns None if model is not loaded
        """
        if self.model is None:
            return None
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image
        img_tensor = self.transform(img_rgb).unsqueeze(0)  # Add batch dimension
        
        # Predict keypoints
        with torch.no_grad():
            outputs = self.model(img_tensor)  # (1, 28)
        
        keypoints = outputs.squeeze().cpu().numpy()  # (28,)
        
        # Extract individual keypoints for midpoint calculation
        kp_4 = keypoints_to_idx(keypoints, 4)
        kp_5 = keypoints_to_idx(keypoints, 5)
        kp_6 = keypoints_to_idx(keypoints, 6)
        kp_7 = keypoints_to_idx(keypoints, 7)
        kp_8 = keypoints_to_idx(keypoints, 8)
        kp_9 = keypoints_to_idx(keypoints, 9)
        kp_10 = keypoints_to_idx(keypoints, 10)
        kp_11 = keypoints_to_idx(keypoints, 11)
        kp_12 = keypoints_to_idx(keypoints, 12)
        kp_13 = keypoints_to_idx(keypoints, 13)
        
        # Calculate additional midpoints (5 new points)
        kp_14 = midpoint(kp_4, kp_6)    # Baseline center (far side)
        kp_15 = midpoint(kp_5, kp_7)    # Baseline center (near side)
        kp_16 = midpoint(kp_12, kp_13)  # Net center
        kp_17 = midpoint(kp_8, kp_10)   # Service line center (far)
        kp_18 = midpoint(kp_9, kp_11)   # Service line center (near)
        
        # Append midpoints to keypoints array
        keypoints = np.append(keypoints, [kp_14[0], kp_14[1]])
        keypoints = np.append(keypoints, [kp_15[0], kp_15[1]])
        keypoints = np.append(keypoints, [kp_16[0], kp_16[1]])
        keypoints = np.append(keypoints, [kp_17[0], kp_17[1]])
        keypoints = np.append(keypoints, [kp_18[0], kp_18[1]])
        
        # Scale keypoints from 224x224 back to original image dimensions
        og_h, og_w = img_rgb.shape[:2]
        keypoints[::2] *= og_w / 224.0   # Scale x coordinates
        keypoints[1::2] *= og_h / 224.0  # Scale y coordinates
        
        return keypoints  # Shape: (38,) - 19 keypoints with (x, y) each
    
    def draw_keypoints(self, image, keypoints, color=(0, 0, 255), radius=5, show_labels=False):
        """
        Draw keypoints on image
        
        Args:
            image: Input frame
            keypoints: 1-D array of keypoint coordinates
            color: BGR color for keypoints
            radius: Circle radius for keypoint markers
            show_labels: Whether to show keypoint index labels
        
        Returns:
            Image with keypoints drawn
        """
        if keypoints is None:
            return image
        
        # Draw only the first 14 primary keypoints (indices 0-27)
        for i in range(0, min(28, len(keypoints)), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            
            # Draw filled circle at keypoint location
            cv2.circle(image, (x, y), radius, color, -1)
            
            # Optionally draw keypoint index label
            if show_labels:
                label = str(i // 2)
                cv2.putText(image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw midpoints with different color (keypoints 14-18)
        midpoint_color = (0, 255, 255)  # Cyan for midpoints
        for i in range(28, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.circle(image, (x, y), radius, midpoint_color, -1)
            
            if show_labels:
                label = str(i // 2)
                cv2.putText(image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, midpoint_color, 2)
        
        return image
    
    def draw_court_lines(self, image, keypoints, color=(255, 255, 255), thickness=2):
        """
        Draw court lines connecting keypoints
        
        Args:
            image: Input frame
            keypoints: 1-D array of keypoint coordinates
            color: BGR color for lines
            thickness: Line thickness
        
        Returns:
            Image with court lines drawn
        """
        if keypoints is None or len(keypoints) < 28:
            return image
        
        def get_point(idx):
            """Helper to get point from keypoints array"""
            return (int(keypoints[idx * 2]), int(keypoints[idx * 2 + 1]))
        
        # Define court line connections (keypoint pairs)
        # Based on standard tennis court structure
        line_connections = [
            # Baselines
            (0, 1),   # Far baseline
            (2, 3),   # Near baseline
            
            # Side lines (doubles)
            (0, 2),   # Left sideline
            (1, 3),   # Right sideline
            
            # Service lines
            (4, 6),   # Far service line
            (5, 7),   # Near service line
            
            # Center service lines
            (8, 9),   # Far half to near half
            
            # Singles sidelines
            (10, 11), # Left singles
            (12, 13), # Right singles (net posts)
        ]
        
        # Draw lines
        for start_idx, end_idx in line_connections:
            if start_idx < 14 and end_idx < 14:  # Only use primary keypoints
                pt1 = get_point(start_idx)
                pt2 = get_point(end_idx)
                cv2.line(image, pt1, pt2, color, thickness)
        
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Draw keypoints on all frames of a video
        
        Args:
            video_frames: List of frames
            keypoints: 1-D array of keypoint coordinates
        
        Returns:
            List of frames with keypoints drawn
        """
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
    
    def get_court_corners(self, keypoints):
        """
        Extract court corners from keypoints for homography
        Uses the 4 corner keypoints of the tennis court
        
        Args:
            keypoints: 1-D array of keypoint coordinates
        
        Returns:
            numpy array of 4 corners [(x,y), (x,y), (x,y), (x,y)]
            Order: top-left, top-right, bottom-right, bottom-left
        """
        if keypoints is None or len(keypoints) < 8:
            return None
        
        # Tennis court keypoint mapping (based on standard court detection):
        # Keypoints 0-3 are typically the baseline corners
        # 0: far-left baseline, 1: far-right baseline
        # 2: near-left baseline, 3: near-right baseline
        
        corners = np.array([
            [keypoints[0], keypoints[1]],    # Top-left (far-left baseline)
            [keypoints[2], keypoints[3]],    # Top-right (far-right baseline)
            [keypoints[6], keypoints[7]],    # Bottom-right (near-right baseline)  
            [keypoints[4], keypoints[5]]     # Bottom-left (near-left baseline)
        ], dtype=np.float32)
        
        return corners
