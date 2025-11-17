"""
Skeleton Visualization for MediaPipe Pose
Renders complete player skeletons with bones and joints
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple
from modules.pose_tracker import PoseDet

# MediaPipe pose connections (bone pairs)
POSE_CONN = mp.solutions.pose.POSE_CONNECTIONS


def draw_bbox(frame: np.ndarray,
              pose_det: Optional[PoseDet],
              color: Tuple[int, int, int],
              label: str,
              thickness: int = 3):
    """
    Draw bounding box around player
    
    Args:
        frame: Image to draw on
        pose_det: PoseDet object with bbox
        color: BGR color for box
        label: Player label
        thickness: Line thickness
    """
    if pose_det is None or pose_det.bbox is None:
        return
    
    x1, y1, x2, y2 = pose_det.bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label with background
    label_text = f"{label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    label_thickness = 2
    
    # Get text size
    text_size = cv2.getTextSize(label_text, font, font_scale, label_thickness)[0]
    
    # Draw label background
    label_y = y1 - 10
    if label_y < text_size[1] + 10:
        label_y = y1 + text_size[1] + 10
    
    cv2.rectangle(frame, 
                  (x1, label_y - text_size[1] - 5), 
                  (x1 + text_size[0] + 10, label_y + 5), 
                  color, -1)
    
    # Draw label text
    cv2.putText(frame, label_text, (x1 + 5, label_y), 
                font, font_scale, (255, 255, 255), label_thickness)


def draw_skeleton(frame: np.ndarray, 
                  pose_det: Optional[PoseDet], 
                  color: Tuple[int, int, int], 
                  label: str,
                  draw_joints: bool = True,
                  draw_bones: bool = True,
                  draw_centroid: bool = True,
                  draw_bbox: bool = False,
                  bone_thickness: int = 2,
                  joint_radius: int = 3):
    """
    Draw complete skeleton on frame
    
    Args:
        frame: Image to draw on
        pose_det: PoseDet object with landmarks
        color: BGR color for skeleton
        label: Player label (e.g., "P1")
        draw_joints: Whether to draw joint circles
        draw_bones: Whether to draw bone lines
        draw_centroid: Whether to draw centroid marker
        draw_bbox: Whether to draw bounding box
        bone_thickness: Thickness of bone lines
        joint_radius: Radius of joint circles
    """
    if pose_det is None:
        return
    
    landmarks = pose_det.landmarks
    
    # Draw bounding box if requested
    if draw_bbox and pose_det.bbox is not None:
        x1, y1, x2, y2 = pose_det.bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw bones (connections between joints)
    if draw_bones:
        for connection in POSE_CONN:
            start_idx, end_idx = connection
            
            # Get landmark coordinates
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            # Check visibility
            if start_lm[3] > 0.5 and end_lm[3] > 0.5:
                start_pt = (int(start_lm[0]), int(start_lm[1]))
                end_pt = (int(end_lm[0]), int(end_lm[1]))
                
                # Draw bone
                cv2.line(frame, start_pt, end_pt, color, bone_thickness)
    
    # Draw joints
    if draw_joints:
        for lm in landmarks:
            if lm[3] > 0.5:  # Check visibility
                center = (int(lm[0]), int(lm[1]))
                cv2.circle(frame, center, joint_radius, color, -1)
                # Draw thin border for better visibility
                cv2.circle(frame, center, joint_radius, (255, 255, 255), 1)
    
    # Draw centroid
    if draw_centroid and pose_det.centroid:
        cx, cy = int(pose_det.centroid[0]), int(pose_det.centroid[1])
        
        # Draw larger filled circle for centroid
        cv2.circle(frame, (cx, cy), 8, color, -1)
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)
        
        # Draw label
        cv2.putText(frame, label, (cx - 20, cy - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw confidence score
        conf_text = f"{pose_det.score:.2f}"
        cv2.putText(frame, conf_text, (cx - 20, cy + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_skeleton_simple(frame: np.ndarray, 
                         pose_det: Optional[PoseDet], 
                         color: Tuple[int, int, int], 
                         label: str):
    """
    Simplified skeleton drawing (just bones and centroid)
    
    Args:
        frame: Image to draw on
        pose_det: PoseDet object
        color: BGR color
        label: Player label
    """
    draw_skeleton(frame, pose_det, color, label, 
                 draw_joints=False, 
                 draw_bones=True, 
                 draw_centroid=True,
                 bone_thickness=2)


def draw_skeleton_detailed(frame: np.ndarray, 
                           pose_det: Optional[PoseDet], 
                           color: Tuple[int, int, int], 
                           label: str):
    """
    Detailed skeleton drawing (bones + joints + centroid)
    
    Args:
        frame: Image to draw on
        pose_det: PoseDet object
        color: BGR color
        label: Player label
    """
    draw_skeleton(frame, pose_det, color, label, 
                 draw_joints=True, 
                 draw_bones=True, 
                 draw_centroid=True,
                 bone_thickness=3,
                 joint_radius=4)


def highlight_keypoints(frame: np.ndarray,
                       pose_det: Optional[PoseDet],
                       keypoint_ids: list,
                       color: Tuple[int, int, int],
                       radius: int = 6,
                       label_keypoints: bool = False):
    """
    Highlight specific keypoints (e.g., wrists for racket tracking)
    
    Args:
        frame: Image to draw on
        pose_det: PoseDet object
        keypoint_ids: List of landmark IDs to highlight
        color: BGR color
        radius: Circle radius
        label_keypoints: Whether to label each keypoint
    """
    if pose_det is None:
        return
    
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    landmarks = pose_det.landmarks
    
    for kp_id in keypoint_ids:
        if kp_id >= len(landmarks):
            continue
        
        lm = landmarks[kp_id]
        if lm[3] > 0.5:  # Check visibility
            center = (int(lm[0]), int(lm[1]))
            
            # Draw highlighted circle
            cv2.circle(frame, center, radius, color, -1)
            cv2.circle(frame, center, radius + 2, (255, 255, 255), 2)
            
            # Optional label
            if label_keypoints and kp_id < len(landmark_names):
                label = landmark_names[kp_id].replace('_', ' ')
                cv2.putText(frame, label, (center[0] + 10, center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def draw_speed_at_centroid(frame: np.ndarray,
                           pose_det: Optional[PoseDet],
                           speed: float,
                           color: Tuple[int, int, int],
                           label: str):
    """
    Draw speed label at player centroid
    
    Args:
        frame: Image to draw on
        pose_det: PoseDet object
        speed: Speed in km/h
        color: BGR color
        label: Player label
    """
    if pose_det is None or pose_det.centroid is None:
        return
    
    cx, cy = int(pose_det.centroid[0]), int(pose_det.centroid[1])
    
    # Draw speed text
    speed_text = f"{label}: {speed:.1f} km/h"
    
    # Add background for readability
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    text_size = cv2.getTextSize(speed_text, font, font_scale, thickness)[0]
    
    # Background rectangle
    padding = 5
    bg_x1 = cx - text_size[0] // 2 - padding
    bg_y1 = cy - 35 - text_size[1] - padding
    bg_x2 = cx + text_size[0] // 2 + padding
    bg_y2 = cy - 35 + padding
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw text
    text_x = cx - text_size[0] // 2
    text_y = cy - 35
    cv2.putText(frame, speed_text, (text_x, text_y), font, font_scale, color, thickness)

