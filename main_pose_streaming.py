"""
Modified main_pose.py with streaming progress updates
Sends frame-by-frame updates via callback for real-time UI updates
"""
import cv2
import numpy as np
import os
from collections import deque

from yolo_inference import BallDetector
from modules.pose_tracker import PoseTracker
from modules.visual_skeleton import draw_skeleton, draw_bbox
from auto_detect_court import detect_court_automatic, draw_court_lines, draw_court_status
from mini_court import MiniCourt
from trackers.physics_ball_tracker import PhysicsBallTracker
from tennis_utils import SpeedTracker, draw_circle, draw_trail, draw_stats_panel
from constants.config import (
    FPS, PIXEL_TO_METER, 
    COLOR_PLAYER1, COLOR_PLAYER2, COLOR_BALL, COLOR_TRAIL,
    SPEED_SMOOTHING_FRAMES
)


def main_with_callback(video_path, output_path=None, calibrate=True, use_pose=True, 
                       show_bbox=False, progress_callback=None):
    """
    Main tennis analysis with progress callback for streaming updates
    
    Args:
        progress_callback: Function(frame_num, total_frames, frame_image) called each frame
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if progress_callback:
            progress_callback(-1, 0, None, error="Could not open video")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output
    if output_path is None:
        output_path = "output_videos/tennis_analysis_pose.mp4"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Use MP4 format for better browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize detectors
    ball_detector = BallDetector()
    pose_tracker = PoseTracker(width, height, min_conf=0.3, smooth=20, num_players=2)
    
    # Warm up
    ret, warm_frame = cap.read()
    if ret:
        for i in range(3):
            pose_L, pose_R = pose_tracker.detect_two(warm_frame)
            if pose_L and pose_R:
                break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pose_tracker.is_first_detection = True
        pose_tracker.frame_count = 0
    
    # Court calibration
    court_detector = None
    use_ml_court = False  # Set to True to use ML-based court detection
    ml_model_path = "models/court_keypoints.pt"  # Path to trained model
    
    if calibrate:
        ret, first_frame = cap.read()
        if ret:
            # Try ML detection if model exists
            if use_ml_court and os.path.exists(ml_model_path):
                court_detector = detect_court_automatic(first_frame, 
                                                       use_ml_detector=True,
                                                       ml_model_path=ml_model_path)
            else:
                court_detector = detect_court_automatic(first_frame)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if use_pose and court_detector:
                pose_tracker.court_detector = court_detector
    
    # Initialize mini-court
    mini_court = None
    if court_detector and court_detector.homography_matrix is not None:
        mini_court = MiniCourt(court_detector)
    
    # Initialize advanced physics-based ball tracker
    ball_tracker = PhysicsBallTracker(court_detector=court_detector, fps=fps)
    ball_speed_tracker = SpeedTracker(window_size=SPEED_SMOOTHING_FRAMES)
    player_speed_trackers = {0: SpeedTracker(), 1: SpeedTracker()}
    
    # History
    ball_trail = deque(maxlen=100)
    player_trails = {0: deque(maxlen=50), 1: deque(maxlen=50)}
    
    # Statistics
    frame_count = 0
    max_ball_speed = 0
    max_player_speeds = {0: 0, 1: 0}
    pose_detections = {0: None, 1: None}
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Pose tracking
            if use_pose:
                pose_L, pose_R = pose_tracker.detect_two(frame)
                pose_detections[0] = pose_L
                pose_detections[1] = pose_R
            
            # Ball detection with advanced physics tracking
            ball_detection = ball_detector.detect(frame)
            ball_center = ball_detection['center'] if ball_detection else None
            
            # Update physics tracker
            ball_result = ball_tracker.update(ball_center, frame_count)
            ball_position = ball_result['position']
            ball_velocity = ball_result['velocity']
            ball_confidence = ball_result['confidence']
            ball_trail.append(ball_position)
            
            # Speed calculations using physics tracker
            if ball_velocity:
                ball_speed = ball_tracker.get_real_speed(ball_velocity)
            else:
                ball_speed = 0.0
            max_ball_speed = max(max_ball_speed, ball_speed)
            
            player_speeds = {}
            player_centers = {}
            
            for player_id, pose_det in pose_detections.items():
                if pose_det is not None:
                    center = pose_det.centroid
                    player_centers[player_id] = center
                    player_trails[player_id].append(center)
                    speed = player_speed_trackers[player_id].update(center, fps, PIXEL_TO_METER)
                    player_speeds[player_id] = speed
                    max_player_speeds[player_id] = max(max_player_speeds[player_id], speed)
            
            # Visualization
            player_sizes = {}
            for player_id, pose_det in pose_detections.items():
                if pose_det is not None and pose_det.bbox is not None:
                    x1, y1, x2, y2 = pose_det.bbox
                    area = (x2 - x1) * (y2 - y1)
                    player_sizes[player_id] = area
            
            near_player_id = None
            far_player_id = None
            if len(player_sizes) == 2:
                near_player_id = max(player_sizes, key=player_sizes.get)
                far_player_id = min(player_sizes, key=player_sizes.get)
            elif len(player_sizes) == 1:
                near_player_id = list(player_sizes.keys())[0]
            
            player_colors = [COLOR_PLAYER1, COLOR_PLAYER2]
            player_labels = ["P1", "P2"]
            
            for player_id, pose_det in pose_detections.items():
                if pose_det is not None:
                    color = player_colors[player_id]
                    label = player_labels[player_id]
                    speed = player_speeds.get(player_id, 0)
                    is_near = (player_id == near_player_id)
                    
                    if is_near:
                        draw_skeleton(frame, pose_det, color, label,
                                    draw_joints=True, draw_bones=True,
                                    draw_centroid=True, draw_bbox=False,
                                    bone_thickness=3, joint_radius=4)
                        # Speed label removed - clean visual
                        pass
                    else:
                        draw_bbox(frame, pose_det, color, label, thickness=3)
                        # Speed label removed - clean visual
                        pass
                    
                    if len(player_trails[player_id]) > 1:
                        trail_color = tuple([int(c*0.7) for c in color])
                        draw_trail(frame, player_trails[player_id], color=trail_color, max_length=20)
            
            # Draw ball
            if ball_position:
                draw_circle(frame, ball_position, radius=8, color=COLOR_BALL)
            
            if len(ball_trail) > 1:
                draw_trail(frame, ball_trail, color=COLOR_TRAIL, max_length=30)
            
            # Stats panel removed - clean visual
            
            # Court keypoints (only if ML detection was used)
            if court_detector and court_detector.keypoints is not None:
                # Draw only keypoints, no lines
                from trackers.court_line_detector import CourtLineDetector
                temp_detector = CourtLineDetector.__new__(CourtLineDetector)
                frame = temp_detector.draw_keypoints(frame, court_detector.keypoints, 
                                                    color=(0, 255, 0), radius=5, show_labels=False)
            # Court status removed - clean visual
            
            # Mini-court
            if mini_court:
                mini_court_img = mini_court.draw({}, ball_position, ball_trail=list(ball_trail))
                frame = mini_court.overlay_on_frame(frame, mini_court_img, position='top_right')
            
            # Write frame
            out.write(frame)
            
            # Send progress update with frame preview
            if progress_callback and frame_count % 5 == 0:  # Every 5 frames
                progress_callback(frame_count, total_frames, frame)
    
    finally:
        cap.release()
        out.release()
        if use_pose:
            pose_tracker.close()
    
    # Collect analysis data
    return {
        'frames_processed': frame_count,
        'fps': fps,
        'max_ball_speed': max_ball_speed,
        'max_player_speeds': max_player_speeds,
        'bounces': [(b[0], b[1], b[2]) for b in ball_tracker.bounces] if hasattr(ball_tracker, 'bounces') else [],
        'ball_trajectory': [(p[0], p[1], p[2]) for p in ball_tracker.trajectory] if hasattr(ball_tracker, 'trajectory') else [],
        'court_detected': court_detector is not None and court_detector.homography_matrix is not None
    }
