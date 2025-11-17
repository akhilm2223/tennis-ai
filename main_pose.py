"""
Main Tennis Match Analysis Pipeline with MediaPipe Pose Tracking
Phase 2: Complete skeleton tracking instead of bounding boxes
"""
import sys
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import pickle
import os
from collections import deque
import argparse

# Ball detection (YOLO)
from yolo_inference import BallDetector
# Pose tracking (MediaPipe)
from modules.pose_tracker import PoseTracker
from modules.visual_skeleton import draw_skeleton, draw_bbox, draw_speed_at_centroid
# Court and visualization
from auto_detect_court import detect_court_automatic, draw_court_lines, draw_court_corners, draw_court_status
from mini_court import MiniCourt
from trackers.physics_ball_tracker import PhysicsBallTracker
from trackers.rally_analyzer import RallyAnalyzer
from trackers.court_line_tracker import CourtLineTracker
from tennis_utils import SpeedTracker, draw_circle, draw_trail, draw_stats_panel
from constants.config import (
    FPS, PIXEL_TO_METER, 
    COLOR_PLAYER1, COLOR_PLAYER2, COLOR_BALL, COLOR_TRAIL,
    SPEED_SMOOTHING_FRAMES
)


def main(video_path, output_path=None, calibrate=True, use_pose=True, show_preview=True, show_bbox=False, court_model_path=None, trigger_box=None, manual_court_calibration=None, court_lines_path=None):
    """
    Main tennis match analysis pipeline with pose tracking
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        calibrate: Whether to perform court calibration
        use_pose: Whether to use pose tracking (True) or bbox tracking (False)
        show_bbox: Whether to show bounding boxes around players
        court_model_path: Path to trained ML court line detector model (.pth)
    """
    print("\n" + "="*60)
    print("TENNIS MATCH ANALYSIS PIPELINE - GHOST TRAIL")
    print("P1 (Near): RED Skeleton | P2 (Far): BLUE Box | Ball: Ghost Trail")
    print("="*60)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video Information:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f} seconds")
    
    # Setup output
    if output_path is None:
        output_path = "output_videos/tennis_analysis_trail.mp4"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Use MP4 format for better browser compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"   Output: {output_path}")
    
    # Initialize detectors
    print("\n🔧 Initializing detectors...")
    
    # Ball detector (YOLO)
    ball_detector = BallDetector()
    print("✅ Ball detector (YOLO) initialized")
    
    # YOLO person detector for Player 2 initialization (then CSRT takes over)
    # Fix for PyTorch 2.6 weights_only issue (same as BallDetector)
    import torch
    from ultralytics import YOLO
    
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    try:
        person_detector = YOLO('yolov8n.pt')
    finally:
        torch.load = original_load
    
    print("✅ YOLO person detector initialized for Player 2 initial detection")
    
    # Pose tracker (MediaPipe) with improved stability
    # NOTE: Will be re-initialized with court_detector after calibration
    if use_pose:
        # Lower confidence (0.3) + heavy smoothing (20) = very sticky, stable tracking
        pose_tracker = PoseTracker(width, height, min_conf=0.3, smooth=20, num_players=2)
        print("✅ Pose tracker (MediaPipe) initialized with enhanced stability")
        
        # Warm up the tracker with first frame for instant detection
        print("🔥 Warming up pose detection for BOTH players...")
        ret, warm_frame = cap.read()
        if ret:
            # Process multiple times to ensure both players detected in warm-up
            for i in range(3):
                pose_L, pose_R = pose_tracker.detect_two(warm_frame)
                if pose_L and pose_R:
                    print(f"   ✅ Both players found in warm-up! (attempt {i+1}/3)")
                    break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            # Reset detection flags for actual video
            pose_tracker.is_first_detection = True
            pose_tracker.frame_count = 0
            pose_tracker.player_L_detected = False
            pose_tracker.player_R_detected = False
            
            print("✅ Tracker warmed up - ready for instant 2-player detection!")
    
    # Court calibration - Manual, ML, or Automatic
    court_detector = None
    if calibrate:
        # Use manual calibration if provided (highest priority)
        if manual_court_calibration:
            print("\n🎯 Loading manual court calibration...")
            ret, first_frame = cap.read()
            if ret:
                from auto_detect_court import CourtDetector
                court_detector = CourtDetector()
                
                # Load corners from manual calibration
                corners = np.array(manual_court_calibration['corners'], dtype=np.float32)
                court_detector.corners = corners
                court_detector.detection_method = 'manual'
                
                # Create homography matrix
                court_width = 400
                court_height = 800
                dst_corners = np.array([
                    [0, 0],
                    [court_width, 0],
                    [court_width, court_height],
                    [0, court_height]
                ], dtype=np.float32)
                
                court_detector.homography_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
                print("✅ Manual court calibration loaded!")
                print(f"   Corners: {corners.shape[0]} points")
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        else:
            # Use ML or automatic detection
            print("\n🎯 Court calibration (automatic)...")
            ret, first_frame = cap.read()
            if ret:
                # Use ML detector if model path provided
                use_ml = court_model_path is not None
                court_detector = detect_court_automatic(first_frame, use_ml_detector=use_ml, ml_model_path=court_model_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        # Re-initialize pose tracker with court info for better player filtering
        if use_pose and court_detector and court_detector.homography_matrix is not None:
            print("🔄 Re-initializing pose tracker with court boundaries...")
            pose_tracker.court_detector = court_detector
            print("   ✅ Court-aware player detection enabled!")
    
    # Initialize mini-court
    mini_court = None
    if court_detector and court_detector.homography_matrix is not None:
        mini_court = MiniCourt(court_detector)
        print("✅ Mini-court initialized")
    else:
        print("⚠ Mini-court disabled (no calibration)")
    
    # Initialize advanced physics-based ball tracker
    ball_tracker = PhysicsBallTracker(court_detector=court_detector, fps=fps)
    
    # Initialize rally analyzer for point/shot tracking
    rally_analyzer = RallyAnalyzer(court_detector=court_detector, fps=fps)
    print("✅ Rally analyzer initialized (point tracking, in/out detection)")
    
    # Initialize perfect court line tracker for stable line visualization
    court_line_tracker = CourtLineTracker(court_detector=court_detector, manual_lines_path=court_lines_path)
    if court_lines_path:
        print("✅ Court line tracker initialized with MANUAL LINES (perfect accuracy!)")
    else:
        print("✅ Court line tracker initialized (temporal smoothing, complete line structure)")
    
    # Initialize speed trackers
    ball_speed_tracker = SpeedTracker(window_size=SPEED_SMOOTHING_FRAMES)
    player_speed_trackers = {0: SpeedTracker(), 1: SpeedTracker()}
    
    # History for visualization
    ball_trail = deque(maxlen=100)
    player_trails = {0: deque(maxlen=50), 1: deque(maxlen=50)}
    
    # Track ball segments between bounces for player-colored trajectories
    current_segment = []
    ball_segments = []
    last_bounce_frame = -1
    current_hitter = None
    
    # === TRIGGER ZONE FOR BALL TRACKING ===
    # Define a box - only start tracking when ball enters this zone
    # Format: (x1, y1, x2, y2) as percentages of frame (0.0 to 1.0)
    if trigger_box is None:
        trigger_box = [0.2, 0.2, 0.8, 0.8]  # Default: center 60% of frame
    
    # Convert to pixel coordinates
    trigger_x1 = int(trigger_box[0] * width)
    trigger_y1 = int(trigger_box[1] * height)
    trigger_x2 = int(trigger_box[2] * width)
    trigger_y2 = int(trigger_box[3] * height)
    
    ball_tracking_started = False  # Flag to track if ball has entered the zone
    print(f"\n📦 Trigger Zone: ({trigger_x1}, {trigger_y1}) to ({trigger_x2}, {trigger_y2})")
    print("   Ball tracking will start when ball enters this zone")
    
    # Temporal consistency for ball detection
    last_ball_center = None
    
    # Statistics
    frame_count = 0
    max_ball_speed = 0
    max_player_speeds = {0: 0, 1: 0}
    
    # Store pose detections for mini-court
    pose_detections = {0: None, 1: None}
    
    # CSRT Tracker for Player 2 (sticky bbox tracking)
    player2_bbox_tracker = None
    player2_bbox_initialized = False
    player2_tracker_failed_count = 0
    
    print("\n▶ Processing video...")
    print("-" * 60)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)", end='\r')
            
            # === COURT TRACKING (every frame for perfect line tracking) ===
            if court_detector:
                # Update court detector (every 5 frames for performance)
                if frame_count % 5 == 0:
                    court_detector.detect_frame(frame)
                
                # Update court line tracker (every frame for smooth visualization)
                court_line_tracker.update(frame, court_detector)
            
            # === POSE TRACKING ===
            # OPTIMIZED: Only track full skeleton for Player 1 (near camera)
            # Player 2 (far camera) uses CSRT tracker for stable bbox
            if use_pose:
                # Player 1 (Near): MediaPipe full skeleton
                pose_L, _ = pose_tracker.detect_two(frame)
                pose_detections[0] = pose_L
                
                # Player 2 (Far): CSRT Tracker for sticky bbox
                # Step 1: Initialize tracker if not already done
                if not player2_bbox_initialized:
                    # Use YOLO to find Player 2 initially
                    mid_height = int(height * 0.55)
                    far_zone = frame[:mid_height, :]
                    
                    results = person_detector(far_zone, conf=0.3, classes=[0], verbose=False)
                    
                    # Find the person with largest bbox in far zone
                    best_person = None
                    max_area = 0
                    
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            area = (x2 - x1) * (y2 - y1)
                            
                            # Filter: reasonable size and position
                            if area > max_area and area > 5000:
                                max_area = area
                                best_person = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Initialize CSRT tracker with first detection
                    if best_person:
                        x1, y1, x2, y2 = best_person
                        w, h = x2 - x1, y2 - y1
                        
                        # Initialize CSRT tracker
                        player2_bbox_tracker = cv2.legacy.TrackerCSRT_create()
                        player2_bbox_tracker.init(frame, (x1, y1, w, h))
                        player2_bbox_initialized = True
                        
                        print(f"   🔒 Player 2 CSRT tracker initialized at ({x1},{y1},{x2},{y2})")
                        
                        # Create PoseDet for first frame
                        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                        from modules.pose_tracker import PoseDet
                        pose_detections[1] = PoseDet(
                            landmarks=[],
                            centroid=centroid,
                            score=1.0,
                            bbox=(float(x1), float(y1), float(x2), float(y2))
                        )
                    else:
                        pose_detections[1] = None
                
                # Step 2: Update tracker every frame after initialization
                elif player2_bbox_tracker is not None:
                    success, box = player2_bbox_tracker.update(frame)
                    
                    if success:
                        x, y, w, h = map(int, box)
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        
                        # Clamp to frame boundaries
                        x1 = max(0, min(x1, width))
                        y1 = max(0, min(y1, height))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))
                        
                        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                        
                        # Create PoseDet with tracked bbox
                        from modules.pose_tracker import PoseDet
                        pose_detections[1] = PoseDet(
                            landmarks=[],
                            centroid=centroid,
                            score=1.0,
                            bbox=(float(x1), float(y1), float(x2), float(y2))
                        )
                        
                        player2_tracker_failed_count = 0
                        
                    else:
                        # Tracker lost - try to re-initialize after 30 frames
                        player2_tracker_failed_count += 1
                        
                        if player2_tracker_failed_count > 30:
                            print(f"   ⚠️  Player 2 tracker lost, re-initializing...")
                            player2_bbox_initialized = False
                            player2_bbox_tracker = None
                            player2_tracker_failed_count = 0
                        
                        pose_detections[1] = None
                else:
                    pose_detections[1] = None
                
                # Debug: Check Player 2 detection every 30 frames
                if frame_count % 30 == 0:
                    if pose_detections[1] is not None:
                        status = "TRACKING" if player2_bbox_initialized else "INITIALIZING"
                        print(f"[Frame {frame_count}] Player 2 (CSRT): {status} ✓ | Bbox: {pose_detections[1].bbox is not None}")
                    else:
                        print(f"[Frame {frame_count}] Player 2 (CSRT): NOT DETECTED ✗")
            
            # === BALL DETECTION with Physics Tracking ===
            # Pass previous center for temporal consistency (reduces false positives)
            # BUT: if we've been predicting for a while, force full-frame search (set prev_center=None)
            use_prev_center = last_ball_center
            
            # Check if tracker is struggling (predicted for multiple frames)
            if ball_trail and len(ball_trail) > 0 and frame_count > 10:
                # Force full-frame search when struggling to re-acquire (MUCH FASTER)
                if ball_tracker.missing_frames > 5:
                    use_prev_center = None  # Full-frame search
                    
                    # Console feedback based on severity
                    if ball_tracker.missing_frames > 12 and frame_count % 30 == 0:
                        print(f"   🚨 DESPERATE MODE: Tracker lost for {ball_tracker.missing_frames} frames (conf=0.05, no filters)")
                    elif ball_tracker.missing_frames > 5 and frame_count % 30 == 0:
                        print(f"   🔍 RE-ACQUISITION: Tracker lost for {ball_tracker.missing_frames} frames (conf=0.08)")
            
            ball_detection = ball_detector.detect(frame, prev_center=use_prev_center, missing_frames=ball_tracker.missing_frames)
            ball_center = ball_detection['center'] if ball_detection else None
            ball_conf = ball_detection.get('conf', 0.0) if ball_detection else 0.0
            
            # Check if ball has entered trigger zone
            if not ball_tracking_started and ball_center:
                bx, by = ball_center
                if trigger_x1 <= bx <= trigger_x2 and trigger_y1 <= by <= trigger_y2:
                    ball_tracking_started = True
                    print(f"\n🎾 BALL ENTERED TRIGGER ZONE at frame {frame_count}!")
                    print(f"   Position: ({int(bx)}, {int(by)})")
                    print(f"   Starting ball tracking...")
            
            # Only process ball if tracking has started
            if not ball_tracking_started:
                ball_center = None  # Ignore detection until ball enters zone
            
            # STABILITY FIX 1: Accept ANY detection once tracking is active (ultra-sticky)
            # Only reject very low confidence when NOT tracking yet
            if ball_center is not None and not ball_tracker.initialized:
                # Not tracking yet - be more selective
                if ball_conf < 0.15:
                    if frame_count % 50 == 0:
                        print(f"   ⚠️ Low confidence ball ({ball_conf:.2f} < 0.15) rejected (not tracking yet) at frame {frame_count}")
                    ball_center = None
            # Once tracking: accept ANY confidence (trust Kalman filter to handle noise)
            
            # STABILITY FIX 2: Detect VERY large jumps (> 350px) and interpolate instead
            # Increased threshold to allow fast serves/smashes to be tracked
            if ball_center is not None and last_ball_center is not None and ball_tracker.initialized:
                jump_distance = np.linalg.norm(np.array(ball_center) - np.array(last_ball_center))
                if jump_distance > 350:  # More lenient - only interpolate on extreme jumps
                    # Interpolate between last and current (50/50 blend)
                    ball_center = (
                        (last_ball_center[0] + ball_center[0]) / 2,
                        (last_ball_center[1] + ball_center[1]) / 2
                    )
                    if frame_count % 50 == 0:
                        print(f"   🔧 Large jump detected ({jump_distance:.1f}px), interpolating at frame {frame_count}")
            
            # Update last_ball_center for next frame (use filtered YOLO or fall back to Kalman)
            if ball_center is not None:
                last_ball_center = ball_center
            
            # Collect player positions for bounce attribution
            player_positions = []
            for player_id in [0, 1]:  # Player 1 (near), Player 2 (far)
                if player_id in pose_detections and pose_detections[player_id] is not None:
                    centroid = pose_detections[player_id].centroid
                    player_positions.append(centroid)
                else:
                    player_positions.append(None)
            
            # Update advanced physics tracker (ENHANCED API with player positions)
            ball_result = ball_tracker.update(ball_center, frame_count, player_positions=player_positions)
            ball_position = ball_result['position']
            ball_velocity = ball_result['velocity']
            ball_confidence = ball_result['confidence']
            ball_is_bounce = ball_result['is_bounce']
            ball_is_predicted = ball_result['is_predicted']
            ball_court_position = ball_result['court_position']
            
            # Update rally analyzer with current frame data
            rally_analyzer.update(
                frame_num=frame_count,
                ball_position=ball_position,
                ball_velocity=ball_velocity,
                court_position=ball_court_position,
                bounces=ball_tracker.get_bounces(),
                player_positions=player_positions,
                is_predicted=ball_is_predicted
            )
            
            # If no YOLO but Kalman has position, update last_ball_center
            if ball_center is None and ball_position is not None:
                last_ball_center = ball_position
            
            # STABILITY FIX 3: After bounce detection, reset smoothing window to avoid spikes
            if ball_is_bounce:
                # Trim trail to last 3 points to reset motion smoothing
                if len(ball_trail) > 3:
                    ball_trail = deque(list(ball_trail)[-3:], maxlen=100)
            
            # Update ball trail (smoothed Kalman positions) - Store as (x, y) tuple ONLY
            if ball_position is not None:
                ball_trail.append(ball_position)  # Just (x, y), not (x, y, is_predicted)
                
                # Track segments between bounces for player-colored trajectories
                current_segment.append(ball_position)
                
                # When bounce detected, save segment and start new one
                if ball_is_bounce:
                    if len(current_segment) > 1:
                        # Get which player hit from bounce data
                        player_who_hit = None
                        if ball_tracker.bounces:
                            last_bounce = ball_tracker.bounces[-1]
                            player_who_hit = last_bounce.get('player')
                        
                        ball_segments.append({
                            'positions': current_segment.copy(),
                            'hit_by': current_hitter,
                            'bounce_frame': frame_count
                        })
                        
                        # Switch hitter for next segment
                        current_hitter = player_who_hit
                        current_segment = [ball_position]  # Start new segment
                    
                    last_bounce_frame = frame_count
            
            # === SPEED CALCULATIONS ===
            # Ball speed using physics tracker
            if ball_velocity:
                ball_speed = ball_tracker.get_real_speed(ball_velocity)
            else:
                ball_speed = 0.0
            max_ball_speed = max(max_ball_speed, ball_speed)
            
            # Player speeds (using centroids from pose)
            player_speeds = {}
            player_centers = {}
            
            for player_id, pose_det in pose_detections.items():
                if pose_det is not None:
                    center = pose_det.centroid
                    player_centers[player_id] = center
                    player_trails[player_id].append(center)
                    
                    # Calculate speed
                    speed = player_speed_trackers[player_id].update(center, fps, PIXEL_TO_METER)
                    player_speeds[player_id] = speed
                    max_player_speeds[player_id] = max(max_player_speeds[player_id], speed)
            
            # === VISUALIZATION ===
            
            # FIXED ASSIGNMENT:
            # Player 0 (ID 0) = near camera = RED = full skeleton
            # Player 1 (ID 1) = far camera = WHITE = bounding box only (no skeleton)
            
            player_labels = ["Player 1", "Player 2"]
            
            for player_id, pose_det in pose_detections.items():
                if pose_det is not None:
                    label = player_labels[player_id]
                    speed = player_speeds.get(player_id, 0)
                    
                    if player_id == 0:
                        # Player 1 (NEAR camera) - Full RED skeleton
                        draw_skeleton(frame, pose_det, COLOR_PLAYER1, label,
                                    draw_joints=True,
                                    draw_bones=True,
                                    draw_centroid=True,
                                    draw_bbox=False,
                                    bone_thickness=3,
                                    joint_radius=4)
                    else:
                        # Player 2 (FAR camera) - BLUE bounding box only (no skeleton, no pose)
                        if pose_det.bbox:
                            x1, y1, x2, y2 = map(int, pose_det.bbox)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue, 2px
                            cv2.putText(frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Draw player trail (no trail for Player 2 to keep it clean)
                    if player_id == 0 and len(player_trails[player_id]) > 1:
                        trail_color = tuple([int(c*0.7) for c in COLOR_PLAYER1])
                        draw_trail(frame, player_trails[player_id], color=trail_color, max_length=20)
            
            # === BALL VISUALIZATION (EXACT COPY FROM example_enhanced_ball_detection.py) ===
            
            # Player colors (broadcast style)
            PLAYER_COLORS = {
                1: (0, 0, 255),      # Player 1 (near) = RED
                2: (255, 0, 0),      # Player 2 (far) = BLUE
                None: (0, 255, 255)  # Unknown = YELLOW
            }
            
            # Draw ball trajectory as connected circles (professional broadcast style)
            num_trail_dots = min(15, len(ball_trail))  # Max 15 dots for clean visualization
            
            if len(ball_trail) > 1:
                # Sample evenly from trail
                step = max(1, len(ball_trail) // num_trail_dots)
                sampled_trail = [ball_trail[i] for i in range(0, len(ball_trail), step)]
                
                # Determine color based on current hitter
                base_color = PLAYER_COLORS.get(current_hitter, PLAYER_COLORS[None])
                
                # Draw dots with fade effect
                for i, pos in enumerate(sampled_trail):
                    # Fade effect: older dots are more transparent
                    alpha = (i + 1) / len(sampled_trail)
                    
                    # Apply fade to color
                    color = tuple(int(c * alpha) for c in base_color)
                    
                    # Determine dot size based on current frame prediction status
                    if ball_is_predicted:
                        radius = max(2, int(3 * alpha))  # Predicted: smaller
                    else:
                        radius = max(3, int(5 * alpha))  # Detected: larger
                    
                    # Draw dot
                    cv2.circle(frame, (int(pos[0]), int(pos[1])), radius, color, -1)
                    
                    # Connect dots with thin line for continuity
                    if i > 0:
                        prev_pos = sampled_trail[i-1]
                        cv2.line(frame, 
                                (int(prev_pos[0]), int(prev_pos[1])),
                                (int(pos[0]), int(pos[1])),
                                color, 1, cv2.LINE_AA)
            
            # Draw current ball position (larger, player-colored)
            if ball_position:
                x, y = int(ball_position[0]), int(ball_position[1])
                
                # Color based on current hitter and detection type
                if ball_is_bounce:
                    ball_color = (255, 0, 255)  # MAGENTA for bounce
                    radius = 14
                    label_text = "BOUNCE!"
                else:
                    # Use player color
                    ball_color = PLAYER_COLORS.get(current_hitter, PLAYER_COLORS[None])
                    radius = 10 if not ball_is_predicted else 8
                    label_text = f"{'P1' if current_hitter == 1 else 'P2' if current_hitter == 2 else 'Ball'}"
                
                # Draw ball with white outline
                cv2.circle(frame, (x, y), radius, ball_color, -1)
                cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 2)
                
                # Draw label
                status = "PREDICTED" if ball_is_predicted else "DETECTED"
                label = f"{label_text} | {status}"
                cv2.putText(frame, label, (x + 15, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2)
            
            # Draw raw YOLO detection (for comparison) - thin gray rectangle
            if ball_detection and ball_detection.get('bbox'):
                x1, y1, x2, y2 = map(int, ball_detection['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
            
            # Trigger zone works silently in background (no visualization)
            # You can add --trigger-box argument to customize the zone
            
            # Draw stats panel (enhanced with rally info)
            detection_rate = (len([e for e in ball_trail if e]) / frame_count * 100) if frame_count > 0 else 0
            
            tracking_status = "ACTIVE" if ball_tracking_started else "WAITING"
            
            # Get current rally info
            rally_info = rally_analyzer.get_live_rally_info()
            
            stats_text = [
                f"Frame: {frame_count}",
                f"Tracking: {tracking_status}",
                f"Ball Trail: {len(ball_trail)} dots",
                f"Detection Rate: {detection_rate:.0f}%",
                f"Bounces: {len(ball_tracker.bounces)}",
                f"Model: Custom best.pt"
            ]
            
            # Add rally info if active
            if rally_info:
                stats_text.extend([
                    "",  # Blank line
                    f"Rally #{rally_info['rally_number']}",
                    f"Shots: {rally_info['shots']}",
                    f"Score: P1 {rally_info['score'][1]} - {rally_info['score'][2]} P2"
                ])
            
            y_offset = 30
            for i, text in enumerate(stats_text):
                if text:  # Skip empty lines for positioning
                    cv2.putText(frame, text, (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, text, (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # 4. Draw legend at bottom
            legend_y = height - 60
            legend_x = 10
            
            # Legend background
            cv2.rectangle(frame, (legend_x, legend_y - 5), 
                         (legend_x + 400, legend_y + 45), (0, 0, 0), -1)
            cv2.rectangle(frame, (legend_x, legend_y - 5), 
                         (legend_x + 400, legend_y + 45), (255, 255, 255), 1)
            
            # Legend items
            cv2.putText(frame, "Ball Path Legend:", (legend_x + 5, legend_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            # Red dots (P1)
            cv2.circle(frame, (legend_x + 5, legend_y + 27), 4, (0, 0, 255), -1)
            cv2.putText(frame, "P1 Shot", (legend_x + 15, legend_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Blue dots (P2)
            cv2.circle(frame, (legend_x + 85, legend_y + 27), 4, (255, 0, 0), -1)
            cv2.putText(frame, "P2 Shot", (legend_x + 95, legend_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Yellow dots (Unknown)
            cv2.circle(frame, (legend_x + 165, legend_y + 27), 4, (0, 255, 255), -1)
            cv2.putText(frame, "Unknown", (legend_x + 175, legend_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Yellow bounce (larger)
            cv2.circle(frame, (legend_x + 255, legend_y + 27), 5, (0, 255, 255), -1)
            cv2.circle(frame, (legend_x + 255, legend_y + 27), 5, (0, 0, 255), 2)
            cv2.putText(frame, "Bounce", (legend_x + 265, legend_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 6. Draw perfect court lines with temporal smoothing
            if court_line_tracker.initialized:
                frame = court_line_tracker.draw(frame, show_all_lines=True, show_labels=False)
            
            # Court status (optional, clean visual)
            # Uncomment below to show calibration status
            # frame = draw_court_status(frame, court_detector, show_status=True)
            
            # 6. Draw mini-court (ball tracker only - no players + bounce markers)
            # Use court_position if available, otherwise fall back to image position
            if mini_court:
                # Format ball position properly for mini-court
                if ball_court_position:
                    # Court coordinates available - pass as dict with court_xy
                    mini_court_ball_pos = {
                        'court_xy': ball_court_position,
                        'image_xy': ball_position
                    }
                elif ball_position:
                    # Only image coordinates available
                    mini_court_ball_pos = {
                        'image_xy': ball_position
                    }
                else:
                    mini_court_ball_pos = None
                
                # ball_trail now stores just (x, y) tuples, so use directly
                trail_positions = list(ball_trail) if ball_trail else []
                
                # Get all bounces from tracker
                all_bounces = ball_tracker.get_bounces()
                
                # Debug: Show bounce count every 30 frames
                if frame_count % 30 == 0 and len(all_bounces) > 0:
                    print(f"   🎯 Frame {frame_count}: Passing {len(all_bounces)} bounces to mini-court")
                    if len(all_bounces) > 0:
                        print(f"      Last bounce: {all_bounces[-1]}")
                
                mini_court_img = mini_court.draw(
                    {},  # Empty dict - no players shown
                    mini_court_ball_pos, 
                    ball_trail=trail_positions,
                    bounces=all_bounces  # Pass bounce data to mini-court
                )
                frame = mini_court.overlay_on_frame(frame, mini_court_img, position='top_right')
            
            # Draw point winner announcement (if any)
            frame = rally_analyzer.draw_point_announcement(frame)
            
            # Write frame
            out.write(frame)
            
            # Live preview
            if show_preview:
                cv2.imshow('Tennis Analysis - 2 Player Tracking (Press Q to quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠ Processing stopped by user (pressed Q)")
                    break
    
    except KeyboardInterrupt:
        print("\n⚠ Processing interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        if use_pose:
            pose_tracker.close()
        cv2.destroyAllWindows()
        
        # === CLEAN MINIMAL JSON EXPORT ===
        # Export only essential match intelligence without circular references
        import json
        
        # Force end current rally if still active
        rally_analyzer.force_end_current_rally(frame_count)
        
        ball_bounces = ball_tracker.get_bounces()
        rally_breakdown = rally_analyzer.get_rally_breakdown()
        
        # Extract score
        score_p1 = rally_breakdown.get('score', {}).get(1, 0)
        score_p2 = rally_breakdown.get('score', {}).get(2, 0)
        
        # Build clean rally summary (no nested objects or circular refs)
        rallies_summary = []
        for rally in rally_breakdown.get('rallies', []):
            rallies_summary.append({
                "rally_number": rally.get('rally_id', 0),
                "start_frame": rally.get('start_frame', 0),
                "end_frame": rally.get('end_frame', 0),
                "shots": rally.get('total_shots', 0),
                "point_winner": rally.get('winner', None),
                "reason": rally.get('outcome', 'Unknown'),
                "max_ball_speed_kmh": float(rally.get('max_speed_kmh', 0.0))
            })
        
        # Build clean bounce list
        bounces_clean = []
        for b in ball_bounces:
            court_xy = b.get("court_xy")
            bounces_clean.append({
                "frame": int(b.get("frame", 0)),
                "court_xy": [float(court_xy[0]), float(court_xy[1])] if court_xy else None,
                "speed_kmh": float(b.get("speed_kmh", 0.0)),
                "player": int(b.get("player")) if b.get("player") else None
            })
        
        # Calculate player statistics
        player_analysis = rally_breakdown.get('player_analysis', {})
        
        def get_player_stat(player_id, key, default=0.0):
            """Safely extract player stat"""
            val = player_analysis.get(player_id, {}).get(key, default)
            if isinstance(val, (np.generic, np.ndarray)):
                return float(val)
            return float(val) if val else default
        
        # Build clean analysis data structure
        analysis_data = {
            "video_info": {
                "filename": video_path,
                "fps": int(fps),
                "duration_seconds": float(total_frames / fps),
                "total_frames": int(total_frames),
                "processed_frames": int(frame_count)
            },
            
            "score": {
                "player_1": int(score_p1),
                "player_2": int(score_p2)
            },
            
            "rallies": rallies_summary,
            
            "bounces": bounces_clean,
            
            "player_stats": {
                "1": {
                    "avg_shot_speed_kmh": get_player_stat(1, 'avg_shot_speed_kmh', 0.0),
                    "max_shot_speed_kmh": get_player_stat(1, 'max_shot_speed_kmh', 0.0),
                    "winners": int(get_player_stat(1, 'winners', 0)),
                    "errors": int(get_player_stat(1, 'unforced_errors', 0) + get_player_stat(1, 'forced_errors', 0)),
                    "points_won": int(score_p1),
                    "total_shots": int(get_player_stat(1, 'total_shots', 0))
                },
                "2": {
                    "avg_shot_speed_kmh": get_player_stat(2, 'avg_shot_speed_kmh', 0.0),
                    "max_shot_speed_kmh": get_player_stat(2, 'max_shot_speed_kmh', 0.0),
                    "winners": int(get_player_stat(2, 'winners', 0)),
                    "errors": int(get_player_stat(2, 'unforced_errors', 0) + get_player_stat(2, 'forced_errors', 0)),
                    "points_won": int(score_p2),
                    "total_shots": int(get_player_stat(2, 'total_shots', 0))
                }
            },
            
            "match_summary": {
                "total_bounces": len(bounces_clean),
                "total_rallies": len(rallies_summary),
                "max_ball_speed_kmh": float(max_ball_speed),
                "max_player1_speed_kmh": float(max_player_speeds[0]),
                "max_player2_speed_kmh": float(max_player_speeds[1])
            }
        }
        
        # Save clean JSON
        json_output_path = output_path.replace('.avi', '.json').replace('.mp4', '.json')
        
        with open(json_output_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\n📄 Saved Match Analysis JSON → {json_output_path}")
        
        print("\n" + "-" * 60)
        print("\n📊 FINAL MATCH STATISTICS")
        print("=" * 60)
        print(f"Frames Processed: {frame_count}/{total_frames}")
        print(f"Tracking Mode: {'OPTIMIZED (P1: MediaPipe, P2: CSRT Tracker)' if use_pose else 'BBOX (YOLO)'}")
        print(f"Match Duration: {total_frames / fps:.1f} seconds")
        
        print(f"\n🏆 FINAL SCORE:")
        print(f"   Player 1: {score_p1} points")
        print(f"   Player 2: {score_p2} points")
        print(f"   {'Player 1 WINS!' if score_p1 > score_p2 else 'Player 2 WINS!' if score_p2 > score_p1 else 'TIED!'}")
        
        print(f"\n🎾 Ball Tracking:")
        print(f"   Total Bounces: {len(bounces_clean)}")
        print(f"   Max Ball Speed: {max_ball_speed:.1f} km/h")
        
        print(f"\n📊 Rally Breakdown:")
        print(f"   Total Rallies: {len(rallies_summary)}")
        if rallies_summary:
            max_shots = max([r['shots'] for r in rallies_summary])
            avg_shots = sum([r['shots'] for r in rallies_summary]) / len(rallies_summary)
            print(f"   Longest Rally: {max_shots} shots")
            print(f"   Average Rally Length: {avg_shots:.1f} shots")
        
        print(f"\n👤 PLAYER 1 STATS:")
        p1 = analysis_data['player_stats']['1']
        print(f"   Points Won: {p1['points_won']}")
        print(f"   Total Shots: {p1['total_shots']}")
        print(f"   Winners: {p1['winners']}")
        print(f"   Errors: {p1['errors']}")
        print(f"   Avg Shot Speed: {p1['avg_shot_speed_kmh']:.1f} km/h")
        print(f"   Max Shot Speed: {p1['max_shot_speed_kmh']:.1f} km/h")
        print(f"   Max Movement Speed: {max_player_speeds[0]:.1f} km/h")
        
        print(f"\n👤 PLAYER 2 STATS:")
        p2 = analysis_data['player_stats']['2']
        print(f"   Points Won: {p2['points_won']}")
        print(f"   Total Shots: {p2['total_shots']}")
        print(f"   Winners: {p2['winners']}")
        print(f"   Errors: {p2['errors']}")
        print(f"   Avg Shot Speed: {p2['avg_shot_speed_kmh']:.1f} km/h")
        print(f"   Max Shot Speed: {p2['max_shot_speed_kmh']:.1f} km/h")
        print(f"   Max Movement Speed: {max_player_speeds[1]:.1f} km/h")
        
        print("\n" + "=" * 60)
        print(f"✅ Video Output: {output_path}")
        print(f"✅ JSON Analysis: {json_output_path}")
        print("=" * 60)
        print("\n🎾 Match analysis complete! 🏆")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tennis Match Analysis Pipeline - Pose Tracking')
    parser.add_argument('--video', type=str, default='input_videos/tennis_input.mp4',
                       help='Path to input video')
    parser.add_argument('--output', type=str, default='output_videos/tennis_analysis_trail.avi',
                       help='Path to output video with ghost ball trail')
    parser.add_argument('--no-calibrate', action='store_true',
                       help='Skip court calibration')
    parser.add_argument('--no-pose', action='store_true',
                       help='Use YOLO bbox tracking instead of pose')
    parser.add_argument('--trigger-box', nargs=4, type=float, metavar=('X1', 'Y1', 'X2', 'Y2'),
                       default=[0.2, 0.2, 0.8, 0.8],
                       help='Trigger zone for ball tracking (x1 y1 x2 y2 as fractions 0.0-1.0). Default: 0.2 0.2 0.8 0.8')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable live preview window (faster processing)')
    parser.add_argument('--bbox', action='store_true',
                       help='Show bounding boxes around players')
    parser.add_argument('--court-model', type=str, default=None,
                       help='Path to trained court line detector model (.pth file) for ML-based detection')
    parser.add_argument('--court-calibration', type=str, default=None,
                       help='Path to manual court calibration file (.json) for perfect accuracy')
    parser.add_argument('--court-lines', type=str, default=None,
                       help='Path to manual court lines definition file (.json) with all line positions')
    
    args = parser.parse_args()
    
    # Check if default video exists, otherwise try to find video
    if not os.path.exists(args.video):
        # Try to find any video in input_videos
        if os.path.exists('input_videos'):
            video_files = [f for f in os.listdir('input_videos') if f.endswith(('.mp4', '.avi', '.mov'))]
            if video_files:
                args.video = os.path.join('input_videos', video_files[0])
                print(f"Using found video: {args.video}")
        
        if not os.path.exists(args.video):
            print(f"❌ Video not found: {args.video}")
            print("Please provide a video file with --video argument")
            exit(1)
    
    # Load manual calibration if provided
    manual_calibration = None
    if args.court_calibration:
        import json
        try:
            with open(args.court_calibration, 'r') as f:
                manual_calibration = json.load(f)
            print(f"✅ Loaded manual court calibration from: {args.court_calibration}")
        except Exception as e:
            print(f"⚠️  Could not load calibration file: {e}")
    
    main(args.video, args.output,
         calibrate=not args.no_calibrate,
         use_pose=not args.no_pose,
         show_preview=not args.no_preview,
         show_bbox=args.bbox,
         court_model_path=args.court_model,
         trigger_box=args.trigger_box,
         manual_court_calibration=manual_calibration,
         court_lines_path=args.court_lines)

