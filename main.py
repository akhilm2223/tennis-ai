"""
Main Tennis Match Analysis Pipeline
"""
import cv2
import numpy as np
import pickle
import os
from collections import deque
import argparse

from yolo_inference import TennisDetector
from auto_detect_court import detect_court_automatic, draw_court_lines, draw_court_corners, draw_court_status
from mini_court import MiniCourt
from trackers.kalman_tracker import BallKalmanTracker, PlayerTracker
from tennis_utils import SpeedTracker, draw_bbox, draw_circle, draw_trail, draw_stats_panel
from constants.config import (
    FPS, PIXEL_TO_METER, 
    COLOR_PLAYER1, COLOR_PLAYER2, COLOR_BALL, COLOR_TRAIL,
    SPEED_SMOOTHING_FRAMES
)


def main(video_path, output_path=None, calibrate=True, show_preview=True):
    """
    Main tennis match analysis pipeline
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video
        calibrate: Whether to perform court calibration
    """
    print("\n" + "="*60)
    print("TENNIS MATCH ANALYSIS PIPELINE")
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
        output_path = "output_videos/tennis_analysis.avi"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"   Output: {output_path}")
    
    # Initialize detector
    print("\n🔧 Initializing detectors...")
    detector = TennisDetector()
    
    # Court calibration
    court_detector = None
    if calibrate:
        print("\n🎯 Court calibration (automatic)...")
        ret, first_frame = cap.read()
        if ret:
            court_detector = detect_court_automatic(first_frame)  # Fully automatic!
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    # Initialize mini-court
    mini_court = None
    if court_detector and court_detector.homography_matrix is not None:
        mini_court = MiniCourt(court_detector)
        print("✅ Mini-court initialized")
    else:
        print("⚠ Mini-court disabled (no calibration)")
    
    # Initialize trackers
    ball_tracker = BallKalmanTracker()
    player_tracker = PlayerTracker()
    
    # Initialize speed trackers
    ball_speed_tracker = SpeedTracker(window_size=SPEED_SMOOTHING_FRAMES)
    player_speed_trackers = {0: SpeedTracker(), 1: SpeedTracker()}
    
    # History for visualization
    ball_trail = deque(maxlen=100)
    player_trails = {0: deque(maxlen=50), 1: deque(maxlen=50)}
    
    # Statistics
    frame_count = 0
    max_ball_speed = 0
    max_player_speeds = {0: 0, 1: 0}
    
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
            
            # Detect players and ball
            detections = detector.detect(frame)
            
            # Track players
            player_bboxes = detections['players']
            tracked_players = player_tracker.update(player_bboxes)
            
            # Track ball
            ball_detection = detections['ball']
            ball_center = ball_detection['center'] if ball_detection else None
            ball_position = ball_tracker.update(ball_center)
            
            # Update ball trail
            ball_trail.append(ball_position)
            
            # Calculate speeds
            ball_speed = ball_speed_tracker.update(ball_position, fps, PIXEL_TO_METER)
            max_ball_speed = max(max_ball_speed, ball_speed)
            
            player_speeds = {}
            player_centers = {}
            
            for player_id, player_data in tracked_players.items():
                center = player_data['center']
                player_centers[player_id] = center
                player_trails[player_id].append(center)
                
                # Calculate speed
                speed = player_speed_trackers[player_id].update(center, fps, PIXEL_TO_METER)
                player_speeds[player_id] = speed
                max_player_speeds[player_id] = max(max_player_speeds[player_id], speed)
            
            # Draw visualizations
            # 1. Draw players
            player_colors = [COLOR_PLAYER1, COLOR_PLAYER2]
            for player_id, player_data in tracked_players.items():
                bbox = player_data['bbox']
                speed = player_speeds.get(player_id, 0)
                color = player_colors[player_id]
                
                label = f"P{player_id+1}: {speed:.1f} km/h"
                draw_bbox(frame, bbox, color=color, label=label)
                
                # Draw player trail
                if len(player_trails[player_id]) > 1:
                    trail_color = tuple([int(c*0.7) for c in color])
                    draw_trail(frame, player_trails[player_id], color=trail_color, max_length=20)
            
            # 2. Draw ball
            if ball_position:
                draw_circle(frame, ball_position, radius=8, color=COLOR_BALL)
                
                # Ball speed label
                cv2.putText(frame, f"Ball: {ball_speed:.1f} km/h", 
                           (int(ball_position[0]) + 15, int(ball_position[1]) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BALL, 2)
            
            # 3. Draw ball trail
            if len(ball_trail) > 1:
                draw_trail(frame, ball_trail, color=COLOR_TRAIL, max_length=30)
            
            # 4. Draw statistics panel
            stats = {
                'Frame': f"{frame_count}/{total_frames}",
                'Ball Speed': f"{ball_speed:.1f} km/h",
                'Max Ball': f"{max_ball_speed:.1f} km/h",
            }
            
            for player_id in range(2):
                if player_id in player_speeds:
                    stats[f'P{player_id+1} Speed'] = f"{player_speeds[player_id]:.1f} km/h"
                    stats[f'P{player_id+1} Max'] = f"{max_player_speeds[player_id]:.1f} km/h"
            
            draw_stats_panel(frame, stats, x=10, y=30)
            
            # 5. Draw court keypoints only (if ML detection was used)
            if court_detector and court_detector.keypoints is not None:
                from trackers.court_line_detector import CourtLineDetector
                temp_detector = CourtLineDetector.__new__(CourtLineDetector)
                frame = temp_detector.draw_keypoints(frame, court_detector.keypoints, 
                                                    color=(0, 255, 0), radius=5, show_labels=False)
            
            # 6. Draw mini-court
            if mini_court:
                mini_court_img = mini_court.draw(
                    player_centers, 
                    ball_position, 
                    ball_trail=list(ball_trail)
                )
                frame = mini_court.overlay_on_frame(frame, mini_court_img, position='top_right')
            
            # Write frame
            out.write(frame)
            
            # Live preview
            if show_preview:
                cv2.imshow('Tennis Analysis (Press Q to quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠ Processing stopped by user (pressed Q)")
                    break
    
    except KeyboardInterrupt:
        print("\n⚠ Processing interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print("\n" + "-" * 60)
        print("\n📊 FINAL STATISTICS")
        print("=" * 60)
        print(f"Frames Processed: {frame_count}/{total_frames}")
        print(f"Max Ball Speed: {max_ball_speed:.1f} km/h")
        for player_id in range(2):
            print(f"Max Player {player_id+1} Speed: {max_player_speeds[player_id]:.1f} km/h")
        print("=" * 60)
        print(f"\n✅ Output saved to: {output_path}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tennis Match Analysis Pipeline')
    parser.add_argument('--video', type=str, default='input_videos/input.mp4',
                       help='Path to input video')
    parser.add_argument('--output', type=str, default='output_videos/tennis_analysis.avi',
                       help='Path to output video')
    parser.add_argument('--no-calibrate', action='store_true',
                       help='Skip court calibration')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable live preview window (faster processing)')
    
    args = parser.parse_args()
    
    # Check if default video exists, otherwise try the uploaded video
    if not os.path.exists(args.video):
        # Try to find any video in the project
        video_files = [f for f in os.listdir('.') if f.endswith(('.mp4', '.avi', '.mov'))]
        if video_files:
            args.video = video_files[0]
            print(f"Using found video: {args.video}")
        else:
            print(f"❌ Video not found: {args.video}")
            print("Please provide a video file with --video argument")
            exit(1)
    
    main(args.video, args.output, calibrate=not args.no_calibrate, show_preview=not args.no_preview)

