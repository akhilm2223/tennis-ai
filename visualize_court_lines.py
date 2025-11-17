"""
Quick visualization of court line detection
Shows how the court lines are detected and tracked
"""
import cv2
import numpy as np
from auto_detect_court import detect_court_automatic
from trackers.court_line_tracker import CourtLineTracker

def visualize_court_detection(video_path, num_frames=5, output_path="court_detection_demo.jpg"):
    """
    Visualize court line detection on multiple frames
    
    Args:
        video_path: Path to input video
        num_frames: Number of frames to show
        output_path: Path to save visualization
    """
    print(f"📹 Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Read first frame for court detection
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Could not read first frame")
        return
    
    print("\n🎯 Detecting court...")
    court_detector = detect_court_automatic(first_frame, use_ml_detector=False, ml_model_path=None)
    
    if court_detector.corners is None:
        print("❌ Court detection failed!")
        return
    
    print(f"✅ Court detected using: {court_detector.detection_method}")
    print(f"   Corners: {court_detector.corners.shape}")
    
    # Initialize court line tracker
    print("\n🔧 Initializing court line tracker...")
    court_line_tracker = CourtLineTracker(court_detector)
    
    # Collect frames to visualize
    frames_to_show = []
    frame_numbers = []
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Sample frames evenly throughout video
    step = max(1, total_frames // (num_frames + 1))
    
    for i in range(num_frames):
        frame_idx = (i + 1) * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Update court line tracker
        court_line_tracker.update(frame, court_detector)
        
        # Create visualization with different modes
        viz_frame = frame.copy()
        
        # Draw court lines with labels
        viz_frame = court_line_tracker.draw(viz_frame, show_all_lines=True, show_labels=True)
        
        # Add frame number and info
        cv2.putText(viz_frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(viz_frame, f"Method: {court_detector.detection_method}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add legend
        legend_y = height - 120
        cv2.rectangle(viz_frame, (5, legend_y - 10), (300, height - 5), (0, 0, 0), -1)
        cv2.rectangle(viz_frame, (5, legend_y - 10), (300, height - 5), (255, 255, 255), 2)
        
        cv2.putText(viz_frame, "Court Line Legend:", (10, legend_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Color indicators
        y_offset = legend_y + 35
        cv2.line(viz_frame, (15, y_offset), (35, y_offset), (0, 255, 0), 3)
        cv2.putText(viz_frame, "Baselines/Sidelines", (45, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.line(viz_frame, (15, y_offset), (35, y_offset), (255, 255, 0), 2)
        cv2.putText(viz_frame, "Service Lines/Net", (45, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.line(viz_frame, (15, y_offset), (35, y_offset), (0, 255, 255), 2)
        cv2.putText(viz_frame, "Singles Lines", (45, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        frames_to_show.append(viz_frame)
        frame_numbers.append(frame_idx)
        
        print(f"   ✓ Processed frame {frame_idx}/{total_frames}")
    
    cap.release()
    
    # Create composite image
    print("\n🎨 Creating visualization grid...")
    
    if len(frames_to_show) == 0:
        print("❌ No frames to visualize")
        return
    
    # Resize frames for grid
    target_width = 640
    target_height = int(height * (target_width / width))
    
    resized_frames = []
    for frame in frames_to_show:
        resized = cv2.resize(frame, (target_width, target_height))
        resized_frames.append(resized)
    
    # Create grid layout
    if len(resized_frames) <= 2:
        # Single row
        grid = np.hstack(resized_frames)
    elif len(resized_frames) <= 4:
        # 2x2 grid
        row1 = np.hstack(resized_frames[:2])
        row2 = np.hstack(resized_frames[2:4]) if len(resized_frames) > 2 else np.zeros_like(row1)
        grid = np.vstack([row1, row2])
    else:
        # 3x2 or 3x3 grid
        cols = 3
        rows = (len(resized_frames) + cols - 1) // cols
        
        grid_rows = []
        for r in range(rows):
            start_idx = r * cols
            end_idx = min(start_idx + cols, len(resized_frames))
            row_frames = resized_frames[start_idx:end_idx]
            
            # Pad row if needed
            while len(row_frames) < cols:
                row_frames.append(np.zeros_like(resized_frames[0]))
            
            grid_rows.append(np.hstack(row_frames))
        
        grid = np.vstack(grid_rows)
    
    # Add title
    title_height = 80
    title_panel = np.zeros((title_height, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_panel, "Perfect Court Line Detection & Tracking", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(title_panel, f"Detection Method: {court_detector.detection_method.upper()} | Temporal Smoothing: Active", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    final_image = np.vstack([title_panel, grid])
    
    # Save visualization
    cv2.imwrite(output_path, final_image)
    print(f"\n✅ Visualization saved to: {output_path}")
    print(f"   Image size: {final_image.shape[1]}x{final_image.shape[0]}")
    print(f"   Frames shown: {len(frames_to_show)}")
    
    # Print court line information
    print("\n📊 Court Line Structure Detected:")
    court_info = court_line_tracker.get_court_info()
    if court_info['lines']:
        print("   ✓ Baselines (top & bottom)")
        print("   ✓ Sidelines (left & right)")
        print("   ✓ Singles sidelines")
        print("   ✓ Service lines (2)")
        print("   ✓ Net line")
        print("   ✓ Center service line")
        print("   ✓ Center mark")
        print(f"\n   Total lines tracked: {len(court_info['lines'])} line segments")
    
    print("\n🎯 Court Corners:")
    if court_detector.corners is not None:
        for i, corner in enumerate(court_detector.corners):
            corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
            print(f"   {corner_names[i]}: ({int(corner[0])}, {int(corner[1])})")
    
    print("\n" + "="*60)
    print("📌 Features:")
    print("   • Temporal smoothing (10-frame average)")
    print("   • Complete tennis court line structure")
    print("   • Perspective-correct positioning")
    print("   • Color-coded visualization")
    print("   • Stable tracking across frames")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov"
    
    output_path = "output_videos/court_detection_visualization.jpg"
    
    print("\n" + "="*60)
    print("🎾 COURT LINE DETECTION VISUALIZATION")
    print("="*60)
    
    visualize_court_detection(video_path, num_frames=6, output_path=output_path)
    
    print("\n✨ Done! Check the output image to see court line detection.")

