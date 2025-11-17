"""
Visualize Court Keypoint/Corner Detection
Shows the actual detected points that define the court
"""
import cv2
import numpy as np
from auto_detect_court import detect_court_automatic

def visualize_court_keypoints(video_path, num_frames=6, output_path="court_keypoints_demo.jpg"):
    """
    Visualize detected court keypoints/corners
    
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
    
    print("\n🎯 Detecting court keypoints...")
    court_detector = detect_court_automatic(first_frame, use_ml_detector=False, ml_model_path=None)
    
    if court_detector.corners is None:
        print("❌ Court detection failed!")
        return
    
    print(f"✅ Court detected using: {court_detector.detection_method}")
    print(f"   Corners detected: {court_detector.corners.shape[0]} points")
    
    # Collect frames to visualize
    frames_to_show = []
    
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
        
        # Update court detection for this frame
        court_detector.detect_frame(frame)
        
        # Create visualization showing keypoints
        viz_frame = frame.copy()
        
        # Draw court boundary polygon (light outline)
        if court_detector.corners is not None:
            corners_int = court_detector.corners.astype(np.int32)
            cv2.polylines(viz_frame, [corners_int], isClosed=True, 
                         color=(100, 100, 100), thickness=2, lineType=cv2.LINE_AA)
        
        # Draw keypoints with labels
        if court_detector.corners is not None:
            corner_names = ["TL", "TR", "BR", "BL"]  # Top-Left, Top-Right, Bottom-Right, Bottom-Left
            colors = [
                (0, 255, 255),    # Yellow for TL
                (0, 255, 0),      # Green for TR
                (255, 0, 0),      # Blue for BR
                (255, 0, 255)     # Magenta for BL
            ]
            
            for idx, (corner, name, color) in enumerate(zip(court_detector.corners, corner_names, colors)):
                x, y = int(corner[0]), int(corner[1])
                
                # Draw large circle for keypoint
                cv2.circle(viz_frame, (x, y), 15, color, -1)  # Filled circle
                cv2.circle(viz_frame, (x, y), 17, (255, 255, 255), 3)  # White outline
                cv2.circle(viz_frame, (x, y), 15, (0, 0, 0), 2)  # Black inner outline
                
                # Draw label
                label_offset_x = -40 if idx in [0, 3] else 20  # Left for TL/BL, right for TR/BR
                label_offset_y = -20 if idx in [0, 1] else 30   # Up for TL/TR, down for BR/BL
                
                label_pos = (x + label_offset_x, y + label_offset_y)
                
                # Background for label
                (label_w, label_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(viz_frame, 
                            (label_pos[0] - 5, label_pos[1] - label_h - 5),
                            (label_pos[0] + label_w + 5, label_pos[1] + 5),
                            (0, 0, 0), -1)
                
                cv2.putText(viz_frame, name, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Draw coordinates
                coord_text = f"({x},{y})"
                coord_pos = (x + label_offset_x, y + label_offset_y + 25)
                
                (coord_w, coord_h), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(viz_frame,
                            (coord_pos[0] - 3, coord_pos[1] - coord_h - 3),
                            (coord_pos[0] + coord_w + 3, coord_pos[1] + 3),
                            (0, 0, 0), -1)
                
                cv2.putText(viz_frame, coord_text, coord_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw ML keypoints if available
        if hasattr(court_detector, 'keypoints') and court_detector.keypoints is not None:
            print(f"   Frame {frame_idx}: {len(court_detector.keypoints)} ML keypoints detected")
            
            for kp_idx, keypoint in enumerate(court_detector.keypoints):
                x, y = int(keypoint[0]), int(keypoint[1])
                
                # Draw small keypoint
                cv2.circle(viz_frame, (x, y), 6, (0, 255, 0), -1)
                cv2.circle(viz_frame, (x, y), 8, (255, 255, 255), 2)
                
                # Draw keypoint number
                cv2.putText(viz_frame, str(kp_idx + 1), (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add frame info
        cv2.putText(viz_frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(viz_frame, f"Detection: {court_detector.detection_method.upper()}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        num_points = court_detector.corners.shape[0]
        if hasattr(court_detector, 'keypoints') and court_detector.keypoints is not None:
            num_points += len(court_detector.keypoints)
        
        cv2.putText(viz_frame, f"Keypoints: {num_points}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add legend
        legend_y = height - 150
        cv2.rectangle(viz_frame, (5, legend_y - 10), (300, height - 5), (0, 0, 0), -1)
        cv2.rectangle(viz_frame, (5, legend_y - 10), (300, height - 5), (255, 255, 255), 2)
        
        cv2.putText(viz_frame, "Court Keypoints:", (10, legend_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Color-coded legend
        y_offset = legend_y + 40
        legend_items = [
            ("TL", (0, 255, 255), "Top-Left"),
            ("TR", (0, 255, 0), "Top-Right"),
            ("BR", (255, 0, 0), "Bottom-Right"),
            ("BL", (255, 0, 255), "Bottom-Left")
        ]
        
        for label, color, desc in legend_items:
            cv2.circle(viz_frame, (20, y_offset), 8, color, -1)
            cv2.circle(viz_frame, (20, y_offset), 10, (255, 255, 255), 2)
            cv2.putText(viz_frame, f"{label}: {desc}", (40, y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_offset += 25
        
        frames_to_show.append(viz_frame)
        print(f"   ✓ Processed frame {frame_idx}/{total_frames}")
    
    cap.release()
    
    # Create composite image
    print("\n🎨 Creating keypoint visualization grid...")
    
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
    
    # Create grid layout (3 columns)
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
    cv2.putText(title_panel, "Court Keypoint Detection (Corner Points)", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(title_panel, f"Method: {court_detector.detection_method.upper()} | 4 Main Corners Tracked", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    final_image = np.vstack([title_panel, grid])
    
    # Save visualization
    cv2.imwrite(output_path, final_image)
    print(f"\n✅ Keypoint visualization saved to: {output_path}")
    print(f"   Image size: {final_image.shape[1]}x{final_image.shape[0]}")
    print(f"   Frames shown: {len(frames_to_show)}")
    
    # Print detected keypoints
    print("\n📍 Detected Court Keypoints:")
    print("="*60)
    
    corner_names_full = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    if court_detector.corners is not None:
        for i, (corner, name) in enumerate(zip(court_detector.corners, corner_names_full)):
            print(f"   {i+1}. {name:15} → ({int(corner[0]):4}, {int(corner[1]):4})")
    
    print("\n" + "="*60)
    print("🎯 Detection Details:")
    print(f"   • Method: {court_detector.detection_method}")
    print(f"   • Main corners: {court_detector.corners.shape[0]}")
    if hasattr(court_detector, 'keypoints') and court_detector.keypoints is not None:
        print(f"   • ML keypoints: {len(court_detector.keypoints)}")
    print(f"   • Homography: {'✓ Available' if court_detector.homography_matrix is not None else '✗ Not available'}")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov"
    
    output_path = "output_videos/court_keypoints_visualization.jpg"
    
    print("\n" + "="*60)
    print("🎯 COURT KEYPOINT DETECTION VISUALIZATION")
    print("="*60)
    
    visualize_court_keypoints(video_path, num_frames=6, output_path=output_path)
    
    print("\n✨ Done! Check the output image to see detected keypoints.")

