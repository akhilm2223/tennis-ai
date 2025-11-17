"""
Test script for Court Line Detector
Demonstrates how to use the ML-based court line detector
"""
import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackers.court_line_detector import CourtLineDetector


def test_court_line_detector(video_path, model_path, output_path=None):
    """
    Test court line detector on a video
    
    Args:
        video_path: Path to input video
        model_path: Path to trained model weights (.pt file)
        output_path: Path to save output video (optional)
    """
    print(f"🎾 Testing Court Line Detector")
    print(f"   Video: {video_path}")
    print(f"   Model: {model_path}")
    
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        print(f"   Please provide a trained ResNet50 model (.pt file)")
        return
    
    # Initialize detector
    detector = CourtLineDetector(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    
    # Setup output video writer if requested
    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"   Output: {output_path}")
    
    # Read first frame and detect keypoints
    ret, first_frame = cap.read()
    if not ret:
        print(f"❌ Error: Could not read first frame")
        cap.release()
        return
    
    print(f"\n🔍 Detecting court keypoints...")
    keypoints = detector.predict(first_frame)
    
    print(f"✅ Detected {len(keypoints)//2} keypoints:")
    for i in range(0, len(keypoints), 2):
        x, y = int(keypoints[i]), int(keypoints[i+1])
        print(f"   Point {i//2}: ({x}, {y})")
    
    # Get court corners
    corners = detector.get_court_corners(keypoints)
    print(f"\n📐 Court corners:")
    for i, corner in enumerate(corners):
        print(f"   Corner {i}: ({int(corner[0])}, {int(corner[1])})")
    
    # Process video
    print(f"\n🎬 Processing video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Draw keypoints only (no lines)
        frame_with_keypoints = detector.draw_keypoints(frame, keypoints, 
                                                       color=(255, 0, 0), 
                                                       radius=5, 
                                                       show_labels=True)
        
        # Write frame if output requested
        if out:
            out.write(frame_with_keypoints)
        
        # Show progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    
    print(f"\n✅ Processing complete!")
    print(f"   Processed {frame_count} frames")
    if output_path:
        print(f"   Output saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 3:
        print("Usage: python test_court_line_detector.py <video_path> <model_path> [output_path]")
        print("\nExample:")
        print("  python scripts/test_court_line_detector.py input_videos/tennis_input.mp4 models/court_keypoints.pt output_videos/court_lines.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    test_court_line_detector(video_path, model_path, output_path)
