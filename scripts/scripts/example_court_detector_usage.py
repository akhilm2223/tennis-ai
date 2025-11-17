"""
Example: How to use the Court Line Detector in your tennis analysis
"""
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trackers.court_line_detector import CourtLineDetector
from auto_detect_court import detect_court_automatic, draw_court_lines


def example_basic_usage():
    """Example 1: Basic keypoint detection"""
    print("=" * 60)
    print("Example 1: Basic Keypoint Detection")
    print("=" * 60)
    
    # Load a frame
    video_path = "input_videos/tennis_input.mp4"
    if not os.path.exists(video_path):
        print(f"⚠ Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read frame")
        return
    
    # Initialize detector (requires trained model)
    model_path = "models/court_keypoints.pt"
    if not os.path.exists(model_path):
        print(f"⚠ Model not found: {model_path}")
        print("   You need a trained ResNet50 model to use ML detection")
        return
    
    detector = CourtLineDetector(model_path)
    
    # Predict keypoints
    keypoints = detector.predict(frame)
    print(f"✅ Detected {len(keypoints)//2} keypoints")
    
    # Draw keypoints only (no lines)
    frame_with_keypoints = detector.draw_keypoints(frame, keypoints, 
                                                   color=(255, 0, 0),
                                                   radius=8,
                                                   show_labels=True)
    
    # Save result
    output_path = "output_videos/example_keypoints.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame_with_keypoints)
    print(f"✅ Saved to: {output_path}")


def example_integrated_detection():
    """Example 2: Using integrated auto-detection"""
    print("\n" + "=" * 60)
    print("Example 2: Integrated Auto-Detection")
    print("=" * 60)
    
    # Load a frame
    video_path = "input_videos/tennis_input.mp4"
    if not os.path.exists(video_path):
        print(f"⚠ Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read frame")
        return
    
    # Method 1: Traditional detection (no model needed)
    print("\n📍 Method 1: Traditional Detection")
    court_detector_traditional = detect_court_automatic(frame)
    print(f"   Detection method: {court_detector_traditional.detection_method}")
    print(f"   Corners detected: {court_detector_traditional.corners is not None}")
    
    # Method 2: ML-based detection (requires model)
    model_path = "models/court_keypoints.pt"
    if os.path.exists(model_path):
        print("\n📍 Method 2: ML-Based Detection")
        court_detector_ml = detect_court_automatic(frame,
                                                   use_ml_detector=True,
                                                   ml_model_path=model_path)
        print(f"   Detection method: {court_detector_ml.detection_method}")
        print(f"   Keypoints detected: {court_detector_ml.keypoints is not None}")
        print(f"   Corners detected: {court_detector_ml.corners is not None}")
        
        # Draw comparison
        frame_traditional = draw_court_lines(frame.copy(), court_detector_traditional,
                                            color=(0, 255, 0), thickness=2)
        frame_ml = draw_court_lines(frame.copy(), court_detector_ml,
                                    color=(0, 255, 255), thickness=2)
        
        # Save comparison
        comparison = cv2.hconcat([frame_traditional, frame_ml])
        output_path = "output_videos/example_comparison.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, comparison)
        print(f"✅ Comparison saved to: {output_path}")
    else:
        print(f"\n⚠ ML model not found: {model_path}")
        print("   Only traditional detection available")


def example_keypoint_usage():
    """Example 3: Using keypoints for custom analysis"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Keypoint Analysis")
    print("=" * 60)
    
    # Load a frame
    video_path = "input_videos/tennis_input.mp4"
    model_path = "models/court_keypoints.pt"
    
    if not os.path.exists(video_path):
        print(f"⚠ Video not found: {video_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"⚠ Model not found: {model_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read frame")
        return
    
    # Detect keypoints
    detector = CourtLineDetector(model_path)
    keypoints = detector.predict(frame)
    
    # Extract specific court features
    print("\n📐 Court Features:")
    
    # Get corners
    corners = detector.get_court_corners(keypoints)
    print(f"   Court corners: {len(corners)} points")
    
    # Calculate court dimensions (in pixels)
    if len(corners) >= 4:
        width = abs(corners[1][0] - corners[0][0])
        height = abs(corners[2][1] - corners[0][1])
        print(f"   Court width: {width:.1f} pixels")
        print(f"   Court height: {height:.1f} pixels")
        print(f"   Aspect ratio: {height/width:.2f}")
    
    # Service line positions (keypoints 4-7)
    if len(keypoints) >= 16:
        service_line_y = (keypoints[9] + keypoints[11]) / 2  # Average y of points 4 and 5
        print(f"   Service line Y: {service_line_y:.1f}")
    
    # Net position (keypoints 8-11)
    if len(keypoints) >= 24:
        net_y = (keypoints[17] + keypoints[19]) / 2  # Average y of points 8 and 9
        print(f"   Net Y: {net_y:.1f}")
    
    print("\n💡 You can use these keypoints to:")
    print("   - Calculate ball position relative to court lines")
    print("   - Detect if ball is in/out")
    print("   - Track player positions on court")
    print("   - Measure shot distances and angles")


if __name__ == "__main__":
    print("\n🎾 Court Line Detector Examples\n")
    
    # Run examples
    example_basic_usage()
    example_integrated_detection()
    example_keypoint_usage()
    
    print("\n" + "=" * 60)
    print("✅ Examples complete!")
    print("=" * 60)
