"""
Test court line detection to diagnose issues
"""
import cv2
import os
from trackers.court_line_detector import CourtLineDetector

print("=" * 70)
print("🎾 COURT LINE DETECTION TEST")
print("=" * 70)
print()

# Check if model exists
model_path = "models/keypoints_model.pth"
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("Please ensure the model file is in the models/ directory")
    exit(1)

print(f"✅ Model found: {model_path}")
print()

# Check if video exists
video_path = "copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov"
if not os.path.exists(video_path):
    print(f"❌ Video not found: {video_path}")
    exit(1)

print(f"✅ Video found: {video_path}")
print()

# Initialize detector
print("🔧 Initializing court line detector...")
try:
    detector = CourtLineDetector(model_path)
    print("✅ Detector initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize detector: {e}")
    exit(1)

print()

# Read first frame
print("📹 Reading first frame from video...")
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to read frame from video")
    exit(1)

print(f"✅ Frame read successfully ({frame.shape[1]}x{frame.shape[0]})")
print()

# Detect keypoints
print("🔍 Detecting court keypoints...")
try:
    keypoints = detector.predict(frame)
    if keypoints is None:
        print("❌ Detection returned None")
        exit(1)
    print(f"✅ Detected {len(keypoints)//2} keypoints ({len(keypoints)} values)")
except Exception as e:
    print(f"❌ Detection failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# Print keypoints
print("📊 DETECTED KEYPOINTS:")
print("-" * 70)
for i in range(0, min(28, len(keypoints)), 2):
    x, y = keypoints[i], keypoints[i+1]
    kp_num = i // 2
    print(f"   Keypoint {kp_num:2d}: ({x:7.2f}, {y:7.2f})")

if len(keypoints) > 28:
    print("\nCalculated Midpoints:")
    for i in range(28, len(keypoints), 2):
        x, y = keypoints[i], keypoints[i+1]
        kp_num = i // 2
        print(f"   Midpoint {kp_num:2d}: ({x:7.2f}, {y:7.2f})")

print()

# Extract corners
print("📐 Extracting court corners...")
try:
    corners = detector.get_court_corners(keypoints)
    if corners is None:
        print("❌ Failed to extract corners")
    else:
        print(f"✅ Extracted {len(corners)} corners")
        print()
        print("CORNERS:")
        for i, corner in enumerate(corners):
            print(f"   Corner {i}: ({corner[0]:7.2f}, {corner[1]:7.2f})")
except Exception as e:
    print(f"❌ Corner extraction failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Draw keypoints
print("🎨 Drawing keypoints on frame...")
try:
    output_frame = frame.copy()
    output_frame = detector.draw_keypoints(output_frame, keypoints, color=(0, 255, 0), radius=8, show_labels=True)
    
    # Save output
    output_path = "output_videos/court_detection_test.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output_frame)
    print(f"✅ Output saved to: {output_path}")
except Exception as e:
    print(f"❌ Drawing failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)


