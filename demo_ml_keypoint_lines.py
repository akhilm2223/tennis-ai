"""
Demo: ML Keypoint Detection with Connected Lines
Shows how 14 ML keypoints are connected into court lines
"""
import cv2
from auto_detect_court import detect_court_automatic

# Load video
video_path = "copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error loading video")
    exit()

print("="*60)
print("🎾 ML KEYPOINT DETECTION WITH LINE CONNECTIONS")
print("="*60)

# Try with ML model (if available)
print("\n🔍 Attempting ML-based detection...")
try:
    court_detector = detect_court_automatic(
        frame, 
        use_ml_detector=True, 
        ml_model_path="models/keypoints_model.pth"
    )
    
    if court_detector.keypoints is not None:
        print(f"✅ ML keypoints detected: {len(court_detector.keypoints)//2} points")
        print(f"   Detection method: {court_detector.detection_method}")
        
        # Show which keypoints form which lines
        print("\n📍 KEYPOINT → LINE CONNECTIONS:")
        print("-"*60)
        connections = [
            ("Baseline (top)", "0 ↔ 1"),
            ("Baseline (bottom)", "2 ↔ 3"),
            ("Sideline (left)", "0 ↔ 2"),
            ("Sideline (right)", "1 ↔ 3"),
            ("Service line (top)", "4 ↔ 6"),
            ("Service line (bottom)", "5 ↔ 7"),
            ("Center service line", "8 ↔ 9"),
            ("Singles sideline (left)", "10 ↔ 11"),
            ("Singles sideline (right)", "12 ↔ 13"),
        ]
        
        for line_name, kp_connection in connections:
            print(f"   {line_name:30} → Keypoints {kp_connection}")
        
        # Create visualization
        from auto_detect_court import draw_court_lines
        
        # Draw connected lines
        viz = frame.copy()
        viz = draw_court_lines(viz, court_detector, color=(0, 255, 0), thickness=3, show_keypoints=True)
        
        # Add title
        cv2.putText(viz, "ML KEYPOINTS CONNECTED INTO COURT LINES", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(viz, f"Method: {court_detector.detection_method.upper()}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save
        output_path = "output_videos/ml_keypoints_connected.jpg"
        cv2.imwrite(output_path, viz)
        
        print(f"\n✅ Visualization saved: {output_path}")
        print("\n📊 What you'll see:")
        print("   • Green lines connecting the ML-detected keypoints")
        print("   • Small red dots showing the 14 keypoint locations")
        print("   • Complete tennis court line structure")
        
    else:
        print("⚠️  ML keypoints not available, using fallback detection")
        
except FileNotFoundError:
    print("⚠️  ML model not found: models/keypoints_model.pth")
    print("   Download from: https://drive.google.com/drive/folders/1kzcLn6nF_X-Jj0O7J8RzVXSIw7G-zJNH")
except Exception as e:
    print(f"⚠️  ML detection error: {e}")

print("\n" + "="*60)
print("💡 HOW IT WORKS:")
print("="*60)
print("1. ResNet-50 detects 14 keypoints (28 coordinates)")
print("2. Keypoints are connected based on tennis court structure")
print("3. Lines are drawn between connected keypoint pairs")
print("4. Result: Perfect court lines from ML detection!")
print("="*60)

