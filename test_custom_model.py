"""
Quick Test Script: Verify Custom Ball Detection Model
Run this to test if your model is working correctly
"""
import os
import sys

def test_model_exists():
    """Check if best.pt model file exists"""
    print("\n1️⃣  Checking if model file exists...")
    if os.path.exists('models/best.pt'):
        size_mb = os.path.getsize('models/best.pt') / (1024 * 1024)
        print(f"   ✅ Model found: models/best.pt ({size_mb:.2f} MB)")
        return True
    else:
        print(f"   ❌ Model NOT found: models/best.pt")
        print(f"   📥 Download from: https://drive.google.com/drive/folders/1kzcLn6nF_X-Jj0O7J8RzVXSIw7G-zJNH")
        return False

def test_dependencies():
    """Check if required packages are installed"""
    print("\n2️⃣  Checking dependencies...")
    required = ['ultralytics', 'cv2', 'pandas', 'numpy']
    missing = []
    
    for package in required:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'ultralytics':
                from ultralytics import YOLO
            elif package == 'pandas':
                import pandas
            elif package == 'numpy':
                import numpy
            print(f"   ✅ {package} installed")
        except ImportError:
            print(f"   ❌ {package} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n   Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    return True

def test_model_loading():
    """Try to load the model"""
    print("\n3️⃣  Testing model loading...")
    try:
        from trackers.ball_tracker import BallTracker
        
        if os.path.exists('models/best.pt'):
            tracker = BallTracker(model_path='models/best.pt')
            print(f"   ✅ Custom model loaded successfully!")
            return True
        else:
            tracker = BallTracker(model_path='yolov8n.pt')
            print(f"   ⚠️  Loaded generic model (custom model not found)")
            return False
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False

def test_detection():
    """Test detection on a dummy frame"""
    print("\n4️⃣  Testing detection on sample frame...")
    try:
        import cv2
        import numpy as np
        from trackers.ball_tracker import BallTracker
        
        # Create a dummy frame
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Load tracker
        model_path = 'models/best.pt' if os.path.exists('models/best.pt') else 'yolov8n.pt'
        tracker = BallTracker(model_path=model_path)
        
        # Run detection
        result = tracker.detect_frame(frame, conf=0.15)
        
        print(f"   ✅ Detection test passed!")
        print(f"   Result: {result if result else 'No ball detected (expected for blank frame)'}")
        return True
    except Exception as e:
        print(f"   ❌ Detection test failed: {e}")
        return False

def test_video_exists():
    """Check if sample video exists"""
    print("\n5️⃣  Checking for sample video...")
    video_files = [f for f in os.listdir('.') if f.endswith(('.mp4', '.mov', '.avi'))]
    
    if video_files:
        print(f"   ✅ Found {len(video_files)} video file(s):")
        for v in video_files[:3]:
            print(f"      - {v}")
        if len(video_files) > 3:
            print(f"      ... and {len(video_files) - 3} more")
        return True
    else:
        print(f"   ⚠️  No video files found in current directory")
        print(f"   Add a video file to test with: python example_custom_ball_detection.py")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("🎾 TENNIS BALL DETECTION - MODEL TEST")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Model File", test_model_exists()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Detection Test", test_detection()))
    results.append(("Sample Video", test_video_exists()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Your system is ready for ball detection!")
        print("\nNext steps:")
        print("1. Run: python example_custom_ball_detection.py")
        print("2. Or integrate into your main script (see BALL_DETECTION_GUIDE.md)")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        print("See BALL_DETECTION_GUIDE.md for help")
    
    print("="*60)

if __name__ == "__main__":
    main()

