"""
Verification script for MediaPipe Pose integration
Tests that pose tracking module loads correctly
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test all necessary imports"""
    print("🔍 Testing imports...")
    
    try:
        import cv2
        print("   ✅ OpenCV")
    except ImportError:
        print("   ❌ OpenCV - install with: pip install opencv-python")
        return False
    
    try:
        import mediapipe as mp
        print("   ✅ MediaPipe")
    except ImportError:
        print("   ❌ MediaPipe - install with: pip install mediapipe")
        return False
    
    try:
        import numpy as np
        print("   ✅ NumPy")
    except ImportError:
        print("   ❌ NumPy - install with: pip install numpy")
        return False
    
    return True


def test_pose_tracker():
    """Test PoseTracker module"""
    print("\n🔍 Testing PoseTracker module...")
    
    try:
        from modules.pose_tracker import PoseTracker, PoseDet
        print("   ✅ Module imports successful")
    except ImportError as e:
        print(f"   ❌ Failed to import: {e}")
        return False
    
    try:
        import numpy as np
        # Create dummy frame
        img = np.zeros((720, 1280, 3), np.uint8)
        
        # Initialize tracker
        pt = PoseTracker(1280, 720, min_conf=0.5, smooth=5)
        print("   ✅ PoseTracker initialized")
        
        # Test detection (will return None on blank image)
        pose_L, pose_R = pt.detect_two(img)
        print("   ✅ detect_two() method works")
        
        # Cleanup
        pt.close()
        print("   ✅ PoseTracker cleanup successful")
        
        return True
    except Exception as e:
        print(f"   ❌ Error during testing: {e}")
        return False


def test_visual_skeleton():
    """Test visual_skeleton module"""
    print("\n🔍 Testing visual_skeleton module...")
    
    try:
        from modules.visual_skeleton import draw_skeleton, draw_skeleton_simple
        print("   ✅ Module imports successful")
        
        import numpy as np
        import cv2
        
        # Create test frame
        frame = np.zeros((720, 1280, 3), np.uint8)
        
        # Test with None pose (should not crash)
        draw_skeleton(frame, None, (0, 0, 255), "P1")
        print("   ✅ draw_skeleton() handles None gracefully")
        
        return True
    except Exception as e:
        print(f"   ❌ Error during testing: {e}")
        return False


def test_mediapipe_pose():
    """Test MediaPipe Pose directly"""
    print("\n🔍 Testing MediaPipe Pose model...")
    
    try:
        import mediapipe as mp
        import numpy as np
        import cv2
        
        # Initialize MediaPipe Pose
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("   ✅ MediaPipe Pose model initialized")
        
        # Test with blank image
        img = np.zeros((480, 640, 3), np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        print("   ✅ Pose processing works")
        
        pose.close()
        print("   ✅ MediaPipe Pose cleanup successful")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Run all verification tests"""
    print("="*60)
    print("MEDIAPIPE POSE INTEGRATION - VERIFICATION")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    if results[-1][1]:  # Only continue if imports succeeded
        results.append(("MediaPipe Pose", test_mediapipe_pose()))
        results.append(("PoseTracker Module", test_pose_tracker()))
        results.append(("Visual Skeleton Module", test_visual_skeleton()))
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou're ready to run:")
        print("   python main.py --video input_videos/tennis_input.mp4")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above.")
        print("Make sure to install: pip install mediapipe")
    
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

