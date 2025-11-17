"""
Quick test to show tracking effects on manual lines
"""
import cv2
import numpy as np
from trackers.court_line_tracker import CourtLineTracker

def test_tracking_effects(video_path='copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov'):
    """Show side-by-side comparison of static vs tracked lines"""
    
    print("=" * 70)
    print("🎾 TESTING TRACKING EFFECTS")
    print("=" * 70)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read video")
        return
    
    # Initialize court line tracker with manual lines
    tracker = CourtLineTracker(manual_lines_path='court_lines_manual.json')
    tracker.update(frame)
    
    # Create two versions: static and tracked
    frame_static = frame.copy()
    frame_tracked = frame.copy()
    
    # Draw without tracking effects
    print("\n📊 Drawing static lines...")
    frame_static = tracker.draw(frame_static, show_all_lines=True, show_tracking_effects=False)
    
    # Draw with tracking effects
    print("📊 Drawing tracked lines (with effects)...")
    frame_tracked = tracker.draw(frame_tracked, show_all_lines=True, show_tracking_effects=True)
    
    # Add labels
    cv2.putText(frame_static, "STATIC LINES", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(frame_tracked, "TRACKED LINES (with effects)", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Combine side by side
    h, w = frame.shape[:2]
    comparison = np.hstack((frame_static, frame_tracked))
    
    # Save
    output_path = 'output_videos/tracking_effects_comparison.jpg'
    cv2.imwrite(output_path, comparison)
    
    print(f"\n✅ Comparison saved: {output_path}")
    print("\n" + "=" * 70)
    print("📋 TRACKING EFFECTS ADDED:")
    print("=" * 70)
    print("  ✅ Slight jitter (±1-2 pixels) - simulates real-time detection")
    print("  ✅ Shadow/glow effect - makes lines appear 3D/dynamic")
    print("  ✅ Detection markers - small circles at line endpoints")
    print("  ✅ Confidence rings - outer circles showing 'detection quality'")
    print("=" * 70)
    print("\n💡 In the video, these effects will make manual lines look like")
    print("   they're being actively tracked frame-by-frame!")
    print("=" * 70)
    
    cap.release()

if __name__ == "__main__":
    test_tracking_effects()

