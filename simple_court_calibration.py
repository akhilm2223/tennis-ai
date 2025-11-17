"""
Simple Court Calibration - Just Click 4 Corners in Order
No console input needed!
"""
import cv2
import numpy as np
import json

# Global variables
points = []
frame = None
original_frame = None
scale_factor = 1.0
window_name = "Court Calibration - Click 4 Corners"

CORNER_NAMES = [
    "Top-Left (far baseline left)",
    "Top-Right (far baseline right)",
    "Bottom-Right (near baseline right)",
    "Bottom-Left (near baseline left)"
]

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Green, Red, Yellow

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks - just click in order!"""
    global points, frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append([x, y])
            corner_idx = len(points) - 1
            
            # Draw point with corner number
            color = COLORS[corner_idx]
            cv2.circle(frame, (x, y), 10, color, -1)
            cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)  # White outline
            
            # Draw label
            label = f"{corner_idx + 1}"
            cv2.putText(frame, label, (x - 20, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(frame, CORNER_NAMES[corner_idx][:12], (x + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw line to previous point
            if len(points) > 1:
                cv2.line(frame, tuple(points[-2]), tuple(points[-1]), color, 2)
            
            # Close the rectangle when 4 points done
            if len(points) == 4:
                cv2.line(frame, tuple(points[3]), tuple(points[0]), COLORS[3], 2)
                
                # Draw success message
                msg_y = 50
                cv2.rectangle(frame, (10, msg_y - 35), (frame.shape[1] - 10, msg_y + 10),
                             (0, 255, 0), -1)
                cv2.putText(frame, "All 4 corners selected! Press 's' to SAVE",
                           (20, msg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            print(f"✓ Point {len(points)}: {CORNER_NAMES[corner_idx]} at ({x}, {y})")
            
            if len(points) < 4:
                print(f"  → Next: Click {CORNER_NAMES[len(points)]}")
            else:
                print("\n✅ All corners selected! Press 's' to save")
            
            cv2.imshow(window_name, frame)

def calibrate_simple(video_path, output_path="court_calibration.json"):
    """Simple calibration - just click 4 corners in order"""
    global points, frame, original_frame, scale_factor
    
    print("=" * 70)
    print("🎾 SIMPLE COURT CALIBRATION")
    print("=" * 70)
    print()
    print("Just click on the 4 corners IN ORDER:")
    print()
    print("  1️⃣  Top-Left (far baseline left)")
    print("  2️⃣  Top-Right (far baseline right)")
    print("  3️⃣  Bottom-Right (near baseline right)")
    print("  4️⃣  Bottom-Left (near baseline left)")
    print()
    print("⚠️  If bottom corners are cut off, estimate their position!")
    print()
    print("Press 's' to save, 'r' to reset, 'q' to quit")
    print("-" * 70)
    print()
    
    # Read first frame
    cap = cv2.VideoCapture(video_path)
    ret, original_frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error: Could not read video")
        return None
    
    # Resize to fit screen (max 1200x800)
    original_h, original_w = original_frame.shape[:2]
    max_width = 1200
    max_height = 800
    
    scale = min(max_width / original_w, max_height / original_h, 1.0)
    if scale < 1.0:
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        display_frame = cv2.resize(original_frame, (new_w, new_h))
        scale_factor = scale
        print(f"📐 Resized: {original_w}x{original_h} → {new_w}x{new_h} (scale={scale:.2f})")
    else:
        display_frame = original_frame.copy()
        scale_factor = 1.0
    
    print()
    print("👆 Click corner #1 (Top-Left)...")
    print()
    
    frame = display_frame.copy()
    points = []
    
    # Draw helpful grid
    h, w = frame.shape[:2]
    for i in range(5):
        y = int(h * i / 4)
        cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)
        x = int(w * i / 4)
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)
    
    # Draw instructions on frame
    cv2.putText(frame, "Click corners 1 -> 2 -> 3 -> 4", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 's' to save | 'r' to reset | 'q' to quit", (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Create window
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Save
        if key == ord('s'):
            if len(points) == 4:
                # Scale back to original coordinates
                scaled_points = np.array([[int(p[0] / scale_factor), int(p[1] / scale_factor)] 
                                         for p in points], dtype=np.float32)
                
                print()
                print("=" * 70)
                print("✅ CALIBRATION SAVED!")
                print()
                print("Court Corners (original coordinates):")
                for i, pt in enumerate(scaled_points):
                    print(f"  {i+1}. {CORNER_NAMES[i]}: ({int(pt[0])}, {int(pt[1])})")
                
                # Save to JSON
                calibration_data = {
                    'corners': scaled_points.tolist(),
                    'video': video_path,
                    'frame_size': [original_w, original_h]
                }
                
                with open(output_path, 'w') as f:
                    json.dump(calibration_data, f, indent=2)
                
                print()
                print(f"💾 Saved to: {output_path}")
                print()
                print("Usage:")
                print(f"  python main_pose.py --video {video_path} \\")
                print(f"      --court-calibration {output_path}")
                print("=" * 70)
                
                cv2.destroyAllWindows()
                return scaled_points
            else:
                print(f"⚠️  Need 4 points, only have {len(points)}")
                print(f"    Click {4 - len(points)} more corners")
        
        # Reset
        elif key == ord('r'):
            print("\n🔄 Resetting... Click corner #1 again")
            points = []
            frame = display_frame.copy()
            
            # Redraw grid
            for i in range(5):
                y = int(h * i / 4)
                cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)
                x = int(w * i / 4)
                cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)
            
            # Redraw instructions
            cv2.putText(frame, "Click corners 1 -> 2 -> 3 -> 4", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 's' to save | 'r' to reset | 'q' to quit", 
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, frame)
        
        # Quit
        elif key == ord('q'):
            print("\n❌ Calibration cancelled")
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Court Calibration')
    parser.add_argument('--video', type=str, required=True,
                       help='Input video path')
    parser.add_argument('--output', type=str, default='court_calibration.json',
                       help='Output calibration file')
    
    args = parser.parse_args()
    
    calibrate_simple(args.video, args.output)


