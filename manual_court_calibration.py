"""
Manual Court Calibration Tool
Click on court corners to get perfect calibration
"""
import cv2
import numpy as np
import json

# Global variables
points = []
frame = None
display_frame = None
window_name = "Manual Court Calibration"
scale_factor = 1.0  # For resizing display

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select court corners"""
    global points, frame, display_frame, scale_factor
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            # Scale coordinates back to original size
            orig_x = int(x / scale_factor)
            orig_y = int(y / scale_factor)
            points.append([orig_x, orig_y])
            print(f"Point {len(points)}: ({orig_x}, {orig_y})")
            
            # Draw point on display frame (scaled for visibility)
            cv2.circle(display_frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display_frame, str(len(points)), (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw lines between points (on display)
            if len(points) > 1:
                pt1 = (int(points[-2][0] * scale_factor), int(points[-2][1] * scale_factor))
                pt2 = (int(points[-1][0] * scale_factor), int(points[-1][1] * scale_factor))
                cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)
            
            # Close rectangle when 4 points selected
            if len(points) == 4:
                pt1 = (int(points[-1][0] * scale_factor), int(points[-1][1] * scale_factor))
                pt2 = (int(points[0][0] * scale_factor), int(points[0][1] * scale_factor))
                cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)
                print("\n✅ All 4 corners selected!")
                print("   Press 's' to save or 'r' to reset")
            
            cv2.imshow(window_name, display_frame)

def calibrate_court_manually(video_path, output_path="court_calibration.json", max_height=900):
    """
    Manual court calibration by clicking corners
    
    Args:
        video_path: Path to video file
        output_path: Where to save calibration data
        max_height: Maximum height for display window (default 900px)
        
    Returns:
        numpy array of 4 corners
    """
    global points, frame, display_frame, scale_factor
    
    print("=" * 70)
    print("🎾 MANUAL COURT CALIBRATION")
    print("=" * 70)
    print()
    print("Instructions:")
    print("1. Click on the 4 corners of the court in order:")
    print("   • Top-Left (far baseline left)")
    print("   • Top-Right (far baseline right)")
    print("   • Bottom-Right (near baseline right)")
    print("   • Bottom-Left (near baseline left)")
    print()
    print("2. Press 's' to save")
    print("3. Press 'r' to reset and start over")
    print("4. Press 'q' to quit without saving")
    print()
    print("-" * 70)
    
    # Read first frame
    cap = cv2.VideoCapture(video_path)
    ret, original_frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error: Could not read video")
        return None
    
    frame = original_frame.copy()
    points = []
    
    # Calculate scale factor to fit window on screen
    orig_height, orig_width = original_frame.shape[:2]
    if orig_height > max_height:
        scale_factor = max_height / orig_height
        display_width = int(orig_width * scale_factor)
        display_height = max_height
        display_frame = cv2.resize(original_frame, (display_width, display_height))
        print(f"📐 Resized display: {orig_width}x{orig_height} → {display_width}x{display_height}")
        print(f"   (Coordinates will be saved in original size)")
    else:
        scale_factor = 1.0
        display_frame = original_frame.copy()
    
    print()
    
    # Create window and set mouse callback
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("📹 Video loaded. Click on the court corners...")
    print()
    
    cv2.imshow(window_name, display_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Save
        if key == ord('s'):
            if len(points) == 4:
                corners = np.array(points, dtype=np.float32)
                
                # Save to JSON
                calibration_data = {
                    'corners': corners.tolist(),
                    'video': video_path,
                    'frame_size': [original_frame.shape[1], original_frame.shape[0]]
                }
                
                with open(output_path, 'w') as f:
                    json.dump(calibration_data, f, indent=2)
                
                print()
                print("=" * 70)
                print("✅ CALIBRATION SAVED!")
                print(f"   File: {output_path}")
                print()
                print("Court Corners:")
                for i, pt in enumerate(points):
                    print(f"   Corner {i}: ({pt[0]}, {pt[1]})")
                print()
                print("Usage in main script:")
                print(f"   python main_pose.py --video {video_path} \\")
                print(f"       --court-calibration {output_path}")
                print("=" * 70)
                
                cv2.destroyAllWindows()
                return corners
            else:
                print(f"⚠️  Need 4 points, only have {len(points)}")
        
        # Reset
        elif key == ord('r'):
            print("\n🔄 Resetting... Click corners again")
            points = []
            frame = original_frame.copy()
            if scale_factor != 1.0:
                display_frame = cv2.resize(original_frame, 
                    (int(orig_width * scale_factor), int(orig_height * scale_factor)))
            else:
                display_frame = original_frame.copy()
            cv2.imshow(window_name, display_frame)
        
        # Quit
        elif key == ord('q'):
            print("\n❌ Calibration cancelled")
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Manual Court Calibration')
    parser.add_argument('--video', type=str, required=True,
                       help='Input video path')
    parser.add_argument('--output', type=str, default='court_calibration.json',
                       help='Output calibration file')
    
    args = parser.parse_args()
    
    calibrate_court_manually(args.video, args.output)


