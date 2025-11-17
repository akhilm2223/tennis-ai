"""
Partial Court Calibration Tool
Works when only 2-3 corners are visible (common in tennis videos)
"""
import cv2
import numpy as np
import json

# Global variables
points = []
point_types = []  # Track which corners these are
frame = None
scale_factor = 1.0  # Display scale factor
window_name = "Partial Court Calibration"
instructions_visible = True

CORNER_NAMES = {
    0: "Top-Left (far baseline left)",
    1: "Top-Right (far baseline right)",
    2: "Bottom-Right (near baseline right)",
    3: "Bottom-Left (near baseline left)"
}

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select visible court corners"""
    global points, point_types, frame
    
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        
        # Ask which corner this is (via console)
        print(f"\nPoint {len(points)} clicked at: ({x}, {y})")
        print("Which corner is this?")
        print("  0 = Top-Left (far baseline left)")
        print("  1 = Top-Right (far baseline right)")
        print("  2 = Bottom-Right (near baseline right)")  
        print("  3 = Bottom-Left (near baseline left)")
        print("  9 = Skip/Undo this point")
        
        while True:
            choice = input("Enter corner number: ").strip()
            if choice in ['0', '1', '2', '3']:
                corner_idx = int(choice)
                point_types.append(corner_idx)
                
                # Draw point
                color = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)][corner_idx]
                cv2.circle(frame, (x, y), 10, color, -1)
                cv2.putText(frame, CORNER_NAMES[corner_idx][:8], 
                           (x+15, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                print(f"✓ Marked as: {CORNER_NAMES[corner_idx]}")
                break
            elif choice == '9':
                points.pop()
                print("Point skipped")
                break
            else:
                print("Invalid choice. Enter 0, 1, 2, 3, or 9")
        
        # Draw lines between points
        if len(points) > 1:
            cv2.line(frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 255), 2)
        
        if len(points) >= 2:
            print(f"\n✓ {len(points)} corners marked")
            print("  Press 's' to save (2+ points needed)")
            print("  Press 'r' to reset")
        
        cv2.imshow(window_name, frame)

def estimate_missing_corners(points, point_types, frame_shape):
    """
    Estimate positions of missing corners based on visible ones
    Uses perspective geometry and typical court proportions
    """
    h, w = frame_shape[:2]
    
    # Create full 4-corner array
    corners = np.zeros((4, 2), dtype=np.float32)
    
    # Fill in known corners
    for i, (pt, pt_type) in enumerate(zip(points, point_types)):
        corners[pt_type] = pt
    
    # Estimate missing corners
    known_mask = np.zeros(4, dtype=bool)
    for pt_type in point_types:
        known_mask[pt_type] = True
    
    # Simple estimation strategies
    if not known_mask[0] and known_mask[1]:  # Missing top-left, have top-right
        # Estimate based on top-right and typical court width
        if known_mask[2] or known_mask[3]:
            # Use bottom corners if available
            if known_mask[3]:
                # Mirror from bottom-left
                corners[0] = [corners[3][0], corners[1][1]]
            elif known_mask[2]:
                # Estimate from top-right and bottom-right
                dx = corners[2][0] - corners[1][0]
                corners[0] = [corners[1][0] - dx, corners[1][1]]
        else:
            # Just guess left edge
            corners[0] = [50, corners[1][1]]
    
    if not known_mask[1] and known_mask[0]:  # Missing top-right, have top-left
        if known_mask[2]:
            corners[1] = [corners[2][0], corners[0][1]]
        elif known_mask[3]:
            dx = corners[0][0] - corners[3][0]
            corners[1] = [corners[0][0] + dx, corners[0][1]]
        else:
            corners[1] = [w - 50, corners[0][1]]
    
    if not known_mask[2] and (known_mask[1] or known_mask[0]):  # Missing bottom-right
        if known_mask[1]:
            corners[2] = [corners[1][0], h - 50]
        elif known_mask[0]:
            corners[2] = [w - 50, h - 50]
    
    if not known_mask[3] and (known_mask[0] or known_mask[2]):  # Missing bottom-left
        if known_mask[0]:
            corners[3] = [corners[0][0], h - 50]
        elif known_mask[2]:
            corners[3] = [50, corners[2][1]]
    
    return corners, known_mask

def calibrate_partial_court(video_path, output_path="court_calibration.json"):
    """
    Calibrate court when only some corners are visible
    """
    global points, point_types, frame, scale_factor
    
    print("=" * 70)
    print("🎾 PARTIAL COURT CALIBRATION")
    print("=" * 70)
    print()
    print("This tool works when bottom corners are CUT OFF or not visible!")
    print()
    print("Instructions:")
    print("1. Click on ANY visible court corners (at least 2)")
    print("2. For each click, specify which corner it is (0-3)")
    print("3. Missing corners will be estimated automatically")
    print()
    print("Corner Numbers:")
    print("  0 = Top-Left (far baseline left)")
    print("  1 = Top-Right (far baseline right)")
    print("  2 = Bottom-Right (near baseline right)")
    print("  3 = Bottom-Left (near baseline left)")
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
    
    # Resize frame to fit on screen (max 1200x800)
    original_h, original_w = original_frame.shape[:2]
    max_width = 1200
    max_height = 800
    
    scale = min(max_width / original_w, max_height / original_h, 1.0)
    if scale < 1.0:
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        display_frame = cv2.resize(original_frame, (new_w, new_h))
        print(f"📐 Resizing video: {original_w}x{original_h} → {new_w}x{new_h} (scale={scale:.2f})")
    else:
        display_frame = original_frame.copy()
        scale = 1.0
    
    frame = display_frame.copy()
    points = []
    point_types = []
    scale_factor = scale  # Store globally for coordinate scaling
    
    # Draw grid to help identify corners
    h, w = frame.shape[:2]
    for i in range(5):
        y = int(h * i / 4)
        cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)
        x = int(w * i / 4)
        cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)
    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("📹 Click on visible court corners...")
    print("(Bottom corners may be cut off - that's OK!)")
    print()
    
    cv2.imshow(window_name, frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Save
        if key == ord('s'):
            if len(points) >= 2:
                # Scale points back to original coordinates
                scaled_points = [[int(p[0] / scale_factor), int(p[1] / scale_factor)] 
                                for p in points]
                
                # Estimate missing corners (using original frame dimensions)
                full_corners, known_mask = estimate_missing_corners(
                    scaled_points, point_types, original_frame.shape
                )
                
                print()
                print("=" * 70)
                print("✅ CALIBRATION CREATED!")
                print()
                print("Known Corners (original coordinates):")
                for i, (pt, pt_type) in enumerate(zip(scaled_points, point_types)):
                    print(f"  ✓ {CORNER_NAMES[pt_type]}: ({pt[0]}, {pt[1]})")
                
                print()
                print("Estimated Corners:")
                for i in range(4):
                    if not known_mask[i]:
                        print(f"  ~ {CORNER_NAMES[i]}: ({int(full_corners[i][0])}, {int(full_corners[i][1])})")
                
                # Save calibration
                calibration_data = {
                    'corners': full_corners.tolist(),
                    'known_corners': known_mask.tolist(),
                    'video': video_path,
                    'frame_size': [original_w, original_h],  # Original dimensions
                    'partial': True,
                    'num_known': int(known_mask.sum())
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
                return full_corners
            else:
                print(f"⚠️  Need at least 2 points, only have {len(points)}")
        
        # Reset
        elif key == ord('r'):
            print("\n🔄 Resetting...")
            points = []
            point_types = []
            frame = display_frame.copy()
            # Redraw grid
            h, w = frame.shape[:2]
            for i in range(5):
                y = int(h * i / 4)
                cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)
                x = int(w * i / 4)
                cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)
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
    
    parser = argparse.ArgumentParser(description='Partial Court Calibration')
    parser.add_argument('--video', type=str, required=True,
                       help='Input video path')
    parser.add_argument('--output', type=str, default='court_calibration.json',
                       help='Output calibration file')
    
    args = parser.parse_args()
    
    calibrate_partial_court(args.video, args.output)

