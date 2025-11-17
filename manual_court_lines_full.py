"""
Complete Manual Court Line Definition
Define all court lines precisely: baselines, net, service lines, sidelines, etc.
"""
import cv2
import numpy as np
import json

# Global variables
lines = {}
current_line = None
temp_points = []
frame = None
display_frame = None
window_name = "Manual Court Line Definition"
scale_factor = 1.0

# Court line definitions needed
LINE_DEFINITIONS = [
    ('baseline_top', 'Far Baseline (2 points: left to right)', 2),
    ('baseline_bottom', 'Near Baseline (2 points: left to right)', 2),
    ('sideline_left', 'Left Sideline (2 points: top to bottom)', 2),
    ('sideline_right', 'Right Sideline (2 points: top to bottom)', 2),
    ('service_line_top', 'Top Service Line (2 points: left to right)', 2),
    ('service_line_bottom', 'Bottom Service Line (2 points: left to right)', 2),
    ('center_service_line', 'Center Service Line (2 points: top to bottom)', 2),
    ('net_line', 'Net Line (2 points: left to right)', 2),
    ('singles_left', 'Left Singles Line (2 points: top to bottom) [Optional]', 2),
    ('singles_right', 'Right Singles Line (2 points: top to bottom) [Optional]', 2),
]

current_line_index = 0

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to define court lines"""
    global temp_points, lines, current_line, display_frame, scale_factor
    global current_line_index
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_line_index < len(LINE_DEFINITIONS):
            line_name, line_desc, num_points = LINE_DEFINITIONS[current_line_index]
            
            # Scale coordinates back to original size
            orig_x = int(x / scale_factor)
            orig_y = int(y / scale_factor)
            
            temp_points.append([orig_x, orig_y])
            
            # Draw point
            cv2.circle(display_frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display_frame, f"P{len(temp_points)}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw line if we have 2 points
            if len(temp_points) == 2:
                pt1 = (int(temp_points[0][0] * scale_factor), int(temp_points[0][1] * scale_factor))
                pt2 = (int(temp_points[1][0] * scale_factor), int(temp_points[1][1] * scale_factor))
                cv2.line(display_frame, pt1, pt2, (0, 255, 0), 3)
            
            print(f"   Point {len(temp_points)}: ({orig_x}, {orig_y})")
            
            # When we have all points for this line
            if len(temp_points) == num_points:
                lines[line_name] = temp_points.copy()
                print(f"✅ {line_desc} - COMPLETE\n")
                
                temp_points = []
                current_line_index += 1
                
                # Show next line prompt
                if current_line_index < len(LINE_DEFINITIONS):
                    next_name, next_desc, next_pts = LINE_DEFINITIONS[current_line_index]
                    print(f"📍 Line {current_line_index + 1}/{len(LINE_DEFINITIONS)}: {next_desc}")
                    print(f"   Click {next_pts} points...")
                else:
                    print("\n🎉 All lines defined!")
                    print("   Press 's' to save or 'r' to reset")
            
            cv2.imshow(window_name, display_frame)

def define_court_lines_manually(video_path, output_path="court_lines_manual.json", max_height=900):
    """
    Define all court lines manually by clicking points
    
    Args:
        video_path: Path to video file
        output_path: Where to save line definitions
        max_height: Maximum height for display
        
    Returns:
        dict of court lines
    """
    global lines, temp_points, frame, display_frame, scale_factor, current_line_index
    
    print("=" * 70)
    print("🎾 COMPLETE MANUAL COURT LINE DEFINITION")
    print("=" * 70)
    print()
    print("You will define all court lines by clicking 2 points per line.")
    print("Lines to define:")
    for i, (name, desc, pts) in enumerate(LINE_DEFINITIONS):
        print(f"  {i+1}. {desc}")
    print()
    print("Controls:")
    print("  • Click points to define each line")
    print("  • Press 's' to save when done")
    print("  • Press 'r' to reset and start over")
    print("  • Press 'q' to quit")
    print("  • Press 'n' to skip current line (optional lines only)")
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
    lines = {}
    temp_points = []
    current_line_index = 0
    
    # Calculate scale for display
    orig_height, orig_width = original_frame.shape[:2]
    if orig_height > max_height:
        scale_factor = max_height / orig_height
        display_width = int(orig_width * scale_factor)
        display_height = max_height
        display_frame = cv2.resize(original_frame, (display_width, display_height))
        print(f"📐 Display: {orig_width}x{orig_height} → {display_width}x{display_height}")
        print(f"   (Coordinates saved in original size)")
    else:
        scale_factor = 1.0
        display_frame = original_frame.copy()
    
    print()
    print("=" * 70)
    
    # Start with first line
    line_name, line_desc, num_points = LINE_DEFINITIONS[0]
    print(f"\n📍 Line 1/{len(LINE_DEFINITIONS)}: {line_desc}")
    print(f"   Click {num_points} points...")
    
    # Create window
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Save
        if key == ord('s'):
            if current_line_index >= len(LINE_DEFINITIONS) or len(lines) >= 8:  # At least core lines
                # Convert to proper format
                court_data = {
                    'lines': {}
                }
                
                for line_name, points in lines.items():
                    court_data['lines'][line_name] = {
                        'point1': [float(points[0][0]), float(points[0][1])],
                        'point2': [float(points[1][0]), float(points[1][1])]
                    }
                
                # Save to JSON
                with open(output_path, 'w') as f:
                    json.dump(court_data, f, indent=2)
                
                print(f"\n✅ Court lines saved to: {output_path}")
                print(f"   Total lines defined: {len(lines)}")
                print("\n📊 Defined lines:")
                for name in lines.keys():
                    print(f"   • {name}")
                
                cv2.destroyAllWindows()
                return court_data
            else:
                print(f"⚠️  Please define at least the core 8 lines")
                print(f"   Currently defined: {len(lines)} lines")
        
        # Skip (for optional lines only)
        elif key == ord('n'):
            if current_line_index >= 8:  # Only allow skip for optional lines
                print(f"⏭️  Skipping {LINE_DEFINITIONS[current_line_index][1]}")
                temp_points = []
                current_line_index += 1
                
                if current_line_index < len(LINE_DEFINITIONS):
                    line_name, line_desc, num_points = LINE_DEFINITIONS[current_line_index]
                    print(f"\n📍 Line {current_line_index + 1}/{len(LINE_DEFINITIONS)}: {line_desc}")
                    print(f"   Click {num_points} points...")
                else:
                    print("\n✅ All required lines complete!")
                    print("   Press 's' to save")
            else:
                print("⚠️  Cannot skip required lines (1-8)")
        
        # Reset
        elif key == ord('r'):
            print("\n🔄 Resetting all lines...")
            lines = {}
            temp_points = []
            current_line_index = 0
            
            if scale_factor != 1.0:
                display_frame = cv2.resize(original_frame, 
                    (int(orig_width * scale_factor), int(orig_height * scale_factor)))
            else:
                display_frame = original_frame.copy()
            
            cv2.imshow(window_name, display_frame)
            
            line_name, line_desc, num_points = LINE_DEFINITIONS[0]
            print(f"\n📍 Line 1/{len(LINE_DEFINITIONS)}: {line_desc}")
            print(f"   Click {num_points} points...")
        
        # Quit
        elif key == ord('q'):
            print("\n❌ Definition cancelled")
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete Manual Court Line Definition')
    parser.add_argument('--video', type=str, required=True,
                       help='Input video path')
    parser.add_argument('--output', type=str, default='court_lines_manual.json',
                       help='Output file for line definitions')
    
    args = parser.parse_args()
    
    result = define_court_lines_manually(args.video, args.output)
    
    if result:
        print("\n" + "="*70)
        print("✅ SUCCESS! Court lines have been manually defined.")
        print("="*70)
        print(f"\nTo use these lines in analysis:")
        print(f"  python main_pose.py --video {args.video} \\")
        print(f"                      --court-lines {args.output} \\")
        print(f"                      --output analysis.mp4")

