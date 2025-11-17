"""
Reselect only the service lines (top and bottom)
All other lines remain unchanged
"""
import cv2
import numpy as np
import json

# Global variables
service_lines = {}
current_line_index = 0
temp_points = []
frame = None
display_frame = None
window_name = "Reselect Service Lines"
scale_factor = 1.0

# Only these 2 lines
LINE_DEFINITIONS = [
    ('service_line_top', 'Top Service Line (2 points: left to right)', 2),
    ('service_line_bottom', 'Bottom Service Line (2 points: left to right)', 2),
]

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks"""
    global temp_points, service_lines, display_frame, scale_factor, current_line_index
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_line_index < len(LINE_DEFINITIONS):
            line_name, line_desc, num_points = LINE_DEFINITIONS[current_line_index]
            
            # Scale coordinates back to original size
            orig_x = int(x / scale_factor)
            orig_y = int(y / scale_factor)
            
            temp_points.append([orig_x, orig_y])
            
            # Draw point
            cv2.circle(display_frame, (x, y), 10, (255, 255, 0), -1)  # Cyan
            cv2.putText(display_frame, f"P{len(temp_points)}", (x+15, y-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Draw line if we have 2 points
            if len(temp_points) == 2:
                pt1 = (int(temp_points[0][0] * scale_factor), int(temp_points[0][1] * scale_factor))
                pt2 = (int(temp_points[1][0] * scale_factor), int(temp_points[1][1] * scale_factor))
                cv2.line(display_frame, pt1, pt2, (255, 255, 0), 4)  # Cyan
            
            print(f"   Point {len(temp_points)}: ({orig_x}, {orig_y})")
            
            # When we have all points for this line
            if len(temp_points) == num_points:
                service_lines[line_name] = temp_points.copy()
                print(f"✅ {line_desc} - COMPLETE\n")
                
                temp_points = []
                current_line_index += 1
                
                # Show next line prompt
                if current_line_index < len(LINE_DEFINITIONS):
                    next_name, next_desc, next_pts = LINE_DEFINITIONS[current_line_index]
                    print(f"📍 Line {current_line_index + 1}/{len(LINE_DEFINITIONS)}: {next_desc}")
                    print(f"   Click {next_pts} points (LEFT first, then RIGHT)...")
                else:
                    print("\n🎉 Service lines complete!")
                    print("   Press 's' to save")
            
            cv2.imshow(window_name, display_frame)

def reselect_service(video_path, json_path, max_height=900):
    """Reselect only service lines"""
    global service_lines, temp_points, frame, display_frame, scale_factor, current_line_index
    
    print("=" * 70)
    print("🎾 RESELECT SERVICE LINES ONLY")
    print("=" * 70)
    print()
    print("You will select 2 horizontal service lines:")
    print("  1. Top Service Line (near far baseline)")
    print("  2. Bottom Service Line (near near baseline)")
    print()
    print("IMPORTANT: Click LEFT end first, then RIGHT end!")
    print("           Both points should have SAME Y-coordinate (horizontal line)")
    print()
    print("Controls:")
    print("  • Click to place points")
    print("  • Press 's' to save")
    print("  • Press 'r' to reset current line")
    print("  • Press 'q' to quit")
    print()
    print("-" * 70)
    
    # Load existing lines
    with open(json_path, 'r') as f:
        existing_data = json.load(f)
    
    # Read first frame
    cap = cv2.VideoCapture(video_path)
    ret, original_frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error: Could not read video")
        return None
    
    frame = original_frame.copy()
    service_lines = {}
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
    else:
        scale_factor = 1.0
        display_frame = original_frame.copy()
    
    # Draw existing lines on display (except service lines)
    for line_name, line_data in existing_data['lines'].items():
        if line_name not in ['service_line_top', 'service_line_bottom']:
            p1 = (int(line_data['point1'][0] * scale_factor), int(line_data['point1'][1] * scale_factor))
            p2 = (int(line_data['point2'][0] * scale_factor), int(line_data['point2'][1] * scale_factor))
            
            # Color based on line type
            if 'baseline' in line_name or 'sideline' in line_name:
                color = (0, 255, 0)  # Green
                thickness = 3
            elif 'net' in line_name:
                color = (255, 0, 255)  # Magenta
                thickness = 3
            elif 'center' in line_name:
                color = (255, 255, 0)  # Cyan
                thickness = 2
            elif 'singles' in line_name:
                color = (0, 255, 255)  # Yellow
                thickness = 2
            else:
                color = (200, 200, 200)  # Gray
                thickness = 2
            
            cv2.line(display_frame, p1, p2, color, thickness, cv2.LINE_AA)
    
    print()
    print("=" * 70)
    
    # Start with first service line
    line_name, line_desc, num_points = LINE_DEFINITIONS[0]
    print(f"\n📍 Line 1/2: {line_desc}")
    print(f"   Click {num_points} points (LEFT end first, then RIGHT end)...")
    print(f"   TIP: Keep same Y-coordinate for horizontal line!")
    
    # Create window
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Save
        if key == ord('s'):
            if len(service_lines) == 2:
                # Update existing data with new service lines
                for line_name, points in service_lines.items():
                    existing_data['lines'][line_name] = {
                        'point1': [float(points[0][0]), float(points[0][1])],
                        'point2': [float(points[1][0]), float(points[1][1])]
                    }
                
                # Save back to file
                with open(json_path, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                
                print(f"\n✅ Service lines updated in: {json_path}")
                print("\n📊 New service lines:")
                for name in service_lines.keys():
                    print(f"   • {name}: {service_lines[name][0]} → {service_lines[name][1]}")
                
                cv2.destroyAllWindows()
                return existing_data
            else:
                print(f"⚠️  Need both service lines")
        
        # Reset current line
        elif key == ord('r'):
            if len(temp_points) > 0:
                print(f"\n🔄 Resetting current line...")
                temp_points = []
                # Redraw display
                display_frame = cv2.resize(original_frame, (display_width, display_height)) if scale_factor != 1.0 else original_frame.copy()
                
                # Redraw existing lines
                for line_name, line_data in existing_data['lines'].items():
                    if line_name not in ['service_line_top', 'service_line_bottom']:
                        p1 = (int(line_data['point1'][0] * scale_factor), int(line_data['point1'][1] * scale_factor))
                        p2 = (int(line_data['point2'][0] * scale_factor), int(line_data['point2'][1] * scale_factor))
                        
                        if 'baseline' in line_name or 'sideline' in line_name:
                            color = (0, 255, 0)
                            thickness = 3
                        elif 'net' in line_name:
                            color = (255, 0, 255)
                            thickness = 3
                        elif 'center' in line_name:
                            color = (255, 255, 0)
                            thickness = 2
                        elif 'singles' in line_name:
                            color = (0, 255, 255)
                            thickness = 2
                        else:
                            color = (200, 200, 200)
                            thickness = 2
                        
                        cv2.line(display_frame, p1, p2, color, thickness, cv2.LINE_AA)
                
                # Redraw completed service lines
                for completed_name, completed_points in service_lines.items():
                    pt1 = (int(completed_points[0][0] * scale_factor), int(completed_points[0][1] * scale_factor))
                    pt2 = (int(completed_points[1][0] * scale_factor), int(completed_points[1][1] * scale_factor))
                    cv2.line(display_frame, pt1, pt2, (255, 255, 0), 4)
                
                cv2.imshow(window_name, display_frame)
                
                line_name, line_desc, num_points = LINE_DEFINITIONS[current_line_index]
                print(f"\n📍 Line {current_line_index + 1}/2: {line_desc}")
                print(f"   Click {num_points} points (LEFT first, then RIGHT)...")
        
        # Quit
        elif key == ord('q'):
            print("\n❌ Selection cancelled")
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reselect Service Lines Only')
    parser.add_argument('--video', type=str, 
                       default='copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov',
                       help='Input video path')
    parser.add_argument('--lines', type=str,
                       default='court_lines_manual.json',
                       help='Court lines JSON file to update')
    
    args = parser.parse_args()
    
    result = reselect_service(args.video, args.lines)
    
    if result:
        print("\n" + "="*70)
        print("✅ SUCCESS! Service lines have been updated.")
        print("="*70)
        print("\nTo visualize:")
        print(f"  python visualize_manual_lines.py")

