"""
Visualize manually defined court lines
"""
import cv2
import json
import numpy as np

def visualize_manual_lines(video_path, lines_json_path, output_path='output_videos/manual_lines_visualization.jpg'):
    """Show the manually defined court lines overlaid on the video"""
    
    print("=" * 70)
    print("🎾 MANUAL COURT LINES VISUALIZATION")
    print("=" * 70)
    
    # Load manual lines
    with open(lines_json_path, 'r') as f:
        data = json.load(f)
    
    lines = data['lines']
    print(f"\n✅ Loaded {len(lines)} manual lines")
    
    # Read first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Error: Could not read video")
        return
    
    # Create overlay
    overlay = frame.copy()
    
    # Color scheme
    COLORS = {
        'baseline_top': (0, 255, 0),       # Green
        'baseline_bottom': (0, 255, 0),    # Green
        'sideline_left': (0, 255, 0),      # Green
        'sideline_right': (0, 255, 0),     # Green
        'service_line_top': (255, 255, 0), # Cyan
        'service_line_bottom': (255, 255, 0), # Cyan
        'center_service_line': (255, 255, 0), # Cyan
        'net_line': (255, 0, 255),         # Magenta
        'singles_left': (0, 255, 255),     # Yellow
        'singles_right': (0, 255, 255),    # Yellow
    }
    
    THICKNESS = {
        'baseline_top': 4,
        'baseline_bottom': 4,
        'sideline_left': 4,
        'sideline_right': 4,
        'service_line_top': 3,
        'service_line_bottom': 3,
        'center_service_line': 3,
        'net_line': 4,
        'singles_left': 2,
        'singles_right': 2,
    }
    
    # Draw all lines
    print("\n📐 Drawing lines:")
    for line_name, line_data in lines.items():
        p1 = tuple(map(int, line_data['point1']))
        p2 = tuple(map(int, line_data['point2']))
        color = COLORS.get(line_name, (255, 255, 255))
        thickness = THICKNESS.get(line_name, 2)
        
        cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
        
        # Draw endpoint circles
        cv2.circle(overlay, p1, 6, color, -1)
        cv2.circle(overlay, p2, 6, color, -1)
        
        # Draw labels
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        
        # Shorten name for display
        short_name = line_name.replace('_', ' ').title()
        cv2.putText(overlay, short_name, (mid_x, mid_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        
        print(f"   • {line_name}: {p1} → {p2}")
    
    # Blend original and overlay
    result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    
    # Add title
    cv2.putText(result, "MANUAL COURT LINES - Perfect Accuracy!", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    
    # Save
    cv2.imwrite(output_path, result)
    print(f"\n✅ Visualization saved: {output_path}")
    
    print("\n" + "=" * 70)
    print("🎯 LEGEND:")
    print("=" * 70)
    print("  🟢 Green Lines   = Baselines & Sidelines (court boundary)")
    print("  🔵 Cyan Lines    = Service Lines & Center Service Line")
    print("  🟣 Magenta Line  = Net Line")
    print("  🟡 Yellow Lines  = Singles Sidelines")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Manual Court Lines')
    parser.add_argument('--video', type=str, 
                       default='copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov',
                       help='Input video path')
    parser.add_argument('--lines', type=str,
                       default='court_lines_manual.json',
                       help='Manual lines JSON file')
    parser.add_argument('--output', type=str,
                       default='output_videos/manual_lines_visualization.jpg',
                       help='Output image path')
    
    args = parser.parse_args()
    visualize_manual_lines(args.video, args.lines, args.output)

