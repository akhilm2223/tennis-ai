"""
Clear visualization of court detection points
Makes keypoints VERY visible and obvious
"""
import cv2
import numpy as np
from auto_detect_court import detect_court_automatic

def visualize_court_points_clear(video_path, output_path="court_points_clear.jpg"):
    """Show court detection points in a very clear way"""
    
    print(f"📹 Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video")
        return
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read frame")
        return
    
    width, height = frame.shape[1], frame.shape[0]
    print(f"📊 Frame size: {width}x{height}")
    
    # Detect court
    print("\n🎯 Detecting court...")
    court_detector = detect_court_automatic(frame, use_ml_detector=False)
    
    if court_detector.corners is None:
        print("❌ No corners detected!")
        return
    
    print(f"✅ Detected {court_detector.corners.shape[0]} corner points")
    
    # Create THREE versions side-by-side
    original = frame.copy()
    points_only = frame.copy()
    annotated = frame.copy()
    
    # VERSION 1: Original frame (no overlay)
    cv2.putText(original, "ORIGINAL", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # VERSION 2: Points only (VERY LARGE AND VISIBLE)
    corner_colors = [
        (0, 255, 255),    # Yellow
        (0, 255, 0),      # Green  
        (255, 0, 0),      # Blue
        (255, 0, 255)     # Magenta
    ]
    corner_names = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
    
    for idx, corner in enumerate(court_detector.corners):
        x, y = int(corner[0]), int(corner[1])
        color = corner_colors[idx]
        
        # Draw HUGE circle
        cv2.circle(points_only, (x, y), 50, color, -1)  # Filled
        cv2.circle(points_only, (x, y), 55, (255, 255, 255), 8)  # White outline
        cv2.circle(points_only, (x, y), 50, (0, 0, 0), 4)  # Black outline
        
        # Draw crosshair
        cv2.line(points_only, (x-80, y), (x+80, y), color, 6)
        cv2.line(points_only, (x, y-80), (x, y+80), color, 6)
        
        # Draw number
        cv2.putText(points_only, str(idx+1), (x-15, y+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    
    cv2.putText(points_only, "DETECTED POINTS", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # VERSION 3: Full annotation with labels and coordinates
    for idx, corner in enumerate(court_detector.corners):
        x, y = int(corner[0]), int(corner[1])
        color = corner_colors[idx]
        name = corner_names[idx]
        
        # Draw HUGE circle
        cv2.circle(annotated, (x, y), 50, color, -1)
        cv2.circle(annotated, (x, y), 55, (255, 255, 255), 8)
        cv2.circle(annotated, (x, y), 50, (0, 0, 0), 4)
        
        # Draw label box
        label_y_offset = -120 if idx < 2 else 150
        label_x = max(100, min(x, width-250))
        label_y = max(150, min(y + label_y_offset, height-50))
        
        # Background rectangle for text
        box_width = 240
        box_height = 100
        cv2.rectangle(annotated,
                     (label_x - 10, label_y - 80),
                     (label_x + box_width, label_y + 20),
                     (0, 0, 0), -1)
        cv2.rectangle(annotated,
                     (label_x - 10, label_y - 80),
                     (label_x + box_width, label_y + 20),
                     color, 5)
        
        # Point number and name
        cv2.putText(annotated, f"POINT {idx+1}", (label_x, label_y - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated, name, (label_x, label_y - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, f"X: {x}, Y: {y}", (label_x, label_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Draw line from point to label
        cv2.line(annotated, (x, y), (label_x + 100, label_y - 10), color, 3)
    
    # Draw court polygon outline
    corners_int = court_detector.corners.astype(np.int32)
    cv2.polylines(annotated, [corners_int], True, (255, 255, 255), 8)
    
    cv2.putText(annotated, "ANNOTATED", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Stack horizontally
    combined = np.hstack([original, points_only, annotated])
    
    # Add title panel
    title_height = 120
    title_panel = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
    
    cv2.putText(title_panel, "COURT KEYPOINT DETECTION", (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    cv2.putText(title_panel, f"Method: {court_detector.detection_method.upper()} | 4 Corner Points Detected", (50, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Add legend panel at bottom
    legend_height = 180
    legend_panel = np.zeros((legend_height, combined.shape[1], 3), dtype=np.uint8)
    
    cv2.putText(legend_panel, "DETECTED KEYPOINTS:", (50, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    y = 80
    for idx, (name, color) in enumerate(zip(corner_names, corner_colors)):
        x_pos = 50 + (idx * 500)
        cv2.circle(legend_panel, (x_pos, y), 30, color, -1)
        cv2.circle(legend_panel, (x_pos, y), 35, (255, 255, 255), 5)
        
        corner = court_detector.corners[idx]
        text = f"{idx+1}. {name}: ({int(corner[0])}, {int(corner[1])})"
        cv2.putText(legend_panel, text, (x_pos + 50, y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(legend_panel, f"These 4 points define the court boundary for homography transformation", (50, 140),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # Combine all
    final = np.vstack([title_panel, combined, legend_panel])
    
    # Save
    cv2.imwrite(output_path, final)
    cap.release()
    
    print(f"\n✅ Visualization saved: {output_path}")
    print(f"   Size: {final.shape[1]}x{final.shape[0]}")
    print(f"\n📍 Corner Points:")
    for idx, (name, corner) in enumerate(zip(corner_names, court_detector.corners)):
        print(f"   {idx+1}. {name:15} → ({int(corner[0]):4}, {int(corner[1]):4})")
    
    print("\n✨ The visualization now shows:")
    print("   • Original frame (left)")
    print("   • Detected points ONLY (middle) - HUGE circles")
    print("   • Annotated with labels (right)")
    print(f"\n🎯 Detection method: {court_detector.detection_method}")


if __name__ == "__main__":
    video_path = "copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov"
    output_path = "output_videos/court_points_CLEAR.jpg"
    
    print("="*60)
    print("🎾 CLEAR COURT KEYPOINT VISUALIZATION")
    print("="*60 + "\n")
    
    visualize_court_points_clear(video_path, output_path)

