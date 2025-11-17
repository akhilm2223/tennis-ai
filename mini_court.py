"""
Mini Court Visualization for Tennis Analysis
Professional tennis court with accurate dimensions and markings
"""
import cv2
import numpy as np


class MiniCourt:
    """
    Mini court visualization showing player and ball positions
    Accurate tennis court dimensions and professional appearance
    """
    
    def __init__(self, court_detector, width=120, height=220):
        """
        Initialize mini court with proper tennis court proportions
        Very compact size to not obstruct main video
        
        Args:
            court_detector: CourtDetector object
            width: Mini court width in pixels (default: 120 - very compact)
            height: Mini court height in pixels (should be ~2x width for tennis)
        """
        self.court_detector = court_detector
        self.width = width
        self.height = height
        
        # Tennis court dimensions (in meters) - OFFICIAL ITF
        self.court_length = 23.77  # meters (78 feet)
        self.court_width = 10.97   # meters (36 feet)
        self.singles_width = 8.23  # meters (27 feet)
        self.service_line_distance = 6.40  # meters from net (21 feet)
        
        # Padding (smaller for compact view)
        self.padding = 10
        
        # Drawing area
        self.draw_width = width - 2 * self.padding
        self.draw_height = height - 2 * self.padding
        
        # Colors (professional tennis court colors)
        self.court_color = (34, 139, 34)  # Forest green
        self.line_color = (255, 255, 255)  # White
        self.bg_color = (20, 80, 20)  # Dark green background
        
        # Display settings
        self.show_players = False  # Only show ball, not players
        self.show_title = True
        
        # Bounce detection
        self.bounce_points = []  # Store all detected bounce points (permanent markers)
        self.max_bounces = 20  # Maximum number of bounces to display
        self.ball_trajectory = []  # Track recent ball positions
        self.trajectory_length = 10  # How many frames to track
        self.min_bounce_distance = 100  # Minimum pixels between bounces
    
    def draw(self, player_positions, ball_position=None, ball_trail=None, bounces=None):
        """
        Draw professional tennis mini court with bounce point markers
        
        Args:
            player_positions: Dict of player_id: (x, y) positions (ignored)
            ball_position: (x, y) ball position or None
            ball_trail: List of ball positions for trail (optional, not used)
            bounces: List of bounce dicts from physics tracker (preferred source)
            
        Returns:
            Mini court image
        """
        # Use physics-based bounces from tracker if provided (much more accurate!)
        if bounces is not None and len(bounces) > 0:
            # Debug: Log bounce count for verification
            if len(bounces) != len(self.bounce_points):
                print(f"   📍 Mini-court updating: {len(bounces)} total bounces")
            
            # Update bounce_points with physics-validated bounces
            # Store both image and court coordinates for accurate mapping
            new_bounce_positions = []
            for bounce in bounces:
                # Store bounce with both coordinate systems AND player attribution
                bounce_entry = {
                    'image_xy': bounce.get('image_xy'),
                    'court_xy': bounce.get('court_xy'),
                    'frame': bounce.get('frame'),
                    'player': bounce.get('player')  # ADD THIS: 1, 2, or None
                }
                new_bounce_positions.append(bounce_entry)
            
            # Replace internal bounce points with physics-based ones
            self.bounce_points = new_bounce_positions[-self.max_bounces:]  # Keep last N bounces
        # Create canvas with dark green background
        canvas = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        
        # Court boundaries
        court_x = self.padding
        court_y = self.padding
        court_w = self.draw_width
        court_h = self.draw_height
        
        # Fill court surface (green)
        cv2.rectangle(canvas, (court_x, court_y), 
                     (court_x + court_w, court_y + court_h),
                     self.court_color, -1)
        
        # === DRAW COURT LINES (WHITE) ===
        line_thickness = 1  # Thinner lines for compact view
        
        # Outer boundary (doubles court)
        cv2.rectangle(canvas, (court_x, court_y), 
                     (court_x + court_w, court_y + court_h),
                     self.line_color, line_thickness)
        
        # Singles sidelines (inner vertical lines)
        singles_margin = int(court_w * (self.court_width - self.singles_width) / (2 * self.court_width))
        cv2.line(canvas, 
                (court_x + singles_margin, court_y),
                (court_x + singles_margin, court_y + court_h),
                self.line_color, line_thickness)
        cv2.line(canvas, 
                (court_x + court_w - singles_margin, court_y),
                (court_x + court_w - singles_margin, court_y + court_h),
                self.line_color, line_thickness)
        
        # Net (center line)
        center_y = court_y + court_h // 2
        cv2.line(canvas, (court_x, center_y), 
                (court_x + court_w, center_y), 
                self.line_color, line_thickness)
        
        # Service lines (horizontal lines at 1/3 and 2/3)
        service_distance_ratio = self.service_line_distance / (self.court_length / 2)
        service_line_y1 = int(court_y + court_h * (0.5 - service_distance_ratio / 2))
        service_line_y2 = int(court_y + court_h * (0.5 + service_distance_ratio / 2))
        
        cv2.line(canvas, (court_x, service_line_y1), 
                (court_x + court_w, service_line_y1), 
                self.line_color, line_thickness)
        cv2.line(canvas, (court_x, service_line_y2), 
                (court_x + court_w, service_line_y2), 
                self.line_color, line_thickness)
        
        # Center service line (vertical line in middle)
        center_x = court_x + court_w // 2
        cv2.line(canvas, (center_x, service_line_y1),
                (center_x, service_line_y2),
                self.line_color, line_thickness)
        
        # === DRAW PERMANENT BOUNCE POINTS (COLOR-CODED BY PLAYER) ===
        # Draw all detected bounce points with player attribution
        for idx, bounce_entry in enumerate(self.bounce_points):
            # Extract position from bounce dict (prefer court_xy for accurate mapping)
            if isinstance(bounce_entry, dict):
                bounce_pos = bounce_entry.get('court_xy') or bounce_entry.get('image_xy')
                player_id = bounce_entry.get('player')  # 1, 2, or None
            else:
                # Legacy tuple/list format: might be [x, y] or [x, y, frame]
                # Always extract only first 2 values
                if isinstance(bounce_entry, (list, tuple)) and len(bounce_entry) >= 2:
                    bounce_pos = (float(bounce_entry[0]), float(bounce_entry[1]))  # ✅ Only x, y
                else:
                    bounce_pos = bounce_entry
                player_id = None
            
            if bounce_pos is None:
                continue
                
            mini_pos = self._world_to_mini(bounce_pos)
            if mini_pos:
                bounce_num = idx + 1
                
                # COLOR-CODE BY PLAYER: RED for Player 1 (near), BLUE for Player 2 (far)
                if player_id == 1:
                    color = (0, 0, 255)  # RED (BGR) - Player 1 (near)
                elif player_id == 2:
                    color = (255, 0, 0)  # BLUE (BGR) - Player 2 (far)
                else:
                    color = (128, 128, 128)  # Gray - Unknown player
                
                # Draw filled circle (permanent marker)
                cv2.circle(canvas, mini_pos, 8, color, -1)
                cv2.circle(canvas, mini_pos, 8, (255, 255, 255), 1)  # White border
                
                # Draw bounce number inside marker
                cv2.putText(canvas, str(bounce_num),
                           (mini_pos[0] - 4, mini_pos[1] + 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # === NO LIVE BALL TRACKING - ONLY BOUNCE POINTS ===
        # (Ball position and players are not shown in real-time)
        
        # === DRAW TITLE AND BOUNCE COUNT ===
        if self.show_title:
            title = f"BOUNCES: {len(self.bounce_points)}"
            cv2.putText(canvas, title, 
                       (court_x + 3, court_y - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return canvas
    
    def _world_to_mini(self, position):
        """
        Convert world coordinates to mini court coordinates using homography
        """
        if position is None:
            return None
        
        # Safety: ensure position is a 2-tuple (handle legacy 3-tuple [x,y,frame])
        if isinstance(position, (list, tuple)):
            if len(position) >= 2:
                x, y = float(position[0]), float(position[1])
            else:
                return None
        else:
            return None
        
        # If we have homography, use it for accurate mapping
        if self.court_detector and self.court_detector.homography_matrix is not None:
            try:
                # Convert point using homography
                point = np.array([[[x, y]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(point, self.court_detector.homography_matrix)
                court_x, court_y = transformed[0][0]
                
                # Map from court coordinates (0-400, 0-800) to mini court pixels
                # Assuming court space is 400x800 (from auto_detect_court.py)
                mini_x = int(self.padding + (court_x / 400) * self.draw_width)
                mini_y = int(self.padding + (court_y / 800) * self.draw_height)
                
                # Clamp to court bounds
                mini_x = max(self.padding, min(mini_x, self.padding + self.draw_width))
                mini_y = max(self.padding, min(mini_y, self.padding + self.draw_height))
                
                return (mini_x, mini_y)
            except:
                pass
        
        # Fallback: simple frame-based mapping with better scaling
        # Get actual frame dimensions from court detector if available
        if self.court_detector and self.court_detector.corners is not None:
            corners = self.court_detector.corners
            # Use court corners to determine frame area
            frame_w = max(corners[:, 0]) - min(corners[:, 0])
            frame_h = max(corners[:, 1]) - min(corners[:, 1])
            offset_x = min(corners[:, 0])
            offset_y = min(corners[:, 1])
            
            # Normalize position relative to court area
            norm_x = (x - offset_x) / frame_w if frame_w > 0 else 0.5
            norm_y = (y - offset_y) / frame_h if frame_h > 0 else 0.5
        else:
            # Use full frame dimensions
            frame_w = 1920  # Typical HD width
            frame_h = 1080  # Typical HD height
            
            # Normalize position
            norm_x = x / frame_w
            norm_y = y / frame_h
        
        # Clamp normalized values
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        # Map to mini court
        mini_x = int(self.padding + norm_x * self.draw_width)
        mini_y = int(self.padding + norm_y * self.draw_height)
        
        # Clamp to court bounds
        mini_x = max(self.padding, min(mini_x, self.padding + self.draw_width))
        mini_y = max(self.padding, min(mini_y, self.padding + self.draw_height))
        
        return (mini_x, mini_y)
    
    def overlay_on_frame(self, frame, mini_court_img, position='top_right', margin=15):
        """
        Overlay mini court on main frame with proper size handling
        
        Args:
            frame: Main frame
            mini_court_img: Mini court image
            position: 'top_right', 'top_left', 'bottom_right', 'bottom_left'
            margin: Margin from edges
            
        Returns:
            Frame with mini court overlay
        """
        h, w = frame.shape[:2]
        mh, mw = mini_court_img.shape[:2]
        
        # Ensure mini court fits in frame
        if mh > h - 2 * margin or mw > w - 2 * margin:
            # Resize mini court to fit
            scale = min((h - 2 * margin) / mh, (w - 2 * margin) / mw)
            new_w = int(mw * scale)
            new_h = int(mh * scale)
            mini_court_img = cv2.resize(mini_court_img, (new_w, new_h))
            mh, mw = new_h, new_w
        
        # Calculate position
        if position == 'top_right':
            y1, y2 = margin, margin + mh
            x1, x2 = w - mw - margin, w - margin
        elif position == 'top_left':
            y1, y2 = margin, margin + mh
            x1, x2 = margin, margin + mw
        elif position == 'bottom_right':
            y1, y2 = h - mh - margin, h - margin
            x1, x2 = w - mw - margin, w - margin
        else:  # bottom_left
            y1, y2 = h - mh - margin, h - margin
            x1, x2 = margin, margin + mw
        
        # Ensure coordinates are within bounds
        y1 = max(0, y1)
        y2 = min(h, y2)
        x1 = max(0, x1)
        x2 = min(w, x2)
        
        # Calculate actual region size
        region_h = y2 - y1
        region_w = x2 - x1
        
        # Resize mini court to exact region size if needed
        if mini_court_img.shape[0] != region_h or mini_court_img.shape[1] != region_w:
            mini_court_img = cv2.resize(mini_court_img, (region_w, region_h))
        
        # Overlay with alpha blending
        try:
            alpha = 0.85
            frame[y1:y2, x1:x2] = cv2.addWeighted(
                frame[y1:y2, x1:x2], 1 - alpha,
                mini_court_img, alpha, 0
            )
        except Exception as e:
            # Fallback: just paste the mini court without blending
            frame[y1:y2, x1:x2] = mini_court_img
        
        return frame
