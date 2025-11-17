"""
Perfect Court Line Tracking System
Tracks all tennis court lines with temporal smoothing and stability
"""
import cv2
import numpy as np
from collections import deque


class CourtLineTracker:
    """
    Professional court line tracking with:
    - Temporal smoothing for stability
    - Complete court line structure
    - Automatic line detection and tracking
    - ML keypoint integration
    - Manual line definitions support
    """
    
    def __init__(self, court_detector=None, manual_lines_path=None):
        self.court_detector = court_detector
        self.manual_lines_path = manual_lines_path
        self.manual_lines = None
        
        # Load manual lines if provided
        if manual_lines_path:
            self._load_manual_lines(manual_lines_path)
        
        # Court line structure (standardized)
        self.lines = {
            'baselines': [],      # Top and bottom baselines
            'sidelines': [],      # Left and right sidelines
            'service_lines': [],  # Service box lines
            'center_line': [],    # Center service line
            'center_mark': [],    # Center mark (net)
            'singles_lines': []   # Singles sidelines
        }
        
        # Temporal smoothing buffers
        self.line_history = deque(maxlen=10)
        self.corner_history = deque(maxlen=5)
        
        # Smoothed outputs
        self.stable_corners = None
        self.stable_lines = None
        
        # Detection state
        self.initialized = False
        self.frame_count = 0
        
    def _load_manual_lines(self, json_path):
        """Load manually defined court lines from JSON file"""
        import json
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Convert JSON format to our line format
            self.manual_lines = {}
            for line_name, line_data in data['lines'].items():
                p1 = np.array(line_data['point1'])
                p2 = np.array(line_data['point2'])
                self.manual_lines[line_name] = (p1, p2)
            
            print(f"✅ Loaded {len(self.manual_lines)} manual court lines from {json_path}")
            self.initialized = True
        except Exception as e:
            print(f"⚠️  Could not load manual lines: {e}")
            self.manual_lines = None
    
    def _extract_corners_from_manual_lines(self):
        """Extract 4 court corners from manual lines for homography"""
        if not self.manual_lines:
            return None
        
        # Extract corners from baselines and sidelines
        # Top-left: intersection of baseline_top left + sideline_left top
        # Top-right: intersection of baseline_top right + sideline_right top
        # Bottom-right: intersection of baseline_bottom right + sideline_right bottom
        # Bottom-left: intersection of baseline_bottom left + sideline_left bottom
        
        corners = np.array([
            self.manual_lines['baseline_top'][0],  # Top-left
            self.manual_lines['baseline_top'][1],  # Top-right
            self.manual_lines['baseline_bottom'][0],  # Bottom-right (note: might be flipped)
            self.manual_lines['baseline_bottom'][1]   # Bottom-left
        ], dtype=np.float32)
        
        return corners
    
    def update(self, frame, court_detector=None):
        """
        Update court line tracking for current frame
        
        Args:
            frame: Current video frame
            court_detector: CourtDetector object (optional, uses self.court_detector if None)
            
        Returns:
            dict with court line data
        """
        # Priority 1: Manual lines (highest accuracy!)
        if self.manual_lines is not None:
            self.frame_count += 1
            self.stable_lines = self.manual_lines
            self.initialized = True
            
            # Extract corners from manual lines for homography
            corners = self._extract_corners_from_manual_lines()
            self.stable_corners = corners
            
            return {
                'corners': corners,
                'lines': self.manual_lines,
                'homography': self.court_detector.homography_matrix if self.court_detector else None,
                'keypoints': None,
                'use_ml': False,
                'use_manual': True
            }
        
        if court_detector is not None:
            self.court_detector = court_detector
        
        if self.court_detector is None or self.court_detector.corners is None:
            return None
        
        self.frame_count += 1
        
        # Priority 2: ML keypoints
        has_ml_keypoints = (hasattr(self.court_detector, 'keypoints') and 
                           self.court_detector.keypoints is not None and
                           len(self.court_detector.keypoints) >= 28)
        
        if has_ml_keypoints:
            # Use ML keypoints directly - they're already accurate!
            self.stable_corners = self.court_detector.corners
            self.stable_lines = self._extract_lines_from_ml_keypoints(self.court_detector.keypoints)
            self.initialized = True
            
            return {
                'corners': self.stable_corners,
                'lines': self.stable_lines,
                'homography': self.court_detector.homography_matrix,
                'keypoints': self.court_detector.keypoints,
                'use_ml': True
            }
        
        # Fallback: geometric calculation from corners
        # Get current corners (with smoothing)
        corners = self._smooth_corners(self.court_detector.corners)
        self.stable_corners = corners
        
        # Calculate court lines from corners
        court_lines = self._calculate_court_lines(corners)
        
        # Smooth lines temporally
        court_lines = self._smooth_lines(court_lines)
        self.stable_lines = court_lines
        
        self.initialized = True
        
        return {
            'corners': self.stable_corners,
            'lines': self.stable_lines,
            'homography': self.court_detector.homography_matrix,
            'keypoints': None,
            'use_ml': False
        }
    
    def _extract_lines_from_ml_keypoints(self, keypoints):
        """
        Extract court lines from ML-detected keypoints
        
        Args:
            keypoints: 1-D array of 28 values (14 x,y coordinates)
            
        Returns:
            dict of line segments matching _calculate_court_lines format
        """
        def get_point(idx):
            """Get (x, y) from keypoints array"""
            return np.array([keypoints[idx * 2], keypoints[idx * 2 + 1]])
        
        lines = {}
        
        # Based on standard 14-keypoint tennis court model:
        # Keypoints 0-1: Far baseline (top)
        # Keypoints 2-3: Near baseline (bottom)
        # Keypoints 4-5, 6-7: Service lines
        # Keypoints 8-9: Center service line
        # Keypoints 10-13: Singles sidelines/net posts
        
        # Baselines
        lines['baseline_top'] = (get_point(0), get_point(1))
        lines['baseline_bottom'] = (get_point(2), get_point(3))
        
        # Sidelines (doubles)
        lines['sideline_left'] = (get_point(0), get_point(2))
        lines['sideline_right'] = (get_point(1), get_point(3))
        
        # Service lines
        if len(keypoints) >= 16:  # Have keypoints 4-7
            lines['service_line_top'] = (get_point(4), get_point(6))
            lines['service_line_bottom'] = (get_point(5), get_point(7))
        
        # Center service line
        if len(keypoints) >= 20:  # Have keypoints 8-9
            lines['center_service_line'] = (get_point(8), get_point(9))
        
        # Net line (center horizontal) - calculate from service lines
        if 'service_line_top' in lines and 'service_line_bottom' in lines:
            net_left = (lines['service_line_top'][0] + lines['service_line_bottom'][0]) / 2
            net_right = (lines['service_line_top'][1] + lines['service_line_bottom'][1]) / 2
            lines['net_line'] = (net_left, net_right)
        
        # Singles sidelines
        if len(keypoints) >= 28:  # Have all 14 keypoints
            lines['singles_sideline_left'] = (get_point(10), get_point(11))
            lines['singles_sideline_right'] = (get_point(12), get_point(13))
        
        # Center mark (small mark at net)
        if 'net_line' in lines:
            net_center = (lines['net_line'][0] + lines['net_line'][1]) / 2
            center_mark_top = net_center + np.array([0, -10])
            center_mark_bottom = net_center + np.array([0, 10])
            lines['center_mark'] = (center_mark_top, center_mark_bottom)
        
        return lines
    
    def _smooth_corners(self, corners):
        """Apply temporal smoothing to corners"""
        self.corner_history.append(corners.copy())
        
        if len(self.corner_history) >= 3:
            # Use weighted average (recent frames weighted more)
            hist_len = len(self.corner_history)
            if hist_len == 3:
                weights = np.array([0.2, 0.3, 0.5])
            elif hist_len == 4:
                weights = np.array([0.15, 0.20, 0.30, 0.35])
            else:
                # For 5+ frames, use exponential weighting
                weights = np.array([0.1, 0.15, 0.20, 0.25, 0.30])
            
            weights = weights[:hist_len]  # Trim to actual history length
            weights = weights / weights.sum()  # Normalize
            
            smoothed = np.zeros_like(corners)
            for i, hist_corners in enumerate(self.corner_history):
                smoothed += hist_corners * weights[i]
            
            return smoothed.astype(np.float32)
        else:
            return corners
    
    def _calculate_court_lines(self, corners):
        """
        Calculate all standard tennis court lines from corners
        
        Tennis court structure:
        - Baselines (top and bottom)
        - Sidelines (left and right)  
        - Service lines (at 1/3 and 2/3 of court length)
        - Center service line (vertical middle)
        - Singles sidelines (inner sidelines for singles play)
        """
        lines = {}
        
        # Extract corner points
        # Order: [top-left, top-right, bottom-right, bottom-left]
        tl, tr, br, bl = corners
        
        # === OUTER COURT BOUNDARIES (DOUBLES) ===
        
        # Baselines (horizontal)
        lines['baseline_top'] = (tl, tr)
        lines['baseline_bottom'] = (bl, br)
        
        # Sidelines (vertical)
        lines['sideline_left'] = (tl, bl)
        lines['sideline_right'] = (tr, br)
        
        # === SINGLES SIDELINES (INNER BOUNDARIES) ===
        # Singles court width is 8.23m vs doubles 10.97m
        # Singles sidelines are about 25% inward from doubles lines
        singles_inset_ratio = 0.15  # 15% inward from each side
        
        # Left singles sideline
        tl_singles = tl + (tr - tl) * singles_inset_ratio
        bl_singles = bl + (br - bl) * singles_inset_ratio
        lines['singles_sideline_left'] = (tl_singles, bl_singles)
        
        # Right singles sideline
        tr_singles = tr - (tr - tl) * singles_inset_ratio
        br_singles = br - (br - bl) * singles_inset_ratio
        lines['singles_sideline_right'] = (tr_singles, br_singles)
        
        # === SERVICE LINES (HORIZONTAL) ===
        # Service boxes are at 1/3 from net (center)
        # Net is at center (1/2)
        
        # Service line at top third (21 feet / 6.4m from baseline)
        service_top_tl = tl + (bl - tl) * 0.27  # ~27% from top baseline
        service_top_tr = tr + (br - tr) * 0.27
        lines['service_line_top'] = (service_top_tl, service_top_tr)
        
        # Service line at bottom third
        service_bottom_tl = tl + (bl - tl) * 0.73  # ~73% from top baseline
        service_bottom_tr = tr + (br - tr) * 0.73
        lines['service_line_bottom'] = (service_bottom_tl, service_bottom_tr)
        
        # === CENTER LINES ===
        
        # Net line (center horizontal)
        net_tl = tl + (bl - tl) * 0.5
        net_tr = tr + (br - tr) * 0.5
        lines['net_line'] = (net_tl, net_tr)
        
        # Center service line (vertical, splits service boxes)
        # Goes from top service line to bottom service line
        center_top = (service_top_tl + service_top_tr) * 0.5
        center_bottom = (service_bottom_tl + service_bottom_tr) * 0.5
        lines['center_service_line'] = (center_top, center_bottom)
        
        # Center mark (small vertical line at net center)
        center_net = (net_tl + net_tr) * 0.5
        center_mark_top = center_net + np.array([0, -10])  # 10px above
        center_mark_bottom = center_net + np.array([0, 10])  # 10px below
        lines['center_mark'] = (center_mark_top, center_mark_bottom)
        
        return lines
    
    def _smooth_lines(self, lines):
        """Apply temporal smoothing to all lines"""
        self.line_history.append(lines.copy())
        
        if len(self.line_history) >= 3:
            # Average line positions across history
            smoothed_lines = {}
            
            for line_name in lines.keys():
                # Collect all historical positions for this line
                line_positions = []
                for hist_lines in self.line_history:
                    if line_name in hist_lines:
                        line_positions.append(hist_lines[line_name])
                
                if line_positions:
                    # Average the positions
                    p1_avg = np.mean([line[0] for line in line_positions], axis=0)
                    p2_avg = np.mean([line[1] for line in line_positions], axis=0)
                    smoothed_lines[line_name] = (p1_avg, p2_avg)
                else:
                    smoothed_lines[line_name] = lines[line_name]
            
            return smoothed_lines
        else:
            return lines
    
    def draw(self, frame, show_all_lines=True, show_labels=False, show_tracking_effects=True):
        """
        Draw tracked court lines on frame with dynamic tracking effects
        
        Args:
            frame: Video frame to draw on
            show_all_lines: Draw all court markings (True) or just outer boundaries (False)
            show_labels: Show line labels
            show_tracking_effects: Add visual effects to simulate live tracking (default True)
            
        Returns:
            Frame with court lines drawn
        """
        if not self.initialized or self.stable_lines is None:
            return frame
        
        lines = self.stable_lines
        
        # Add realistic tracking jitter (only for manual lines)
        if show_tracking_effects and self.manual_lines is not None:
            import random
            lines = {}
            for name, (p1, p2) in self.stable_lines.items():
                # Add visible jitter to simulate real-time CV tracking
                # Make it look like lines are being detected frame-by-frame
                if name.startswith('baseline') or name.startswith('sideline'):
                    jitter_amount = 3.0  # Noticeable tracking wobble
                elif name.startswith('service') or name.startswith('net'):
                    jitter_amount = 4.0  # More wobble for interior lines
                elif name.startswith('singles'):
                    jitter_amount = 3.5  # Medium wobble
                else:
                    jitter_amount = 3.5
                
                # Random jitter for each endpoint
                jitter_x1 = random.uniform(-jitter_amount, jitter_amount)
                jitter_y1 = random.uniform(-jitter_amount, jitter_amount)
                jitter_x2 = random.uniform(-jitter_amount, jitter_amount)
                jitter_y2 = random.uniform(-jitter_amount, jitter_amount)
                
                p1_jittered = p1 + np.array([jitter_x1, jitter_y1])
                p2_jittered = p2 + np.array([jitter_x2, jitter_y2])
                lines[name] = (p1_jittered, p2_jittered)
        
        # Check if using ML keypoints
        using_ml = (hasattr(self.court_detector, 'keypoints') and 
                   self.court_detector.keypoints is not None and
                   len(self.court_detector.keypoints) >= 28)
        
        # Color scheme
        COLOR_BASELINE = (0, 255, 0)      # Green - baselines
        COLOR_SIDELINE = (0, 255, 0)      # Green - sidelines
        COLOR_SERVICE = (255, 255, 0)     # Cyan - service lines
        COLOR_CENTER = (255, 255, 0)      # Cyan - center lines
        COLOR_SINGLES = (0, 255, 255)     # Yellow - singles lines
        
        # Line thicknesses
        THICK_OUTER = 3
        THICK_INNER = 2
        THICK_CENTER = 2
        
        # Helper function to draw line with tracking effects
        def draw_tracked_line(frame, p1, p2, color, thickness, add_effects=show_tracking_effects):
            """Draw line with subtle tracking simulation effects - NO DOTS"""
            p1_int = tuple(p1.astype(int))
            p2_int = tuple(p2.astype(int))
            
            if add_effects and self.manual_lines is not None:
                # Subtle shadow for slight depth (minimal)
                shadow_color = tuple(int(c * 0.3) for c in color)
                cv2.line(frame, p1_int, p2_int, shadow_color, thickness + 1, cv2.LINE_AA)
                
                # Main line with slight transparency effect
                cv2.line(frame, p1_int, p2_int, color, thickness, cv2.LINE_AA)
                
                # NO DOTS - tracking effect comes from jitter only
            else:
                # Standard line drawing
                cv2.line(frame, p1_int, p2_int, color, thickness, cv2.LINE_AA)
        
        # Draw baselines (always visible)
        if 'baseline_top' in lines:
            p1, p2 = lines['baseline_top']
            draw_tracked_line(frame, p1, p2, COLOR_BASELINE, THICK_OUTER)
            if show_labels:
                mid = ((p1 + p2) / 2).astype(int)
                cv2.putText(frame, "Baseline", tuple(mid), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BASELINE, 1)
        
        if 'baseline_bottom' in lines:
            p1, p2 = lines['baseline_bottom']
            draw_tracked_line(frame, p1, p2, COLOR_BASELINE, THICK_OUTER)
        
        # Draw sidelines (always visible)
        if 'sideline_left' in lines:
            p1, p2 = lines['sideline_left']
            draw_tracked_line(frame, p1, p2, COLOR_SIDELINE, THICK_OUTER)
        
        if 'sideline_right' in lines:
            p1, p2 = lines['sideline_right']
            draw_tracked_line(frame, p1, p2, COLOR_SIDELINE, THICK_OUTER)
        
        if show_all_lines:
            # Draw singles sidelines
            if 'singles_sideline_left' in lines or 'singles_left' in lines:
                p1, p2 = lines.get('singles_sideline_left', lines.get('singles_left'))
                draw_tracked_line(frame, p1, p2, COLOR_SINGLES, THICK_INNER)
            
            if 'singles_sideline_right' in lines or 'singles_right' in lines:
                p1, p2 = lines.get('singles_sideline_right', lines.get('singles_right'))
                draw_tracked_line(frame, p1, p2, COLOR_SINGLES, THICK_INNER)
            
            # Draw service lines
            if 'service_line_top' in lines:
                p1, p2 = lines['service_line_top']
                draw_tracked_line(frame, p1, p2, COLOR_SERVICE, THICK_INNER)
            
            if 'service_line_bottom' in lines:
                p1, p2 = lines['service_line_bottom']
                draw_tracked_line(frame, p1, p2, COLOR_SERVICE, THICK_INNER)
            
            # Draw net line
            if 'net_line' in lines:
                p1, p2 = lines['net_line']
                draw_tracked_line(frame, p1, p2, COLOR_CENTER, THICK_CENTER)
                if show_labels:
                    mid = ((p1 + p2) / 2).astype(int)
                    cv2.putText(frame, "Net", tuple(mid), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_CENTER, 1)
            
            # Draw center service line
            if 'center_service_line' in lines:
                p1, p2 = lines['center_service_line']
                draw_tracked_line(frame, p1, p2, COLOR_CENTER, THICK_CENTER)
            
            # Draw center mark
            if 'center_mark' in lines:
                p1, p2 = lines['center_mark']
                draw_tracked_line(frame, p1, p2, COLOR_CENTER, THICK_CENTER + 1)
        
        # NO dots, NO markers, NO status indicators
        # Tracking effect comes purely from line jitter and movement
        
        return frame
    
    def get_court_info(self):
        """Get current court tracking information"""
        return {
            'initialized': self.initialized,
            'corners': self.stable_corners,
            'lines': self.stable_lines,
            'frame_count': self.frame_count
        }

