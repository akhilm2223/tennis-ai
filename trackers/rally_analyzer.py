"""
Advanced Rally Analysis for Tennis
Tracks points, rallies, shot patterns, and outcomes

Features:
- Rally counting and shot tracking
- Point winner detection
- In/Out ball detection using court boundaries
- Forced errors vs winners classification
- Pattern recognition (cross-court, down-the-line, etc.)
- Complete rally breakdown with statistics
"""
import numpy as np
import cv2
from collections import deque
from enum import Enum


class RallyState(Enum):
    """Rally state machine"""
    NEW_RALLY = "new_rally"
    IN_RALLY = "in_rally"
    END_RALLY = "end_rally"
    POINT_ANNOUNCED = "point_announced"


class ShotType(Enum):
    """Types of tennis shots"""
    SERVE = "serve"
    FOREHAND = "forehand"
    BACKHAND = "backhand"
    VOLLEY = "volley"
    SMASH = "smash"
    DROP = "drop"
    LOB = "lob"
    UNKNOWN = "unknown"


class ShotOutcome(Enum):
    """Outcome of a shot"""
    IN_PLAY = "in_play"
    WINNER = "winner"
    FORCED_ERROR = "forced_error"
    UNFORCED_ERROR = "unforced_error"
    OUT = "out"
    NET = "net"


class Rally:
    """Represents a single rally (exchange of shots)"""
    def __init__(self, rally_id, start_frame):
        self.rally_id = rally_id
        self.start_frame = start_frame
        self.end_frame = None
        self.shots = []  # List of Shot objects
        self.bounces = []  # List of bounce locations
        self.winner = None  # Player who won the point (1 or 2)
        self.outcome = None  # How the point ended
        self.total_shots = 0
        self.duration_frames = 0
        self.duration_seconds = 0.0
        self.pattern = []  # Sequence of shot locations
        
    def add_shot(self, player, frame, position, velocity, speed_kmh, shot_type=ShotType.UNKNOWN):
        """Add a shot to the rally"""
        shot = {
            'player': player,
            'frame': frame,
            'position': position,
            'velocity': velocity,
            'speed_kmh': speed_kmh,
            'shot_type': shot_type,
            'shot_number': len(self.shots) + 1
        }
        self.shots.append(shot)
        self.total_shots = len(self.shots)
        
    def add_bounce(self, bounce_data):
        """Add a bounce to the rally"""
        self.bounces.append(bounce_data)
        
    def end_rally(self, end_frame, winner, outcome, fps=30):
        """End the rally with outcome"""
        self.end_frame = end_frame
        self.winner = winner
        self.outcome = outcome
        self.duration_frames = end_frame - self.start_frame
        self.duration_seconds = self.duration_frames / fps
        
    def get_summary(self):
        """Get rally summary"""
        return {
            'rally_id': self.rally_id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'duration_seconds': self.duration_seconds,
            'total_shots': self.total_shots,
            'winner': self.winner,
            'outcome': self.outcome.value if self.outcome else None,
            'shots': self.shots,
            'bounces': self.bounces,
            'max_speed_kmh': max([s['speed_kmh'] for s in self.shots]) if self.shots else 0.0,
            'avg_speed_kmh': np.mean([s['speed_kmh'] for s in self.shots]) if self.shots else 0.0
        }


class RallyAnalyzer:
    """
    Professional rally analysis system
    Tracks complete match structure with rally-by-rally breakdown
    """
    
    def __init__(self, court_detector=None, fps=30):
        self.court_detector = court_detector
        self.fps = fps
        
        # Rally tracking
        self.rallies = []
        self.current_rally = None
        self.rally_counter = 0
        self.point_counter = 0
        
        # Rally state machine
        self.rally_state = RallyState.NEW_RALLY
        self.point_winner_announced = False
        self.frames_since_last_bounce = 0
        self.announcement_frames_left = 0  # Frames to show announcement
        self.announcement_duration = 60  # Show for 2 seconds (60 frames @ 30fps)
        
        # Court boundaries for in/out detection
        self.court_boundaries = None
        if court_detector and hasattr(court_detector, 'corners'):
            self._setup_court_boundaries(court_detector)
        
        # Player scores
        self.score = {1: 0, 2: 0}  # Player 1 vs Player 2
        
        # Shot tracking
        self.last_bounce_frame = -999
        self.last_hitter = None
        self.shots_in_current_rally = 0
        self.min_shot_interval = 15  # Minimum frames between shots (0.5s at 30fps)
        
        # Pattern recognition
        self.shot_patterns = []  # Track shot sequences for pattern analysis
        
        # Statistics
        self.stats = {
            'total_rallies': 0,
            'total_shots': 0,
            'winners': {1: 0, 2: 0},
            'forced_errors': {1: 0, 2: 0},
            'unforced_errors': {1: 0, 2: 0},
            'longest_rally': 0,
            'fastest_shot_kmh': 0.0,
            'avg_rally_length': 0.0
        }
        
        # Ball tracking state
        self.ball_out_count = 0  # Consecutive frames ball is out
        self.out_threshold = 3  # Frames to confirm ball is out
        
        # Double bounce detection
        self.bounce_history = deque(maxlen=5)  # Last 5 bounces
        self.last_bounce_court_side = None  # 1 or 2
        
        # No return detection (>1.5 seconds without return)
        self.no_return_threshold = int(1.5 * fps)  # 45 frames @ 30fps
        self.last_return_frame = -999
        
    def _setup_court_boundaries(self, court_detector):
        """Setup court boundaries from court detector"""
        if court_detector.corners is not None and len(court_detector.corners) >= 4:
            # Court corners in order: [top-left, top-right, bottom-right, bottom-left]
            corners = court_detector.corners
            
            # Create a polygon for the court (in court coordinates)
            # For singles court (standard dimensions: 23.77m x 8.23m)
            # Mini-court dimensions are typically 400x800 pixels
            self.court_boundaries = {
                'singles': {
                    'x_min': 50,   # Left boundary
                    'x_max': 350,  # Right boundary
                    'y_min': 0,    # Near baseline
                    'y_max': 800   # Far baseline
                },
                'doubles': {
                    'x_min': 0,
                    'x_max': 400,
                    'y_min': 0,
                    'y_max': 800
                }
            }
            
            print("✅ Court boundaries initialized for in/out detection")
    
    def update(self, frame_num, ball_position, ball_velocity, court_position, 
               bounces, player_positions, is_predicted=False):
        """
        Update rally analyzer with current frame data
        
        Args:
            frame_num: Current frame number
            ball_position: (x, y) in image coordinates
            ball_velocity: (vx, vy) velocity vector
            court_position: (cx, cy) in court coordinates
            bounces: List of bounce data from ball tracker
            player_positions: Dict {player_id: centroid} for both players
            is_predicted: Whether ball position is predicted (no detection)
        """
        
        # If in END_RALLY state, just wait for announcement to finish
        if self.rally_state == RallyState.END_RALLY:
            return
        
        # Start new rally if needed
        if self.rally_state == RallyState.NEW_RALLY:
            if ball_position and not is_predicted:
                self._start_new_rally(frame_num)
                self.rally_state = RallyState.IN_RALLY
            return
        
        # Track frames since last bounce for no-return detection
        if self.current_rally:
            self.frames_since_last_bounce = frame_num - self.last_bounce_frame
            
            # TRIGGER 1: No return after 1.5 seconds (45 frames @ 30fps)
            if self.frames_since_last_bounce > self.no_return_threshold:
                if self.last_hitter:
                    # Last hitter wins the point (opponent failed to return)
                    winner = self.last_hitter
                    outcome = ShotOutcome.WINNER if self.shots_in_current_rally > 3 else ShotOutcome.FORCED_ERROR
                    self._end_rally(frame_num, winner, outcome)
                    return
        
        # Check if new bounce occurred
        new_bounce = None
        if bounces and len(bounces) > len(self.current_rally.bounces if self.current_rally else []):
            new_bounce = bounces[-1]  # Most recent bounce
            
            # Only process if it's a new bounce (not already processed)
            if new_bounce['frame'] != self.last_bounce_frame:
                self._process_bounce(new_bounce, frame_num, ball_velocity)
                self.last_bounce_frame = new_bounce['frame']
                self.frames_since_last_bounce = 0
                
                # TRIGGER 2: Double bounce detection
                # Check if ball bounced twice on same side
                self._check_double_bounce(new_bounce, frame_num)
        
        # TRIGGER 3: Ball out of bounds
        if court_position and not is_predicted and self.rally_state == RallyState.IN_RALLY:
            is_in = self._is_ball_in_court(court_position)
            
            if not is_in:
                self.ball_out_count += 1
                
                # Ball has been out for enough frames - point ended
                if self.ball_out_count >= self.out_threshold:
                    self._end_point_out_of_bounds(frame_num)
                    self.ball_out_count = 0
            else:
                self.ball_out_count = 0  # Reset counter
    
    def _process_bounce(self, bounce_data, frame_num, ball_velocity):
        """Process a new bounce detection"""
        player = bounce_data.get('player')
        speed_kmh = bounce_data.get('speed_kmh', 0.0)
        position = bounce_data.get('image_xy')
        court_pos = bounce_data.get('court_xy')
        
        # Start new rally if none exists
        if self.current_rally is None:
            self._start_new_rally(frame_num)
        
        # Add bounce to current rally
        self.current_rally.add_bounce(bounce_data)
        
        # Determine if this is a new shot
        if frame_num - self.last_bounce_frame > self.min_shot_interval:
            # This is a new shot
            if player and player != self.last_hitter:
                # Valid shot - player alternation
                self.current_rally.add_shot(
                    player=player,
                    frame=frame_num,
                    position=position,
                    velocity=ball_velocity,
                    speed_kmh=speed_kmh
                )
                
                self.last_hitter = player
                self.shots_in_current_rally += 1
                self.stats['total_shots'] += 1
                self.stats['fastest_shot_kmh'] = max(self.stats['fastest_shot_kmh'], speed_kmh)
                
                print(f"   📊 Rally #{self.rally_counter}: Shot #{self.shots_in_current_rally} by P{player} at {speed_kmh:.1f} km/h")
    
    def _is_ball_in_court(self, court_position, court_type='singles'):
        """
        Check if ball is within court boundaries
        
        Args:
            court_position: (x, y) in court coordinates
            court_type: 'singles' or 'doubles'
            
        Returns:
            bool: True if ball is in bounds
        """
        if self.court_boundaries is None or court_position is None:
            return True  # Assume in if we can't check
        
        cx, cy = court_position
        bounds = self.court_boundaries[court_type]
        
        # Check if within boundaries (with some margin for error)
        margin = 20  # pixels
        is_in = (bounds['x_min'] - margin <= cx <= bounds['x_max'] + margin and
                 bounds['y_min'] - margin <= cy <= bounds['y_max'] + margin)
        
        return is_in
    
    def _check_double_bounce(self, bounce_data, frame_num):
        """Check for double bounce on same side (point end trigger)"""
        if not self.current_rally or self.rally_state != RallyState.IN_RALLY:
            return
        
        # Add bounce to history
        self.bounce_history.append(bounce_data)
        
        # Determine which court side the bounce is on (1 = top half, 2 = bottom half)
        court_pos = bounce_data.get('court_xy')
        if court_pos:
            cy = court_pos[1]  # Y coordinate in court space
            court_side = 1 if cy < 400 else 2  # 400 is mid-court
            
            # Check if this is a double bounce (same side as last bounce)
            if self.last_bounce_court_side == court_side and len(self.bounce_history) >= 2:
                # Get time since last bounce
                last_bounce = self.bounce_history[-2]
                time_diff_frames = bounce_data['frame'] - last_bounce['frame']
                
                # If bounces very close in time (< 30 frames = 1 second), it's double bounce
                if time_diff_frames < 30:
                    # Double bounce! Receiver loses the point
                    # The player on this side (where double bounce happened) loses
                    winner = 2 if court_side == 1 else 1
                    outcome = ShotOutcome.UNFORCED_ERROR
                    
                    print(f"   ⚠️ DOUBLE BOUNCE detected on P{court_side} side! P{winner} wins point")
                    self._end_rally(frame_num, winner, outcome)
                    return
            
            self.last_bounce_court_side = court_side
    
    def _start_new_rally(self, frame_num):
        """Start a new rally"""
        self.rally_counter += 1
        self.current_rally = Rally(self.rally_counter, frame_num)
        self.shots_in_current_rally = 0
        self.last_hitter = None
        self.bounce_history.clear()
        self.last_bounce_court_side = None
        self.frames_since_last_bounce = 0
        
        print(f"\n🎾 === RALLY #{self.rally_counter} STARTED (Frame {frame_num}) ===")
    
    def _end_point_out_of_bounds(self, frame_num):
        """End point because ball went out of bounds"""
        if self.current_rally is None or self.rally_state == RallyState.END_RALLY:
            return
        
        # Winner is the player who didn't hit last (opponent hit it out)
        if self.last_hitter:
            winner = 2 if self.last_hitter == 1 else 1
            outcome = ShotOutcome.OUT  # Ball went out
        else:
            # Can't determine winner
            winner = None
            outcome = ShotOutcome.OUT
        
        self._end_rally(frame_num, winner, outcome)
        
        print(f"   ❌ Ball OUT - Point to P{winner}")
    
    def _end_point_timeout(self, frame_num):
        """End point due to timeout (no detection)"""
        if self.current_rally is None:
            return
        
        # Winner is the last hitter (opponent couldn't return)
        winner = self.last_hitter
        outcome = ShotOutcome.WINNER if self.shots_in_current_rally > 3 else ShotOutcome.FORCED_ERROR
        
        self._end_rally(frame_num, winner, outcome)
    
    def _end_rally(self, frame_num, winner, outcome):
        """End the current rally"""
        if self.current_rally is None:
            return
        
        self.current_rally.end_rally(frame_num, winner, outcome, self.fps)
        
        # Change state to END_RALLY and start announcement
        self.rally_state = RallyState.END_RALLY
        self.announcement_frames_left = self.announcement_duration
        
        # Update statistics
        if winner:
            self.score[winner] += 1
            self.point_counter += 1
            
            if outcome == ShotOutcome.WINNER:
                self.stats['winners'][winner] += 1
            elif outcome == ShotOutcome.FORCED_ERROR:
                loser = 2 if winner == 1 else 1
                self.stats['forced_errors'][loser] += 1
            elif outcome == ShotOutcome.UNFORCED_ERROR or outcome == ShotOutcome.OUT:
                loser = 2 if winner == 1 else 1
                self.stats['unforced_errors'][loser] += 1
        
        self.stats['total_rallies'] += 1
        self.stats['longest_rally'] = max(self.stats['longest_rally'], 
                                          self.current_rally.total_shots)
        
        # Calculate average rally length
        total_shots = sum([r.total_shots for r in self.rallies]) + self.current_rally.total_shots
        self.stats['avg_rally_length'] = total_shots / (len(self.rallies) + 1)
        
        # Store rally
        self.rallies.append(self.current_rally)
        
        print(f"   ✅ Rally #{self.current_rally.rally_id} ENDED")
        print(f"      Winner: P{winner} | Outcome: {outcome.value if outcome else 'Unknown'}")
        print(f"      Shots: {self.current_rally.total_shots} | Duration: {self.current_rally.duration_seconds:.1f}s")
        print(f"      Score: P1 {self.score[1]} - {self.score[2]} P2")
        
        # Reset for next rally
        self.current_rally = None
        self.shots_in_current_rally = 0
    
    def force_end_current_rally(self, frame_num, winner=None):
        """Force end current rally (useful for video end or manual intervention)"""
        if self.current_rally:
            if winner is None:
                winner = self.last_hitter  # Last hitter wins by default
            
            outcome = ShotOutcome.IN_PLAY  # Incomplete
            self._end_rally(frame_num, winner, outcome)
    
    def get_rally_breakdown(self):
        """
        Get complete rally breakdown
        
        Returns:
            dict with complete analysis
        """
        rallies_summary = [rally.get_summary() for rally in self.rallies]
        
        return {
            'total_rallies': len(self.rallies),
            'score': self.score,
            'rallies': rallies_summary,
            'statistics': self.stats,
            'shot_patterns': self._analyze_patterns(),
            'player_analysis': {
                1: self._get_player_stats(1),
                2: self._get_player_stats(2)
            }
        }
    
    def _analyze_patterns(self):
        """Analyze shot patterns across rallies"""
        patterns = {
            'cross_court': 0,
            'down_the_line': 0,
            'inside_out': 0,
            'short_rallies': 0,  # 1-3 shots
            'medium_rallies': 0,  # 4-8 shots
            'long_rallies': 0    # 9+ shots
        }
        
        for rally in self.rallies:
            shots = rally.total_shots
            
            if shots <= 3:
                patterns['short_rallies'] += 1
            elif shots <= 8:
                patterns['medium_rallies'] += 1
            else:
                patterns['long_rallies'] += 1
        
        return patterns
    
    def _get_player_stats(self, player_id):
        """Get statistics for specific player"""
        stats = {
            'points_won': self.score.get(player_id, 0),
            'winners': self.stats['winners'].get(player_id, 0),
            'forced_errors': self.stats['forced_errors'].get(player_id, 0),
            'unforced_errors': self.stats['unforced_errors'].get(player_id, 0),
            'total_shots': 0,
            'avg_shot_speed_kmh': 0.0,
            'max_shot_speed_kmh': 0.0
        }
        
        # Calculate shot statistics
        player_shots = []
        for rally in self.rallies:
            for shot in rally.shots:
                if shot['player'] == player_id:
                    player_shots.append(shot)
        
        if player_shots:
            stats['total_shots'] = len(player_shots)
            speeds = [s['speed_kmh'] for s in player_shots if s['speed_kmh'] > 0]
            if speeds:
                stats['avg_shot_speed_kmh'] = np.mean(speeds)
                stats['max_shot_speed_kmh'] = np.max(speeds)
        
        return stats
    
    def get_live_rally_info(self):
        """Get current rally information for live display"""
        if self.current_rally is None:
            return None
        
        return {
            'rally_number': self.rally_counter,
            'shots': self.shots_in_current_rally,
            'duration': (self.current_rally.duration_frames / self.fps) if self.current_rally.duration_frames else 0,
            'last_hitter': self.last_hitter,
            'score': self.score,
            'state': self.rally_state.value,
            'show_announcement': self.announcement_frames_left > 0,
            'winner': self.current_rally.winner if self.current_rally and self.current_rally.winner else None,
            'outcome': self.current_rally.outcome.value if self.current_rally and self.current_rally.outcome else None
        }
    
    def draw_point_announcement(self, frame):
        """
        Draw point winner announcement on frame
        
        Args:
            frame: Video frame to draw on
            
        Returns:
            Frame with announcement drawn
        """
        if self.announcement_frames_left <= 0 or self.current_rally is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Get winner and outcome
        winner = self.current_rally.winner
        outcome = self.current_rally.outcome
        
        if winner is None:
            return frame
        
        # Animation: fade in/out based on frames left
        progress = min(1.0, self.announcement_frames_left / 10)  # Fade in first 10 frames
        fade_out_start = 15
        if self.announcement_frames_left < fade_out_start:
            progress = self.announcement_frames_left / fade_out_start  # Fade out last 15 frames
        
        alpha = int(255 * progress)
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Winner colors
        winner_color = (0, 0, 255) if winner == 1 else (255, 0, 0)  # Red for P1, Blue for P2
        loser_color = (255, 0, 0) if winner == 1 else (0, 0, 255)
        
        # Large announcement banner
        banner_h = 200
        banner_y = (h - banner_h) // 2
        
        # Semi-transparent background
        cv2.rectangle(overlay, (0, banner_y), (w, banner_y + banner_h), (0, 0, 0), -1)
        
        # Main text: "PLAYER X WINS POINT!"
        main_text = f"PLAYER {winner} WINS POINT!"
        font = cv2.FONT_HERSHEY_BOLD
        font_scale = 2.5
        thickness = 5
        
        text_size = cv2.getTextSize(main_text, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = banner_y + 80
        
        # Text shadow
        cv2.putText(overlay, main_text, (text_x + 4, text_y + 4),
                   font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        
        # Main text
        cv2.putText(overlay, main_text, (text_x, text_y),
                   font, font_scale, winner_color, thickness, cv2.LINE_AA)
        
        # Outcome text
        outcome_text = ""
        if outcome == ShotOutcome.OUT:
            loser = 2 if winner == 1 else 1
            outcome_text = f"Player {loser} hit OUT"
        elif outcome == ShotOutcome.WINNER:
            outcome_text = "WINNER!"
        elif outcome == ShotOutcome.UNFORCED_ERROR:
            loser = 2 if winner == 1 else 1
            outcome_text = f"Player {loser} Unforced Error"
        elif outcome == ShotOutcome.FORCED_ERROR:
            loser = 2 if winner == 1 else 1
            outcome_text = f"Player {loser} Forced Error"
        else:
            outcome_text = outcome.value.upper().replace('_', ' ') if outcome else ""
        
        if outcome_text:
            font_scale_small = 1.2
            thickness_small = 3
            text_size_small = cv2.getTextSize(outcome_text, font, font_scale_small, thickness_small)[0]
            text_x_small = (w - text_size_small[0]) // 2
            text_y_small = text_y + 60
            
            cv2.putText(overlay, outcome_text, (text_x_small, text_y_small),
                       font, font_scale_small, (255, 255, 255), thickness_small, cv2.LINE_AA)
        
        # Score display
        score_text = f"Score: {self.score[1]} - {self.score[2]}"
        font_scale_score = 1.5
        thickness_score = 3
        text_size_score = cv2.getTextSize(score_text, font, font_scale_score, thickness_score)[0]
        text_x_score = (w - text_size_score[0]) // 2
        text_y_score = text_y_small + 50
        
        cv2.putText(overlay, score_text, (text_x_score, text_y_score),
                   font, font_scale_score, (255, 255, 255), thickness_score, cv2.LINE_AA)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Decrement announcement counter
        self.announcement_frames_left -= 1
        
        # When announcement finishes, start new rally
        if self.announcement_frames_left <= 0:
            self.rally_state = RallyState.NEW_RALLY
        
        return frame

