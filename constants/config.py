"""
Configuration constants for Tennis Match Analysis Pipeline
"""

# Model paths
PLAYER_MODEL = "yolov8n.pt"  # Using COCO pretrained model
BALL_MODEL = "yolov8n.pt"    # Same model, different class

# Class IDs from COCO dataset
PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32  # Sports ball in COCO

# Detection thresholds
CONF_THRESH = 0.15  # Lowered from 0.25 for better ball detection
IOU_THRESH = 0.45

# Video settings
FPS = 30

# Court dimensions (singles tennis court in meters)
COURT_WIDTH = 8.23   # meters
COURT_LENGTH = 23.77  # meters

# Calibration
PIXEL_TO_METER = 0.02  # Default, will be updated by homography

# Mini-court visualization settings
MINI_COURT_WIDTH = 90   # Compact size (50% of previous)
MINI_COURT_HEIGHT = 180  # Maintains 1:2 aspect ratio
MINI_COURT_PADDING = 8   # Proportional padding

# Tracking settings
MAX_MISSING_FRAMES = 15  # Increased from 10 to keep ball tracking longer
KALMAN_PROCESS_NOISE = 0.05  # Reduced for smoother predictions
KALMAN_MEASUREMENT_NOISE = 0.1

# Colors (BGR format)
COLOR_PLAYER1 = (0, 0, 255)    # Red
COLOR_PLAYER2 = (255, 0, 0)    # Blue
COLOR_BALL = (255, 255, 255)   # White
COLOR_TRAIL = (0, 255, 255)    # Yellow

# Speed calculation
SPEED_SMOOTHING_FRAMES = 5  # Rolling average window

