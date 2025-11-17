"""
Configuration for MediaPipe Pose tracking
"""

# MediaPipe Pose settings (optimized for stability)
POSE_MIN_DETECTION_CONF = 0.3    # Minimum detection confidence (lower = more sticky)
POSE_MIN_TRACKING_CONF = 0.3     # Minimum tracking confidence (lower = keeps tracking)
POSE_MODEL_COMPLEXITY = 1        # 0=lite, 1=full, 2=heavy
POSE_SMOOTH_LANDMARKS = True     # Enable landmark smoothing
POSE_ENABLE_SEGMENTATION = False # Disable segmentation (faster)

# Centroid smoothing (increased for stability)
POSE_CENTROID_SMOOTH_FRAMES = 10  # Smoothing window for centroids (higher = more stable)

# Tracking continuity
POSE_MAX_MISSING_FRAMES = 15     # Keep last pose for up to N frames when detection fails

# Player filtering (to ignore audience/non-players)
POSE_MIN_PLAYER_SIZE = 0.01      # Minimum pose size as fraction of frame (1%)
POSE_IGNORE_TOP_PERCENT = 0.20   # Ignore detections in top 20% (audience)
POSE_IGNORE_BOTTOM_PERCENT = 0.05 # Ignore detections in bottom 5% (scoreboard)
POSE_MIN_VISIBLE_LANDMARKS = 10  # Minimum visible joints to count as player

# Skeleton visualization
SKELETON_BONE_THICKNESS = 3      # Thickness of skeleton bones
SKELETON_JOINT_RADIUS = 4        # Radius of joint circles
SKELETON_DRAW_JOINTS = True      # Draw joint circles
SKELETON_DRAW_BONES = True       # Draw bone connections
SKELETON_DRAW_CENTROID = True    # Draw centroid marker

# Keypoints for analysis (MediaPipe landmark IDs)
KEYPOINT_LEFT_WRIST = 15
KEYPOINT_RIGHT_WRIST = 16
KEYPOINT_LEFT_SHOULDER = 11
KEYPOINT_RIGHT_SHOULDER = 12
KEYPOINT_LEFT_HIP = 23
KEYPOINT_RIGHT_HIP = 24
KEYPOINT_LEFT_KNEE = 25
KEYPOINT_RIGHT_KNEE = 26
KEYPOINT_LEFT_ANKLE = 27
KEYPOINT_RIGHT_ANKLE = 28

# Centroid calculation points (shoulders + hips)
CENTROID_KEYPOINTS = [11, 12, 23, 24]  # L_SHOULDER, R_SHOULDER, L_HIP, R_HIP

# Visibility threshold
MIN_KEYPOINT_VISIBILITY = 0.5    # Minimum visibility to use a keypoint

