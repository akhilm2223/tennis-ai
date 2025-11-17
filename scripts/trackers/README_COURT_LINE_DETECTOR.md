# Court Line Detector

ML-based tennis court line detector using ResNet50 to predict 14 keypoints representing court lines and corners.

## Overview

The `CourtLineDetector` class uses a pre-trained ResNet50 model fine-tuned to detect 14 keypoints on a tennis court:
- **Points 0-3**: Outer court corners (baseline corners)
- **Points 4-7**: Service line corners
- **Points 8-11**: Net line corners  
- **Points 12-13**: Center service line points

## Model Architecture

- **Base Model**: ResNet50 (pretrained on ImageNet)
- **Output Layer**: Fully connected layer with 28 outputs (14 keypoints × 2 coordinates)
- **Input**: 224×224 RGB image (normalized)
- **Output**: 28 values [x0, y0, x1, y1, ..., x13, y13] scaled to original frame size

## Usage

### Basic Usage

```python
from trackers.court_line_detector import CourtLineDetector
import cv2

# Initialize detector with trained model
detector = CourtLineDetector('models/court_keypoints.pt')

# Read frame
frame = cv2.imread('tennis_frame.jpg')

# Predict keypoints
keypoints = detector.predict(frame)

# Draw keypoints (points only, no lines)
frame_with_keypoints = detector.draw_keypoints(frame, keypoints)

# Get court corners for homography
corners = detector.get_court_corners(keypoints)
```

### Integration with Auto Detection

The court line detector is integrated into `auto_detect_court.py`:

```python
from auto_detect_court import detect_court_automatic

# Use ML-based detection
court_detector = detect_court_automatic(
    frame, 
    use_ml_detector=True,
    ml_model_path='models/court_keypoints.pt'
)

# Access detected keypoints
keypoints = court_detector.keypoints
corners = court_detector.corners
```

### Testing the Detector

Use the test script to visualize keypoint detection:

```bash
python scripts/test_court_line_detector.py input_videos/tennis_input.mp4 models/court_keypoints.pt output_videos/court_lines.mp4
```

## Model Training

To train your own model, you'll need:

1. **Dataset**: Tennis court images with annotated keypoints
2. **Training Script**: PyTorch training loop with:
   - ResNet50 base model
   - MSE loss for keypoint regression
   - Data augmentation (rotation, scaling, color jitter)
   - Learning rate scheduling

Example training structure:

```python
import torch
import torch.nn as nn
from torchvision import models

# Create model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 28)  # 14 keypoints × 2

# Loss function
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for images, keypoints in dataloader:
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        # ... backprop and optimization
```

## Model File Format

The model should be saved as a PyTorch state dict:

```python
torch.save(model.state_dict(), 'court_keypoints.pt')
```

Place the trained model in the `models/` directory.

## Keypoint Layout

Standard tennis court keypoint numbering:

```
        0 -------- 1
        |          |
        4 -------- 5
        |    12    |
    8 ------ 13 ------ 9
        |          |
        6 -------- 7
        |          |
        2 -------- 3
```

- **0-3**: Outer boundary corners
- **4-7**: Service line intersections
- **8-11**: Net line endpoints
- **12-13**: Center service line points

## Performance

- **Inference Speed**: ~10-20ms per frame (CPU)
- **Accuracy**: Depends on training data quality
- **Robustness**: Works best with clear court visibility

## Fallback Behavior

If ML detection fails or model is not available, the system automatically falls back to traditional detection methods:
1. Color-based court detection (green/blue courts)
2. Line-based detection (Hough transform)
3. Default court region estimation

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
```

## Notes

- The detector expects a trained model file. If you don't have one, the system will use traditional detection methods.
- For best results, ensure the court is clearly visible in the frame.
- The model can be retrained on your specific court types for better accuracy.
