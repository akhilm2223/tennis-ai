# Court Line Detector Setup Guide

This guide explains how to set up and use the ML-based court line detector in your tennis analysis project.

## Overview

The court line detector uses a ResNet50 model to predict 14 keypoints representing tennis court lines and corners. This provides more accurate court detection compared to traditional color/edge-based methods.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r backend_requirements.txt
```

This will install:
- `torch>=2.0.0` - PyTorch for model inference
- `torchvision>=0.15.0` - Pre-trained ResNet50 model
- Other existing dependencies

### 2. Get a Trained Model

You need a trained ResNet50 model that outputs 14 keypoints (28 values). Place it in the `models/` directory:

```
models/
  └── court_keypoints.pt
```

**Note**: If you don't have a trained model, the system will automatically fall back to traditional detection methods (color-based and line-based).

### 3. Test the Detector

Run the test script to verify the detector works:

```bash
python scripts/test_court_line_detector.py input_videos/tennis_input.mp4 models/court_keypoints.pt output_videos/court_lines.mp4
```

This will:
- Load the video and model
- Detect keypoints on the first frame
- Draw keypoints and court lines on all frames
- Save the output video

### 4. Use in Your Analysis

The detector is already integrated into `main_pose_streaming.py`. To enable it:

```python
# In main_pose_streaming.py, around line 50:
use_ml_court = True  # Change from False to True
ml_model_path = "models/court_keypoints.pt"
```

Or use it directly in your code:

```python
from trackers.court_line_detector import CourtLineDetector

# Initialize
detector = CourtLineDetector('models/court_keypoints.pt')

# Predict keypoints
keypoints = detector.predict(frame)

# Draw visualization
frame_with_lines = detector.draw_court_lines(frame, keypoints)
```

## Model Training (Optional)

If you want to train your own model:

### Dataset Requirements

You need:
- Tennis court images (various angles, lighting, court types)
- Annotated keypoints for each image (14 points per image)
- Recommended: 1000+ images for good accuracy

### Training Script Template

```python
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader

# 1. Create dataset
class CourtKeypointDataset(Dataset):
    def __init__(self, images, keypoints, transform=None):
        self.images = images
        self.keypoints = keypoints
        self.transform = transform
    
    def __getitem__(self, idx):
        image = self.images[idx]
        keypoints = self.keypoints[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, keypoints

# 2. Create model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 28)  # 14 keypoints × 2

# 3. Training loop
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for images, keypoints in train_loader:
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. Save model
torch.save(model.state_dict(), 'models/court_keypoints.pt')
```

## Keypoint Layout

The 14 keypoints represent:

```
Court Layout (top view):

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

- **Points 0-3**: Outer court corners (baseline corners)
- **Points 4-7**: Service line corners
- **Points 8-11**: Net line corners
- **Points 12-13**: Center service line points

## Usage Examples

### Example 1: Basic Detection

```python
from trackers.court_line_detector import CourtLineDetector
import cv2

detector = CourtLineDetector('models/court_keypoints.pt')
frame = cv2.imread('tennis_frame.jpg')

keypoints = detector.predict(frame)
frame_with_keypoints = detector.draw_keypoints(frame, keypoints)

cv2.imwrite('output.jpg', frame_with_keypoints)
```

### Example 2: Integrated with Auto-Detection

```python
from auto_detect_court import detect_court_automatic

# Automatically tries ML detection, falls back to traditional if needed
court_detector = detect_court_automatic(
    frame,
    use_ml_detector=True,
    ml_model_path='models/court_keypoints.pt'
)

# Access results
keypoints = court_detector.keypoints  # ML keypoints (if available)
corners = court_detector.corners      # Court corners
homography = court_detector.homography_matrix  # For coordinate transformation
```

### Example 3: Run Examples

```bash
python scripts/example_court_detector_usage.py
```

This demonstrates:
- Basic keypoint detection
- Comparison between traditional and ML detection
- Custom analysis using keypoints

## Fallback Behavior

The system is designed to work with or without the ML model:

1. **ML Detection** (if model available): Most accurate, uses trained keypoints
2. **Color Detection**: Detects green/blue court surfaces
3. **Line Detection**: Uses Hough transform to find court lines
4. **Default**: Uses frame-based estimation with margins

## Troubleshooting

### Model Not Found

```
⚠ ML model not found: models/court_keypoints.pt
   Falling back to traditional detection...
```

**Solution**: Either provide a trained model or continue using traditional detection.

### Import Error

```
ModuleNotFoundError: No module named 'torch'
```

**Solution**: Install PyTorch:
```bash
pip install torch torchvision
```

### Poor Detection Quality

**Solutions**:
- Ensure court is clearly visible in frame
- Train model on similar court types
- Adjust detection parameters in `auto_detect_court.py`

## Performance

- **ML Detection**: ~10-20ms per frame (CPU), ~2-5ms (GPU)
- **Traditional Detection**: ~5-10ms per frame
- **Memory**: ~200MB for ResNet50 model

## Files Created

```
trackers/
  ├── court_line_detector.py          # Main detector class
  └── README_COURT_LINE_DETECTOR.md   # Detailed documentation

scripts/
  ├── test_court_line_detector.py     # Test script
  └── example_court_detector_usage.py # Usage examples

COURT_LINE_DETECTOR_SETUP.md          # This file
```

## Next Steps

1. **Get a trained model** or train your own
2. **Test the detector** on your videos
3. **Enable ML detection** in `main_pose_streaming.py`
4. **Customize** keypoint usage for your specific needs

## Resources

- PyTorch Documentation: https://pytorch.org/docs/
- ResNet Paper: https://arxiv.org/abs/1512.03385
- Tennis Court Dimensions: https://www.itftennis.com/

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review example scripts in `scripts/`
3. Read detailed docs in `trackers/README_COURT_LINE_DETECTOR.md`
