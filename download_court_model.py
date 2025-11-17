"""
Download the Court Line Detection Model (keypoints_model.pth)

This model is a ResNet-50 CNN trained to detect 14 keypoints on a tennis court.
"""

print("=" * 70)
print("🎾 COURT LINE DETECTION MODEL DOWNLOAD INSTRUCTIONS")
print("=" * 70)
print()
print("The court line detection model is NOT included in this repository")
print("due to its large size (90.2 MB).")
print()
print("📥 DOWNLOAD INSTRUCTIONS:")
print()
print("1. Go to Google Drive:")
print("   https://drive.google.com/drive/folders/1kzcLn6nF_X-Jj0O7J8RzVXSIw7G-zJNH")
print()
print("2. Find and download:")
print("   📄 keypoints_model.pth (90.2 MB)")
print()
print("3. Place it in the models/ folder:")
print("   models/keypoints_model.pth")
print()
print("=" * 70)
print()
print("🔍 WHAT IS THIS MODEL?")
print()
print("• Type: ResNet-50 CNN (NOT YOLO!)")
print("• Task: Keypoint detection (14 points on tennis court)")
print("• Output: 28 values (x,y for each of 14 keypoints)")
print("• Training: Custom dataset of tennis courts with labeled keypoints")
print("• Framework: PyTorch")
print()
print("=" * 70)
print()
print("📦 CURRENT STATUS:")
print()

import os

if os.path.exists("models/keypoints_model.pth"):
    size_mb = os.path.getsize("models/keypoints_model.pth") / (1024 * 1024)
    print(f"✅ Model found: models/keypoints_model.pth ({size_mb:.1f} MB)")
    print()
    print("You're ready to use court line detection!")
    print()
    print("Run with:")
    print("  python main_pose.py --video your_video.mp4 --court-model models/keypoints_model.pth")
else:
    print("❌ Model NOT found: models/keypoints_model.pth")
    print()
    print("Please download it from the link above.")
    print()
    print("After downloading:")
    print("  1. Create models/ folder if it doesn't exist")
    print("  2. Move keypoints_model.pth to models/")
    print("  3. Run this script again to verify")

print()
print("=" * 70)

