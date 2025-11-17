"""
Download Custom Tennis Ball Detection Model (best.pt)

This script helps you download the custom-trained YOLOv5 model
for superior tennis ball detection accuracy.
"""
import os
import sys

def main():
    print("=" * 60)
    print("🎾 CUSTOM TENNIS BALL MODEL DOWNLOAD")
    print("=" * 60)
    print()
    
    # Check if model already exists
    if os.path.exists("models/best.pt"):
        print("✅ Custom model already exists: models/best.pt")
        
        # Test the model
        print("\n🧪 Testing model...")
        try:
            from ultralytics import YOLO
            model = YOLO('models/best.pt')
            print("✅ Model loaded successfully!")
            print(f"   Model type: {type(model)}")
            print()
            print("You're all set! The custom model is ready to use.")
            return
        except Exception as e:
            print(f"⚠️  Model file exists but failed to load: {e}")
            print("   Please re-download the model.")
    
    print("📥 DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("The custom tennis ball model (best.pt) must be downloaded")
    print("from Google Drive due to its size (~14 MB).")
    print()
    print("🔗 DOWNLOAD LINK:")
    print("   https://drive.google.com/drive/folders/1kzcLn6nF_X-Jj0O7J8RzVXSIw7G-zJNH")
    print()
    print("📋 STEPS:")
    print("   1. Click the link above")
    print("   2. Find and download 'best.pt'")
    print("   3. Place it in the 'models/' directory")
    print()
    print("📁 EXPECTED LOCATION:")
    print("   models/best.pt")
    print()
    print("=" * 60)
    print()
    
    # Try automated download with gdown (if available)
    try:
        import gdown
        print("🚀 Attempting automated download with gdown...")
        print()
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Google Drive file ID (you'll need to update this with the actual file ID)
        # For now, we provide instructions
        print("⚠️  Automated download requires the specific Google Drive file ID.")
        print("   Please download manually using the link above.")
        print()
        
    except ImportError:
        print("💡 TIP: Install 'gdown' for automated downloads:")
        print("   pip install gdown")
        print()
    
    print("After downloading, run this script again to verify the model.")
    print()

if __name__ == "__main__":
    main()

