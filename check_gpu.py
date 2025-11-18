
import torch
import sys

print("=" * 60)
print("GPU/CUDA Status Check")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("\n⚠️  CUDA not available - will use CPU")
    print("   (This is fine, but processing will be slower)")

# Check MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f"\n✅ Apple Silicon (MPS) available: {torch.backends.mps.is_available()}")
elif sys.platform == 'darwin':
    print("\n⚠️  MPS not available (Apple Silicon GPU)")

print("\n" + "=" * 60)

