"""
AGGRESSIVE CLEANUP - Remove ALL non-essential files
Keep ONLY what's needed to run main_pose.py
"""
import os
import shutil

# Documentation files (can delete - rewrite later)
MARKDOWN_FILES = [
    "ADVANCED_ANALYSIS_FEATURES.md",
    "APPLICATION_DESCRIPTION_PARAGRAPH.md",
    "CHANGELOG.md",
    "COACH_INTEGRATION_GUIDE.md",
    "COMPLETE_APPLICATION_DESCRIPTION.md",
    "COMPLETE_TRACKING_SUMMARY.md",
    "COURT_DETECTION_OPTIONS.md",
    "COURT_LINE_DETECTOR_SETUP.md",
    "COURT_LINE_TRACKING.md",
    "COURT_LINES_EXPLAINED.md",
    "FEATURES_SUMMARY.md",
    "FULLSTACK_SETUP.md",
    "GPU_STATUS.md",
    "MANUAL_CALIBRATION_GUIDE.md",
    "ML_KEYPOINT_CONNECTIONS.md",
    "POINT_WINNER_DETECTION.md",
    "PORT_CHANGE_README.md",
    "START_FULLSTACK.md",
    "SYSTEM_OVERVIEW.md",
]

# Backend/server files (not needed for analysis)
BACKEND_FILES = [
    "backend_server.py",
    "voice_coach_server.py",
    "tts_service.py",
    "run_backend.py",
    "backend_requirements.txt",
]

# Calibration tools (not needed if you have court_lines_manual.json)
CALIBRATION_FILES = [
    "manual_court_calibration.py",  # Already have calibrated lines
    "court_calibration.json",  # Old calibration
]

# Misc files
MISC_FILES = [
    "ball.glb",  # 3D model (not used in main_pose.py)
    "package-lock.json",  # Not Python
    "CLEANUP_LIST.txt",  # Already ran cleanup
]

# Entire folders to remove
FOLDERS_TO_DELETE = [
    "RAG_MentalCoach",  # Separate coach system - keep if building AI coach
    "tennis-3d-viewer",  # 3D viewer (separate React app)
    "scripts",  # Example scripts (not needed)
    "uploads",  # Empty folder
    "venv",  # Python virtual environment (can recreate)
    "__pycache__",  # Python cache
    "modules/__pycache__",  
    "constants/__pycache__",
    "trackers/__pycache__",
]

def main():
    print("=" * 70)
    print("AGGRESSIVE CLEANUP - Remove ALL non-essential files")
    print("=" * 70)
    
    total_files = len(MARKDOWN_FILES) + len(BACKEND_FILES) + len(CALIBRATION_FILES) + len(MISC_FILES)
    total_folders = len(FOLDERS_TO_DELETE)
    
    print(f"\n📄 Markdown docs to delete: {len(MARKDOWN_FILES)}")
    print(f"🖥️  Backend files to delete: {len(BACKEND_FILES)}")
    print(f"📐 Calibration files to delete: {len(CALIBRATION_FILES)}")
    print(f"📦 Misc files to delete: {len(MISC_FILES)}")
    print(f"📁 Folders to delete: {len(FOLDERS_TO_DELETE)}")
    print(f"\n🗑️  TOTAL: {total_files} files + {total_folders} folders")
    
    print("\n⚠️  WARNING: This will delete:")
    print("   - All .md documentation files")
    print("   - Backend/server files")
    print("   - RAG_MentalCoach folder (can restore if needed)")
    print("   - tennis-3d-viewer folder")
    print("   - venv folder (can recreate)")
    print("   - scripts folder")
    
    print("\n✅ Files that will be KEPT:")
    print("   - main_pose.py")
    print("   - court_lines_manual.json")
    print("   - copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov")
    print("   - output_videos/ (analysis results)")
    print("   - All core modules (trackers/, modules/, yolo, etc.)")
    print("   - manual_court_lines_full.py (for recalibration)")
    print("   - visualize_manual_lines.py (verification)")
    print("   - requirements.txt")
    
    response = input("\n⚠️  Proceed with AGGRESSIVE cleanup? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("❌ Cleanup cancelled")
        return
    
    deleted_files = 0
    deleted_folders = 0
    failed = 0
    
    print("\n🗑️  Deleting files...\n")
    
    # Delete markdown files
    all_files = MARKDOWN_FILES + BACKEND_FILES + CALIBRATION_FILES + MISC_FILES
    for file_path in all_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"   ✅ {file_path}")
                deleted_files += 1
            except Exception as e:
                print(f"   ❌ Failed: {file_path} - {e}")
                failed += 1
    
    # Delete folders
    print("\n🗑️  Deleting folders...\n")
    for folder in FOLDERS_TO_DELETE:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"   ✅ Deleted folder: {folder}/")
                deleted_folders += 1
            except Exception as e:
                print(f"   ❌ Failed: {folder}/ - {e}")
                failed += 1
    
    print("\n" + "=" * 70)
    print("✅ AGGRESSIVE CLEANUP COMPLETE!")
    print(f"   Files deleted: {deleted_files}")
    print(f"   Folders deleted: {deleted_folders}")
    print(f"   Failed: {failed}")
    print("=" * 70)
    
    print("\n📂 Your folder is now MINIMAL - only essential files remain!")
    print("   You can rewrite documentation later as needed.")

if __name__ == "__main__":
    main()

