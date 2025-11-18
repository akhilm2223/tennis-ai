"""
Safe cleanup script - Removes test files, duplicates, and old outputs
Keeps ONLY essential files for tennis analysis
"""
import os
import glob

# Files to delete
FILES_TO_DELETE = [
    # Test files
    "test_exact_main_pose.py",
    "test_exact_points.py",
    "test_loading.py",
    "test_main_pose_frame1.py",
    "test_tracking_effects.py",
    "test_court_detection.py",
    "test_custom_model.py",
    "quick_test_10_frames.py",
    "check_video_frame.py",
    "check_gpu.py",
    "check_system_gpu.py",
    "chatbot_test.py",
    
    # Duplicates/old
    "court_lines_manual_NEW.json",
    "main_pose_debug.log",
    "main.py",
    "main_pose_streaming.py",
    
    # Visualization tests
    "visualize_court_keypoints.py",
    "visualize_court_lines.py",
    "visualize_court_points_clear.py",
    "demo_ml_keypoint_lines.py",
    
    # Alternative calibration tools
    "partial_court_calibration.py",
    "simple_court_calibration.py",
    "reselect_service_lines.py",
    "reselect_singles_lines.py",
    
    # Backend (optional - uncomment if not using)
    # "backend_server.py",
    # "voice_coach_server.py",
    # "tts_service.py",
    # "run_backend.py",
    # "backend_requirements.txt",
    
    # Daytona/GPU
    "daytona_example.py",
    "daytona_integration.py",
    "download_court_model.py",
    "download_custom_model.py",
    
    # Screen recordings
    "ScreenRecording_11-18-2025 10-42-57_1.mp4",
    
    # Old output videos (test outputs)
    "output_videos/TEST_10_FRAMES.mp4",
    "output_videos/PERFECT_LINES_NEW.mp4",
    "output_videos/CHECK_PERFECT_LINES_FRAME1.jpg",
    "output_videos/PROOF_EXACT_POINTS.jpg",
    "output_videos/TEST_EXACT_SIMULATION.jpg",
    "output_videos/TEST_MAIN_POSE_FRAME1.jpg",
]

# Old analysis videos (keep only the latest)
OLD_ANALYSIS_PATTERN = "output_videos/analyzed_copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B_*.mp4"

def main():
    deleted_count = 0
    failed_count = 0
    
    print("=" * 60)
    print("TENNIS AI - FOLDER CLEANUP")
    print("=" * 60)
    print(f"\nThis will delete {len(FILES_TO_DELETE)} test/duplicate files")
    print("\n⚠️  Files to delete:")
    for f in FILES_TO_DELETE[:10]:
        print(f"   - {f}")
    print(f"   ... and {len(FILES_TO_DELETE) - 10} more\n")
    
    response = input("Proceed with cleanup? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("❌ Cleanup cancelled")
        return
    
    print("\n🗑️  Deleting files...\n")
    
    # Delete individual files
    for file_path in FILES_TO_DELETE:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"   ✅ Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {file_path}: {e}")
                failed_count += 1
        else:
            print(f"   ⏭️  Already gone: {file_path}")
    
    # Delete old analysis videos
    old_videos = glob.glob(OLD_ANALYSIS_PATTERN)
    if old_videos:
        print(f"\n🗑️  Found {len(old_videos)} old analysis videos")
        for video in old_videos:
            try:
                os.remove(video)
                print(f"   ✅ Deleted: {video}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                failed_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Cleanup complete!")
    print(f"   Deleted: {deleted_count} files")
    print(f"   Failed: {failed_count} files")
    print("=" * 60)
    
    print("\n📁 Essential files kept:")
    print("   ✅ main_pose.py")
    print("   ✅ court_lines_manual.json")
    print("   ✅ output_videos/tennis_analysis_trail.json")
    print("   ✅ output_videos/tennis_analysis_trail.avi")
    print("   ✅ All core modules (trackers/, modules/, yolo, etc.)")

if __name__ == "__main__":
    main()

