"""
Organize videos into fall and non-fall folders based on video_selection_100.xlsx
"""

import pandas as pd
import shutil
from pathlib import Path

# Read the Excel file
excel_path = Path(r"c:\Users\shams\Documents\Horseracing\video_selection_100.xlsx")
df = pd.read_excel(excel_path, sheet_name='Video Selection')

# Define paths
base_dir = Path(r"c:\Users\shams\Documents\Horseracing")
fall_dir = base_dir / "fall"
non_fall_dir = base_dir / "non-fall"

# Create non-fall directory if it doesn't exist
non_fall_dir.mkdir(exist_ok=True)

# Also check old folder names
old_fall_dir = base_dir / "Fell"
old_pulled_dir = base_dir / "Pulled-up"

print("="*80)
print("VIDEO ORGANIZATION SCRIPT")
print("="*80)

# Get list of all video files in fall-related directories
video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
all_videos = []

for search_dir in [fall_dir, old_fall_dir, old_pulled_dir, base_dir]:
    if search_dir.exists():
        for ext in video_extensions:
            all_videos.extend(search_dir.glob(f'*{ext}'))

print(f"\nFound {len(all_videos)} video files total")

# Create mapping from Excel
video_class_map = {}
for _, row in df.iterrows():
    video_id = row['Video_ID']
    video_class = row['Class']
    video_class_map[video_id] = video_class

print(f"Excel file contains {len(video_class_map)} videos")
print(f"  - FALL: {sum(1 for v in video_class_map.values() if v == 'FALL')}")
print(f"  - NO_FALL: {sum(1 for v in video_class_map.values() if v == 'NO_FALL')}")

# Match videos to their class
moved_count = 0
not_found_count = 0
already_correct = 0

print(f"\n{'='*80}")
print("PROCESSING VIDEOS")
print(f"{'='*80}\n")

for video_path in all_videos:
    video_name = video_path.stem  # filename without extension
    
    # Try to match with video_id from Excel
    matched_class = None
    for video_id, video_class in video_class_map.items():
        # Match by video_id (may need to handle slight variations)
        if video_id in video_name or video_name in video_id:
            matched_class = video_class
            break
    
    if matched_class is None:
        print(f"[WARNING] NOT IN EXCEL: {video_path.name}")
        not_found_count += 1
        continue
    
    # Determine target directory
    if matched_class == 'FALL':
        target_dir = fall_dir
    else:
        target_dir = non_fall_dir
    
    # Check if already in correct location
    if video_path.parent == target_dir:
        print(f"[OK] ALREADY CORRECT: {video_path.name} -> {target_dir.name}/")
        already_correct += 1
        continue
    
    # Move the video
    target_path = target_dir / video_path.name
    
    try:
        shutil.move(str(video_path), str(target_path))
        print(f"[MOVED] {video_path.name}")
        print(f"   FROM: {video_path.parent.name}/")
        print(f"   TO:   {target_dir.name}/")
        moved_count += 1
    except Exception as e:
        print(f"[ERROR] moving {video_path.name}: {e}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total videos processed: {len(all_videos)}")
print(f"Moved to correct folder: {moved_count}")
print(f"Already in correct location: {already_correct}")
print(f"Not found in Excel: {not_found_count}")

# Verify final structure
print(f"\n{'='*80}")
print("FINAL DIRECTORY STRUCTURE")
print(f"{'='*80}")

if fall_dir.exists():
    fall_videos = list(fall_dir.glob('*.mp4')) + list(fall_dir.glob('*.avi'))
    print(f"\n{fall_dir}:")
    print(f"  {len(fall_videos)} videos")

if non_fall_dir.exists():
    non_fall_videos = list(non_fall_dir.glob('*.mp4')) + list(non_fall_dir.glob('*.avi'))
    print(f"\n{non_fall_dir}:")
    print(f"  {len(non_fall_videos)} videos")

print(f"\n{'='*80}")
print("DONE!")
print(f"{'='*80}")
