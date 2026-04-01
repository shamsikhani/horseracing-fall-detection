"""
Run feature extraction on all 100 videos using the proper pipeline function.
"""

import json
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.feature_extraction import run_feature_extraction
from pipeline.config import CLIP_META_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Load all clip metadata
    all_clips = {}
    clip_files = sorted(CLIP_META_DIR.glob("*_clips.json"))
    
    for clip_file in clip_files:
        if clip_file.name == "all_clips.json":
            continue
        
        video_id = clip_file.stem.replace("_clips", "")
        with open(clip_file) as f:
            clips = json.load(f)
        all_clips[video_id] = clips
    
    print(f"Loaded {len(all_clips)} videos with clip metadata")
    
    # Run feature extraction
    run_feature_extraction(all_clips, force=False)

if __name__ == "__main__":
    main()
