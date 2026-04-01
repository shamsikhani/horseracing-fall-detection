"""
Run feature extraction on all 100 videos.
"""

import json
import logging
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.feature_extraction import extract_features_for_video
from pipeline.config import CLIP_META_DIR, OUTPUT_DIR, FEATURES_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load all clip metadata files
    clip_files = sorted(CLIP_META_DIR.glob("*_clips.json"))
    clip_files = [f for f in clip_files if f.name != "all_clips.json"]
    
    logger.info(f"Found {len(clip_files)} videos to process")
    
    # Check which videos already have features
    existing_features = set(f.stem for f in FEATURES_DIR.glob("*.npy"))
    logger.info(f"Already extracted features for {len(existing_features)} videos")
    
    # Process each video
    processed = 0
    skipped = 0
    
    for clip_file in clip_files:
        video_id = clip_file.stem.replace("_clips", "")
        
        # Skip if already processed
        if video_id in existing_features:
            logger.info(f"Skipping {video_id} (already processed)")
            skipped += 1
            continue
        
        logger.info(f"Processing {video_id} ({processed + 1}/{len(clip_files) - len(existing_features)})")
        
        try:
            # Load clip metadata for this video
            with open(clip_file) as f:
                clips = json.load(f)
            
            # Extract features
            extract_features_for_video(video_id, clips)
            processed += 1
            
            logger.info(f"✓ Completed {video_id} ({len(clips)} clips)")
            
        except Exception as e:
            logger.error(f"✗ Error processing {video_id}: {e}")
            continue
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Feature extraction complete!")
    logger.info(f"Processed: {processed} videos")
    logger.info(f"Skipped: {skipped} videos (already done)")
    logger.info(f"Total features: {len(list(FEATURES_DIR.glob('*.npy')))} videos")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()
