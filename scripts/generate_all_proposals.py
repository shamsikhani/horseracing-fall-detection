"""
Generate cold-start proposals for all 100 videos.
"""

import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cold_start import generate_cold_start_proposals
from pipeline.config import CLIP_META_DIR, PROPOSALS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Load all clip metadata
    clip_files = sorted(CLIP_META_DIR.glob("*_clips.json"))
    clip_files = [f for f in clip_files if f.name != "all_clips.json"]
    
    print(f"Generating proposals for {len(clip_files)} videos...")
    
    all_proposals = {}
    
    for i, clip_file in enumerate(clip_files, 1):
        video_id = clip_file.stem.replace("_clips", "")
        
        with open(clip_file) as f:
            clips = json.load(f)
        
        print(f"[{i}/{len(clip_files)}] Generating proposals for {video_id}...")
        
        proposal = generate_cold_start_proposals(video_id, clips)
        all_proposals[video_id] = proposal
    
    # Save all proposals
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    proposals_file = PROPOSALS_DIR / "all_proposals.json"
    
    with open(proposals_file, "w") as f:
        json.dump(all_proposals, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Cold-start proposals generated for {len(all_proposals)} videos")
    print(f"Saved to: {proposals_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
