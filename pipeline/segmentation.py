"""
Stage 2: Temporal Segmentation.

Partitions each preprocessed video into overlapping clips using a sliding window.
- Window duration: 3.0 s
- Stride: 1.0 s
- Overlap: ~66.7%

No physical clip files are created; only metadata is persisted.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

from .config import CONFIG, CLIP_META_DIR, PREPROCESSED_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


def segment_video(video_meta: dict) -> List[Dict]:
    """Generate clip metadata for a single video using sliding window."""
    cfg = CONFIG.segmentation
    duration = video_meta["duration"]
    fps = video_meta.get("fps", 25.0)
    video_id = video_meta["video_id"]

    clips = []
    clip_idx = 0
    start = 0.0

    while start + cfg.min_clip_duration <= duration:
        end = min(start + cfg.clip_duration, duration)
        clip_dur = end - start

        if clip_dur < cfg.min_clip_duration:
            break

        clip = {
            "clip_id": f"{video_id}_clip_{clip_idx:04d}",
            "video_id": video_id,
            "clip_index": clip_idx,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration": round(clip_dur, 3),
            "start_frame": int(start * fps),
            "end_frame": int(end * fps),
            "fps": fps,
            "binary_label": video_meta.get("binary_label", -1),
            "label_name": video_meta.get("label_name", "unknown"),
        }
        clips.append(clip)
        clip_idx += 1
        start += cfg.stride

    return clips


def run_segmentation(video_metadata_list: list) -> Dict[str, List[Dict]]:
    """Run segmentation on all preprocessed videos.
    Returns dict mapping video_id -> list of clip metadata dicts.
    """
    CLIP_META_DIR.mkdir(parents=True, exist_ok=True)
    all_clips = {}

    for meta in video_metadata_list:
        video_id = meta["video_id"]
        clips = segment_video(meta)
        all_clips[video_id] = clips

        # Save per-video clip metadata
        clip_file = CLIP_META_DIR / f"{video_id}_clips.json"
        with open(clip_file, "w") as f:
            json.dump(clips, f, indent=2)

        logger.info(f"Segmented {video_id}: {len(clips)} clips "
                     f"(duration={meta['duration']:.1f}s)")

    # Save combined clip index
    combined = CLIP_META_DIR / "all_clips.json"
    with open(combined, "w") as f:
        json.dump(all_clips, f, indent=2)

    total = sum(len(c) for c in all_clips.values())
    logger.info(f"Segmentation complete: {total} clips from {len(all_clips)} videos")
    return all_clips
