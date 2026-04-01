"""
Cold-Start Heuristic Proposal Generation.

Before any trained model is available, generates initial candidate proposals
using a domain-informed heuristic: incidents typically occur between 40%-80%
of the race duration.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np

from .config import CONFIG, PROPOSALS_DIR, CLIP_META_DIR

logger = logging.getLogger(__name__)


def generate_cold_start_proposals(
    video_id: str,
    clips: List[Dict],
    video_duration: float = None,
) -> Dict:
    """Generate heuristic proposals for a single video (cold start)."""
    cfg = CONFIG.cold_start

    if not clips:
        return {}

    if video_duration is None:
        video_duration = clips[-1]["end_time"]

    # Focus region: [0.4 * D, 0.8 * D]
    focus_start = cfg.focus_start * video_duration
    focus_end = cfg.focus_end * video_duration

    # Filter clips within focus region
    focus_clips = [
        c for c in clips
        if c["start_time"] >= focus_start and c["start_time"] <= focus_end
    ]

    if not focus_clips:
        focus_clips = clips  # fallback to all clips

    # Select K clips uniformly spaced across focus region
    K = min(cfg.num_candidates, len(focus_clips))
    if K == 0:
        return {}

    indices = np.linspace(0, len(focus_clips) - 1, K, dtype=int)
    selected_clips = [focus_clips[i] for i in indices]

    # Build candidates with uniform scoring
    candidates = []
    for rank, clip in enumerate(selected_clips, 1):
        candidates.append({
            "clip_id": clip["clip_id"],
            "clip_index": clip["clip_index"],
            "start_time": clip["start_time"],
            "end_time": clip["end_time"],
            "duration": clip["duration"],
            "clip_prob": cfg.default_probability,
            "attention_weight": 1.0 / K,
            "rank": rank,
        })

    # Build uniform probabilities for all clips
    all_probs = [cfg.default_probability if (
        c["start_time"] >= focus_start and c["start_time"] <= focus_end
    ) else 0.1 for c in clips]

    all_attention = [1.0 / len(clips)] * len(clips)

    result = {
        "video_id": video_id,
        "bag_prob": cfg.default_probability,
        "binary_label": clips[0].get("binary_label", -1),
        "label_name": clips[0].get("label_name", "unknown"),
        "num_clips": len(clips),
        "candidates": candidates,
        "all_clip_probs": all_probs,
        "all_attention_weights": all_attention,
        "source": "cold_start_heuristic",
    }

    return result


def run_cold_start(all_clips: Dict[str, List[Dict]] = None) -> Dict[str, Dict]:
    """Generate cold-start proposals for all videos."""
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)

    if all_clips is None:
        clips_file = CLIP_META_DIR / "all_clips.json"
        if clips_file.exists():
            with open(clips_file) as f:
                all_clips = json.load(f)
        else:
            logger.error("No clip metadata found")
            return {}

    all_proposals = {}
    for video_id, clips in all_clips.items():
        proposals = generate_cold_start_proposals(video_id, clips)
        if proposals:
            all_proposals[video_id] = proposals
            prop_file = PROPOSALS_DIR / f"{video_id}_proposals.json"
            with open(prop_file, "w") as f:
                json.dump(proposals, f, indent=2)

    # Save combined
    combined = PROPOSALS_DIR / "all_proposals.json"
    with open(combined, "w") as f:
        json.dump(all_proposals, f, indent=2)

    logger.info(f"Cold-start proposals generated for {len(all_proposals)} videos")
    return all_proposals
