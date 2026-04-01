"""
Candidate Proposal Generation with Non-Maximum Suppression.

After MIL inference, clips are ranked by clip-level probability.
Temporal NMS removes overlapping detections to produce diverse proposals.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch

from .config import CONFIG, FEATURES_DIR, PROPOSALS_DIR, CLIP_META_DIR, MODEL_DIR
from .model import AttentionMIL
from .train import load_trained_model
from .ood_filter import detect_ood_clips, filter_proposals_with_ood
from .clip_classifier import load_clip_classifier, predict_video_clips

logger = logging.getLogger(__name__)


def temporal_nms(
    candidates: List[Dict],
    nms_window: float = 2.0,
    top_k: int = 5,
    min_probability: float = 0.1,
) -> List[Dict]:
    """
    Temporal Non-Maximum Suppression.
    candidates: list of dicts with 'start_time', 'clip_prob' keys, sorted by prob desc.
    Returns top-K diverse candidates.
    """
    if not candidates:
        return []

    # Sort by clip probability descending
    candidates = sorted(candidates, key=lambda c: c["clip_prob"], reverse=True)

    selected = []
    suppressed = set()

    for i, cand in enumerate(candidates):
        if i in suppressed:
            continue
        selected.append(cand)

        # Suppress nearby candidates
        for j in range(i + 1, len(candidates)):
            if j in suppressed:
                continue
            if abs(cand["start_time"] - candidates[j]["start_time"]) < nms_window:
                suppressed.add(j)

    # Filter by minimum probability
    selected = [c for c in selected if c["clip_prob"] >= min_probability]

    # Take top-K
    selected = selected[:top_k]

    # Assign ranks
    for rank, cand in enumerate(selected, 1):
        cand["rank"] = rank

    return selected


@torch.no_grad()
def generate_proposals_for_video(
    video_id: str,
    clips: List[Dict],
    model: AttentionMIL,
    device: str,
) -> Dict:
    """Generate proposals for a single video using the trained model."""
    feat_path = FEATURES_DIR / f"{video_id}.npy"
    if not feat_path.exists():
        logger.warning(f"No features found for {video_id}")
        return {}

    features = np.load(feat_path)  # (num_clips, 768)
    features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, M, 768)
    mask = torch.ones(1, features.shape[0], dtype=torch.bool).to(device)

    model.eval()
    output = model(features_t, mask)

    bag_prob = output["bag_prob"].item()
    clip_probs = output["clip_probs"][0].cpu().numpy()
    attention = output["attention"][0].cpu().numpy()

    # Detect OOD clips
    ood_results = detect_ood_clips(features, clips)

    # Build candidate list
    candidates = []
    for idx, clip in enumerate(clips):
        if idx < len(clip_probs):
            candidates.append({
                "clip_id": clip["clip_id"],
                "clip_index": idx,
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "duration": clip["duration"],
                "clip_prob": float(clip_probs[idx]),
                "attention_weight": float(attention[idx]),
            })

    # Downweight OOD clips before NMS
    candidates = filter_proposals_with_ood(candidates, ood_results)

    # Apply NMS
    cfg = CONFIG.proposals
    selected = temporal_nms(
        candidates,
        nms_window=cfg.nms_window,
        top_k=cfg.top_k,
        min_probability=cfg.min_probability,
    )

    result = {
        "video_id": video_id,
        "bag_prob": bag_prob,
        "binary_label": clips[0]["binary_label"] if clips else -1,
        "label_name": clips[0]["label_name"] if clips else "unknown",
        "num_clips": len(clips),
        "candidates": selected,
        "all_clip_probs": clip_probs.tolist(),
        "all_attention_weights": attention.tolist(),
        "source": "model",
    }

    return result


def generate_proposals_clip_clf(
    video_id: str,
    clips: List[Dict],
    model_dict: dict,
) -> Dict:
    """Generate proposals for a single video using the clip classifier."""
    clip_probs, bag_prob = predict_video_clips(video_id, clips, model_dict)

    if len(clip_probs) == 0:
        return {}

    # Build candidate list
    candidates = []
    for idx, clip in enumerate(clips):
        if idx < len(clip_probs):
            candidates.append({
                "clip_id": clip["clip_id"],
                "clip_index": idx,
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "duration": clip["duration"],
                "clip_prob": float(clip_probs[idx]),
                "attention_weight": float(clip_probs[idx]),
            })

    # Apply NMS
    cfg = CONFIG.proposals
    selected = temporal_nms(
        candidates,
        nms_window=cfg.nms_window,
        top_k=cfg.top_k,
        min_probability=cfg.min_probability,
    )

    result = {
        "video_id": video_id,
        "bag_prob": bag_prob,
        "binary_label": clips[0]["binary_label"] if clips else -1,
        "label_name": clips[0]["label_name"] if clips else "unknown",
        "num_clips": len(clips),
        "candidates": selected,
        "all_clip_probs": clip_probs.tolist(),
        "all_attention_weights": clip_probs.tolist(),
        "source": "clip_classifier",
    }

    return result


def run_proposal_generation(
    all_clips: Dict[str, List[Dict]] = None,
    model: AttentionMIL = None,
    device: str = None,
    clip_clf: dict = None,
) -> Dict[str, Dict]:
    """Generate proposals for all videos. Returns dict of video_id -> proposal dict."""
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)

    device = device or CONFIG.features.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load clips if not provided
    if all_clips is None:
        clips_file = CLIP_META_DIR / "all_clips.json"
        if clips_file.exists():
            with open(clips_file) as f:
                all_clips = json.load(f)
        else:
            logger.error("No clip metadata found")
            return {}

    # Prefer clip classifier over MIL model
    if clip_clf is None:
        clip_clf = load_clip_classifier()

    use_clip_clf = clip_clf is not None

    if not use_clip_clf:
        # Fall back to MIL model
        if model is None:
            model = load_trained_model(device=device)
            if model is None:
                logger.error("No trained model available for proposal generation")
                return {}
        model = model.to(device)

    all_proposals = {}
    for video_id, clips in all_clips.items():
        if use_clip_clf:
            proposals = generate_proposals_clip_clf(video_id, clips, clip_clf)
        else:
            proposals = generate_proposals_for_video(video_id, clips, model, device)
        if proposals:
            all_proposals[video_id] = proposals
            # Save per-video proposals
            prop_file = PROPOSALS_DIR / f"{video_id}_proposals.json"
            with open(prop_file, "w") as f:
                json.dump(proposals, f, indent=2)

    # Save combined proposals
    combined = PROPOSALS_DIR / "all_proposals.json"
    with open(combined, "w") as f:
        json.dump(all_proposals, f, indent=2)

    logger.info(f"Generated proposals for {len(all_proposals)} videos")
    return all_proposals
