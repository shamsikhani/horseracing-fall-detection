"""
Inference module: run trained model on videos to produce predictions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import CONFIG, FEATURES_DIR, CLIP_META_DIR, MODEL_DIR
from .model import AttentionMIL
from .train import load_trained_model

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_inference_single(
    video_id: str,
    clips: List[Dict],
    model: AttentionMIL,
    device: str,
) -> Dict:
    """Run inference on a single video. Returns detailed predictions."""
    feat_path = FEATURES_DIR / f"{video_id}.npy"
    if not feat_path.exists():
        return {}

    features = np.load(feat_path)
    features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.ones(1, features.shape[0], dtype=torch.bool).to(device)

    model.eval()
    output = model(features_t, mask)

    bag_prob = output["bag_prob"].item()
    clip_probs = output["clip_probs"][0].cpu().numpy()
    attention = output["attention"][0].cpu().numpy()

    # Per-clip details
    clip_details = []
    for idx, clip in enumerate(clips):
        if idx < len(clip_probs):
            clip_details.append({
                "clip_id": clip["clip_id"],
                "clip_index": idx,
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "clip_prob": float(clip_probs[idx]),
                "attention_weight": float(attention[idx]),
            })

    return {
        "video_id": video_id,
        "bag_prob": bag_prob,
        "predicted_class": 1 if bag_prob > 0.5 else 0,
        "binary_label": clips[0]["binary_label"] if clips else -1,
        "label_name": clips[0]["label_name"] if clips else "unknown",
        "clip_details": clip_details,
        "all_clip_probs": clip_probs.tolist(),
        "all_attention_weights": attention.tolist(),
    }


def run_inference_all(
    all_clips: Dict[str, List[Dict]] = None,
    model: AttentionMIL = None,
    device: str = None,
) -> Dict[str, Dict]:
    """Run inference on all videos."""
    device = device or CONFIG.features.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if model is None:
        model = load_trained_model(device=device)
        if model is None:
            logger.error("No trained model for inference")
            return {}

    model = model.to(device)

    if all_clips is None:
        clips_file = CLIP_META_DIR / "all_clips.json"
        if clips_file.exists():
            with open(clips_file) as f:
                all_clips = json.load(f)
        else:
            return {}

    results = {}
    for video_id, clips in all_clips.items():
        result = run_inference_single(video_id, clips, model, device)
        if result:
            results[video_id] = result

    # Summary
    correct = sum(
        1 for r in results.values()
        if r["predicted_class"] == r["binary_label"]
    )
    total = len(results)
    logger.info(f"Inference: {correct}/{total} correct ({correct/max(total,1)*100:.1f}%)")

    return results
