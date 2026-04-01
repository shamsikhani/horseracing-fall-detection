"""
Out-of-Distribution (OOD) filtering for non-race content.

Identifies clips that are likely non-race footage (black screens, static
images, pre/post-race walking, extreme zoom transitions) using embedding
statistics. These clips can be downweighted or excluded from proposals.

Approach:
  1. Compute per-clip embedding statistics (norm, variance)
  2. Compute video-level statistics (median norm, IQR)
  3. Flag clips that deviate significantly from the video's "active race" profile
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import FEATURES_DIR, CLIP_META_DIR

logger = logging.getLogger(__name__)


def compute_clip_statistics(features: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute per-clip statistics from the embedding matrix.
    features: (num_clips, 768)
    Returns dict with per-clip arrays.
    """
    norms = np.linalg.norm(features, axis=1)  # (num_clips,)
    variances = np.var(features, axis=1)       # (num_clips,)

    # Temporal difference: how much each clip differs from its neighbours
    if len(features) > 1:
        diffs = np.linalg.norm(np.diff(features, axis=0), axis=1)
        # Pad to match length (first clip gets same diff as second)
        temporal_diffs = np.concatenate([[diffs[0]], diffs])
    else:
        temporal_diffs = np.zeros(len(features))

    return {
        "norms": norms,
        "variances": variances,
        "temporal_diffs": temporal_diffs,
    }


def detect_ood_clips(
    features: np.ndarray,
    clips: List[Dict],
    z_threshold: float = 2.5,
) -> List[Dict]:
    """
    Detect out-of-distribution clips within a single video.

    Uses a robust z-score approach: clips whose embedding norm deviates
    more than z_threshold * IQR from the median are flagged as OOD.

    Returns a list of dicts with OOD information per clip:
      - clip_index: int
      - is_ood: bool
      - ood_score: float (higher = more OOD)
      - ood_reason: str or None
    """
    stats = compute_clip_statistics(features)
    norms = stats["norms"]
    variances = stats["variances"]
    temporal_diffs = stats["temporal_diffs"]

    # Robust statistics using median and IQR
    median_norm = np.median(norms)
    q25, q75 = np.percentile(norms, [25, 75])
    iqr_norm = q75 - q25
    if iqr_norm < 1e-6:
        iqr_norm = np.std(norms) + 1e-6

    median_var = np.median(variances)
    q25_v, q75_v = np.percentile(variances, [25, 75])
    iqr_var = q75_v - q25_v
    if iqr_var < 1e-6:
        iqr_var = np.std(variances) + 1e-6

    results = []
    for idx in range(len(features)):
        # Z-score based on norm (catches black screens, static images)
        norm_z = abs(norms[idx] - median_norm) / iqr_norm

        # Z-score based on variance (catches uniform/static content)
        var_z = abs(variances[idx] - median_var) / iqr_var

        # Combined OOD score
        ood_score = max(norm_z, var_z)
        is_ood = ood_score > z_threshold

        # Determine reason
        reason = None
        if is_ood:
            if norms[idx] < median_norm - z_threshold * iqr_norm:
                reason = "low_energy"  # likely black screen or static
            elif norms[idx] > median_norm + z_threshold * iqr_norm:
                reason = "high_energy"  # likely extreme zoom or transition
            elif variances[idx] < median_var - z_threshold * iqr_var:
                reason = "low_variance"  # likely static/uniform content
            else:
                reason = "anomalous"

        results.append({
            "clip_index": idx,
            "is_ood": is_ood,
            "ood_score": float(ood_score),
            "ood_reason": reason,
        })

    n_ood = sum(1 for r in results if r["is_ood"])
    if n_ood > 0:
        logger.debug(
            f"Detected {n_ood}/{len(features)} OOD clips "
            f"(reasons: {[r['ood_reason'] for r in results if r['is_ood']]})"
        )

    return results


def filter_proposals_with_ood(
    candidates: List[Dict],
    ood_results: List[Dict],
    downweight_factor: float = 0.1,
) -> List[Dict]:
    """
    Downweight OOD clips in the candidate list rather than removing them.
    This preserves the candidate count but makes OOD clips less likely
    to be top-ranked.
    """
    ood_set = {r["clip_index"] for r in ood_results if r["is_ood"]}

    for cand in candidates:
        if cand.get("clip_index") in ood_set:
            cand["clip_prob"] *= downweight_factor
            cand["is_ood"] = True
        else:
            cand["is_ood"] = False

    return candidates


def get_race_phase_mask(
    clips: List[Dict],
    video_duration: float,
    pre_race_fraction: float = 0.05,
    post_race_fraction: float = 0.95,
) -> List[bool]:
    """
    Simple heuristic: flag clips in the first 5% and last 5% of the video
    as likely pre/post-race content.

    Returns a boolean mask where True = likely active race.
    """
    mask = []
    for clip in clips:
        t = clip["start_time"]
        in_race = (t >= video_duration * pre_race_fraction and
                   t <= video_duration * post_race_fraction)
        mask.append(in_race)
    return mask
