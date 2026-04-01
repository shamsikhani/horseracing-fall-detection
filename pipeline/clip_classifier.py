"""
Ensemble clip classifier for fall detection.

Trains two GradientBoosting classifiers:
  1. Temporal GBM on 6 temporal-difference features (detects fall REGIONS)
  2. Embedding GBM on within-video-normalised VideoMAE embeddings
     (detects fall FRAMES even when temporal change is subtle)

Final score = alpha * temporal_prob + (1-alpha) * embedding_prob.

Within-video normalisation removes inter-video variance (camera, lighting,
turf) so the embedding classifier focuses on *what deviates from this
video's norm*, not video-specific visual patterns.

LOGO evaluation (ensemble alpha=0.5):
  R@10/30s = 94%, R@10/5s = 46% (temporal-only was 26%).
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, FEATURES_DIR, MODEL_DIR, CLIP_META_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

FALL_TYPES = {"fall", "fell", "brought_down", "unseated_rider", "slipped_up"}
NEG_TYPES = {"no_incident", "no_fall", "pulled_up", "pull_up", "refused",
             "carried_out", "ran_out"}
JUNK_TYPES = {"not_visible", "non_race_footage", "riderless_horse"}

ANNOTATIONS_FILE = OUTPUT_DIR / "annotations.jsonl"
CLF_PATH = MODEL_DIR / "clip_classifier.pkl"

# Race-phase boundaries (normalised position in video)
RACE_START_FRAC = 0.05
RACE_END_FRAC = 0.85

# Ensemble weight: alpha * temporal_prob + (1-alpha) * embedding_prob.
# alpha=0.5 gives best R@10/5s (46%) while keeping R@10/30s at 94%.
# alpha=0.7 gives best R@10/30s (97%) but worse 5s precision.
ENSEMBLE_ALPHA = 0.5

# Hard negative mining: per no-fall video, include the N clips with
# highest temporal change as negatives (camera transitions, replays).
# LOGO sweep: 3 gives best R@10/30s = 93.9%.
HARD_NEG_PER_VIDEO = 3



# ---------------------------------------------------------------------------
# Temporal feature extraction
# ---------------------------------------------------------------------------

def compute_temporal_features(features: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute per-clip temporal anomaly features.

    For each clip returns 6 scalar features:
        0: diff_norm_prev  — L2 distance to previous clip embedding
        1: diff_norm_next  — L2 distance to next clip embedding
        2: deviation_norm  — L2 distance to local context mean
        3: temp_var        — mean variance of embeddings in local window
        4: position        — normalised position in video [0, 1]
        5: max_diff_window — max consecutive-clip L2 distance in local window

    Parameters
    ----------
    features : (N, D) array of clip embeddings
    window   : half-size of the local context window

    Returns
    -------
    (N, 6) array of temporal features
    """
    n, d = features.shape

    # Consecutive-clip L2 distances
    diff_prev = np.zeros(n)
    diff_next = np.zeros(n)
    for i in range(1, n):
        diff_prev[i] = np.linalg.norm(features[i] - features[i - 1])
    for i in range(n - 1):
        diff_next[i] = np.linalg.norm(features[i] - features[i + 1])

    # Context deviation
    deviation_norm = np.zeros(n)
    for i in range(n):
        s = max(0, i - window)
        e = min(n, i + window + 1)
        ctx = np.mean(features[s:e], axis=0)
        deviation_norm[i] = np.linalg.norm(features[i] - ctx)

    # Local temporal variance
    temp_var = np.zeros(n)
    for i in range(n):
        s = max(0, i - window)
        e = min(n, i + window + 1)
        temp_var[i] = np.mean(np.var(features[s:e], axis=0))

    # Position (normalised)
    position = np.arange(n, dtype=np.float64) / max(n - 1, 1)

    # Max consecutive-clip distance in local window
    max_diff_window = np.zeros(n)
    for i in range(n):
        s = max(0, i - window)
        e = min(n, i + window + 1)
        local = features[s:e]
        if len(local) > 1:
            diffs = np.linalg.norm(np.diff(local, axis=0), axis=1)
            max_diff_window[i] = np.max(diffs)

    return np.column_stack([
        diff_prev, diff_next, deviation_norm,
        temp_var, position, max_diff_window,
    ])


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _load_annotations() -> Dict[str, list]:
    """Load confirmed annotations grouped by video_id."""
    annotations: Dict[str, list] = defaultdict(list)
    if not ANNOTATIONS_FILE.exists():
        return annotations
    with open(ANNOTATIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)
            if ann.get("confirmed"):
                annotations[ann["video_id"]].append(ann)
    return dict(annotations)


def _load_all_clips() -> Dict[str, list]:
    clips_file = CLIP_META_DIR / "all_clips.json"
    if clips_file.exists():
        with open(clips_file) as f:
            return json.load(f)
    return {}


def _normalize_within_video(features: np.ndarray) -> np.ndarray:
    """Z-score normalise embeddings within a video.
    Removes inter-video variance (camera, lighting, turf colour).
    What remains: how each clip deviates from the video average."""
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    return (features - mean) / std


def build_clip_training_data(
    exclude_vid: Optional[str] = None,
    neg_samples_per_video: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build training dataset from annotations + sampled negatives from
    confirmed-negative videos.

    Returns (X_temp, y, X_norm_emb) where:
        X_temp     is (N, 6) temporal features
        y          is (N,) binary labels
        X_norm_emb is (N, D) within-video normalised embeddings
    """
    annotations = _load_annotations()
    all_clips = _load_all_clips()

    X_temp_list: List[np.ndarray] = []
    X_norm_emb_list: List[np.ndarray] = []
    y_list: List[int] = []

    # --- Clips from temporal annotations ---
    for vid, anns in annotations.items():
        if vid == exclude_vid:
            continue
        feat_path = FEATURES_DIR / f"{vid}.npy"
        if not feat_path.exists():
            continue
        features = np.load(feat_path)
        clips = all_clips.get(vid, [])
        if len(clips) != len(features):
            continue

        temp_feats = compute_temporal_features(features)
        norm_emb = _normalize_within_video(features)

        for ann in anns:
            ts = ann["timestamp"]
            end_ts = ann.get("end_timestamp") or (ts + 3)
            lt = ann["label_type"]
            if lt in JUNK_TYPES:
                continue
            if lt not in FALL_TYPES and lt not in NEG_TYPES:
                continue
            label = 1 if lt in FALL_TYPES else 0
            for idx, clip in enumerate(clips):
                if clip["start_time"] < end_ts and clip["end_time"] > ts:
                    X_temp_list.append(temp_feats[idx])
                    X_norm_emb_list.append(norm_emb[idx])
                    y_list.append(label)

    # --- Sampled negatives from confirmed-negative videos ---
    from .label_utils import get_corrected_binary_labels
    corrected = get_corrected_binary_labels()
    rng = np.random.RandomState(42)

    for vid, label in corrected.items():
        if label == 1:
            continue
        if vid == exclude_vid:
            continue
        feat_path = FEATURES_DIR / f"{vid}.npy"
        if not feat_path.exists():
            continue
        features = np.load(feat_path)
        clips = all_clips.get(vid, [])
        if len(clips) != len(features):
            continue

        temp_feats = compute_temporal_features(features)
        norm_emb = _normalize_within_video(features)
        n = len(temp_feats)

        # Sample from race portion only
        start = int(n * RACE_START_FRAC)
        end = int(n * RACE_END_FRAC)
        if end <= start:
            continue
        indices = rng.choice(
            range(start, end),
            size=min(neg_samples_per_video, end - start),
            replace=False,
        )
        for idx in indices:
            X_temp_list.append(temp_feats[idx])
            X_norm_emb_list.append(norm_emb[idx])
            y_list.append(0)

        # Hard negatives: clips with highest temporal change
        # (camera transitions, broadcast replays, angle changes).
        # These look like anomalies but are NOT falls.
        if HARD_NEG_PER_VIDEO > 0:
            race_temp = temp_feats[start:end]
            change_score = race_temp[:, 0] + race_temp[:, 2]  # diff_prev + deviation
            top_k = min(HARD_NEG_PER_VIDEO, len(change_score))
            hard_indices = np.argsort(change_score)[-top_k:]
            for local_idx in hard_indices:
                idx = start + local_idx
                X_temp_list.append(temp_feats[idx])
                X_norm_emb_list.append(norm_emb[idx])
                y_list.append(0)

    if not X_temp_list:
        return np.empty((0, 6)), np.empty(0, dtype=int), np.empty((0, 768))

    return np.array(X_temp_list), np.array(y_list), np.array(X_norm_emb_list)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_clip_classifier() -> Tuple[Optional[object], dict]:
    """
    Train an ensemble of two GradientBoosting classifiers:
      1. Temporal GBM on 6 temporal-difference features
      2. Embedding GBM on within-video normalised VideoMAE embeddings

    The temporal model detects fall REGIONS (good at 30s matching).
    The embedding model detects fall FRAMES (good at 5s matching).
    At inference they are blended with ENSEMBLE_ALPHA.

    Returns (model_dict, metrics) or (None, {}) on failure.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_temp, y, X_norm_emb = build_clip_training_data()

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    logger.info(f"Clip classifier training data: {len(X_temp)} clips "
                f"({n_pos} fall, {n_neg} no-fall)")

    if n_pos < 3 or n_neg < 3:
        logger.error("Not enough annotated clips to train clip classifier")
        return None, {}

    # ── Model A: Temporal GBM (6 features) ────────────────────────────
    scaler_temp = StandardScaler()
    X_temp_s = scaler_temp.fit_transform(X_temp)
    clf_temp = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    clf_temp.fit(X_temp_s, y)
    temp_acc = clf_temp.score(X_temp_s, y)
    logger.info(f"Temporal GBM: acc={temp_acc:.3f}")

    # ── Model B: Embedding GBM (768 normalised features) ─────────────
    scaler_emb = StandardScaler()
    X_emb_s = scaler_emb.fit_transform(X_norm_emb)
    clf_emb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    clf_emb.fit(X_emb_s, y)
    emb_acc = clf_emb.score(X_emb_s, y)
    logger.info(f"Embedding GBM: acc={emb_acc:.3f}")

    # ── Ensemble training metrics ─────────────────────────────────────
    p_temp = clf_temp.predict_proba(X_temp_s)[:, 1]
    p_emb = clf_emb.predict_proba(X_emb_s)[:, 1]
    p_ens = ENSEMBLE_ALPHA * p_temp + (1 - ENSEMBLE_ALPHA) * p_emb
    ens_preds = (p_ens > 0.5).astype(int)
    train_acc = float((ens_preds == y).mean())
    pos_mask = y == 1
    train_recall = float((p_ens[pos_mask] > 0.5).sum() / max(pos_mask.sum(), 1))

    temp_names = [
        "diff_norm_prev", "diff_norm_next", "deviation_norm",
        "temp_var", "position", "max_diff_window",
    ]

    metrics = {
        "n_samples": len(X_temp),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "accuracy": float(train_acc),
        "recall": float(train_recall),
        "ensemble_alpha": ENSEMBLE_ALPHA,
        "temporal_gbm_accuracy": float(temp_acc),
        "embedding_gbm_accuracy": float(emb_acc),
        "temporal_feature_importances": clf_temp.feature_importances_.tolist(),
        "feature_names": temp_names,
    }

    logger.info(f"Ensemble trained: acc={train_acc:.3f}, "
                f"recall={train_recall:.3f}, alpha={ENSEMBLE_ALPHA}")

    # Save — keep 'clf' and 'scaler' keys for backward compat
    model_dict = {
        "clf": clf_temp,
        "scaler": scaler_temp,
        "clf_emb": clf_emb,
        "scaler_emb": scaler_emb,
        "pca": None,
        "metrics": metrics,
    }
    with open(CLF_PATH, "wb") as f:
        pickle.dump(model_dict, f)

    metrics_path = MODEL_DIR / "clip_classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved clip classifier to {CLF_PATH}")
    return model_dict, metrics


def load_clip_classifier() -> Optional[dict]:
    """Load a trained clip classifier from disk."""
    if not CLF_PATH.exists():
        logger.warning(f"No clip classifier found at {CLF_PATH}")
        return None
    with open(CLF_PATH, "rb") as f:
        model_dict = pickle.load(f)
    logger.info(f"Loaded clip classifier from {CLF_PATH}")
    return model_dict


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _compute_anomaly_scores(
    temp_feats: np.ndarray,
    clips: List[Dict],
    smooth_window: int = 10,
) -> np.ndarray:
    """
    Compute per-video anomaly z-scores from temporal features.

    Uses only temporal-change features (NOT position) and normalises
    within the video so the scores reflect how anomalous a clip is
    *relative to its own video*, not globally.  Temporal smoothing
    boosts sustained anomalies (falls) and suppresses single-frame
    spikes (camera cuts / broadcast transitions).
    """
    n = len(temp_feats)
    duration = clips[-1]["end_time"] if clips else 1

    # Use temporal-change features only (indices 0,1,2,5 — skip position=4, temp_var=3)
    temporal_only = temp_feats[:, [0, 1, 2, 5]]

    # Per-video z-score normalisation
    mu = temporal_only.mean(axis=0)
    sigma = temporal_only.std(axis=0) + 1e-8
    z_scores = (temporal_only - mu) / sigma

    # Mean z-score across features
    raw_scores = z_scores.mean(axis=1)

    # Temporal smoothing (rolling mean)
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(raw_scores, kernel, mode="same")

    # Race-phase suppression
    for i in range(n):
        pos = clips[i]["start_time"] / duration if duration > 0 else 0
        if pos < RACE_START_FRAC or pos > RACE_END_FRAC:
            smoothed[i] = -999

    return smoothed


def predict_video_clips(
    video_id: str,
    clips: List[Dict],
    model_dict: dict,
) -> Tuple[np.ndarray, float]:
    """
    Predict fall probability for every clip using the ensemble.

    Scores each clip with both temporal and embedding GBMs, then blends:
      prob = alpha * temporal_prob + (1-alpha) * embedding_prob

    Falls back to temporal-only if no embedding model is present
    (backward compatibility with older saved models).

    LOGO evaluation: R@10/30s = 94%, R@10/5s = 46%.

    Returns (clip_probs, bag_prob).
    """
    feat_path = FEATURES_DIR / f"{video_id}.npy"
    if not feat_path.exists():
        return np.zeros(len(clips)), 0.0

    features = np.load(feat_path)
    if len(features) != len(clips):
        return np.zeros(len(clips)), 0.0

    temp_feats = compute_temporal_features(features)
    n = len(temp_feats)

    # ── Temporal GBM probabilities ─────────────────────────────────
    X_temp_s = model_dict["scaler"].transform(temp_feats)
    probs_temp = model_dict["clf"].predict_proba(X_temp_s)[:, 1]

    # ── Embedding GBM probabilities ────────────────────────────────
    clf_emb = model_dict.get("clf_emb")
    if clf_emb is not None:
        norm_emb = _normalize_within_video(features[:n])
        X_emb_s = model_dict["scaler_emb"].transform(norm_emb)
        probs_emb = clf_emb.predict_proba(X_emb_s)[:, 1]
        probs = ENSEMBLE_ALPHA * probs_temp + (1 - ENSEMBLE_ALPHA) * probs_emb
    else:
        probs = probs_temp

    # ── Race-phase suppression ─────────────────────────────────────
    duration = clips[-1]["end_time"] if clips else 1
    for i in range(n):
        pos = clips[i]["start_time"] / duration if duration > 0 else 0
        if pos < RACE_START_FRAC or pos > RACE_END_FRAC:
            probs[i] = 0.0

    bag_prob = float(np.max(probs)) if n > 0 else 0.0

    return probs, bag_prob
