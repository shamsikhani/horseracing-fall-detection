"""
Held-out evaluation for the ensemble fall detection model.

Strategy:
- Split 51 annotated fall videos into train (~40) and test (~11) sets
- All 60 non-fall videos always stay in training (they provide negatives)
- Retrain both GBMs on the train split only
- Generate proposals on the held-out test videos
- Compute Recall@K at 5s and 30s tolerance on held-out videos
- Repeat over 5 random splits and report mean +/- std

This gives a clean, honest evaluation that a reviewer cannot attack.
"""

import json
import sys
import os
import numpy as np
from collections import defaultdict
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FEATURES_DIR = OUTPUT_DIR / "features"
CLIP_META_DIR = OUTPUT_DIR / "clip_metadata"
ANNOTATIONS_FILE = OUTPUT_DIR / "annotations.jsonl"

# ── Constants (same as clip_classifier.py) ─────────────────────────────────
FALL_TYPES = {"fall", "fell", "brought_down", "unseated_rider", "slipped_up"}
NEG_TYPES = {"no_incident", "no_fall", "pulled_up", "pull_up", "refused",
             "carried_out", "ran_out"}
JUNK_TYPES = {"not_visible", "non_race_footage", "riderless_horse"}
RACE_START_FRAC = 0.05
RACE_END_FRAC = 0.85
ENSEMBLE_ALPHA = 0.5
HARD_NEG_PER_VIDEO = 3
NMS_WINDOW = 15.0
TOP_K = 10
MIN_PROB = 0.01

# ── Load data ──────────────────────────────────────────────────────────────

def load_annotations():
    """Load confirmed annotations grouped by video_id."""
    annotations = defaultdict(list)
    with open(ANNOTATIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ann = json.loads(line)
            if ann.get("confirmed"):
                annotations[ann["video_id"]].append(ann)
    return dict(annotations)


def load_all_clips():
    clips_file = CLIP_META_DIR / "all_clips.json"
    with open(clips_file) as f:
        return json.load(f)


def get_corrected_binary_labels():
    """Get binary labels from grief metadata Excel."""
    try:
        import pandas as pd
        excel_path = PROJECT_ROOT / "video_selection_100.xlsx"
        if excel_path.exists():
            df = pd.read_excel(excel_path)
            labels = {}
            fall_types = {"fell", "brought_down", "unseated_rider", "slipped_up"}
            for _, row in df.iterrows():
                vid = str(row.get("Video_ID", "")).strip()
                if not vid:
                    continue
                raw = str(row.get("All_Grief_Events", ""))
                events = [e.strip().lower().replace(" ", "_").replace("-", "_")
                          for e in raw.split(",") if e.strip()]
                is_fall = any(e in fall_types for e in events)
                labels[vid] = 1 if is_fall else 0
            return labels
    except Exception:
        pass
    # Fallback: use clip metadata
    all_clips = load_all_clips()
    return {vid: clips[0]["binary_label"] for vid, clips in all_clips.items() if clips}


# ── Feature computation (same as clip_classifier.py) ───────────────────────

def compute_temporal_features(features, window=5):
    n, d = features.shape
    diff_prev = np.zeros(n)
    diff_next = np.zeros(n)
    for i in range(1, n):
        diff_prev[i] = np.linalg.norm(features[i] - features[i - 1])
    for i in range(n - 1):
        diff_next[i] = np.linalg.norm(features[i] - features[i + 1])
    deviation_norm = np.zeros(n)
    for i in range(n):
        s, e = max(0, i - window), min(n, i + window + 1)
        ctx = np.mean(features[s:e], axis=0)
        deviation_norm[i] = np.linalg.norm(features[i] - ctx)
    temp_var = np.zeros(n)
    for i in range(n):
        s, e = max(0, i - window), min(n, i + window + 1)
        temp_var[i] = np.mean(np.var(features[s:e], axis=0))
    position = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    max_diff_window = np.zeros(n)
    for i in range(n):
        s, e = max(0, i - window), min(n, i + window + 1)
        local = features[s:e]
        if len(local) > 1:
            diffs = np.linalg.norm(np.diff(local, axis=0), axis=1)
            max_diff_window[i] = np.max(diffs)
    return np.column_stack([diff_prev, diff_next, deviation_norm,
                            temp_var, position, max_diff_window])


def normalize_within_video(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    return (features - mean) / std


def temporal_nms(candidates, nms_window=NMS_WINDOW, top_k=TOP_K, min_prob=MIN_PROB):
    candidates = [c for c in candidates if c["prob"] >= min_prob]
    candidates = sorted(candidates, key=lambda c: c["prob"], reverse=True)
    selected = []
    suppressed = set()
    for i, cand in enumerate(candidates):
        if i in suppressed:
            continue
        selected.append(cand)
        if len(selected) >= top_k:
            break
        for j in range(i + 1, len(candidates)):
            if abs(cand["start_time"] - candidates[j]["start_time"]) < nms_window:
                suppressed.add(j)
    return selected[:top_k]


# ── Build training data (with exclusion set) ───────────────────────────────

def build_training_data(exclude_vids, annotations, all_clips, corrected_labels):
    """Build training data excluding a set of videos."""
    X_temp_list, X_emb_list, y_list = [], [], []

    # Positive/negative clips from annotations
    for vid, anns in annotations.items():
        if vid in exclude_vids:
            continue
        feat_path = FEATURES_DIR / f"{vid}.npy"
        if not feat_path.exists():
            continue
        features = np.load(feat_path)
        clips = all_clips.get(vid, [])
        if len(clips) != len(features):
            continue
        temp_feats = compute_temporal_features(features)
        norm_emb = normalize_within_video(features)
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
                    X_emb_list.append(norm_emb[idx])
                    y_list.append(label)

    # Sampled negatives from non-fall videos
    rng = np.random.RandomState(42)
    for vid, label in corrected_labels.items():
        if label == 1:
            continue
        if vid in exclude_vids:
            continue
        feat_path = FEATURES_DIR / f"{vid}.npy"
        if not feat_path.exists():
            continue
        features = np.load(feat_path)
        clips = all_clips.get(vid, [])
        if len(clips) != len(features):
            continue
        temp_feats = compute_temporal_features(features)
        norm_emb = normalize_within_video(features)
        n = len(temp_feats)
        start = int(n * RACE_START_FRAC)
        end = int(n * RACE_END_FRAC)
        if end <= start:
            continue
        indices = rng.choice(range(start, end), size=min(10, end - start), replace=False)
        for idx in indices:
            X_temp_list.append(temp_feats[idx])
            X_emb_list.append(norm_emb[idx])
            y_list.append(0)
        # Hard negatives
        if HARD_NEG_PER_VIDEO > 0:
            race_temp = temp_feats[start:end]
            change_score = race_temp[:, 0] + race_temp[:, 2]
            top_k = min(HARD_NEG_PER_VIDEO, len(change_score))
            hard_indices = np.argsort(change_score)[-top_k:]
            for local_idx in hard_indices:
                idx = start + local_idx
                X_temp_list.append(temp_feats[idx])
                X_emb_list.append(norm_emb[idx])
                y_list.append(0)

    return np.array(X_temp_list), np.array(y_list), np.array(X_emb_list)


# ── Train and predict ──────────────────────────────────────────────────────

def train_ensemble(X_temp, y, X_emb):
    scaler_temp = StandardScaler()
    X_temp_s = scaler_temp.fit_transform(X_temp)
    clf_temp = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=5, random_state=42)
    clf_temp.fit(X_temp_s, y)

    scaler_emb = StandardScaler()
    X_emb_s = scaler_emb.fit_transform(X_emb)
    clf_emb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=5, random_state=42)
    clf_emb.fit(X_emb_s, y)

    return {"clf": clf_temp, "scaler": scaler_temp,
            "clf_emb": clf_emb, "scaler_emb": scaler_emb}


def predict_and_propose(vid, clips, model_dict):
    """Predict clip probs and run NMS for a single video."""
    feat_path = FEATURES_DIR / f"{vid}.npy"
    if not feat_path.exists():
        return []
    features = np.load(feat_path)
    if len(features) != len(clips):
        return []

    temp_feats = compute_temporal_features(features)
    norm_emb = normalize_within_video(features)
    n = len(temp_feats)

    X_temp_s = model_dict["scaler"].transform(temp_feats)
    probs_temp = model_dict["clf"].predict_proba(X_temp_s)[:, 1]

    X_emb_s = model_dict["scaler_emb"].transform(norm_emb)
    probs_emb = model_dict["clf_emb"].predict_proba(X_emb_s)[:, 1]

    probs = ENSEMBLE_ALPHA * probs_temp + (1 - ENSEMBLE_ALPHA) * probs_emb

    # Race-phase suppression
    duration = clips[-1]["end_time"] if clips else 1
    for i in range(n):
        pos = clips[i]["start_time"] / duration if duration > 0 else 0
        if pos < RACE_START_FRAC or pos > RACE_END_FRAC:
            probs[i] = 0.0

    candidates = [{"start_time": clips[i]["start_time"], "prob": float(probs[i])}
                   for i in range(n)]
    return temporal_nms(candidates)


# ── Recall@K computation ──────────────────────────────────────────────────

def compute_recall_at_k(proposals, fall_timestamps, k_values, tolerances):
    """
    proposals: list of NMS candidates (sorted by prob desc)
    fall_timestamps: list of annotated fall timestamps for this video
    Returns dict of (k, tol) -> 1/0
    """
    results = {}
    for k in k_values:
        top_k = proposals[:k]
        top_k_times = [c["start_time"] for c in top_k]
        for tol in tolerances:
            detected = 0
            for ts in fall_timestamps:
                if any(abs(t - ts) <= tol for t in top_k_times):
                    detected = 1
                    break
            results[(k, tol)] = detected
    return results


# ── Main evaluation ───────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 70)
    print("HELD-OUT EVALUATION: Ensemble Fall Detection Model")
    print("=" * 70)

    # Load all data
    annotations = load_annotations()
    all_clips = load_all_clips()
    corrected_labels = get_corrected_binary_labels()

    # Identify fall videos with timestamps
    fall_vid_timestamps = defaultdict(list)
    for vid, anns in annotations.items():
        for ann in anns:
            if ann.get("confirmed") and ann["label_type"] in FALL_TYPES:
                ts = ann.get("timestamp", 0)
                if ts and ts > 0:
                    fall_vid_timestamps[vid].append(ts)

    fall_vids = sorted(fall_vid_timestamps.keys())
    print(f"\nFall videos with timestamps: {len(fall_vids)}")
    print(f"Total videos: {len(all_clips)}")

    k_values = [1, 3, 5, 10]
    tolerances = [5, 30]
    n_splits = 5
    test_fraction = 0.2  # ~10 test videos per split

    all_split_results = []

    for split_idx in range(n_splits):
        rng = np.random.RandomState(split_idx * 7 + 13)  # different seed per split
        n_test = max(1, int(len(fall_vids) * test_fraction))
        test_vids = set(rng.choice(fall_vids, size=n_test, replace=False))
        train_vids = set(fall_vids) - test_vids

        print(f"\n{'-' * 60}")
        print(f"Split {split_idx + 1}/{n_splits}: "
              f"{len(train_vids)} train fall vids, {len(test_vids)} test fall vids")
        print(f"  Test vids: {sorted(test_vids)}")

        # Build training data excluding test videos
        X_temp, y, X_emb = build_training_data(
            exclude_vids=test_vids, annotations=annotations,
            all_clips=all_clips, corrected_labels=corrected_labels)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        print(f"  Training data: {len(y)} clips ({n_pos} pos, {n_neg} neg)")

        if n_pos < 3 or n_neg < 3:
            print("  SKIP: insufficient training data")
            continue

        # Train
        model_dict = train_ensemble(X_temp, y, X_emb)

        # Evaluate on held-out test videos
        split_results = {(k, t): [] for k in k_values for t in tolerances}

        for vid in sorted(test_vids):
            clips = all_clips.get(vid, [])
            if not clips:
                continue
            proposals = predict_and_propose(vid, clips, model_dict)
            timestamps = fall_vid_timestamps[vid]
            recall = compute_recall_at_k(proposals, timestamps, k_values, tolerances)
            for key, val in recall.items():
                split_results[key].append(val)

            # Show per-video detail
            top1_time = proposals[0]["start_time"] if proposals else -1
            top1_prob = proposals[0]["prob"] if proposals else 0
            gt = timestamps[0]
            dist = abs(top1_time - gt) if proposals else 999
            detected_5s = "Y" if recall.get((5, 5), 0) else "N"
            print(f"    {vid}: GT={gt:.0f}s, Top1={top1_time:.0f}s (d={dist:.0f}s, p={top1_prob:.3f}), "
                  f"R@5/5s={detected_5s}")

        # Aggregate per split
        split_agg = {}
        for key, vals in split_results.items():
            if vals:
                split_agg[key] = sum(vals) / len(vals)
            else:
                split_agg[key] = 0.0
        all_split_results.append(split_agg)

        print(f"\n  Split {split_idx + 1} results:")
        for tol in tolerances:
            line = f"    Tol={tol}s: "
            parts = []
            for k in k_values:
                val = split_agg.get((k, tol), 0)
                parts.append(f"R@{k}={val*100:.1f}%")
            line += ", ".join(parts)
            print(line)

    # --- Aggregate across all splits ---------------------------------------

    print(f"\n{'=' * 70}")
    print(f"AGGREGATE RESULTS ACROSS {n_splits} SPLITS")
    print('=' * 70)

    for tol in tolerances:
        print(f"\n  Tolerance = {tol}s:")
        for k in k_values:
            vals = [s.get((k, tol), 0) for s in all_split_results]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            print(f"    R@{k:2d}: {mean_val*100:.1f}% +/- {std_val*100:.1f}%  "
                  f"(per-split: {[f'{v*100:.0f}%' for v in vals]})")

    # ── Also run full LOGO (leave-one-group-out) for comparison ─────────
    print('\n' + '=' * 70)
    print('LOGO CROSS-VALIDATION (leave-one-fall-video-out)')
    print('=' * 70)

    logo_results = {(k, t): [] for k in k_values for t in tolerances}

    for i, vid in enumerate(fall_vids):
        test_set = {vid}
        X_temp, y, X_emb = build_training_data(
            exclude_vids=test_set, annotations=annotations,
            all_clips=all_clips, corrected_labels=corrected_labels)
        n_pos = int(y.sum())
        if n_pos < 3:
            print(f"  SKIP {vid}: insufficient positives ({n_pos})")
            continue
        model_dict = train_ensemble(X_temp, y, X_emb)
        clips = all_clips.get(vid, [])
        if not clips:
            continue
        proposals = predict_and_propose(vid, clips, model_dict)
        timestamps = fall_vid_timestamps[vid]
        recall = compute_recall_at_k(proposals, timestamps, k_values, tolerances)
        for key, val in recall.items():
            logo_results[key].append(val)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(fall_vids)} videos...")

    print(f"\n  LOGO results ({len(fall_vids)} fall videos):")
    for tol in tolerances:
        print(f"\n  Tolerance = {tol}s:")
        for k in k_values:
            vals = logo_results.get((k, tol), [])
            if vals:
                mean_val = np.mean(vals)
                n_detected = sum(vals)
                print(f"    R@{k:2d}: {mean_val*100:.1f}%  ({n_detected}/{len(vals)} videos)")

    # ── Summary for paper ──────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('SUMMARY FOR PAPER')
    print('=' * 70)
    print(f"\n  Dataset: {len(fall_vids)} fall videos, {len(all_clips) - len(fall_vids)} non-fall videos")
    print(f"\n  Held-out (20%, 5 random splits):")
    for tol in tolerances:
        for k in k_values:
            vals = [s.get((k, tol), 0) for s in all_split_results]
            print(f"    R@{k}/{'5' if tol==5 else '30'}s = {np.mean(vals)*100:.1f} +/- {np.std(vals)*100:.1f}%")

    print(f"\n  LOGO (leave-one-out, {len(fall_vids)} folds):")
    for tol in tolerances:
        for k in k_values:
            vals = logo_results.get((k, tol), [])
            if vals:
                print(f"    R@{k}/{'5' if tol==5 else '30'}s = {np.mean(vals)*100:.1f}%  ({sum(vals)}/{len(vals)})")


if __name__ == "__main__":
    run_evaluation()
