"""
Label utilities for grief-event-based video reclassification.

The original dataset organises videos by folder (Fell/ vs Pulled-up/),
but many Pulled-up videos contain fall-type incidents (unseated rider,
brought down, slipped up). This module provides the corrected labels
based on the actual grief event metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .config import PROJECT_ROOT, ANNOTATIONS_FILE, CLIP_META_DIR

logger = logging.getLogger(__name__)

# ── Grief event taxonomy ────────────────────────────────────────────────────
FALL_TYPES: Set[str] = {"fell", "brought_down", "unseated_rider", "slipped_up"}
HARD_NEGATIVE_TYPES: Set[str] = {"pulled_up", "refused", "carried_out", "ran_out"}

EXCEL_PATH = PROJECT_ROOT / "video_selection_100.xlsx"


def _normalise_event(event: str) -> str:
    """Normalise a grief event string: lowercase, strip, replace spaces."""
    return event.strip().lower().replace(" ", "_").replace("-", "_")


def load_grief_labels() -> Dict[str, Dict]:
    """
    Load grief event metadata from the Excel file and return a dict
    mapping video_id -> {
        'events': list of normalised event strings,
        'is_fall': bool (True if any event is a fall type),
        'binary_label': int (1 for fall, 0 for non-fall),
        'primary_event': str (first event),
    }
    """
    if not EXCEL_PATH.exists():
        logger.warning(f"Excel file not found: {EXCEL_PATH}")
        return {}

    df = pd.read_excel(EXCEL_PATH)
    labels = {}

    for _, row in df.iterrows():
        video_id = str(row.get("Video_ID", "")).strip()
        if not video_id:
            continue

        raw_events = str(row.get("All_Grief_Events", ""))
        events = [_normalise_event(e) for e in raw_events.split(",") if e.strip()]
        is_fall = any(e in FALL_TYPES for e in events)

        labels[video_id] = {
            "events": events,
            "is_fall": is_fall,
            "binary_label": 1 if is_fall else 0,
            "primary_event": events[0] if events else "unknown",
        }

    n_fall = sum(1 for v in labels.values() if v["is_fall"])
    n_nofall = len(labels) - n_fall
    logger.info(
        f"Loaded grief labels for {len(labels)} videos: "
        f"{n_fall} fall, {n_nofall} non-fall"
    )
    return labels


def get_corrected_binary_labels() -> Dict[str, int]:
    """
    Return a mapping of video_id -> corrected binary label (0 or 1).
    Falls back to folder-based labels from clip metadata if Excel is unavailable.
    """
    grief = load_grief_labels()
    if grief:
        return {vid: info["binary_label"] for vid, info in grief.items()}

    # Fallback: use clip metadata (folder-based labels)
    logger.warning("Using folder-based labels as fallback")
    clips_file = CLIP_META_DIR / "all_clips.json"
    if clips_file.exists():
        with open(clips_file) as f:
            all_clips = json.load(f)
        return {
            vid: clips[0]["binary_label"]
            for vid, clips in all_clips.items()
            if clips
        }
    return {}


def load_temporal_annotations() -> Dict[str, List[Dict]]:
    """
    Load confirmed temporal annotations from the JSONL file.
    Returns a dict mapping video_id -> list of annotation records.
    Each record has: timestamp, end_timestamp, label_type, confirmed.
    """
    annotations: Dict[str, List[Dict]] = {}

    if not ANNOTATIONS_FILE.exists():
        return annotations

    with open(ANNOTATIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not record.get("confirmed", False):
                continue

            vid = record.get("video_id", "")
            if not vid:
                continue

            annotations.setdefault(vid, []).append({
                "timestamp": record.get("timestamp", 0),
                "end_timestamp": record.get("end_timestamp"),
                "label_type": record.get("label_type", ""),
            })

    logger.info(
        f"Loaded temporal annotations for {len(annotations)} videos "
        f"({sum(len(v) for v in annotations.values())} total annotations)"
    )
    return annotations


def build_clip_level_labels(
    clips: List[Dict],
    video_annotations: List[Dict],
    video_binary_label: int,
) -> List[int]:
    """
    Build clip-level supervision labels from temporal annotations.

    Returns a list of length len(clips) where:
      1  = clip overlaps with a fall/incident annotation (positive)
      0  = clip overlaps with a no_incident annotation (negative)
     -1  = unsupervised (no annotation covers this clip)

    IMPORTANT: Only videos with explicit annotations get clip-level
    supervision. Unannotated videos (even negatives) stay unsupervised
    to avoid flooding the clip loss with trivial negatives.
    """
    n_clips = len(clips)
    clip_labels = [-1] * n_clips  # default: unsupervised

    # For negative videos without annotations, all clips are negative
    if video_binary_label == 0 and not video_annotations:
        return [0] * n_clips

    # For positive videos without annotations, stay unsupervised
    if not video_annotations:
        return clip_labels

    # Fall-type label types
    fall_label_types = {
        "fall", "fell", "brought_down", "unseated_rider", "slipped_up",
    }
    # Negative label types
    neg_label_types = {
        "no_incident", "no_fall", "pulled_up", "pull_up",
        "refused", "carried_out", "ran_out",
        "not_visible", "non_race_footage", "riderless_horse",
    }

    for ann in video_annotations:
        t_start = ann["timestamp"]
        t_end = ann.get("end_timestamp") or (t_start + 3.0)
        label_type = ann["label_type"].lower().strip()

        is_positive = label_type in fall_label_types
        is_negative = label_type in neg_label_types

        if not is_positive and not is_negative:
            continue

        # Find clips that overlap with this annotation
        for idx, clip in enumerate(clips):
            clip_start = clip["start_time"]
            clip_end = clip["end_time"]

            # Check temporal overlap
            if clip_start < t_end and clip_end > t_start:
                if is_positive:
                    clip_labels[idx] = 1
                elif is_negative and clip_labels[idx] == -1:
                    # Only mark negative if not already marked positive
                    clip_labels[idx] = 0

    return clip_labels


# ── Annotation label constants (for the UI) ────────────────────────────────
ANNOTATION_LABELS = [
    "fall", "brought_down", "unseated_rider", "slipped_up",
    "pulled_up", "refused", "carried_out", "ran_out",
    "no_incident",
    "not_visible", "non_race_footage", "riderless_horse",
]

ANNOTATION_DISPLAY = {
    "fall":              "🔴 Fall",
    "brought_down":      "🔴 Brought down",
    "unseated_rider":    "🔴 Unseated rider",
    "slipped_up":        "🔴 Slipped up",
    "pulled_up":         "🟡 Pulled up",
    "refused":           "🟡 Refused",
    "carried_out":       "🟡 Carried out",
    "ran_out":           "🟡 Ran out",
    "no_incident":       "🟢 No incident (clean)",
    "not_visible":       "⚫ Not visible (off-camera)",
    "non_race_footage":  "⚫ Non-race footage (junk)",
    "riderless_horse":   "⚫ Riderless horse (post-incident)",
}
