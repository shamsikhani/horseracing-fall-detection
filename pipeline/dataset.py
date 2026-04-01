"""
MIL Dataset and custom collation for variable-length sequences.

Each sample is a bag (one video's clip embeddings) with a binary label.
Shorter bags are zero-padded to the longest bag in the batch.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import CONFIG, FEATURES_DIR, CLIP_META_DIR
from .label_utils import (
    get_corrected_binary_labels,
    load_temporal_annotations,
    build_clip_level_labels,
)

logger = logging.getLogger(__name__)


class MILVideoDataset(Dataset):
    """Dataset for MIL training: each item is (features, label, video_id, clip_labels)."""

    def __init__(self, video_entries: List[Dict]):
        """
        video_entries: list of dicts with keys:
          - video_id: str
          - binary_label: int (0 or 1)
          - feature_path: str or Path to .npy file
          - clip_labels: optional list of int (-1=unsupervised, 0=neg, 1=pos)
        """
        self.entries = []
        for entry in video_entries:
            feat_path = Path(entry["feature_path"])
            if feat_path.exists():
                self.entries.append(entry)
            else:
                logger.warning(f"Feature file not found: {feat_path}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        features = np.load(entry["feature_path"])  # (num_clips, 768)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(entry["binary_label"], dtype=torch.float32)
        clip_labels = entry.get("clip_labels")
        if clip_labels is not None:
            clip_labels = torch.tensor(clip_labels, dtype=torch.float32)
        else:
            clip_labels = torch.full((features.shape[0],), -1.0)
        return features, label, entry["video_id"], clip_labels


def mil_collate_fn(batch):
    """Custom collation: zero-pad to the longest sequence in the batch.
    Returns (features, labels, masks, video_ids, clip_labels).
    """
    features_list, labels_list, video_ids, clip_labels_list = zip(*batch)
    batch_size = len(features_list)
    max_len = max(f.shape[0] for f in features_list)
    feat_dim = features_list[0].shape[1]

    # Pad features, masks, and clip labels
    padded_features = torch.zeros(batch_size, max_len, feat_dim)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    padded_clip_labels = torch.full((batch_size, max_len), -1.0)

    for i, (feat, cl) in enumerate(zip(features_list, clip_labels_list)):
        length = feat.shape[0]
        padded_features[i, :length] = feat
        masks[i, :length] = True
        padded_clip_labels[i, :cl.shape[0]] = cl

    labels = torch.stack(labels_list)
    return padded_features, labels, masks, list(video_ids), padded_clip_labels


def build_dataset_entries(all_clips: Dict[str, List[Dict]] = None) -> List[Dict]:
    """Build dataset entries from clip metadata, corrected labels, and annotations."""
    entries = []

    if all_clips is None:
        clips_file = CLIP_META_DIR / "all_clips.json"
        if clips_file.exists():
            with open(clips_file) as f:
                all_clips = json.load(f)
        else:
            logger.error("No clip metadata found")
            return entries

    # Load corrected labels from grief event metadata
    corrected_labels = get_corrected_binary_labels()
    # Load temporal annotations for clip-level supervision
    temporal_anns = load_temporal_annotations()

    n_corrected = 0
    n_with_clip_supervision = 0

    for video_id, clips in all_clips.items():
        if not clips:
            continue

        # Use corrected label if available, else fall back to folder-based
        folder_label = clips[0]["binary_label"]
        corrected = corrected_labels.get(video_id, folder_label)
        if corrected != folder_label:
            n_corrected += 1

        feat_path = FEATURES_DIR / f"{video_id}.npy"

        # Build clip-level labels from temporal annotations
        video_anns = temporal_anns.get(video_id, [])
        clip_labels = build_clip_level_labels(clips, video_anns, corrected)
        if any(cl >= 0 for cl in clip_labels):
            n_with_clip_supervision += 1

        entry = {
            "video_id": video_id,
            "binary_label": corrected,
            "label_name": clips[0]["label_name"],
            "feature_path": str(feat_path),
            "num_clips": len(clips),
            "clip_labels": clip_labels,
        }
        entries.append(entry)

    if n_corrected > 0:
        logger.info(f"Corrected labels for {n_corrected} videos (grief-event reclassification)")
    logger.info(f"{n_with_clip_supervision}/{len(entries)} videos have clip-level supervision")

    return entries


def create_data_loaders(
    entries: List[Dict],
    val_split: float = None,
    batch_size: int = None,
    seed: int = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders with stratified split."""
    cfg = CONFIG.training
    val_split = val_split or cfg.val_split
    batch_size = batch_size or cfg.batch_size
    seed = seed or cfg.seed

    # Stratified split
    pos_entries = [e for e in entries if e["binary_label"] == 1]
    neg_entries = [e for e in entries if e["binary_label"] == 0]

    rng = np.random.RandomState(seed)
    rng.shuffle(pos_entries)
    rng.shuffle(neg_entries)

    n_val_pos = max(1, int(len(pos_entries) * val_split))
    n_val_neg = max(1, int(len(neg_entries) * val_split))

    val_entries = pos_entries[:n_val_pos] + neg_entries[:n_val_neg]
    train_entries = pos_entries[n_val_pos:] + neg_entries[n_val_neg:]

    logger.info(f"Train: {len(train_entries)} videos "
                f"({sum(1 for e in train_entries if e['binary_label']==1)} pos, "
                f"{sum(1 for e in train_entries if e['binary_label']==0)} neg)")
    logger.info(f"Val:   {len(val_entries)} videos "
                f"({sum(1 for e in val_entries if e['binary_label']==1)} pos, "
                f"{sum(1 for e in val_entries if e['binary_label']==0)} neg)")

    train_dataset = MILVideoDataset(train_entries)
    val_dataset = MILVideoDataset(val_entries)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=mil_collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=mil_collate_fn, drop_last=False,
    )

    return train_loader, val_loader
