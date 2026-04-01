"""
Training loop for the AttentionMIL model.

Uses AdamW optimizer, binary cross-entropy loss at the bag level,
and early stopping based on validation loss.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from .config import CONFIG, MODEL_DIR
from .model import AttentionMIL
from .dataset import build_dataset_entries, create_data_loaders

logger = logging.getLogger(__name__)


# Weight for clip-level supervision loss (relative to bag-level loss)
CLIP_LOSS_WEIGHT = 0.3


def compute_clip_level_loss(clip_probs, clip_labels, masks):
    """
    Compute weighted BCE loss on clips that have supervision (clip_labels >= 0).
    Positive clips are upweighted to handle class imbalance.
    clip_probs: (batch, num_clips) - model clip predictions
    clip_labels: (batch, num_clips) - supervision labels (-1=ignore, 0=neg, 1=pos)
    masks: (batch, num_clips) - valid clip mask
    Returns scalar loss or None if no supervised clips.
    """
    supervised_mask = (clip_labels >= 0) & masks
    if supervised_mask.sum() == 0:
        return None

    pred = clip_probs[supervised_mask]
    target = clip_labels[supervised_mask]

    # Compute positive class weight to handle imbalance
    n_pos = (target == 1).sum().float()
    n_neg = (target == 0).sum().float()
    if n_pos > 0 and n_neg > 0:
        pos_weight = n_neg / n_pos  # upweight rare positive clips
        pos_weight = min(pos_weight, 20.0)  # cap to prevent instability
        weight = torch.where(target == 1, pos_weight, 1.0)
        loss = nn.functional.binary_cross_entropy(pred, target, weight=weight)
    else:
        loss = nn.functional.binary_cross_entropy(pred, target)

    return loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_bag_loss = 0.0
    total_clip_loss = 0.0
    num_batches = 0
    num_clip_batches = 0

    for features, labels, masks, video_ids, clip_labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        clip_labels = clip_labels.to(device)

        optimizer.zero_grad()
        output = model(features, masks)

        # Bag-level loss (always computed)
        bag_loss = criterion(output["bag_prob"], labels)

        # Clip-level loss (only on supervised clips)
        clip_loss = compute_clip_level_loss(
            output["clip_probs"], clip_labels, masks
        )

        # Combined loss
        loss = bag_loss
        if clip_loss is not None:
            loss = loss + CLIP_LOSS_WEIGHT * clip_loss
            total_clip_loss += clip_loss.item()
            num_clip_batches += 1

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bag_loss += bag_loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_bag = total_bag_loss / max(num_batches, 1)
    avg_clip = total_clip_loss / max(num_clip_batches, 1) if num_clip_batches > 0 else 0.0
    return avg_loss, avg_bag, avg_clip


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model. Returns average loss, accuracy, and clip-level accuracy."""
    model.eval()
    total_loss = 0.0
    sum_bag_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    clip_correct = 0
    clip_total = 0

    for features, labels, masks, video_ids, clip_labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        clip_labels = clip_labels.to(device)

        output = model(features, masks)

        # Bag-level loss and accuracy
        bag_loss = criterion(output["bag_prob"], labels)
        clip_loss = compute_clip_level_loss(
            output["clip_probs"], clip_labels, masks
        )
        loss = bag_loss
        if clip_loss is not None:
            loss = loss + CLIP_LOSS_WEIGHT * clip_loss

        total_loss += loss.item()
        sum_bag_loss += bag_loss.item()
        preds = (output["bag_prob"] > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        num_batches += 1

        # Clip-level accuracy on supervised clips
        supervised = (clip_labels >= 0) & masks
        if supervised.sum() > 0:
            clip_preds = (output["clip_probs"][supervised] > 0.5).float()
            clip_targets = clip_labels[supervised]
            clip_correct += (clip_preds == clip_targets).sum().item()
            clip_total += supervised.sum().item()

    avg_loss = total_loss / max(num_batches, 1)
    avg_bag_loss = sum_bag_loss / max(num_batches, 1)
    accuracy = correct / max(total, 1)
    clip_acc = clip_correct / max(clip_total, 1) if clip_total > 0 else None
    return avg_loss, avg_bag_loss, accuracy, clip_acc


def train_model(
    all_clips: Dict = None,
    model_path: Path = None,
    device: str = None,
) -> Tuple[AttentionMIL, dict]:
    """
    Train the AttentionMIL model.
    Returns (trained_model, training_history).
    """
    cfg_model = CONFIG.model
    cfg_train = CONFIG.training
    device = device or CONFIG.features.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_path or MODEL_DIR / "attention_mil.pt"

    # Build dataset
    entries = build_dataset_entries(all_clips)
    if len(entries) < 2:
        logger.error(f"Need at least 2 videos, got {len(entries)}")
        return None, {}

    train_loader, val_loader = create_data_loaders(entries)

    # Create model
    model = AttentionMIL(
        input_dim=cfg_model.input_dim,
        hidden_dim=cfg_model.hidden_dim,
        attention_dim=cfg_model.attention_dim,
        dropout=cfg_model.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_train.learning_rate,
        weight_decay=cfg_train.weight_decay,
    )
    criterion = nn.BCELoss()

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    logger.info(f"Training AttentionMIL on {device} for up to {cfg_train.max_epochs} epochs")
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(cfg_train.max_epochs):
        train_loss, train_bag, train_clip = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_bag_loss, val_acc, val_clip_acc = validate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        clip_str = f" | Clip Acc: {val_clip_acc:.3f}" if val_clip_acc is not None else ""
        logger.info(
            f"Epoch {epoch+1:3d}/{cfg_train.max_epochs} | "
            f"Train Loss: {train_loss:.4f} (bag={train_bag:.4f} clip={train_clip:.4f}) | "
            f"Val BagLoss: {val_bag_loss:.4f} | "
            f"Val Acc: {val_acc:.3f}{clip_str}"
        )

        # Early stopping based on BAG-LEVEL val loss only
        # (clip loss can be noisy with sparse supervision)
        if val_bag_loss < best_val_loss:
            best_val_loss = val_bag_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": {
                    "input_dim": cfg_model.input_dim,
                    "hidden_dim": cfg_model.hidden_dim,
                    "attention_dim": cfg_model.attention_dim,
                    "dropout": cfg_model.dropout,
                },
            }, model_path)
            logger.info(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg_train.patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={cfg_train.patience})")
                break

    # Load best model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Save training history
    history_path = MODEL_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    return model, history


def load_trained_model(model_path: Path = None, device: str = None) -> Optional[AttentionMIL]:
    """Load a trained AttentionMIL model from checkpoint."""
    model_path = model_path or MODEL_DIR / "attention_mil.pt"
    device = device or CONFIG.features.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if not model_path.exists():
        logger.warning(f"No trained model found at {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    model = AttentionMIL(
        input_dim=cfg.get("input_dim", 768),
        hidden_dim=cfg.get("hidden_dim", 256),
        attention_dim=cfg.get("attention_dim", 128),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded model from {model_path} (epoch={checkpoint.get('epoch', '?')})")
    return model
