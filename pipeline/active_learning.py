"""
Active Learning Controller.

Manages the iterative improvement cycle:
- Monitors annotation count
- Triggers retraining when threshold is met
- Transitions from cold-start to model-based proposals
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from .config import (
    CONFIG,
    ANNOTATIONS_FILE,
    ACTIVE_LEARNING_STATE,
    MODEL_DIR,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


class ActiveLearningController:
    """Orchestrates the active learning feedback loop."""

    def __init__(self):
        self.cfg = CONFIG.active_learning
        self.state_path = ACTIVE_LEARNING_STATE
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load persistent state."""
        if self.state_path.exists():
            with open(self.state_path) as f:
                return json.load(f)
        return {
            "last_training_count": 0,
            "total_retrains": 0,
            "mode": "cold_start",  # "cold_start" or "model"
        }

    def _save_state(self):
        """Persist state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def count_annotations(self) -> int:
        """Count confirmed annotations in the JSONL file."""
        if not ANNOTATIONS_FILE.exists():
            return 0
        count = 0
        with open(ANNOTATIONS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if record.get("confirmed", False):
                            count += 1
                    except json.JSONDecodeError:
                        continue
        return count

    def should_retrain(self) -> bool:
        """Check if retraining conditions are met."""
        total = self.count_annotations()
        last = self.state["last_training_count"]

        cond1 = total >= self.cfg.min_training_samples
        cond2 = (total - last) >= self.cfg.retraining_threshold

        if cond1 and cond2:
            logger.info(
                f"Retraining triggered: {total} total annotations, "
                f"{total - last} new since last training"
            )
            return True
        return False

    def get_mode(self) -> str:
        """Return current mode: 'cold_start' or 'model'."""
        clip_clf_path = MODEL_DIR / "clip_classifier.pkl"
        model_path = MODEL_DIR / "attention_mil.pt"
        if (clip_clf_path.exists() or model_path.exists()) and self.count_annotations() >= self.cfg.min_training_samples:
            self.state["mode"] = "model"
        else:
            self.state["mode"] = "cold_start"
        self._save_state()
        return self.state["mode"]

    def trigger_retrain(self) -> bool:
        """
        Execute the retraining cycle:
        1. Retrain clip classifier (temporal-difference GBM)
        2. Regenerate proposals
        3. Update state
        Returns True if retraining was performed.
        """
        if not self.should_retrain():
            logger.info("Retraining conditions not met, skipping")
            return False

        from .clip_classifier import train_clip_classifier
        from .proposals import run_proposal_generation

        logger.info("=== Active Learning: Retraining cycle started ===")

        # Step 1: Retrain clip classifier
        clip_clf, metrics = train_clip_classifier()
        if clip_clf is None:
            logger.error("Retraining failed")
            return False

        # Step 2: Regenerate proposals
        run_proposal_generation(clip_clf=clip_clf)

        # Step 3: Update state
        self.state["last_training_count"] = self.count_annotations()
        self.state["total_retrains"] += 1
        self.state["mode"] = "model"
        self._save_state()

        logger.info(
            f"=== Active Learning: Cycle complete "
            f"(retrain #{self.state['total_retrains']}) ==="
        )
        return True

    def get_status(self) -> dict:
        """Return current active learning status."""
        total = self.count_annotations()
        last = self.state["last_training_count"]
        return {
            "mode": self.get_mode(),
            "total_annotations": total,
            "annotations_since_last_train": total - last,
            "total_retrains": self.state["total_retrains"],
            "retrain_needed": self.should_retrain(),
            "min_training_samples": self.cfg.min_training_samples,
            "retraining_threshold": self.cfg.retraining_threshold,
        }
