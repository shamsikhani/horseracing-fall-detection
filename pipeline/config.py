"""
Centralised configuration for the Horse Racing Incident Detection Pipeline.
All hyperparameters and paths are defined here using dataclasses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ── Root paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT

# ── Output directories ──────────────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "output"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
CLIP_META_DIR = OUTPUT_DIR / "clip_metadata"
FEATURES_DIR = OUTPUT_DIR / "features"
MODEL_DIR = OUTPUT_DIR / "models"
PROPOSALS_DIR = OUTPUT_DIR / "proposals"
ANNOTATIONS_FILE = OUTPUT_DIR / "annotations.jsonl"
ACTIVE_LEARNING_STATE = OUTPUT_DIR / "active_learning_state.json"


@dataclass
class PreprocessingConfig:
    target_fps: int = 25
    target_resolution: tuple = (224, 224)
    codec: str = "libx264"
    crf: int = 23


@dataclass
class SegmentationConfig:
    clip_duration: float = 3.0      # seconds
    stride: float = 1.0             # seconds
    min_clip_duration: float = 1.0  # seconds


@dataclass
class FeatureExtractionConfig:
    model_name: str = "MCG-NJU/videomae-base"
    frames_per_clip: int = 16
    embedding_dim: int = 768
    batch_size: int = 4
    device: str = "cuda"


@dataclass
class MILModelConfig:
    input_dim: int = 768
    hidden_dim: int = 256
    attention_dim: int = 128   # hidden_dim / 2
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    max_epochs: int = 50
    patience: int = 10
    val_split: float = 0.2
    seed: int = 42


@dataclass
class ProposalConfig:
    top_k: int = 10
    nms_window: float = 15.0      # seconds — 15s allows nearby but distinct events
    min_probability: float = 0.01


@dataclass
class ColdStartConfig:
    focus_start: float = 0.4      # fraction of video duration
    focus_end: float = 0.8
    num_candidates: int = 5
    default_probability: float = 0.5


@dataclass
class ActiveLearningConfig:
    min_training_samples: int = 5
    retraining_threshold: int = 10


@dataclass
class GriefTaxonomy:
    """Maps grief labels to binary classes for the two-class setup:
       Fell = Positive (1), Pulled-up = Negative (0).
    """
    positive_labels: List[str] = field(default_factory=lambda: ["fell"])
    negative_labels: List[str] = field(default_factory=lambda: ["pulled-up", "pulled up", "pu"])
    ignored_labels: List[str] = field(default_factory=lambda: ["unknown", "void", "empty"])

    label_to_binary: Dict[str, int] = field(default_factory=lambda: {
        "fell": 1,
        "pulled-up": 0,
        "pulled up": 0,
        "pu": 0,
    })

    label_to_id: Dict[str, int] = field(default_factory=lambda: {
        "fell": 0,
        "pulled-up": 1,
        "pulled up": 1,
    })

    def map_label(self, label: str) -> int:
        """Map a grief label string to binary class (0 or 1)."""
        label_lower = label.strip().lower()
        for key, val in self.label_to_binary.items():
            if key in label_lower:
                return val
        return -1  # ignored


@dataclass
class PipelineConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    features: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    model: MILModelConfig = field(default_factory=MILModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    proposals: ProposalConfig = field(default_factory=ProposalConfig)
    cold_start: ColdStartConfig = field(default_factory=ColdStartConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    taxonomy: GriefTaxonomy = field(default_factory=GriefTaxonomy)


# Singleton config instance
CONFIG = PipelineConfig()
