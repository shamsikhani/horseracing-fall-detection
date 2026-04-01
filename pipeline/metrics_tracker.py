"""
Comprehensive Metrics Tracking System for Publication.

Tracks all aspects of the active learning pipeline:
- Annotation efficiency (time savings, human effort)
- Model performance over iterations (precision, recall, temporal error)
- Active learning effectiveness (annotation reduction, sample efficiency)
- Class-specific performance (fall vs no-fall)
- Dataset statistics and evolution
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass, asdict

from .config import OUTPUT_DIR

logger = logging.getLogger(__name__)

METRICS_DIR = OUTPUT_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_LOG = METRICS_DIR / "experiment_log.jsonl"
ANNOTATION_LOG = METRICS_DIR / "annotation_metrics.jsonl"
TRAINING_LOG = METRICS_DIR / "training_metrics.jsonl"
ACTIVE_LEARNING_LOG = METRICS_DIR / "active_learning_metrics.jsonl"
SUMMARY_FILE = METRICS_DIR / "metrics_summary.json"


@dataclass
class AnnotationMetrics:
    """Metrics for a single annotation event."""
    timestamp: str
    video_id: str
    annotator_id: str = "default"
    
    # Annotation details
    annotation_time_seconds: float = 0.0
    incident_timestamp: float = 0.0
    incident_end_timestamp: Optional[float] = None
    label_type: str = ""
    confirmed: bool = False
    
    # Model assistance
    candidate_rank_selected: int = 0  # 0 = manual, 1+ = model suggestion
    model_bag_probability: float = 0.0
    model_source: str = ""  # "cold_start" or "trained_model"
    
    # Temporal accuracy
    temporal_error_seconds: Optional[float] = None  # If ground truth available
    
    # Session info
    session_id: str = ""
    iteration_number: int = 0
    total_annotations_so_far: int = 0


@dataclass
class TrainingMetrics:
    """Metrics for a single training iteration."""
    timestamp: str
    iteration: int
    
    # Training data
    num_training_samples: int
    num_fall_samples: int
    num_no_fall_samples: int
    
    # Training performance
    num_epochs: int
    final_train_loss: float
    final_val_loss: float
    final_val_accuracy: float
    best_val_loss: float
    best_epoch: int
    training_time_seconds: float
    
    # Model architecture
    hidden_dim: int
    attention_dim: int
    dropout: float
    learning_rate: float
    
    # Early stopping
    early_stopped: bool
    patience: int


@dataclass
class ActiveLearningMetrics:
    """Metrics for active learning iteration."""
    timestamp: str
    iteration: int
    
    # Dataset state
    total_videos: int
    annotated_videos: int
    unannotated_videos: int
    
    # Class distribution
    fall_videos_annotated: int
    no_fall_videos_annotated: int
    
    # Model performance (on validation/test set if available)
    recall_at_1: Optional[float] = None
    recall_at_3: Optional[float] = None
    recall_at_5: Optional[float] = None
    precision_at_1: Optional[float] = None
    mean_temporal_error: Optional[float] = None
    
    # Active learning efficiency
    annotation_reduction_percent: Optional[float] = None  # vs random sampling
    samples_to_reach_target_performance: Optional[int] = None
    
    # Annotation time
    total_annotation_time_minutes: float = 0.0
    avg_annotation_time_per_video: float = 0.0
    
    # Model assistance effectiveness
    model_suggestions_accepted: int = 0
    model_suggestions_rejected: int = 0
    model_acceptance_rate: float = 0.0


class MetricsTracker:
    """Central metrics tracking system."""
    
    def __init__(self):
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.annotation_start_times = {}  # Track when annotation started
    
    def log_annotation_start(self, video_id: str):
        """Mark when annotation started for a video."""
        self.annotation_start_times[video_id] = datetime.utcnow()
    
    def log_annotation(
        self,
        video_id: str,
        label_type: str,
        incident_timestamp: float,
        incident_end_timestamp: Optional[float],
        confirmed: bool,
        candidate_rank: int,
        bag_prob: float,
        source: str,
        iteration: int,
        total_annotations: int,
        ground_truth_timestamp: Optional[float] = None,
    ):
        """Log a single annotation event."""
        
        # Calculate annotation time
        annotation_time = 0.0
        if video_id in self.annotation_start_times:
            time_delta = datetime.utcnow() - self.annotation_start_times[video_id]
            annotation_time = time_delta.total_seconds()
            del self.annotation_start_times[video_id]
        
        # Calculate temporal error if ground truth available
        temporal_error = None
        if ground_truth_timestamp is not None and confirmed:
            temporal_error = abs(incident_timestamp - ground_truth_timestamp)
        
        metrics = AnnotationMetrics(
            timestamp=datetime.utcnow().isoformat(),
            video_id=video_id,
            annotation_time_seconds=annotation_time,
            incident_timestamp=incident_timestamp,
            incident_end_timestamp=incident_end_timestamp,
            label_type=label_type,
            confirmed=confirmed,
            candidate_rank_selected=candidate_rank,
            model_bag_probability=bag_prob,
            model_source=source,
            temporal_error_seconds=temporal_error,
            session_id=self.session_id,
            iteration_number=iteration,
            total_annotations_so_far=total_annotations,
        )
        
        # Append to log
        with open(ANNOTATION_LOG, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")
        
        logger.info(f"Logged annotation for {video_id} (time: {annotation_time:.1f}s)")
    
    def log_training(
        self,
        iteration: int,
        num_train: int,
        num_fall: int,
        num_no_fall: int,
        training_history: Dict,
        training_time: float,
        config: Dict,
    ):
        """Log training iteration metrics."""
        
        # Extract best performance
        val_losses = training_history.get("val_loss", [])
        val_accs = training_history.get("val_acc", [])
        
        best_epoch = int(np.argmin(val_losses)) if val_losses else 0
        best_val_loss = float(np.min(val_losses)) if val_losses else 0.0
        
        metrics = TrainingMetrics(
            timestamp=datetime.utcnow().isoformat(),
            iteration=iteration,
            num_training_samples=num_train,
            num_fall_samples=num_fall,
            num_no_fall_samples=num_no_fall,
            num_epochs=len(val_losses),
            final_train_loss=float(training_history.get("train_loss", [0])[-1]),
            final_val_loss=float(val_losses[-1]) if val_losses else 0.0,
            final_val_accuracy=float(val_accs[-1]) if val_accs else 0.0,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            training_time_seconds=training_time,
            hidden_dim=config.get("hidden_dim", 256),
            attention_dim=config.get("attention_dim", 128),
            dropout=config.get("dropout", 0.3),
            learning_rate=config.get("learning_rate", 1e-4),
            early_stopped=len(val_losses) < config.get("max_epochs", 100),
            patience=config.get("patience", 10),
        )
        
        with open(TRAINING_LOG, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")
        
        logger.info(f"Logged training iteration {iteration}")
    
    def log_active_learning_iteration(
        self,
        iteration: int,
        total_videos: int,
        annotated_videos: int,
        fall_annotated: int,
        no_fall_annotated: int,
        annotations: List[Dict],
        evaluation_results: Optional[Dict] = None,
    ):
        """Log active learning iteration metrics."""
        
        # Calculate annotation time statistics
        total_time = sum(a.get("annotation_time_seconds", 0) for a in annotations)
        avg_time = total_time / len(annotations) if annotations else 0
        
        # Calculate model assistance effectiveness
        suggestions_accepted = sum(1 for a in annotations if a.get("candidate_rank", 0) > 0)
        suggestions_rejected = sum(1 for a in annotations if a.get("candidate_rank", 0) == 0)
        acceptance_rate = suggestions_accepted / len(annotations) if annotations else 0
        
        metrics = ActiveLearningMetrics(
            timestamp=datetime.utcnow().isoformat(),
            iteration=iteration,
            total_videos=total_videos,
            annotated_videos=annotated_videos,
            unannotated_videos=total_videos - annotated_videos,
            fall_videos_annotated=fall_annotated,
            no_fall_videos_annotated=no_fall_annotated,
            total_annotation_time_minutes=total_time / 60.0,
            avg_annotation_time_per_video=avg_time,
            model_suggestions_accepted=suggestions_accepted,
            model_suggestions_rejected=suggestions_rejected,
            model_acceptance_rate=acceptance_rate,
        )
        
        # Add evaluation results if available
        if evaluation_results:
            metrics.recall_at_1 = evaluation_results.get("recall@1")
            metrics.recall_at_3 = evaluation_results.get("recall@3")
            metrics.recall_at_5 = evaluation_results.get("recall@5")
            metrics.precision_at_1 = evaluation_results.get("precision@1")
            metrics.mean_temporal_error = evaluation_results.get("mean_temporal_error")
        
        with open(ACTIVE_LEARNING_LOG, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")
        
        logger.info(f"Logged active learning iteration {iteration}")
    
    def log_experiment_config(self, config: Dict):
        """Log experiment configuration and hyperparameters."""
        experiment_config = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "config": config,
        }
        
        with open(EXPERIMENT_LOG, "a") as f:
            f.write(json.dumps(experiment_config) + "\n")
    
    def generate_summary(self) -> Dict:
        """Generate summary statistics for publication."""
        
        summary = {
            "generated_at": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
        }
        
        # Load all metrics
        annotations = self._load_jsonl(ANNOTATION_LOG)
        training_runs = self._load_jsonl(TRAINING_LOG)
        al_iterations = self._load_jsonl(ACTIVE_LEARNING_LOG)
        
        # Annotation statistics
        if annotations:
            summary["annotation_stats"] = {
                "total_annotations": len(annotations),
                "total_time_hours": sum(a.get("annotation_time_seconds", 0) for a in annotations) / 3600,
                "avg_time_per_annotation_seconds": np.mean([a.get("annotation_time_seconds", 0) for a in annotations]),
                "median_time_per_annotation_seconds": np.median([a.get("annotation_time_seconds", 0) for a in annotations]),
                "model_acceptance_rate": np.mean([1 if a.get("candidate_rank_selected", 0) > 0 else 0 for a in annotations]),
                "confirmed_annotations": sum(1 for a in annotations if a.get("confirmed", False)),
            }
        
        # Training statistics
        if training_runs:
            summary["training_stats"] = {
                "total_iterations": len(training_runs),
                "total_training_time_minutes": sum(t.get("training_time_seconds", 0) for t in training_runs) / 60,
                "avg_epochs_per_iteration": np.mean([t.get("num_epochs", 0) for t in training_runs]),
                "best_val_accuracy": max(t.get("final_val_accuracy", 0) for t in training_runs),
                "final_val_accuracy": training_runs[-1].get("final_val_accuracy", 0) if training_runs else 0,
            }
        
        # Active learning statistics
        if al_iterations:
            summary["active_learning_stats"] = {
                "total_iterations": len(al_iterations),
                "final_annotated_videos": al_iterations[-1].get("annotated_videos", 0) if al_iterations else 0,
                "annotation_efficiency": {
                    "total_time_saved_hours": 0,  # Calculate based on baseline
                    "videos_annotated_per_hour": 0,
                },
            }
            
            # Performance progression
            if any(it.get("recall_at_3") for it in al_iterations):
                summary["performance_progression"] = {
                    "recall@3": [it.get("recall_at_3") for it in al_iterations if it.get("recall_at_3")],
                    "iterations": [it.get("iteration") for it in al_iterations if it.get("recall_at_3")],
                }
        
        # Save summary
        with open(SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file."""
        if not filepath.exists():
            return []
        
        data = []
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data


# Global tracker instance
_tracker = None

def get_tracker() -> MetricsTracker:
    """Get global metrics tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = MetricsTracker()
    return _tracker
