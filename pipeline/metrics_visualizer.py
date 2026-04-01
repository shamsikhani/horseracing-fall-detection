"""
Metrics Visualization for Publication-Quality Analysis.

Generates plots and tables for research paper:
- Annotation efficiency over time
- Model performance progression
- Active learning curves
- Class-specific performance
- Temporal error analysis
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .config import OUTPUT_DIR

logger = logging.getLogger(__name__)

METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = METRICS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class MetricsVisualizer:
    """Generate publication-quality visualizations."""
    
    def __init__(self):
        self.annotation_log = METRICS_DIR / "annotation_metrics.jsonl"
        self.training_log = METRICS_DIR / "training_metrics.jsonl"
        self.al_log = METRICS_DIR / "active_learning_metrics.jsonl"
    
    def load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file."""
        if not filepath.exists():
            return []
        
        data = []
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def plot_annotation_efficiency(self, save_path: Path = None):
        """Plot annotation time efficiency over iterations."""
        annotations = self.load_jsonl(self.annotation_log)
        if not annotations:
            logger.warning("No annotation data available")
            return
        
        # Group by iteration
        iterations = {}
        for ann in annotations:
            it = ann.get("iteration_number", 0)
            if it not in iterations:
                iterations[it] = []
            iterations[it].append(ann.get("annotation_time_seconds", 0))
        
        # Calculate statistics per iteration
        iter_nums = sorted(iterations.keys())
        avg_times = [np.mean(iterations[i]) for i in iter_nums]
        std_times = [np.std(iterations[i]) for i in iter_nums]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(iter_nums, avg_times, yerr=std_times, marker='o', 
                    capsize=5, capthick=2, linewidth=2, markersize=8)
        ax.set_xlabel('Active Learning Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Avg. Annotation Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Annotation Efficiency Over Active Learning Iterations', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = PLOTS_DIR / "annotation_efficiency.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved annotation efficiency plot to {save_path}")
    
    def plot_model_performance_progression(self, save_path: Path = None):
        """Plot model performance over active learning iterations."""
        al_data = self.load_jsonl(self.al_log)
        if not al_data:
            logger.warning("No active learning data available")
            return
        
        iterations = [d["iteration"] for d in al_data]
        recall_1 = [d.get("recall_at_1", 0) for d in al_data]
        recall_3 = [d.get("recall_at_3", 0) for d in al_data]
        recall_5 = [d.get("recall_at_5", 0) for d in al_data]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, recall_1, marker='o', linewidth=2, label='Recall@1', markersize=8)
        ax.plot(iterations, recall_3, marker='s', linewidth=2, label='Recall@3', markersize=8)
        ax.plot(iterations, recall_5, marker='^', linewidth=2, label='Recall@5', markersize=8)
        
        ax.set_xlabel('Active Learning Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Progression', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        if save_path is None:
            save_path = PLOTS_DIR / "model_performance_progression.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved performance progression plot to {save_path}")
    
    def plot_training_curves(self, save_path: Path = None):
        """Plot training loss and accuracy curves."""
        training_data = self.load_jsonl(self.training_log)
        if not training_data:
            logger.warning("No training data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot validation loss progression
        iterations = [d["iteration"] for d in training_data]
        val_losses = [d["best_val_loss"] for d in training_data]
        val_accs = [d["final_val_accuracy"] for d in training_data]
        
        ax1.plot(iterations, val_losses, marker='o', linewidth=2, markersize=8, color='#e74c3c')
        ax1.set_xlabel('Active Learning Iteration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Best Validation Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Validation Loss Over Iterations', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(iterations, val_accs, marker='s', linewidth=2, markersize=8, color='#27ae60')
        ax2.set_xlabel('Active Learning Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Final Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Validation Accuracy Over Iterations', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        if save_path is None:
            save_path = PLOTS_DIR / "training_curves.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training curves to {save_path}")
    
    def plot_annotation_time_distribution(self, save_path: Path = None):
        """Plot distribution of annotation times."""
        annotations = self.load_jsonl(self.annotation_log)
        if not annotations:
            logger.warning("No annotation data available")
            return
        
        times = [a.get("annotation_time_seconds", 0) for a in annotations if a.get("annotation_time_seconds", 0) > 0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(times, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
        ax.axvline(np.median(times), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(times):.1f}s')
        ax.axvline(np.mean(times), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(times):.1f}s')
        
        ax.set_xlabel('Annotation Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Annotation Times', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path is None:
            save_path = PLOTS_DIR / "annotation_time_distribution.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved annotation time distribution to {save_path}")
    
    def plot_model_assistance_effectiveness(self, save_path: Path = None):
        """Plot how often model suggestions are accepted."""
        annotations = self.load_jsonl(self.annotation_log)
        if not annotations:
            logger.warning("No annotation data available")
            return
        
        # Group by iteration
        iterations = {}
        for ann in annotations:
            it = ann.get("iteration_number", 0)
            if it not in iterations:
                iterations[it] = {"accepted": 0, "rejected": 0}
            
            if ann.get("candidate_rank_selected", 0) > 0:
                iterations[it]["accepted"] += 1
            else:
                iterations[it]["rejected"] += 1
        
        iter_nums = sorted(iterations.keys())
        acceptance_rates = [
            iterations[i]["accepted"] / (iterations[i]["accepted"] + iterations[i]["rejected"]) 
            if (iterations[i]["accepted"] + iterations[i]["rejected"]) > 0 else 0
            for i in iter_nums
        ]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(iter_nums, acceptance_rates, alpha=0.7, edgecolor='black', color='#9b59b6')
        ax.set_xlabel('Active Learning Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model Suggestion Acceptance Rate', fontsize=12, fontweight='bold')
        ax.set_title('Effectiveness of Model Assistance', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path is None:
            save_path = PLOTS_DIR / "model_assistance_effectiveness.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved model assistance plot to {save_path}")
    
    def plot_class_distribution(self, save_path: Path = None):
        """Plot class distribution over iterations."""
        al_data = self.load_jsonl(self.al_log)
        if not al_data:
            logger.warning("No active learning data available")
            return
        
        iterations = [d["iteration"] for d in al_data]
        fall_counts = [d["fall_videos_annotated"] for d in al_data]
        no_fall_counts = [d["no_fall_videos_annotated"] for d in al_data]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        x = np.arange(len(iterations))
        
        ax.bar(x - width/2, fall_counts, width, label='Fall', alpha=0.8, color='#e74c3c')
        ax.bar(x + width/2, no_fall_counts, width, label='No-Fall', alpha=0.8, color='#27ae60')
        
        ax.set_xlabel('Active Learning Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Annotated Videos', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution Over Iterations', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(iterations)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path is None:
            save_path = PLOTS_DIR / "class_distribution.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved class distribution plot to {save_path}")
    
    def generate_all_plots(self):
        """Generate all publication-quality plots."""
        logger.info("Generating all metrics visualizations...")
        
        self.plot_annotation_efficiency()
        self.plot_model_performance_progression()
        self.plot_training_curves()
        self.plot_annotation_time_distribution()
        self.plot_model_assistance_effectiveness()
        self.plot_class_distribution()
        
        logger.info(f"All plots saved to {PLOTS_DIR}")
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table of key metrics for paper."""
        al_data = self.load_jsonl(self.al_log)
        training_data = self.load_jsonl(self.training_log)
        annotations = self.load_jsonl(self.annotation_log)
        
        if not al_data:
            return "% No data available"
        
        latex = r"""
\begin{table}[h]
\centering
\caption{Active Learning Performance Summary}
\label{tab:al_performance}
\begin{tabular}{lccccc}
\hline
\textbf{Iteration} & \textbf{Annotations} & \textbf{Recall@3} & \textbf{Val Acc} & \textbf{Avg Time (s)} & \textbf{Model Accept} \\
\hline
"""
        
        for i, al_iter in enumerate(al_data):
            iteration = al_iter["iteration"]
            n_annotations = al_iter["annotated_videos"]
            recall_3 = al_iter.get("recall_at_3", 0)
            
            # Get corresponding training data
            val_acc = 0
            if i < len(training_data):
                val_acc = training_data[i]["final_val_accuracy"]
            
            # Get annotation stats for this iteration
            iter_annotations = [a for a in annotations if a.get("iteration_number") == iteration]
            avg_time = np.mean([a.get("annotation_time_seconds", 0) for a in iter_annotations]) if iter_annotations else 0
            acceptance = np.mean([1 if a.get("candidate_rank_selected", 0) > 0 else 0 for a in iter_annotations]) if iter_annotations else 0
            
            latex += f"{iteration} & {n_annotations} & {recall_3:.3f} & {val_acc:.3f} & {avg_time:.1f} & {acceptance:.2f} \\\\\n"
        
        latex += r"""\hline
\end{tabular}
\end{table}
"""
        
        # Save to file
        table_path = METRICS_DIR / "performance_table.tex"
        with open(table_path, "w") as f:
            f.write(latex)
        
        logger.info(f"Saved LaTeX table to {table_path}")
        return latex


def generate_publication_materials():
    """Generate all materials needed for publication."""
    visualizer = MetricsVisualizer()
    visualizer.generate_all_plots()
    visualizer.generate_latex_table()
    
    # Generate summary statistics
    from .metrics_tracker import get_tracker
    tracker = get_tracker()
    summary = tracker.generate_summary()
    
    logger.info("Publication materials generated successfully")
    return summary
