"""
Generate comprehensive metrics dashboard for publication analysis.
Run this script to create all visualizations and summary statistics.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.metrics_visualizer import generate_publication_materials
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("=" * 80)
    print("Generating Publication-Quality Metrics Dashboard")
    print("=" * 80)
    
    summary = generate_publication_materials()
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if "annotation_stats" in summary:
        print("\n📝 Annotation Statistics:")
        stats = summary["annotation_stats"]
        print(f"  Total annotations: {stats.get('total_annotations', 0)}")
        print(f"  Total time: {stats.get('total_time_hours', 0):.2f} hours")
        print(f"  Avg time per annotation: {stats.get('avg_time_per_annotation_seconds', 0):.1f}s")
        print(f"  Model acceptance rate: {stats.get('model_acceptance_rate', 0):.1%}")
    
    if "training_stats" in summary:
        print("\n🤖 Training Statistics:")
        stats = summary["training_stats"]
        print(f"  Total iterations: {stats.get('total_iterations', 0)}")
        print(f"  Total training time: {stats.get('total_training_time_minutes', 0):.1f} minutes")
        print(f"  Best validation accuracy: {stats.get('best_val_accuracy', 0):.1%}")
    
    if "active_learning_stats" in summary:
        print("\n🎯 Active Learning Statistics:")
        stats = summary["active_learning_stats"]
        print(f"  Total iterations: {stats.get('total_iterations', 0)}")
        print(f"  Final annotated videos: {stats.get('final_annotated_videos', 0)}")
    
    print("\n" + "=" * 80)
    print("✅ All metrics and visualizations generated successfully!")
    print(f"📊 Plots saved to: {PROJECT_ROOT / 'output' / 'metrics' / 'plots'}")
    print(f"📄 LaTeX table saved to: {PROJECT_ROOT / 'output' / 'metrics' / 'performance_table.tex'}")
    print(f"📈 Summary saved to: {PROJECT_ROOT / 'output' / 'metrics' / 'metrics_summary.json'}")
    print("=" * 80)
