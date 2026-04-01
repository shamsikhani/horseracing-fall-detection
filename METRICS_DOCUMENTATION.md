# Comprehensive Metrics Tracking System

## Overview

This system tracks **all aspects** of the active learning pipeline to support publication in top-tier journals (Nature AI, CVPR, etc.). Every annotation, training iteration, and model decision is logged with detailed metrics.

## 📊 Tracked Metrics

### 1. **Annotation Metrics** (`annotation_metrics.jsonl`)

Tracks every single annotation event with:

- **Timing Data**
  - `annotation_time_seconds`: How long the annotator spent on this video
  - `timestamp`: When the annotation was made
  - `session_id`: Unique session identifier

- **Annotation Details**
  - `incident_timestamp`: When the incident occurred (in seconds)
  - `incident_end_timestamp`: End time of incident (optional)
  - `label_type`: "fall" or "no_fall"
  - `confirmed`: Whether incident was confirmed or marked as no incident

- **Model Assistance**
  - `candidate_rank_selected`: Which model suggestion was used (0 = manual, 1+ = model)
  - `model_bag_probability`: Model's confidence score
  - `model_source`: "cold_start" or "trained_model"

- **Quality Metrics**
  - `temporal_error_seconds`: Error vs ground truth (if available)
  - `iteration_number`: Which active learning iteration
  - `total_annotations_so_far`: Cumulative count

**Key Questions Answered:**
- How much time does annotation take?
- Does annotation time decrease with model assistance?
- How often do annotators accept model suggestions?
- What is the temporal accuracy of annotations?

---

### 2. **Training Metrics** (`training_metrics.jsonl`)

Tracks each model training iteration:

- **Dataset Composition**
  - `num_training_samples`: Total videos used for training
  - `num_fall_samples`: Number of fall videos
  - `num_no_fall_samples`: Number of no-fall videos

- **Training Performance**
  - `num_epochs`: How many epochs trained
  - `final_train_loss`: Final training loss
  - `final_val_loss`: Final validation loss
  - `final_val_accuracy`: Final validation accuracy
  - `best_val_loss`: Best validation loss achieved
  - `best_epoch`: Which epoch achieved best performance
  - `training_time_seconds`: Total training time

- **Model Configuration**
  - `hidden_dim`: Hidden layer dimension (256)
  - `attention_dim`: Attention mechanism dimension (128)
  - `dropout`: Dropout rate (0.3)
  - `learning_rate`: Learning rate (1e-4)

- **Early Stopping**
  - `early_stopped`: Whether early stopping was triggered
  - `patience`: Early stopping patience value

**Key Questions Answered:**
- How does model performance improve with more annotations?
- What is the training efficiency (time per iteration)?
- Does the model converge faster in later iterations?
- What is the class balance in training data?

---

### 3. **Active Learning Metrics** (`active_learning_metrics.jsonl`)

Tracks the overall active learning loop:

- **Dataset State**
  - `total_videos`: Total videos in dataset (100)
  - `annotated_videos`: How many have been annotated
  - `unannotated_videos`: Remaining videos
  - `fall_videos_annotated`: Fall class count
  - `no_fall_videos_annotated`: No-fall class count

- **Model Performance**
  - `recall_at_1`: Recall@1 (top-1 proposal correct)
  - `recall_at_3`: Recall@3 (incident in top-3 proposals)
  - `recall_at_5`: Recall@5 (incident in top-5 proposals)
  - `precision_at_1`: Precision of top proposal
  - `mean_temporal_error`: Average temporal localization error

- **Efficiency Metrics**
  - `total_annotation_time_minutes`: Cumulative annotation time
  - `avg_annotation_time_per_video`: Average time per video
  - `model_suggestions_accepted`: How many suggestions accepted
  - `model_suggestions_rejected`: How many rejected
  - `model_acceptance_rate`: Acceptance rate (0-1)

- **Sample Efficiency**
  - `annotation_reduction_percent`: Reduction vs random sampling
  - `samples_to_reach_target_performance`: How many annotations needed

**Key Questions Answered:**
- How sample-efficient is active learning vs random sampling?
- How much annotation time is saved?
- Does model performance improve with each iteration?
- What is the learning curve (performance vs annotations)?

---

## 📈 Generated Visualizations

All plots are saved as **publication-quality** PNG files (300 DPI) in `output/metrics/plots/`:

### 1. **Annotation Efficiency** (`annotation_efficiency.png`)
- X-axis: Active learning iteration
- Y-axis: Average annotation time (seconds)
- Shows: How annotation time changes as model improves
- Error bars: Standard deviation across annotations

### 2. **Model Performance Progression** (`model_performance_progression.png`)
- X-axis: Active learning iteration
- Y-axis: Recall (0-1)
- Lines: Recall@1, Recall@3, Recall@5
- Shows: How model accuracy improves with more annotations

### 3. **Training Curves** (`training_curves.png`)
- Two subplots:
  - Validation loss over iterations
  - Validation accuracy over iterations
- Shows: Model convergence and stability

### 4. **Annotation Time Distribution** (`annotation_time_distribution.png`)
- Histogram of annotation times
- Median and mean lines
- Shows: Typical annotation effort

### 5. **Model Assistance Effectiveness** (`model_assistance_effectiveness.png`)
- Bar chart of model suggestion acceptance rate per iteration
- Shows: How useful model suggestions become over time

### 6. **Class Distribution** (`class_distribution.png`)
- Stacked bar chart of fall vs no-fall annotations per iteration
- Shows: Dataset balance evolution

---

## 📄 LaTeX Table for Paper

Automatically generates a publication-ready LaTeX table (`performance_table.tex`):

```latex
\begin{table}[h]
\centering
\caption{Active Learning Performance Summary}
\label{tab:al_performance}
\begin{tabular}{lccccc}
\hline
\textbf{Iteration} & \textbf{Annotations} & \textbf{Recall@3} & \textbf{Val Acc} & \textbf{Avg Time (s)} & \textbf{Model Accept} \\
\hline
0 & 10 & 0.450 & 0.650 & 45.2 & 0.00 \\
1 & 20 & 0.650 & 0.750 & 38.5 & 0.35 \\
2 & 30 & 0.800 & 0.850 & 32.1 & 0.58 \\
...
\hline
\end{tabular}
\end{table}
```

---

## 🚀 Usage

### During Annotation (Automatic)

Metrics are **automatically logged** when you:
1. Select a video to annotate
2. Submit an annotation
3. Trigger model retraining

**No manual action required!**

### Generate Publication Materials

Run this command to generate all plots and tables:

```bash
python scripts/generate_metrics_dashboard.py
```

This will create:
- All 6 publication-quality plots
- LaTeX performance table
- JSON summary of all metrics

### Access Metrics Programmatically

```python
from pipeline.metrics_tracker import get_tracker

tracker = get_tracker()

# Generate summary
summary = tracker.generate_summary()

# Access specific metrics
print(f"Total annotations: {summary['annotation_stats']['total_annotations']}")
print(f"Model acceptance rate: {summary['annotation_stats']['model_acceptance_rate']:.1%}")
```

---

## 📊 Key Metrics for Publication

### Sample Efficiency
- **Annotation Reduction**: How many fewer annotations needed vs random sampling
- **Learning Curve**: Performance vs number of annotations
- **Time to Target Performance**: Annotations needed to reach X% recall

### Annotation Efficiency
- **Time Savings**: Total time saved with model assistance
- **Acceptance Rate**: How often annotators use model suggestions
- **Time per Video**: Average annotation time (decreases with iterations)

### Model Performance
- **Recall@K**: Percentage of incidents found in top-K proposals
- **Temporal Error**: Average error in incident localization (seconds)
- **Class-Specific Performance**: Fall vs no-fall accuracy

### Active Learning Effectiveness
- **Performance Gain per Iteration**: How much model improves
- **Convergence Rate**: How quickly model reaches plateau
- **Sample Selection Quality**: Are selected samples informative?

---

## 📝 Publication Checklist

For your Nature AI / top-tier journal paper, you'll have:

✅ **Quantitative Results**
- Learning curves (performance vs annotations)
- Annotation time savings (hours saved)
- Model acceptance rates (human-AI collaboration)
- Temporal localization accuracy

✅ **Visualizations**
- 6 publication-quality plots (300 DPI)
- LaTeX-formatted tables
- Training curves and convergence analysis

✅ **Reproducibility**
- All hyperparameters logged
- Dataset composition tracked
- Random seeds and configurations saved

✅ **Statistical Rigor**
- Mean and standard deviation for all metrics
- Per-iteration breakdowns
- Class-specific performance analysis

---

## 📂 File Structure

```
output/metrics/
├── annotation_metrics.jsonl          # Every annotation event
├── training_metrics.jsonl            # Every training iteration
├── active_learning_metrics.jsonl    # Every AL iteration
├── experiment_log.jsonl             # Experiment configurations
├── metrics_summary.json             # Overall summary
├── performance_table.tex            # LaTeX table
└── plots/                           # Publication-quality plots
    ├── annotation_efficiency.png
    ├── model_performance_progression.png
    ├── training_curves.png
    ├── annotation_time_distribution.png
    ├── model_assistance_effectiveness.png
    └── class_distribution.png
```

---

## 🎯 Example Research Questions Answered

1. **"How sample-efficient is our approach?"**
   - Compare annotations needed vs random sampling
   - Show learning curves (Recall@3 vs annotations)

2. **"Does active learning reduce annotation time?"**
   - Plot annotation time per iteration
   - Calculate total time saved

3. **"How effective is human-AI collaboration?"**
   - Model suggestion acceptance rate
   - Time saved when using model suggestions

4. **"What is the temporal localization accuracy?"**
   - Mean temporal error in seconds
   - Distribution of errors

5. **"How does performance scale with data?"**
   - Performance vs dataset size
   - Convergence analysis

---

## 💡 Tips for Publication

1. **Run multiple experiments** with different random seeds for statistical significance
2. **Compare against baselines**: Random sampling, uncertainty sampling, etc.
3. **Report confidence intervals**: Use standard deviation across runs
4. **Class-specific analysis**: Report fall vs no-fall performance separately
5. **Ablation studies**: Test different components (cold-start, attention mechanism, etc.)

---

## 🔬 Advanced Analysis

For deeper analysis, load the JSONL files directly:

```python
import json
import pandas as pd

# Load annotation metrics
annotations = []
with open('output/metrics/annotation_metrics.jsonl') as f:
    for line in f:
        annotations.append(json.loads(line))

df = pd.DataFrame(annotations)

# Analyze annotation time by iteration
time_by_iteration = df.groupby('iteration_number')['annotation_time_seconds'].agg(['mean', 'std'])
print(time_by_iteration)

# Model acceptance rate over time
acceptance_by_iteration = df.groupby('iteration_number').apply(
    lambda x: (x['candidate_rank_selected'] > 0).mean()
)
print(acceptance_by_iteration)
```

---

## 📧 Questions?

All metrics are automatically tracked. Just annotate videos normally and the system handles everything!

To generate publication materials at any time:
```bash
python scripts/generate_metrics_dashboard.py
```
