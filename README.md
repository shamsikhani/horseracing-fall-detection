# Horse Racing Incident Detection Pipeline

Human-in-the-Loop Weakly Supervised Temporal Action Localisation for detecting **Fell** (positive) and **Pulled-up** (negative) incidents in broadcast horse racing video.

## System Requirements

- **GPU**: NVIDIA RTX A2000 8GB (or similar)
- **RAM**: 32 GB
- **Python**: 3.11+

## Setup

```bash
cd c:\Users\shams\Documents\Horseracing
pip install -r requirements.txt
```

## Pipeline Stages

### Run Full Pipeline (Stages 1-4)

```bash
python -m pipeline.run_pipeline
```

### Run Individual Stages

```bash
python -m pipeline.run_pipeline --stage 1   # Preprocessing
python -m pipeline.run_pipeline --stage 2   # Segmentation
python -m pipeline.run_pipeline --stage 3   # Feature Extraction (VideoMAE)
python -m pipeline.run_pipeline --stage 4   # MIL Training + Proposals
```

### Launch Streamlit Review App (Stage 5)

```bash
streamlit run app.py
```

## Project Structure

```
Horseracing/
├── Fell/                    # Positive class videos (6 videos)
├── Pulled-up/               # Negative class videos (6 videos)
├── pipeline/
│   ├── config.py            # All hyperparameters & paths
│   ├── preprocessing.py     # Stage 1: Video normalisation
│   ├── segmentation.py      # Stage 2: Sliding window clips
│   ├── feature_extraction.py# Stage 3: VideoMAE embeddings
│   ├── model.py             # Stage 4: AttentionMIL architecture
│   ├── dataset.py           # MIL dataset + collation
│   ├── train.py             # Training loop + early stopping
│   ├── proposals.py         # NMS candidate proposals
│   ├── cold_start.py        # Heuristic proposals (pre-model)
│   ├── active_learning.py   # Retraining controller
│   ├── inference.py         # Model inference
│   └── run_pipeline.py      # End-to-end orchestrator
├── app.py                   # Stage 5: Streamlit review UI
├── requirements.txt
└── output/                  # Generated artifacts
    ├── preprocessed/        # Normalised videos
    ├── clip_metadata/       # Clip JSON metadata
    ├── features/            # VideoMAE .npy embeddings
    ├── models/              # Trained AttentionMIL checkpoints
    ├── proposals/           # Candidate proposals
    └── annotations.jsonl    # Human annotations
```

## Methodology

1. **Preprocessing**: Videos normalised to 25fps, 224x224, H.264
2. **Segmentation**: 3s clips with 1s stride (66.7% overlap)
3. **Feature Extraction**: VideoMAE-Base (ViT-B/16) → 768-dim embeddings
4. **MIL Training**: Attention-based MIL with bag-level BCE loss
5. **Review**: Streamlit app with timeline, candidates, annotation form
6. **Active Learning**: Auto-retrain after 10 new annotations
