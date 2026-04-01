"""
End-to-end pipeline orchestrator.

Runs all stages in sequence:
  Stage 1: Preprocessing
  Stage 2: Segmentation
  Stage 3: Feature extraction (VideoMAE)
  Stage 4: MIL training + proposal generation
  Stage 5: Launch Streamlit review app
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import CONFIG, OUTPUT_DIR
from pipeline.preprocessing import run_preprocessing
from pipeline.segmentation import run_segmentation
from pipeline.feature_extraction import run_feature_extraction
from pipeline.train import train_model
from pipeline.proposals import run_proposal_generation
from pipeline.cold_start import run_cold_start
from pipeline.active_learning import ActiveLearningController

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger("pipeline")


def run_stage1(force: bool = False):
    """Stage 1: Video Ingestion and Preprocessing."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Video Ingestion and Preprocessing")
    logger.info("=" * 60)
    video_metadata = run_preprocessing(force=force, metadata_only=True)
    logger.info(f"Stage 1 complete: {len(video_metadata)} videos")
    return video_metadata


def run_stage2(video_metadata: list):
    """Stage 2: Temporal Segmentation."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Temporal Segmentation")
    logger.info("=" * 60)
    all_clips = run_segmentation(video_metadata)
    total_clips = sum(len(c) for c in all_clips.values())
    logger.info(f"Stage 2 complete: {total_clips} clips from {len(all_clips)} videos")
    return all_clips


def run_stage3(all_clips: dict, force: bool = False):
    """Stage 3: Feature Extraction (VideoMAE)."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Feature Extraction (VideoMAE)")
    logger.info("=" * 60)
    feature_paths = run_feature_extraction(all_clips, force=force)
    logger.info(f"Stage 3 complete: {len(feature_paths)} feature files")
    return feature_paths


def run_stage4(all_clips: dict):
    """Stage 4: MIL Training + Proposal Generation."""
    logger.info("=" * 60)
    logger.info("STAGE 4: MIL Training + Proposal Generation")
    logger.info("=" * 60)

    al_controller = ActiveLearningController()
    mode = al_controller.get_mode()

    if mode == "cold_start":
        logger.info("Cold-start mode: generating heuristic proposals")
        proposals = run_cold_start(all_clips)
    else:
        logger.info("Model mode: training and generating model-based proposals")

    # Always attempt training if features exist
    model, history = train_model(all_clips)
    if model is not None:
        proposals = run_proposal_generation(all_clips, model=model)
        logger.info(f"Stage 4 complete: model trained, {len(proposals)} proposal sets")
    else:
        if mode == "cold_start":
            logger.info("Stage 4 complete: cold-start proposals only")
        else:
            logger.warning("Stage 4: training failed, using existing proposals")

    return proposals


def main():
    parser = argparse.ArgumentParser(description="Horse Racing Incident Detection Pipeline")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run a specific stage only")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing of all stages")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip video preprocessing (use originals)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Horse Racing Incident Detection Pipeline")
    logger.info(f"Project root: {Path(__file__).resolve().parent.parent}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    if args.stage:
        logger.info(f"Running stage {args.stage} only")

    # Run pipeline stages
    video_metadata = None
    all_clips = None
    proposals = None

    if args.stage is None or args.stage == 1:
        video_metadata = run_stage1(force=args.force)

    if args.stage is None or args.stage == 2:
        if video_metadata is None:
            import json
            meta_file = OUTPUT_DIR / "video_metadata" / "all_videos.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    video_metadata = json.load(f)
            else:
                logger.error("No video metadata found. Run stage 1 first.")
                return
        all_clips = run_stage2(video_metadata)

    if args.stage is None or args.stage == 3:
        if all_clips is None:
            import json
            clips_file = OUTPUT_DIR / "clip_metadata" / "all_clips.json"
            if clips_file.exists():
                with open(clips_file) as f:
                    all_clips = json.load(f)
            else:
                logger.error("No clip metadata found. Run stage 2 first.")
                return
        run_stage3(all_clips, force=args.force)

    if args.stage is None or args.stage == 4:
        if all_clips is None:
            import json
            clips_file = OUTPUT_DIR / "clip_metadata" / "all_clips.json"
            if clips_file.exists():
                with open(clips_file) as f:
                    all_clips = json.load(f)
            else:
                logger.error("No clip metadata found. Run stages 1-2 first.")
                return
        proposals = run_stage4(all_clips)

    if args.stage == 5:
        logger.info("Launch the Streamlit app with: streamlit run app.py")

    logger.info("Pipeline execution complete!")


if __name__ == "__main__":
    main()
