"""
Stage 3: Spatiotemporal Feature Extraction using VideoMAE.

For each clip, uniformly samples 16 frames, encodes them through the frozen
VideoMAE-Base (ViT-B/16) backbone, and mean-pools over tokens to produce a
768-dimensional clip embedding.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import av
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEModel

from .config import CONFIG, FEATURES_DIR, PREPROCESSED_DIR, CLIP_META_DIR, DATA_ROOT

logger = logging.getLogger(__name__)


def load_videomae(device: str = None):
    """Load the pretrained VideoMAE model and image processor."""
    cfg = CONFIG.features
    device = device or cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")

    logger.info(f"Loading VideoMAE model: {cfg.model_name} on {device}")
    processor = VideoMAEImageProcessor.from_pretrained(cfg.model_name)
    model = VideoMAEModel.from_pretrained(cfg.model_name)
    model = model.to(device)
    model.eval()
    return processor, model, device


def decode_all_frames(
    video_path: Path,
    clips: List[Dict],
    num_frames_per_clip: int = 16,
    target_size: tuple = (224, 224),
) -> List[List[np.ndarray]]:
    """Decode a video ONCE and extract frames for all clips in a single pass.
    Returns a list of frame-lists, one per clip.
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 25.0

    # Pre-compute which frame indices each clip needs
    clip_frame_specs = []  # list of (clip_idx, sorted np array of frame indices)
    all_needed = set()
    for ci, clip in enumerate(clips):
        sf = int(clip["start_time"] * fps)
        ef = int(clip["end_time"] * fps)
        total = max(ef - sf, 1)
        if total >= num_frames_per_clip:
            indices = np.linspace(sf, ef - 1, num_frames_per_clip, dtype=int)
        else:
            indices = list(range(sf, ef))
            while len(indices) < num_frames_per_clip:
                indices.append(indices[-1])
            indices = np.array(indices[:num_frames_per_clip], dtype=int)
        clip_frame_specs.append(indices)
        all_needed.update(indices.tolist())

    if not all_needed:
        container.close()
        blank = np.zeros((*target_size, 3), dtype=np.uint8)
        return [[blank] * num_frames_per_clip for _ in clips]

    max_needed = max(all_needed)

    # Single-pass decode — only convert frames we actually need
    decoded = {}
    frame_idx = 0
    for frame in container.decode(video=0):
        if frame_idx in all_needed:
            decoded[frame_idx] = np.array(frame.to_image().resize(target_size))
        if frame_idx >= max_needed:
            break
        frame_idx += 1
    container.close()

    # Assemble per-clip frame lists
    blank = np.zeros((*target_size, 3), dtype=np.uint8)
    all_clip_frames = []
    for indices in clip_frame_specs:
        frames = []
        last = None
        for idx in indices:
            if idx in decoded:
                last = decoded[idx]
                frames.append(last)
            elif last is not None:
                frames.append(last)
            else:
                frames.append(blank)
        all_clip_frames.append(frames)

    return all_clip_frames


def extract_features_for_video(
    video_id: str,
    clips: List[Dict],
    processor,
    model,
    device: str,
) -> np.ndarray:
    """Extract features for all clips of a single video.
    Decodes the video once, then batches clips through VideoMAE.
    Returns array of shape (num_clips, 768).
    """
    cfg = CONFIG.features
    from .preprocessing import resolve_video_path
    video_path = resolve_video_path(video_id)
    if video_path is None:
        logger.error(f"Video file not found for {video_id}")
        return np.zeros((len(clips), cfg.embedding_dim), dtype=np.float32)

    # Decode ALL frames for ALL clips in one pass
    logger.info(f"  {video_id}: decoding video frames (single pass)...")
    all_clip_frames = decode_all_frames(
        video_path, clips,
        num_frames_per_clip=cfg.frames_per_clip,
        target_size=CONFIG.preprocessing.target_resolution,
    )

    # Batch through VideoMAE
    all_embeddings = []
    num_clips = len(clips)

    for batch_start in range(0, num_clips, cfg.batch_size):
        batch_frames = all_clip_frames[batch_start:batch_start + cfg.batch_size]

        inputs = processor(batch_frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            embeddings = hidden.mean(dim=1)  # (batch, 768)

        all_embeddings.append(embeddings.cpu().numpy())

        processed = min(batch_start + cfg.batch_size, num_clips)
        if processed % 100 == 0 or processed == num_clips:
            logger.info(f"  {video_id}: {processed}/{num_clips} clips encoded")

    features = np.concatenate(all_embeddings, axis=0)  # (num_clips, 768)
    return features


def run_feature_extraction(all_clips: Dict[str, List[Dict]], force: bool = False) -> Dict[str, Path]:
    """Run feature extraction for all videos.
    Returns dict mapping video_id -> path to saved .npy feature file.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check which videos need extraction
    videos_to_process = []
    feature_paths = {}

    for video_id, clips in all_clips.items():
        feat_path = FEATURES_DIR / f"{video_id}.npy"
        feature_paths[video_id] = feat_path
        if feat_path.exists() and not force:
            logger.info(f"Skipping {video_id} (features already extracted)")
        else:
            videos_to_process.append((video_id, clips))

    if not videos_to_process:
        logger.info("All features already extracted")
        return feature_paths

    # Load model once
    processor, model, device = load_videomae()

    for video_id, clips in videos_to_process:
        logger.info(f"Extracting features for {video_id} ({len(clips)} clips)...")
        features = extract_features_for_video(video_id, clips, processor, model, device)
        feat_path = FEATURES_DIR / f"{video_id}.npy"
        np.save(feat_path, features)
        feature_paths[video_id] = feat_path
        logger.info(f"Saved features for {video_id}: shape={features.shape}")

    logger.info(f"Feature extraction complete: {len(feature_paths)} videos")
    return feature_paths
