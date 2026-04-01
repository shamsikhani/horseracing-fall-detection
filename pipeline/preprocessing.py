"""
Stage 1: Video Ingestion and Preprocessing.

Normalises raw broadcast videos to a consistent format:
- Frame rate: 25 fps
- Resolution: 224x224
- Codec: H.264 / CRF 23
"""

import json
import logging
from pathlib import Path

import av

from .config import (
    CONFIG,
    DATA_ROOT,
    PREPROCESSED_DIR,
    OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


def get_video_metadata(video_path: Path) -> dict:
    """Extract metadata from a video file using PyAV."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    duration = float(container.duration) / av.time_base if container.duration else 0.0
    meta = {
        "path": str(video_path),
        "filename": video_path.name,
        "duration": duration,
        "fps": float(stream.average_rate) if stream.average_rate else 25.0,
        "width": stream.codec_context.width,
        "height": stream.codec_context.height,
        "num_frames": stream.frames if stream.frames > 0 else int(duration * float(stream.average_rate or 25)),
    }
    container.close()
    return meta


def preprocess_video(input_path: Path, output_path: Path) -> dict:
    """
    Preprocess a single video: re-encode to target fps, resolution, and codec.
    Returns metadata dict for the preprocessed video.
    """
    cfg = CONFIG.preprocessing
    target_w, target_h = cfg.target_resolution

    input_container = av.open(str(input_path))
    input_stream = input_container.streams.video[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_container = av.open(str(output_path), mode="w")
    output_stream = output_container.add_stream(cfg.codec, rate=cfg.target_fps)
    output_stream.width = target_w
    output_stream.height = target_h
    output_stream.pix_fmt = "yuv420p"
    output_stream.options = {"crf": str(cfg.crf)}

    frame_count = 0
    for frame in input_container.decode(video=0):
        img = frame.to_image()
        img = img.resize((target_w, target_h))
        new_frame = av.VideoFrame.from_image(img)
        new_frame.pts = frame_count
        for packet in output_stream.encode(new_frame):
            output_container.mux(packet)
        frame_count += 1

    # Flush
    for packet in output_stream.encode():
        output_container.mux(packet)

    output_container.close()
    input_container.close()

    meta = get_video_metadata(output_path)
    meta["original_path"] = str(input_path)
    logger.info(f"Preprocessed {input_path.name} -> {output_path.name} ({frame_count} frames)")
    return meta


def discover_videos() -> list:
    """Discover all videos in the fall and non-fall directories.
    Returns list of (video_path, label_name, binary_label) tuples.
    """
    videos = []
    fall_dir = DATA_ROOT / "fall"
    non_fall_dir = DATA_ROOT / "non-fall"
    
    # Also check old folder names for backward compatibility
    old_fell_dir = DATA_ROOT / "Fell"
    old_pullup_dir = DATA_ROOT / "Pulled-up"

    # Check new folder structure first
    if fall_dir.exists():
        for vpath in sorted(fall_dir.glob("*.mp4")):
            videos.append((vpath, "fall", 1))
    elif old_fell_dir.exists():
        for vpath in sorted(old_fell_dir.glob("*.mp4")):
            videos.append((vpath, "fall", 1))

    if non_fall_dir.exists():
        for vpath in sorted(non_fall_dir.glob("*.mp4")):
            videos.append((vpath, "no_fall", 0))
    elif old_pullup_dir.exists():
        for vpath in sorted(old_pullup_dir.glob("*.mp4")):
            videos.append((vpath, "no_fall", 0))

    return videos


def resolve_video_path(video_id: str) -> Path:
    """Resolve the actual video file path from a video_id (stem).
    Checks preprocessed dir first, then original fall/non-fall dirs.
    """
    # Check preprocessed
    preprocessed = PREPROCESSED_DIR / f"{video_id}.mp4"
    if preprocessed.exists():
        return preprocessed
    # Check new folder structure
    for subdir in ["fall", "non-fall"]:
        original = DATA_ROOT / subdir / f"{video_id}.mp4"
        if original.exists():
            return original
    # Check old folder structure for backward compatibility
    for subdir in ["Fell", "Pulled-up"]:
        original = DATA_ROOT / subdir / f"{video_id}.mp4"
        if original.exists():
            return original
    return None


def run_preprocessing(force: bool = False, metadata_only: bool = True) -> list:
    """Run preprocessing on all discovered videos.
    If metadata_only=True, just extract metadata from originals (fast).
    If metadata_only=False, re-encode videos to target format.
    Returns list of metadata dicts.
    """
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    meta_dir = OUTPUT_DIR / "video_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos()
    all_meta = []

    for video_path, label_name, binary_label in videos:
        if metadata_only:
            logger.info(f"Extracting metadata for {video_path.name} (no re-encode)")
            meta = get_video_metadata(video_path)
        else:
            output_path = PREPROCESSED_DIR / video_path.name
            if output_path.exists() and not force:
                logger.info(f"Skipping {video_path.name} (already preprocessed)")
                meta = get_video_metadata(output_path)
            else:
                logger.info(f"Preprocessing {video_path.name} ...")
                meta = preprocess_video(video_path, output_path)

        meta["label_name"] = label_name
        meta["binary_label"] = binary_label
        meta["video_id"] = video_path.stem
        meta["original_path"] = str(video_path)
        all_meta.append(meta)

        # Save per-video metadata
        meta_file = meta_dir / f"{video_path.stem}.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    # Save combined metadata
    combined_file = meta_dir / "all_videos.json"
    with open(combined_file, "w") as f:
        json.dump(all_meta, f, indent=2)

    logger.info(f"Preprocessing complete: {len(all_meta)} videos processed")
    return all_meta
