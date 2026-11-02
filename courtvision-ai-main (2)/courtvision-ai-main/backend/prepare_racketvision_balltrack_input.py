"""prepare frames and a median background for racketvision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MODEL_WIDTH = 512
MODEL_HEIGHT = 288


def parse_arguments() -> argparse.Namespace:
    """read command-line options."""
    parser = argparse.ArgumentParser(
        description="Convert a CourtVision video to RacketVision BallTrack input."
    )
    parser.add_argument("--video", type=Path, default=Path("data/raw/test_rally.mp4"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/racketvision_input")
    )
    parser.add_argument("--match-id", default="courtvision_match")
    parser.add_argument("--rally-id", default="test_rally")
    parser.add_argument(
        "--median-samples",
        type=int,
        default=120,
        help="Evenly spaced frames used to estimate RacketVision's background image.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse a complete prepared input for a retry instead of extracting frames again.",
    )
    return parser.parse_args()


def collect_median_frame(frame: np.ndarray) -> np.ndarray:
    """resize a frame for the median background."""
    return cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_AREA)


def main() -> None:
    """run the script's main workflow."""
    args = parse_arguments()
    video_path = args.video.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Could not find input video: {video_path}")
    if args.median_samples < 1:
        raise ValueError("--median-samples must be at least 1.")

    frame_dir = output_root / "tennis" / "all" / args.match_id / "frame" / args.rally_id
    info_dir = output_root / "tennis" / "info"
    median_path = output_root / "tennis" / "all" / args.match_id / "median.npz"
    split_path = info_dir / "test.json"
    if frame_dir.exists() and any(frame_dir.iterdir()) and args.reuse_existing:
        if not median_path.is_file() or not split_path.is_file():
            raise FileNotFoundError(
                "Prepared frames exist but the matching median background or test split is missing. "
                "Choose a new --rally-id rather than mixing incomplete inputs."
            )
        print(f"Reusing prepared frames: {frame_dir}")
        print(f"Reusing median background: {median_path}")
        return
    if frame_dir.exists() and any(frame_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite prepared frames: {frame_dir}\n"
            "Choose a new --rally-id or remove that one generated input folder."
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count < 1 or fps <= 0:
        raise RuntimeError("The video does not report a usable frame count and FPS.")

    # evenly spaced samples reduce the effect of nearby duplicate frames.
    sample_indices = set(
        np.linspace(0, frame_count - 1, min(args.median_samples, frame_count), dtype=int).tolist()
    )
    median_frames: list[np.ndarray] = []
    frame_dir.mkdir(parents=True, exist_ok=False)
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        jpeg_path = frame_dir / f"{index:06d}.jpg"
        if not cv2.imwrite(str(jpeg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write frame: {jpeg_path}")
        if index in sample_indices:
            median_frames.append(collect_median_frame(frame))
        index += 1
    capture.release()
    if not median_frames:
        raise RuntimeError("No frames were decoded for the median background.")

    median_path.parent.mkdir(parents=True, exist_ok=True)
    # the pixel-wise median reduces the influence of moving players.
    median = np.median(np.stack(median_frames, axis=0), axis=0).astype(np.uint8)
    np.savez_compressed(median_path, median=median)
    info_dir.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps([[args.match_id, args.rally_id]], indent=2), encoding="utf-8"
    )
    manifest = {
        "video": str(video_path),
        "match_id": args.match_id,
        "rally_id": args.rally_id,
        "decoded_frames": index,
        "fps": fps,
        "median_samples": len(median_frames),
        "median_size": [MODEL_WIDTH, MODEL_HEIGHT],
        "output_root": str(output_root),
    }
    (output_root / "courtvision_racketvision_input_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Prepared {index} source-resolution frames: {frame_dir}")
    print(f"Saved median background: {median_path}")
    print(f"Saved RacketVision test split: {split_path}")


if __name__ == "__main__":
    main()
