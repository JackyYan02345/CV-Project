"""check court calibration on frame 450."""

import json
from pathlib import Path

import cv2
import numpy as np

# reuse the function that draws court lines from a homography.
from calibrate_validation_court import create_verification_image


VIDEO_PATH = Path("data/raw/validation_rally_01.mp4")

CALIBRATION_PATH = Path(
    "outputs/validation_rally_01_calibration_frame_450/"
    "image_to_court_homography.json"
)

OUTPUT_DIR = Path(
    "outputs/validation_rally_01_calibration_frame_450/nearby_checks"
)

# inspect views before, at, and after the calibration frame.
FRAMES_TO_CHECK = [375, 425, 450, 475, 525]


def main():
    """run the script's main workflow."""
    with CALIBRATION_PATH.open("r", encoding="utf-8") as file:
        calibration = json.load(file)

    image_to_meters = np.array(
        calibration["image_to_court_meters_matrix"],
        dtype=np.float64,
    )

    if image_to_meters.shape != (3, 3):
        raise ValueError("The calibration matrix must be 3 by 3.")

    # check all output paths before processing any frames.
    output_paths = [
        OUTPUT_DIR / f"frame_{frame_number}.jpg"
        for frame_number in FRAMES_TO_CHECK
    ]

    existing = [path for path in output_paths if path.exists()]

    if existing:
        raise FileExistsError(
            f"An inspection image already exists: {existing[0]}\n"
            "Choose a different output directory before running again."
        )

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for frame_number, output_path in zip(
            FRAMES_TO_CHECK,
            output_paths,
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()

            if not success:
                raise RuntimeError(
                    f"Could not read video frame {frame_number}"
                )

            # draw the same frame-450 calibration on this video frame.
            verification = create_verification_image(
                frame,
                image_to_meters,
            )

            if not cv2.imwrite(str(output_path), verification):
                raise RuntimeError(f"Could not save: {output_path}")

            print(f"Saved frame {frame_number}: {output_path}")

    finally:
        capture.release()


if __name__ == "__main__":
    main()