"""compare court calibration across selected frames."""

import json
from pathlib import Path

import cv2
import numpy as np

# reuse the court-line drawing function you just created.
from calibrate_validation_court import create_verification_image


VIDEO_PATH = Path("data/raw/validation_rally_01.mp4")

CALIBRATION_PATH = Path(
    "outputs/validation_rally_01_calibration/"
    "image_to_court_homography.json"
)

OUTPUT_DIR = Path(
    "outputs/validation_rally_01_calibration/time_checks"
)

# these sample the beginning, middle, and end of the 902-frame clip.
FRAMES_TO_CHECK = [0, 225, 450, 675, 901]


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

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for frame_number in FRAMES_TO_CHECK:
            output_path = OUTPUT_DIR / f"frame_{frame_number}.jpg"

            # keep any earlier inspection image rather than overwriting it.
            if output_path.exists():
                raise FileExistsError(
                    f"Inspection image already exists: {output_path}"
                )

            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = capture.read()

            if not success:
                raise RuntimeError(
                    f"Could not read video frame {frame_number}"
                )

            verification = create_verification_image(
                frame,
                image_to_meters,
            )

            if not cv2.imwrite(str(output_path), verification):
                raise RuntimeError(
                    f"Could not save verification image: {output_path}"
                )

            print(f"Saved frame {frame_number}: {output_path}")

    finally:
        capture.release()


if __name__ == "__main__":
    main()