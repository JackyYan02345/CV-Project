"""generate court mappings from tracked court edges."""

import json
from pathlib import Path

import cv2
import numpy as np

# reuse the model-loading and prediction functions from the visual test.
from learn_test_automatic_court_detection import (
    VIDEO_PATH,
    create_image_preprocessor,
    load_court_model,
    predict_landmarks,
)


OUTPUT_PATH = Path(
    "outputs/validation_rally_01_automatic_court_mappings.json"
)

COURT_WIDTH_METERS = 8.23
COURT_LENGTH_METERS = 23.77

# model landmark indexes for the singles-court corners, ordered:
# far-left, far-right, near-right, near-left.
SINGLES_CORNER_INDEXES = [4, 6, 7, 5]


def calculate_mapping(landmarks, frame_width, frame_height):
    """map image points to court coordinates."""
    corners = landmarks[SINGLES_CORNER_INDEXES].astype(np.float32)

    if not np.isfinite(corners).all():
        return None, "invalid_nonfinite_corners"

    # require the predicted corners to lie within the video frame.
    for x, y in corners:
        if not (0 <= x < frame_width and 0 <= y < frame_height):
            return None, "invalid_corner_outside_frame"

    # a court outline should form a non-degenerate convex quadrilateral.
    contour = corners.reshape(-1, 1, 2)

    if not cv2.isContourConvex(contour):
        return None, "invalid_court_shape"

    if cv2.contourArea(contour) < 100:
        return None, "invalid_court_area"

    court_corners_meters = np.float32([
        [0, 0],
        [COURT_WIDTH_METERS, 0],
        [COURT_WIDTH_METERS, COURT_LENGTH_METERS],
        [0, COURT_LENGTH_METERS],
    ])

    matrix = cv2.getPerspectiveTransform(
        corners,
        court_corners_meters,
    )

    if not np.isfinite(matrix).all():
        return None, "invalid_mapping"

    if abs(np.linalg.det(matrix)) < 1e-12:
        return None, "invalid_singular_mapping"

    return matrix, "experimental_mapping_created"


def main():
    """generate mappings for each video frame and save the results."""
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Choose a new output path before running again."
        )

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Could not find video: {VIDEO_PATH}")

    model = load_court_model()
    preprocess = create_image_preprocessor()

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    frame_results = []

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            raise RuntimeError("The video does not report a valid frame rate.")

        frame_number = 0

        # predict the court corners and mapping independently for each frame.
        while True:
            success, frame = capture.read()

            if not success:
                break

            height, width = frame.shape[:2]

            landmarks = predict_landmarks(
                frame,
                model,
                preprocess,
            )

            matrix, status = calculate_mapping(
                landmarks,
                width,
                height,
            )

            frame_results.append({
                "frame": frame_number,
                "time_seconds": frame_number / fps,
                "status": status,
                "predicted_singles_corners_pixels": (
                    landmarks[SINGLES_CORNER_INDEXES].tolist()
                ),
                "image_to_court_meters_matrix": (
                    matrix.tolist() if matrix is not None else None
                ),
            })

            frame_number += 1

            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames")

    finally:
        capture.release()

    if not frame_results:
        raise RuntimeError("No video frames were processed.")

    valid_count = sum(
        result["status"] == "experimental_mapping_created"
        for result in frame_results
    )

    output_data = {
        "schema_version": 1,
        "source_video": str(VIDEO_PATH),
        "court_width_meters": COURT_WIDTH_METERS,
        "court_length_meters": COURT_LENGTH_METERS,
        "important_limit": (
            "Experimental mappings generated from model predictions. "
            "Basic geometry checks do not prove court-line alignment. "
            "Do not treat these mappings as independently verified."
        ),
        "summary": {
            "processed_frames": len(frame_results),
            "mappings_created": valid_count,
            "invalid_frames": len(frame_results) - valid_count,
        },
        "frames": frame_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # exclusive creation protects any existing results.
    with OUTPUT_PATH.open("x", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

    print(f"\nProcessed frames: {len(frame_results)}")
    print(f"Experimental mappings created: {valid_count}")
    print(f"Invalid frames: {len(frame_results) - valid_count}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()