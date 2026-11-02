"""inspect selected validation-screen cases."""

import json
from pathlib import Path

import cv2
from ultralytics import YOLO

import learn_screen_racket_contacts as screen


VIDEO_PATH = Path("data/raw/validation_rally_01.mp4")

CANDIDATES_PATH = Path(
    "outputs/validation_rally_01_bounce_review_candidates.json"
)

OUTPUT_DIR = Path("outputs/validation_screen_inspection")

FRAMES_TO_INSPECT = [746, 864]


def load_candidates_by_frame():
    """
    load the automatic candidates and look them up by frame number.
    """
    with CANDIDATES_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        candidate["candidate_frame"]: candidate
        for candidate in data["candidates"]
    }


def inspect_frame(capture, model, candidate):
    """inspect the selected video frame."""
    frame_number = candidate["candidate_frame"]

    frame = screen.read_frame_at(capture, frame_number)

    if frame is None:
        raise RuntimeError(f"Could not read frame {frame_number}")

    pose_result = model(
        frame,
        classes=[0],
        conf=screen.PERSON_CONFIDENCE_THRESHOLD,
        verbose=False,
    )[0]

    # result.plot() draws yolo's detected people and pose landmarks.
    annotated = pose_result.plot()

    ball_x = round(candidate["x_pixels"])
    ball_y = round(candidate["y_pixels"])

    # opencv uses bgr colors: (0, 0, 255) is red.
    cv2.circle(
        annotated,
        (ball_x, ball_y),
        8,
        (0, 0, 255),
        -1,
    )

    cv2.putText(
        annotated,
        f"Candidate ball - frame {frame_number}",
        (ball_x + 12, max(ball_y - 12, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    wrists = screen.extract_reliable_wrists(pose_result)

    nearest_wrist = screen.find_nearest_wrist(
        ball_x,
        ball_y,
        wrists,
    )

    if nearest_wrist is None:
        print(f"Frame {frame_number}: no reliable wrist detected")
    else:
        wrist_x = round(nearest_wrist["x_pixels"])
        wrist_y = round(nearest_wrist["y_pixels"])
        distance = nearest_wrist["ball_to_wrist_distance_pixels"]

        # yellow line connects the candidate ball and its nearest wrist.
        cv2.line(
            annotated,
            (ball_x, ball_y),
            (wrist_x, wrist_y),
            (0, 255, 255),
            2,
        )

        print(
            f"Frame {frame_number}: "
            f"nearest wrist {distance:.1f} pixels away "
            f"({nearest_wrist['wrist_name']}, "
            f"confidence {nearest_wrist['confidence']:.2f})"
        )

    return annotated


def main():
    """run the script's main workflow."""
    candidates = load_candidates_by_frame()

    for frame_number in FRAMES_TO_INSPECT:
        if frame_number not in candidates:
            raise ValueError(
                f"Frame {frame_number} is missing from the candidate file."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # avoid overwriting an image from an earlier inspection.
    output_paths = [
        OUTPUT_DIR / f"frame_{frame_number}.jpg"
        for frame_number in FRAMES_TO_INSPECT
    ]

    existing = [path for path in output_paths if path.exists()]

    if existing:
        raise FileExistsError(
            f"Inspection image already exists: {existing[0]}"
        )

    model = YOLO(str(screen.MODEL_PATH))
    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    try:
        for frame_number, output_path in zip(
            FRAMES_TO_INSPECT,
            output_paths,
        ):
            annotated = inspect_frame(
                capture,
                model,
                candidates[frame_number],
            )

            if not cv2.imwrite(str(output_path), annotated):
                raise RuntimeError(f"Could not save: {output_path}")

            print(f"Saved: {output_path}")
    finally:
        capture.release()


if __name__ == "__main__":
    main()