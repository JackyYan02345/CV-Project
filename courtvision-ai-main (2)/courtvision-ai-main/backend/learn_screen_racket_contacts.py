"""screen racket-contact candidates using wrist distance."""

import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO


CANDIDATES_PATH = Path("outputs/learned_bounce_review_candidates.json")
VIDEO_PATH = Path("data/raw/test_rally.mp4")
MODEL_PATH = Path("yolo11n-pose.pt")
OUTPUT_PATH = Path(
    "outputs/bounce_candidates_with_contact_screening.json"
)

PERSON_CONFIDENCE_THRESHOLD = 0.25
WRIST_CONFIDENCE_THRESHOLD = 0.25

# derived from the manually reviewed evidence in this camera setup.
# largest measured contact: 102.1 px
# smallest measured bounce: 161.4 px
CONTACT_DISTANCE_THRESHOLD_PIXELS = 130.0

# coco pose-keypoint indexes used by yolo pose.
LEFT_WRIST_INDEX = 9
RIGHT_WRIST_INDEX = 10


def load_candidates():
    """load the reviewed contact candidates."""
    with CANDIDATES_PATH.open("r", encoding="utf-8") as file:
        candidate_data = json.load(file)

    candidates = candidate_data["candidates"]

    if not candidates:
        raise ValueError("No bounce-review candidates were found.")

    return candidate_data, candidates


def open_source_video():
    """
    open the source rally video once for efficient frame access.
    """
    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {VIDEO_PATH}")

    return capture


def read_frame_at(capture, frame_number):
    """
    read one exact zero-based video frame.
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = capture.read()

    if not success:
        return None

    return frame


def extract_reliable_wrists(pose_result):
    """return wrist points with reliable pose scores."""
    if (
        pose_result.keypoints is None
        or pose_result.keypoints.xy is None
        or pose_result.keypoints.conf is None
    ):
        return []

    wrist_records = []

    all_positions = pose_result.keypoints.xy.cpu().tolist()
    all_confidences = pose_result.keypoints.conf.cpu().tolist()

    for person_index, person_keypoints in enumerate(all_positions):
        person_confidences = all_confidences[person_index]

        for wrist_name, wrist_index in (
            ("left_wrist", LEFT_WRIST_INDEX),
            ("right_wrist", RIGHT_WRIST_INDEX),
        ):
            wrist_x, wrist_y = person_keypoints[wrist_index]
            wrist_confidence = person_confidences[wrist_index]

            # ignore missing landmarks and very low-confidence estimates.
            if (
                wrist_confidence >= WRIST_CONFIDENCE_THRESHOLD
                and (wrist_x != 0 or wrist_y != 0)
            ):
                wrist_records.append(
                    {
                        "person_index": person_index,
                        "wrist_name": wrist_name,
                        "x_pixels": wrist_x,
                        "y_pixels": wrist_y,
                        "confidence": wrist_confidence,
                    }
                )

    return wrist_records


def find_nearest_wrist(ball_x, ball_y, wrist_records):
    """
    measure ball distance to every usable wrist and return the nearest one.
    """
    if not wrist_records:
        return None

    measured_wrists = []

    for wrist in wrist_records:
        distance_pixels = math.hypot(
            ball_x - wrist["x_pixels"],
            ball_y - wrist["y_pixels"],
        )

        measured_wrist = dict(wrist)
        measured_wrist["ball_to_wrist_distance_pixels"] = (
            distance_pixels
        )

        measured_wrists.append(measured_wrist)

    return min(
        measured_wrists,
        key=lambda wrist: wrist["ball_to_wrist_distance_pixels"],
    )


def assign_screening_outcome(nearest_wrist):
    """classify a candidate using its wrist distances."""
    if nearest_wrist is None:
        return "unknown_no_reliable_wrist"

    distance = nearest_wrist["ball_to_wrist_distance_pixels"]

    if distance <= CONTACT_DISTANCE_THRESHOLD_PIXELS:
        return "likely_racket_contact"

    return "not_close_to_detected_wrist"


def screen_candidates(candidates, pose_model, capture):
    """
    add independent wrist-proximity screening evidence to every candidate.
    """
    screened_candidates = []

    for candidate in candidates:
        candidate_frame = candidate["candidate_frame"]
        ball_x = candidate["x_pixels"]
        ball_y = candidate["y_pixels"]

        frame = read_frame_at(capture, candidate_frame)

        candidate_with_screening = dict(candidate)

        if frame is None:
            candidate_with_screening["racket_contact_screening"] = {
                "screening_status": "screening_evidence_only",
                "screening_outcome": "unknown_video_frame_unavailable",
                "nearest_wrist": None,
            }

            print(
                f"Frame {candidate_frame}: "
                "unknown_video_frame_unavailable"
            )

            screened_candidates.append(candidate_with_screening)
            continue

        pose_result = pose_model(
            frame,
            classes=[0],
            conf=PERSON_CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]

        reliable_wrists = extract_reliable_wrists(pose_result)

        nearest_wrist = find_nearest_wrist(
            ball_x,
            ball_y,
            reliable_wrists,
        )

        screening_outcome = assign_screening_outcome(nearest_wrist)

        candidate_with_screening["racket_contact_screening"] = {
            "screening_status": "screening_evidence_only",
            "screening_outcome": screening_outcome,
            "contact_distance_threshold_pixels": (
                CONTACT_DISTANCE_THRESHOLD_PIXELS
            ),
            "reliable_wrist_count": len(reliable_wrists),
            "nearest_wrist": nearest_wrist,
        }

        if nearest_wrist is None:
            print(
                f"Frame {candidate_frame}: "
                f"{screening_outcome}"
            )
        else:
            distance = nearest_wrist["ball_to_wrist_distance_pixels"]

            print(
                f"Frame {candidate_frame}: "
                f"{screening_outcome} | "
                f"nearest wrist = {distance:.1f} px"
            )

        screened_candidates.append(candidate_with_screening)

    return screened_candidates


def build_summary(screened_candidates):
    """
    count screening outcomes without interpreting them as confirmed events.
    """
    outcome_counts = {}

    for candidate in screened_candidates:
        outcome = candidate["racket_contact_screening"][
            "screening_outcome"
        ]

        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    return {
        "candidate_count": len(screened_candidates),
        "likely_racket_contact_count": outcome_counts.get(
            "likely_racket_contact",
            0,
        ),
        "not_close_to_detected_wrist_count": outcome_counts.get(
            "not_close_to_detected_wrist",
            0,
        ),
        "unknown_no_reliable_wrist_count": outcome_counts.get(
            "unknown_no_reliable_wrist",
            0,
        ),
        "unknown_video_frame_unavailable_count": outcome_counts.get(
            "unknown_video_frame_unavailable",
            0,
        ),
    }


def main():
    """run the script's main workflow."""
    _, candidates = load_candidates()

    pose_model = YOLO(str(MODEL_PATH))
    capture = open_source_video()

    try:
        screened_candidates = screen_candidates(
            candidates,
            pose_model,
            capture,
        )
    finally:
        capture.release()

    output_data = {
        "schema_version": 1,
        "important_limit": (
            "This output contains racket-contact screening evidence, not "
            "automatic bounce confirmations. Candidates still need review "
            "unless independently confirmed."
        ),
        "source": {
            "candidate_file": str(CANDIDATES_PATH),
            "video_file": str(VIDEO_PATH),
            "pose_model": str(MODEL_PATH),
        },
        "screening_configuration": {
            "person_confidence_threshold": (
                PERSON_CONFIDENCE_THRESHOLD
            ),
            "wrist_confidence_threshold": (
                WRIST_CONFIDENCE_THRESHOLD
            ),
            "contact_distance_threshold_pixels": (
                CONTACT_DISTANCE_THRESHOLD_PIXELS
            ),
        },
        "summary": build_summary(screened_candidates),
        "candidates": screened_candidates,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output_data, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved contact-screened candidates: {OUTPUT_PATH}")
    print(f"Summary: {output_data['summary']}")


if __name__ == "__main__":
    main()