"""measure wrist distances for contact candidates."""

import json
import math
from pathlib import Path

import cv2
from ultralytics import YOLO


CANDIDATES_PATH = Path("outputs/learned_bounce_review_candidates.json")
VIDEO_PATH = Path("data/raw/test_rally.mp4")
MODEL_PATH = Path("yolo11n-pose.pt")

# coco pose model landmark numbers:
# 9 = left wrist, 10 = right wrist.
LEFT_WRIST_INDEX = 9
RIGHT_WRIST_INDEX = 10

OUTPUT_PATH = Path("outputs/candidate_wrist_distance_measurements.json")

def load_reviewed_candidates():
    """load candidates marked for review."""
    with CANDIDATES_PATH.open("r", encoding="utf-8") as file:
        candidate_data = json.load(file)

    reviewed_candidates = [
        candidate
        for candidate in candidate_data["candidates"]
        if candidate.get("manual_review_label") is not None
    ]

    if not reviewed_candidates:
        raise ValueError(
            "No manually reviewed candidates were found. "
            "Check that manual_review_label exists in the candidate JSON."
        )

    return reviewed_candidates

def open_source_video():
    """open the source video for frame inspection."""
    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {VIDEO_PATH}")

    return capture

def read_frame_at(capture, frame_number: int):
    """read the video frame at the requested index."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = capture.read()

    if not success:
        return None

    return frame

WRIST_CONFIDENCE_THRESHOLD = 0.25

def extract_reliable_wrists(pose_result):
    """return wrist points with reliable pose scores."""
    if pose_result.keypoints is None or pose_result.keypoints.xy is None:
        return []

    wrist_records = []

    # one [17 keypoints] list per detected person.
    all_keypoint_positions = pose_result.keypoints.xy.cpu().tolist()

    # yolo normally supplies one confidence score per keypoint.
    all_keypoint_confidences = pose_result.keypoints.conf.cpu().tolist()

    for person_index, person_keypoints in enumerate(all_keypoint_positions):
        person_confidences = all_keypoint_confidences[person_index]

        for wrist_name, wrist_index in (
            ("left_wrist", LEFT_WRIST_INDEX),
            ("right_wrist", RIGHT_WRIST_INDEX),
        ):
            wrist_x, wrist_y = person_keypoints[wrist_index]
            wrist_confidence = person_confidences[wrist_index]

            # a low-confidence or (0, 0) landmark is not trustworthy evidence.
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

def find_nearest_wrist(ball_x: float, ball_y: float, wrist_records: list):
    """find the wrist nearest to the ball."""
    if not wrist_records:
        return None

    measured_wrists = []

    for wrist in wrist_records:
        distance_pixels = math.hypot(
            ball_x - wrist["x_pixels"],
            ball_y - wrist["y_pixels"],
        )

        # make a copy so the original wrist record stays unchanged.
        measured_wrist = dict(wrist)
        measured_wrist["ball_to_wrist_distance_pixels"] = distance_pixels
        measured_wrists.append(measured_wrist)

    return min(
        measured_wrists,
        key=lambda wrist: wrist["ball_to_wrist_distance_pixels"],
    )

def main():
    """run the script's main workflow."""
    reviewed_candidates = load_reviewed_candidates()

    print(f"Manually reviewed candidates: {len(reviewed_candidates)}")

    # load the pose model once, rather than once per candidate frame.
    pose_model = YOLO(str(MODEL_PATH))

    # open the mp4 once, rather than once per candidate frame.
    capture = open_source_video()

    return reviewed_candidates, pose_model, capture

if __name__ == "__main__":
    reviewed_candidates, pose_model, capture = main()

    # one measurement record per manually reviewed candidate.
    measurements = []

    try:
        for candidate in reviewed_candidates:
            candidate_frame = candidate["candidate_frame"]
            ball_x = candidate["x_pixels"]
            ball_y = candidate["y_pixels"]
            manual_label = candidate["manual_review_label"]

            frame = read_frame_at(capture, candidate_frame)

            # start a record even if video reading or pose detection fails.
            # this preserves the fact that the event was attempted.
            measurement = {
                "candidate_frame": candidate_frame,
                "candidate_time_seconds": candidate["candidate_time_seconds"],
                "manual_review_label": manual_label,
                "automatic_status": candidate["automatic_status"],
                "ball_x_pixels": ball_x,
                "ball_y_pixels": ball_y,
                "reliable_wrist_detected": False,
                "nearest_wrist": None,
            }

            if frame is None:
                measurement["measurement_status"] = "video_frame_unavailable"

                print(
                    f"Frame {candidate_frame}: could not read video frame "
                    f"({manual_label})"
                )

                measurements.append(measurement)
                continue

            pose_result = pose_model(
                frame,
                classes=[0],
                conf=0.25,
                verbose=False,
            )[0]

            reliable_wrists = extract_reliable_wrists(pose_result)
            nearest_wrist = find_nearest_wrist(
                ball_x,
                ball_y,
                reliable_wrists,
            )

            if nearest_wrist is None:
                measurement["measurement_status"] = "no_reliable_wrist_detected"

                print(
                    f"Frame {candidate_frame}: {manual_label} | "
                    "no reliable wrist detected"
                )
            else:
                measurement["measurement_status"] = "measured"
                measurement["reliable_wrist_detected"] = True
                measurement["nearest_wrist"] = nearest_wrist

                distance = nearest_wrist["ball_to_wrist_distance_pixels"]

                print(
                    f"Frame {candidate_frame}: {manual_label} | "
                    f"nearest wrist = {distance:.1f} px | "
                    f"{nearest_wrist['wrist_name']} "
                    f"(person {nearest_wrist['person_index']}, "
                    f"confidence {nearest_wrist['confidence']:.2f})"
                )

            measurements.append(measurement)

    finally:
        capture.release()

    # save the raw evidence. this file does not make automatic event decisions.
    output_data = {
        "schema_version": 1,
        "purpose": (
            "Ball-to-wrist measurements for manually reviewed tennis "
            "direction-change candidates. These are analysis evidence, "
            "not automatic bounce classifications."
        ),
        "source": {
            "candidate_file": str(CANDIDATES_PATH),
            "video_file": str(VIDEO_PATH),
            "pose_model": str(MODEL_PATH),
        },
        "summary": {
            "manually_reviewed_candidate_count": len(measurements),
            "reliable_wrist_measurement_count": sum(
                measurement["reliable_wrist_detected"]
                for measurement in measurements
            ),
        },
        "measurements": measurements,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output_data, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved wrist-distance evidence: {OUTPUT_PATH}")