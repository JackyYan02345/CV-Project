"""evaluate wrist-distance contact screening against reviewed labels."""

import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

# import the screening functions and settings you already created.
import learn_screen_racket_contacts as screen


CANDIDATES_PATH = Path(
    "outputs/validation_rally_01_bounce_review_candidates.json"
)

VIDEO_PATH = Path("data/raw/validation_rally_01.mp4")

OUTPUT_PATH = Path(
    "outputs/validation_rally_01_contact_evaluation.json"
)


# your video-review decisions, keyed by candidate frame number.
#
# using frame numbers connects each label to a specific event, even if
# the order of the candidates in the json file changes.
MANUAL_LABELS_BY_FRAME = {
    23: "confirmed_bounce",
    31: "confirmed_racket_contact",
    58: "confirmed_bounce",
    103: "confirmed_bounce",
    139: "confirmed_bounce",
    175: "confirmed_bounce",
    183: "rejected_bad_detection",
    193: "confirmed_racket_contact",
    258: "confirmed_racket_contact",
    280: "confirmed_bounce",
    325: "confirmed_racket_contact",
    349: "confirmed_bounce",
    374: "hit_net",
    385: "confirmed_bounce",
    449: "confirmed_racket_contact",
    478: "confirmed_bounce",
    509: "confirmed_bounce",
    553: "confirmed_bounce",
    588: "confirmed_bounce",
    598: "confirmed_racket_contact",
    622: "rejected_bad_detection",
    635: "confirmed_bounce",
    746: "confirmed_bounce",
    754: "confirmed_racket_contact",
    845: "confirmed_bounce",
    864: "confirmed_racket_contact",
    892: "confirmed_bounce",
}


def load_and_check_candidates():
    """load candidates and check their required fields."""
    with CANDIDATES_PATH.open("r", encoding="utf-8") as file:
        candidate_data = json.load(file)

    candidates = candidate_data["candidates"]

    candidate_frames = [
        candidate["candidate_frame"]
        for candidate in candidates
    ]

    if len(candidate_frames) != len(set(candidate_frames)):
        raise ValueError("The candidate file contains duplicate frame numbers.")

    actual_frames = set(candidate_frames)
    labeled_frames = set(MANUAL_LABELS_BY_FRAME)

    if actual_frames != labeled_frames:
        missing_labels = sorted(actual_frames - labeled_frames)
        unexpected_labels = sorted(labeled_frames - actual_frames)

        raise ValueError(
            "The validation candidates and manual labels do not match.\n"
            f"Candidate frames without labels: {missing_labels}\n"
            f"Label frames not present in candidates: {unexpected_labels}"
        )

    return candidates


def evaluate_results(screened_candidates):
    """compare predictions with the reviewed labels."""
    results = []
    counts_by_label_and_outcome = {}

    for candidate in screened_candidates:
        frame = candidate["candidate_frame"]
        manual_label = MANUAL_LABELS_BY_FRAME[frame]

        screening = candidate["racket_contact_screening"]
        outcome = screening["screening_outcome"]
        nearest_wrist = screening["nearest_wrist"]

        # preserve both the human decision and the model evidence.
        result = {
            "candidate_frame": frame,
            "candidate_time_seconds": candidate["candidate_time_seconds"],
            "manual_review_label": manual_label,
            "screening_outcome": outcome,
            "nearest_wrist": nearest_wrist,
        }

        results.append(result)

        # build a count for each label and screening outcome.
        if manual_label not in counts_by_label_and_outcome:
            counts_by_label_and_outcome[manual_label] = Counter()

        counts_by_label_and_outcome[manual_label][outcome] += 1

    # counter behaves like a dictionary, but regular dictionaries are easier
    # to read in the json output.
    summary = {
        label: dict(outcome_counts)
        for label, outcome_counts in counts_by_label_and_outcome.items()
    }

    return results, summary


def print_evaluation(summary):
    """
    show how the screen performed for each manually reviewed event type.
    """
    print("\nValidation results by manual label:\n")

    for label in sorted(summary):
        print(f"{label}:")

        for outcome, count in sorted(summary[label].items()):
            print(f"  {outcome}: {count}")

        print()

    contacts = summary.get("confirmed_racket_contact", {})
    flagged = contacts.get("likely_racket_contact", 0)
    contact_total = sum(contacts.values())

    print(
        f"Confirmed contacts flagged: {flagged}/{contact_total}"
    )

    bounces = summary.get("confirmed_bounce", {})
    bounce_flagged = bounces.get("likely_racket_contact", 0)
    bounce_total = sum(bounces.values())

    print(
        f"Confirmed bounces flagged as likely contact: "
        f"{bounce_flagged}/{bounce_total}"
    )

    print(
        "\nUnknown outcomes are not counted as correct or incorrect "
        "contact decisions."
    )


def main():
    # avoid accidentally replacing an earlier evaluation.
    """run the script's main workflow."""
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Choose a different output filename before running again."
        )

    candidates = load_and_check_candidates()

    print(f"Candidates matched to manual labels: {len(candidates)}")
    print(
        "Contact-distance threshold: "
        f"{screen.CONTACT_DISTANCE_THRESHOLD_PIXELS:.1f} pixels"
    )

    # load the same pose model and use the same screening functions and
    # thresholds as the original experiment.
    pose_model = YOLO(str(screen.MODEL_PATH))
    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    try:
        # this function does not use manual_labels_by_frame.
        screened_candidates = screen.screen_candidates(
            candidates,
            pose_model,
            capture,
        )
    finally:
        capture.release()

    # compare predictions with your labels only after screening is complete.
    results, summary = evaluate_results(screened_candidates)

    output_data = {
        "schema_version": 1,
        "source": {
            "candidate_file": str(CANDIDATES_PATH),
            "video_file": str(VIDEO_PATH),
            "pose_model": str(screen.MODEL_PATH),
        },
        "screening_configuration": {
            "contact_distance_threshold_pixels": (
                screen.CONTACT_DISTANCE_THRESHOLD_PIXELS
            ),
            "person_confidence_threshold": (
                screen.PERSON_CONFIDENCE_THRESHOLD
            ),
            "wrist_confidence_threshold": (
                screen.WRIST_CONFIDENCE_THRESHOLD
            ),
        },
        "important_limit": (
            "This evaluates the wrist screen on manually reviewed candidate "
            "events from one validation rally. It does not measure events "
            "missed by the candidate generator."
        ),
        "summary_by_manual_label": summary,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # "x" creates a new file and refuses to overwrite an existing one.
    with OUTPUT_PATH.open("x", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

    print_evaluation(summary)
    print(f"\nSaved validation evaluation: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()