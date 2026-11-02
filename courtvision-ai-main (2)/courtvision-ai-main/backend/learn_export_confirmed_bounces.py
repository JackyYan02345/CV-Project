"""export manually confirmed bounce labels."""

import json
from pathlib import Path


CANDIDATES_PATH = Path("outputs/learned_bounce_review_candidates.json")
OUTPUT_PATH = Path("outputs/confirmed_bounces.json")

CONFIRMED_BOUNCE_LABEL = "confirmed_bounce"


def load_review_candidates():
    """
    load the candidate file containing automatic proposals and manual labels.
    """
    with CANDIDATES_PATH.open("r", encoding="utf-8") as file:
        candidate_data = json.load(file)

    return candidate_data


def build_confirmed_bounce_events(candidates):
    """build events from confirmed bounce labels."""
    confirmed_bounces = []

    for candidate in candidates:
        if candidate.get("manual_review_label") != CONFIRMED_BOUNCE_LABEL:
            continue

        bounce_number = len(confirmed_bounces) + 1

        confirmed_bounces.append(
            {
                "bounce_id": f"bounce_{bounce_number:03d}",
                "event_status": "confirmed_bounce_manual_review",
                "source_candidate_frame": candidate["candidate_frame"],
                "time_seconds": candidate["candidate_time_seconds"],
                "image_position_pixels": {
                    "x": candidate["x_pixels"],
                    "y": candidate["y_pixels"],
                },
                # a bounce occurs on the court plane, so the existing court
                # homography is appropriate for these confirmed events.
                "court_position_meters": {
                    "x": candidate["court_x_meters"],
                    "y": candidate["court_y_meters"],
                },
                "trajectory_evidence": {
                    "turn_angle_degrees": candidate["turn_angle_degrees"],
                    "incoming_image_speed_pixels_per_second": (
                        candidate["incoming_image_speed_pixels_per_second"]
                    ),
                    "outgoing_image_speed_pixels_per_second": (
                        candidate["outgoing_image_speed_pixels_per_second"]
                    ),
                },
                "review_evidence": {
                    "manual_review_label": candidate["manual_review_label"],
                    "automatic_candidate_status": candidate[
                        "automatic_status"
                    ],
                },
            }
        )

    return confirmed_bounces


def main():
    """run the script's main workflow."""
    candidate_data = load_review_candidates()
    candidates = candidate_data["candidates"]

    reviewed_candidate_count = sum(
        candidate.get("manual_review_label") is not None
        for candidate in candidates
    )

    confirmed_bounces = build_confirmed_bounce_events(candidates)

    output_data = {
        "schema_version": 1,
        "important_limit": (
            "These bounce events were confirmed through manual video review. "
            "They are not automatic bounce detections."
        ),
        "source": {
            "review_candidate_file": str(CANDIDATES_PATH),
            "ball_tracking_file": (
                "outputs/racketvision_ball_tracking_data.json"
            ),
            "court_homography_file": (
                "outputs/image_to_court_homography.json"
            ),
        },
        "summary": {
            "reviewed_candidate_count": reviewed_candidate_count,
            "confirmed_bounce_count": len(confirmed_bounces),
        },
        "confirmed_bounces": confirmed_bounces,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output_data, indent=2),
        encoding="utf-8",
    )

    print(f"Reviewed candidates: {reviewed_candidate_count}")
    print(f"Confirmed bounces exported: {len(confirmed_bounces)}")
    print(f"Saved confirmed bounce events: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()