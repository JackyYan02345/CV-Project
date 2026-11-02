"""prepare review evidence for one tennis shot."""

import json
from pathlib import Path


POSITIONS_PATH = Path("outputs/validation_player_positions.json")
PHASES_PATH = Path("outputs/validation_movement_phases.json")
OUTPUT_PATH = Path("outputs/validation_shot_evidence.json")


def read_json(path):
    """open and read a json file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Run the position and "
            "movement-phase scripts first."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    """run the script's main workflow."""
    positions_data = read_json(POSITIONS_PATH)
    phases_data = read_json(PHASES_PATH)

    # both summaries should describe the same tracking sequence and video.
    if positions_data["source_file"] != phases_data["source_file"]:
        raise ValueError(
            "The position and phase summaries use different "
            "tracking sequences."
        )

    if positions_data["source_video"] != phases_data["source_video"]:
        raise ValueError(
            "The position and phase summaries use different videos."
        )

    positions = positions_data["positions"]
    phases = phases_data["phases"]

    # check that the phase boundaries match the frames described
    # in the position summary.
    expected_boundaries = {
        "approach_to_bounce": ("start", "near_bounce"),
        "bounce_to_contact": ("near_bounce", "near_contact"),
        "after_contact": ("near_contact", "after_contact"),
    }

    for phase_name, (start_name, end_name) in (
        expected_boundaries.items()
    ):
        phase = phases[phase_name]

        if (
            phase["start_frame"] != positions[start_name]["frame"]
            or phase["end_frame"] != positions[end_name]["frame"]
        ):
            raise ValueError(
                f"The frame boundaries for {phase_name} do not "
                "match the position summary. Regenerate both "
                "summaries from the same sequence."
            )

    evidence = {
        "source_video": positions_data["source_video"],
        "source_tracking_file": positions_data["source_file"],

        # identified these events while reviewing the video.
        # the frames are nearby reference points, not exact event times.
        "reviewed_event_anchors": {
            "bounce": {
                "frame": positions["near_bounce"]["frame"],
                "note": (
                    "Bounce observed shortly before this frame."
                ),
            },
            "racket_contact": {
                "frame": positions["near_contact"]["frame"],
                "note": (
                    "Frame identified as a racket-contact candidate."
                ),
            },
        },

        "player_position_estimates": positions,
        "ankle_movement_phases": phases,

        "after_contact_midpoint_displacement_meters": (
            positions_data[
                "after_contact_midpoint_displacement_meters"
            ]
        ),

        "limitations": [
            "Player positions are midpoints between projected ankle "
            "estimates.",

            "Ankle position does not establish foot-ground contact.",

            "Displacement compares phase endpoints and does not "
            "measure the complete path traveled.",

            "Event frames are review anchors and may differ from "
            "the exact event times.",

            "The measurements do not establish whether a movement "
            "was tactically or technically correct.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(evidence, file, indent=2, allow_nan=False)

    print(f"Saved shot evidence: {OUTPUT_PATH}")
    print(f"Source video: {evidence['source_video']}")
    print(
        "Bounce review anchor: "
        f"frame {evidence['reviewed_event_anchors']['bounce']['frame']}"
    )
    print(
        "Contact review anchor: "
        f"frame "
        f"{evidence['reviewed_event_anchors']['racket_contact']['frame']}"
    )
    print(
        "After-contact midpoint displacement: "
        f"{evidence['after_contact_midpoint_displacement_meters']:.2f} m"
        if evidence["after_contact_midpoint_displacement_meters"]
        is not None
        else "After-contact midpoint displacement: unavailable"
    )


if __name__ == "__main__":
    main()