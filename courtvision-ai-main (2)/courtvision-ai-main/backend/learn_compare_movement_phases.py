"""compare ankle movement across shot phases."""

import json
import math
from pathlib import Path


INPUT_PATH = Path("outputs/validation_shot_449_feet.json")
OUTPUT_PATH = Path("outputs/validation_shot_449_movement_phases.json")

FRAME_START = 430
FRAME_NEAR_BOUNCE = 441
FRAME_NEAR_CONTACT = 449
FRAME_END = 475

ANKLE_NAMES = ["left ankle", "right ankle"]


def load_frames(path):
    """load the sequence and create a lookup using frame numbers."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Run "
            "learn_track_player_feet.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    frames = data.get("frames", [])
    frame_lookup = {}

    for frame in frames:
        frame_number = frame["frame"]

        if frame_number in frame_lookup:
            raise ValueError(
                f"Frame {frame_number} appears more than once."
            )

        frame_lookup[frame_number] = frame

    return data, frame_lookup


def get_position(frame, ankle_name):
    """return a court position, or none if it was unavailable."""
    ankle = frame.get("ankles", {}).get(ankle_name, {})
    return ankle.get("court_position_meters")


def compare_phase(frame_lookup, start_number, end_number):
    """measure endpoint displacement for each ankle in one phase."""
    start = frame_lookup[start_number]
    end = frame_lookup[end_number]

    duration = end["time_seconds"] - start["time_seconds"]

    if duration <= 0:
        raise ValueError(
            f"Frame {end_number} must occur after frame {start_number}."
        )

    phase = {
        "start_frame": start_number,
        "end_frame": end_number,
        "duration_seconds": duration,
        "ankles": {},
    }

    for ankle_name in ANKLE_NAMES:
        start_position = get_position(start, ankle_name)
        end_position = get_position(end, ankle_name)

        if start_position is None or end_position is None:
            phase["ankles"][ankle_name] = {
                "status": "unavailable",
                "reason": "An endpoint position was missing.",
            }
            continue

        delta_x = end_position["x"] - start_position["x"]
        delta_y = end_position["y"] - start_position["y"]

        phase["ankles"][ankle_name] = {
            "status": "measured",
            "change_x_meters": delta_x,
            "change_y_meters": delta_y,
            "straight_line_displacement_meters": math.hypot(
                delta_x,
                delta_y,
            ),
        }

    return phase


def print_phase(name, phase):
    """show the movement measurements in a readable form."""
    print(f"\n{name}")
    print(
        f"  Frames {phase['start_frame']}–{phase['end_frame']} "
        f"({phase['duration_seconds']:.3f} seconds)"
    )

    for ankle_name, measurement in phase["ankles"].items():
        if measurement["status"] != "measured":
            print(f"  {ankle_name}: unavailable")
            continue

        print(
            f"  {ankle_name}: "
            f"x {measurement['change_x_meters']:+.2f} m, "
            f"y {measurement['change_y_meters']:+.2f} m, "
            f"displacement "
            f"{measurement['straight_line_displacement_meters']:.2f} m"
        )


def main():
    """run the script's main workflow."""
    data, frame_lookup = load_frames(INPUT_PATH)

    required_frames = [
        FRAME_START,
        FRAME_NEAR_BOUNCE,
        FRAME_NEAR_CONTACT,
        FRAME_END,
    ]

    missing_frames = [
        number for number in required_frames
        if number not in frame_lookup
    ]

    if missing_frames:
        raise ValueError(
            f"These frames are missing from the sequence: "
            f"{missing_frames}"
        )

    phases = {
        "approach_to_bounce": compare_phase(
            frame_lookup,
            FRAME_START,
            FRAME_NEAR_BOUNCE,
        ),
        "bounce_to_contact": compare_phase(
            frame_lookup,
            FRAME_NEAR_BOUNCE,
            FRAME_NEAR_CONTACT,
        ),
        "after_contact": compare_phase(
            frame_lookup,
            FRAME_NEAR_CONTACT,
            FRAME_END,
        ),
    }

    print_phase("Approaching the bounce", phases["approach_to_bounce"])
    print_phase("Near bounce to near contact", phases["bounce_to_contact"])
    print_phase("After the contact candidate", phases["after_contact"])

    output = {
        "source_file": str(INPUT_PATH),
        "source_video": data.get("source_video"),
        "event_frame_note": (
            "Frames 746 and 754 are review anchors. "
            "Exact event timing may differ."
        ),
        "phases": phases,
        "important_limit": (
            "These measurements compare projected ankle positions "
            "at phase endpoints. They do not establish foot-ground "
            "contact or measure the complete path traveled."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, allow_nan=False)

    print(f"\nSaved phase measurements: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()