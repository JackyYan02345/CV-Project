"""summarize player position and movement across a shot."""

import json
import math
from pathlib import Path


INPUT_PATH = Path("outputs/validation_shot_449_feet.json")
OUTPUT_PATH = Path("outputs/validation_shot_449_player_positions.json")

FRAMES_TO_INSPECT = {
    "start": 430,
    "near_bounce": 441,
    "near_contact": 449,
    "after_contact": 475,
}

# the singles court is 8.23 meters wide.
COURT_WIDTH_METERS = 8.23

# the near baseline is at y = 23.77 meters in this court map.
COURT_LENGTH_METERS = 23.77


def load_frame_lookup(path):
    """read the ankle data and organize records by frame number."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Run "
            "learn_track_player_feet.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    frame_lookup = {}

    for frame in data.get("frames", []):
        frame_number = frame["frame"]

        if frame_number in frame_lookup:
            raise ValueError(
                f"Frame {frame_number} appears more than once."
            )

        frame_lookup[frame_number] = frame

    return data, frame_lookup


def ankle_midpoint(frame):
    """find the midpoint between two ankles."""
    ankles = frame.get("ankles", {})

    left = ankles.get("left ankle", {}).get(
        "court_position_meters"
    )

    right = ankles.get("right ankle", {}).get(
        "court_position_meters"
    )

    if left is None or right is None:
        return None

    return {
        "x": (left["x"] + right["x"]) / 2,
        "y": (left["y"] + right["y"]) / 2,
    }


def describe_position(frame):
    """summarize one frame's approximate player position."""
    midpoint = ankle_midpoint(frame)

    result = {
        "frame": frame["frame"],
        "time_seconds": frame["time_seconds"],
        "court_registration_status": frame[
            "court_registration_status"
        ],
        "ankle_midpoint_meters": midpoint,
        "distance_from_center_line_meters": None,
        "distance_from_near_baseline_meters": None,
    }

    if midpoint is None:
        return result

    # positive means the midpoint is on the right side of the court
    # in the image's court-coordinate system.
    result["distance_from_center_line_meters"] = (
        midpoint["x"] - COURT_WIDTH_METERS / 2
    )

    # positive means beyond the near baseline.
    # negative means inside the near baseline.
    result["distance_from_near_baseline_meters"] = (
        midpoint["y"] - COURT_LENGTH_METERS
    )

    return result


def displacement(start, end):
    """calculate straight-line displacement between two midpoints."""
    if start is None or end is None:
        return None

    return math.hypot(
        end["x"] - start["x"],
        end["y"] - start["y"],
    )


def main():
    """run the script's main workflow."""
    data, frame_lookup = load_frame_lookup(INPUT_PATH)

    missing = [
        frame_number
        for frame_number in FRAMES_TO_INSPECT.values()
        if frame_number not in frame_lookup
    ]

    if missing:
        raise ValueError(
            f"These frames are missing from the sequence: {missing}"
        )

    positions = {}

    for name, frame_number in FRAMES_TO_INSPECT.items():
        positions[name] = describe_position(
            frame_lookup[frame_number]
        )

    print("Approximate position between the ankles:")

    for name, position in positions.items():
        midpoint = position["ankle_midpoint_meters"]

        print(f"\n{name} — frame {position['frame']}")

        if midpoint is None:
            print("  Position unavailable: an ankle estimate is missing.")
            continue

        print(
            f"  Court position: "
            f"x={midpoint['x']:.2f} m, "
            f"y={midpoint['y']:.2f} m"
        )
        print(
            "  Relative to court center line: "
            f"{position['distance_from_center_line_meters']:+.2f} m"
        )
        print(
            "  Relative to near baseline: "
            f"{position['distance_from_near_baseline_meters']:+.2f} m"
        )

    contact_position = positions["near_contact"][
        "ankle_midpoint_meters"
    ]

    after_position = positions["after_contact"][
        "ankle_midpoint_meters"
    ]

    after_contact_displacement = displacement(
        contact_position,
        after_position,
    )

    print("\nAfter-contact midpoint displacement:")

    if after_contact_displacement is None:
        print("  Unavailable: an endpoint position is missing.")
    else:
        print(f"  {after_contact_displacement:.2f} m")

    output = {
        "source_file": str(INPUT_PATH),
        "source_video": data.get("source_video"),
        "positions": positions,
        "after_contact_midpoint_displacement_meters": (
            after_contact_displacement
        ),
        "important_limit": (
            "Positions are midpoints between projected ankle estimates. "
            "They are approximate and do not establish foot-ground contact. "
            "Event frames are review anchors rather than exact event times."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, allow_nan=False)

    print(f"\nSaved player positions: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()