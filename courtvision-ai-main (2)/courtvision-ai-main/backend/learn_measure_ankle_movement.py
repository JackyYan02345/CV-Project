"""measure ankle positions around selected shot events."""

import json
import math
from pathlib import Path


INPUT_PATH = Path("outputs/validation_feet_sequence.json")
OUTPUT_PATH = Path("outputs/validation_ankle_movement.json")

ANKLE_NAMES = ["left ankle", "right ankle"]


def load_sequence(path):
    """read the saved ankle positions."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Run "
            "learn_track_player_feet.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    frames = data.get("frames", [])

    if len(frames) < 2:
        raise ValueError(
            "The sequence must contain at least two frames."
        )

    return data


def get_ankle_position(frame, ankle_name):
    """return an ankle position when available."""
    ankle = frame.get("ankles", {}).get(ankle_name, {})
    return ankle.get("court_position_meters")


def measure_ankle(start_frame, end_frame, ankle_name):
    """compare one ankle's positions in the two endpoint frames."""
    start = get_ankle_position(start_frame, ankle_name)
    end = get_ankle_position(end_frame, ankle_name)

    if start is None or end is None:
        return {
            "status": "unavailable",
            "reason": (
                "An ankle court position was missing in the "
                "first or last frame."
            ),
        }

    # positive x movement follows the court's x-axis.
    delta_x = end["x"] - start["x"]

    # positive y movement follows the court's y-axis.
    delta_y = end["y"] - start["y"]

    # pythagorean theorem gives the straight-line distance between
    # the two estimated positions on the court plane.
    displacement = math.hypot(delta_x, delta_y)

    return {
        "status": "measured",
        "start_court_position_meters": start,
        "end_court_position_meters": end,
        "change_x_meters": delta_x,
        "change_y_meters": delta_y,
        "straight_line_displacement_meters": displacement,
    }


def main():
    """run the script's main workflow."""
    data = load_sequence(INPUT_PATH)
    frames = data["frames"]

    start_frame = frames[0]
    end_frame = frames[-1]

    start_time = start_frame["time_seconds"]
    end_time = end_frame["time_seconds"]
    duration = end_time - start_time

    if duration <= 0:
        raise ValueError(
            "The last frame must come after the first frame."
        )

    print(
        f"Comparing frames {start_frame['frame']} "
        f"and {end_frame['frame']}"
    )
    print(f"Time between frames: {duration:.3f} seconds")

    measurements = {}

    for ankle_name in ANKLE_NAMES:
        measurement = measure_ankle(
            start_frame,
            end_frame,
            ankle_name,
        )

        measurements[ankle_name] = measurement

        print(f"\n{ankle_name}:")

        if measurement["status"] == "unavailable":
            print(f"  {measurement['reason']}")
            continue

        print(
            "  Change across the court: "
            f"{measurement['change_x_meters']:+.2f} m"
        )
        print(
            "  Change along the court: "
            f"{measurement['change_y_meters']:+.2f} m"
        )
        print(
            "  Straight-line displacement: "
            f"{measurement['straight_line_displacement_meters']:.2f} m"
        )

    output = {
        "source_file": str(INPUT_PATH),
        "source_video": data.get("source_video"),
        "start_frame": start_frame["frame"],
        "end_frame": end_frame["frame"],
        "duration_seconds": duration,
        "measurements": measurements,
        "important_limit": (
            "These are approximate changes in projected ankle "
            "positions. They do not measure the full distance "
            "traveled or confirm foot-ground contact."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, allow_nan=False)

    print(f"\nSaved measurements: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()