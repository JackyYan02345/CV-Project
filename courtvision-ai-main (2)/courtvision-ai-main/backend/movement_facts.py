"""describe measured movement without assigning coaching meaning."""

import argparse
import json
import math
from pathlib import Path


def baseline_statement(distance):
    """describe distance from the near baseline."""
    if distance > 0:
        return (
            f"The estimated ankle midpoint was "
            f"{distance:.2f} m behind the near baseline."
        )

    if distance < 0:
        return (
            f"The estimated ankle midpoint was "
            f"{abs(distance):.2f} m inside the near baseline."
        )

    return "The estimated ankle midpoint was on the near baseline."


def center_statement(distance):
    """describe position relative to court center."""
    if distance < 0:
        return (
            f"The estimated ankle midpoint was "
            f"{abs(distance):.2f} m left of the court center line."
        )

    if distance > 0:
        return (
            f"The estimated ankle midpoint was "
            f"{distance:.2f} m right of the court center line."
        )

    return "The estimated ankle midpoint was on the court center line."


def ankle_separation(frame):
    """measure the distance between projected ankles."""
    ankles = frame.get("ankles", {})

    left = ankles.get("left ankle", {}).get(
        "court_position_meters"
    )

    right = ankles.get("right ankle", {}).get(
        "court_position_meters"
    )

    if left is None or right is None:
        return None

    return math.hypot(
        right["x"] - left["x"],
        right["y"] - left["y"],
    )


def load_tracking_frames(summary):
    """load the tracking frames for this shot."""
    tracking_path = Path(summary["tracking_data"])

    if not tracking_path.is_file():
        raise FileNotFoundError(
            f"Tracking data not found: {tracking_path}"
        )

    with tracking_path.open("r", encoding="utf-8") as file:
        tracking = json.load(file)

    if tracking["source_video"] != summary["source_video"]:
        raise ValueError(
            "The tracking data and shot summary refer to "
            "different source videos."
        )

    frame_lookup = {}

    for frame in tracking["frames"]:
        frame_number = frame["frame"]

        if frame_number in frame_lookup:
            raise ValueError(
                f"Duplicate frame in tracking data: {frame_number}"
            )

        frame_lookup[frame_number] = frame

    return frame_lookup


def build_facts(summary):
    """build position and ankle-separation facts."""
    frame_lookup = load_tracking_frames(summary)
    facts = {}

    for anchor_name, position in summary["player_positions"].items():
        frame_number = position["frame"]

        if frame_number not in frame_lookup:
            raise ValueError(
                f"Anchor frame {frame_number} is missing "
                "from the tracking data."
            )

        baseline_distance = position.get(
            "distance_from_near_baseline_meters"
        )

        center_distance = position.get(
            "distance_from_center_line_meters"
        )

        separation = ankle_separation(frame_lookup[frame_number])

        fact = {
            "frame": frame_number,
            "time_seconds": position["time_seconds"],
            "baseline_statement": None,
            "baseline_distance_meters": baseline_distance,
            "center_statement": None,
            "center_distance_meters": center_distance,
            "ankle_separation_meters": separation,
            "ankle_separation_statement": None,
        }

        if baseline_distance is not None:
            fact["baseline_statement"] = baseline_statement(
                baseline_distance
            )

        if center_distance is not None:
            fact["center_statement"] = center_statement(
                center_distance
            )

        if separation is not None:
            fact["ankle_separation_statement"] = (
                "The estimated projected ankle positions were "
                f"{separation:.2f} m apart."
            )

        facts[anchor_name] = fact

    return facts


def main():
    """create ankle-separation facts from a shot summary."""
    parser = argparse.ArgumentParser(
        description="Create position facts from a shot summary."
    )

    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to a shot_summary.json file.",
    )

    args = parser.parse_args()

    if not args.summary.is_file():
        raise FileNotFoundError(
            f"Shot summary not found: {args.summary}"
        )

    with args.summary.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    facts = build_facts(summary)

    output = {
        "source_summary": str(args.summary),
        "source_tracking_data": summary["tracking_data"],
        "facts": facts,
        "important_limit": (
            "Distances are calculated from projected ankle estimates. "
            "They do not establish foot-ground contact or prove how "
            "wide the player's planted stance was."
        ),
    }

    # keep the earlier movement_facts.json file unchanged.
    output_path = args.summary.parent / "ankle_separation_facts.json"

    with output_path.open("x", encoding="utf-8") as file:
        json.dump(output, file, indent=2, allow_nan=False)

    for anchor_name, fact in facts.items():
        print(f"\n{anchor_name} — frame {fact['frame']}")

        for key in [
            "baseline_statement",
            "center_statement",
            "ankle_separation_statement",
        ]:
            statement = fact[key]

            if statement is None:
                print("  Measurement unavailable.")
            else:
                print(f"  {statement}")

    print(f"\nSaved facts: {output_path}")


if __name__ == "__main__":
    main()