"""analyze a manually reviewed tennis shot."""

import argparse
import json
import movement_facts
from pathlib import Path

import learn_track_player_feet as tracking
import learn_compare_movement_phases as movement
import learn_summarize_player_position as position


def parse_arguments():
    """read the video, frame numbers, and output folder."""
    parser = argparse.ArgumentParser(
        description="Analyze one manually reviewed shot."
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the source video.",
    )

    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--bounce", type=int, required=True)
    parser.add_argument("--contact", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New folder where this shot's results will be saved.",
    )

    args = parser.parse_args()

    if not (
        0 <= args.start
        < args.bounce
        < args.contact
        < args.end
    ):
        parser.error(
            "Frame numbers must satisfy "
            "0 <= start < bounce < contact < end."
        )

    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")

    if args.output_dir.exists():
        parser.error(
            f"Output folder already exists: {args.output_dir}. "
            "Choose a new folder name."
        )

    return args


def main():
    """run the script's main workflow."""
    args = parse_arguments()

    # a separate folder keeps this shot's files together and protects
    # earlier results from accidental replacement.
    args.output_dir.mkdir(parents=True)

    tracking_video = args.output_dir / "feet_review.mp4"
    tracking_data = args.output_dir / "feet_tracking.json"
    summary_path = args.output_dir / "shot_summary.json"

    # supply this shot's settings to the existing tracking script.
    # importing that script does not start its main() automatically.
    tracking.VIDEO_PATH = args.video
    tracking.START_FRAME = args.start
    tracking.END_FRAME = args.end
    tracking.OUTPUT_VIDEO = tracking_video
    tracking.OUTPUT_DATA = tracking_data

    print("\nTracking ankles and checking the court...")
    tracking.main()

    # read the records produced for this exact shot.
    with tracking_data.open("r", encoding="utf-8") as file:
        tracking_result = json.load(file)

    frame_lookup = {
        frame["frame"]: frame
        for frame in tracking_result["frames"]
    }

    anchors = {
        "start": args.start,
        "near_bounce": args.bounce,
        "near_contact": args.contact,
        "after_contact": args.end,
    }

    missing = [
        frame_number
        for frame_number in anchors.values()
        if frame_number not in frame_lookup
    ]

    if missing:
        raise ValueError(
            f"Tracking output is missing required frames: {missing}"
        )

    # reuse your position function for all four selected frames.
    positions = {}

    for name, frame_number in anchors.items():
        positions[name] = position.describe_position(
            frame_lookup[frame_number]
        )

    # reuse your existing ankle movement calculation.
    phases = {
        "approach_to_bounce": movement.compare_phase(
            frame_lookup,
            args.start,
            args.bounce,
        ),
        "bounce_to_contact": movement.compare_phase(
            frame_lookup,
            args.bounce,
            args.contact,
        ),
        "after_contact": movement.compare_phase(
            frame_lookup,
            args.contact,
            args.end,
        ),
    }

    contact_midpoint = positions["near_contact"][
        "ankle_midpoint_meters"
    ]
    end_midpoint = positions["after_contact"][
        "ankle_midpoint_meters"
    ]

    after_contact_displacement = position.displacement(
        contact_midpoint,
        end_midpoint,
    )

    # count missing information across the complete window, not only
    # at the four frames used for the position summary.
    frames = tracking_result["frames"]

    quality = {
        "frames_processed": len(frames),
        "frames_without_selected_player": sum(
            not frame["player_detected"]
            for frame in frames
        ),
        "frames_without_court_mapping": sum(
            frame["court_registration_status"] != "paint_verified"
            for frame in frames
        ),
        "frames_without_both_ankle_positions": sum(
            position.ankle_midpoint(frame) is None
            for frame in frames
        ),
    }

    summary = {
        "source_video": str(args.video),
        "tracking_data": str(tracking_data),
        "review_video": str(tracking_video),
        "event_anchors": anchors,
        "event_note": (
            "The bounce and contact frames are supplied review "
            "anchors. Exact event timing may differ."
        ),
        "quality": quality,
        "player_positions": positions,
        "ankle_movement_phases": phases,
        "after_contact_midpoint_displacement_meters": (
            after_contact_displacement
        ),
        "limitations": [
            "The player is selected independently in each frame. "
            "Review the video to check player identity.",

            "Court positions are projections of estimated ankles. "
            "They do not establish foot-ground contact.",

            "Displacement compares endpoints and does not measure "
            "the complete path traveled.",

            "These measurements do not determine whether the "
            "player's movement or shot choice was correct.",
        ],
    }

    # generate position statements directly from the saved measurements.
    summary["movement_facts"] = movement_facts.build_facts(summary)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    print("\nMovement summary")
    movement.print_phase(
        "Approaching the bounce",
        phases["approach_to_bounce"],
    )
    movement.print_phase(
        "Near bounce to near contact",
        phases["bounce_to_contact"],
    )
    movement.print_phase(
        "After the contact anchor",
        phases["after_contact"],
    )

    print("\nData availability")
    for name, value in quality.items():
        print(f"  {name}: {value}")

    print(f"\nSaved review video: {tracking_video}")
    print(f"Saved per-frame data: {tracking_data}")
    print(f"Saved shot summary: {summary_path}")


if __name__ == "__main__":
    main()