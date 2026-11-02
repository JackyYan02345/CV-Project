"""prepare an evidence-based feedback prompt from a shot summary."""

import argparse
import json
from pathlib import Path

import movement_facts


def parse_arguments():
    """read the path to one saved shot summary."""
    parser = argparse.ArgumentParser(
        description="Prepare an evidence-based tennis feedback prompt."
    )

    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to a shot_summary.json file.",
    )

    return parser.parse_args()


def load_summary(path):
    """load and validate the shot summary."""
    if not path.is_file():
        raise FileNotFoundError(f"Shot summary not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    required_keys = [
        "source_video",
        "tracking_data",
        "event_anchors",
        "quality",
        "player_positions",
        "ankle_movement_phases",
        "limitations",
    ]

    missing = [
        key for key in required_keys
        if key not in summary
    ]

    if missing:
        raise ValueError(
            f"The shot summary is missing these fields: {missing}"
        )

    return summary


def main():
    """prepare the evidence-based feedback prompt."""
    args = parse_arguments()
    summary = load_summary(args.summary)

    positions = summary["player_positions"]

    start_seconds = positions["start"]["time_seconds"]
    end_seconds = positions["after_contact"]["time_seconds"]

    # refresh facts so older summaries include ankle separation.
    calculated_facts = movement_facts.build_facts(summary)

    evidence = {
        "event_anchors": summary["event_anchors"],
        "data_availability": summary["quality"],
        "player_position_estimates": positions,
        "calculated_position_facts": calculated_facts,
        "ankle_movement_phases": summary[
            "ankle_movement_phases"
        ],
        "limitations": summary["limitations"],
    }

    evidence_text = json.dumps(
        evidence,
        indent=2,
        allow_nan=False,
    )

    prompt = f"""Analyze the near player's movement in the attached video.

Focus on the interval from approximately {start_seconds:.2f} seconds
to {end_seconds:.2f} seconds. The near player is the player closest
to the camera.

Use the tracking evidence below as approximate supporting information.
The event frames are review anchors, and their exact timing may differ.
A missing value means the measurement was unavailable.

Respond in four parts:

1. Observations
   Describe up to three things you can see about the player's position
   and movement. For each observation, give the approximate time or
   frame and explain whether it is supported by the video, tracking
   evidence, or both.

2. Uncertainty
   Identify any disagreement between the video and measurements.
   Say when the clip or evidence does not support a conclusion.

3. Coaching question
   Offer at most one question or tentative suggestion for a coach to
   investigate. Explain what additional context would help evaluate it.

4. Evidence used
   List the specific measurements and video moments you relied on.

The ankle-separation values compare the two projected ankle positions
at individual frames. They do not establish whether either foot is
planted. Do not infer a wider or narrower stance from the movement
of only one ankle.

Do not infer the player's intentions, exact foot-ground contact,
shot quality, or tactical success unless the evidence supports it.
Do not assume a movement is correct or incorrect based on one shot.

TRACKING EVIDENCE:

{evidence_text}
"""

    # keep any earlier prompt unchanged.
    output_path = (
        args.summary.parent / "ai_feedback_prompt_with_separation.txt"
    )

    with output_path.open("x", encoding="utf-8") as file:
        file.write(prompt)

    print(f"Source video: {summary['source_video']}")
    print(
        f"Review interval: "
        f"{start_seconds:.2f}–{end_seconds:.2f} seconds"
    )
    print(f"Saved prompt: {output_path}")


if __name__ == "__main__":
    main()