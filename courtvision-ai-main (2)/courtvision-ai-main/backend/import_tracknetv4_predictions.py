"""import direct tracknetv4 ball observations into courtvision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


DIRECT_PROVENANCE = "direct_temporal_model_observation"


def normalise_column_name(column_name: str) -> str:
    """make csv heading comparisons tolerant of capitalisation and spaces."""
    return "".join(character for character in column_name.lower() if character.isalnum())


def parse_visibility(value: str, line_number: int) -> bool:
    """parse common boolean and 0/1 visibility values without guessing."""
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "visible"}:
        return True
    if normalised in {"0", "false", "no", "notvisible", ""}:
        return False
    raise ValueError(f"CSV line {line_number}: unsupported Visibility value {value!r}.")


def parse_integer(value: str, field_name: str, line_number: int) -> int:
    """parse a whole-number frame value."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(
            f"CSV line {line_number}: {field_name} must be a number, got {value!r}."
        ) from error
    if not parsed.is_integer():
        raise ValueError(
            f"CSV line {line_number}: {field_name} must be a whole frame number, got {value!r}."
        )
    return int(parsed)


def parse_float(value: str, field_name: str, line_number: int) -> float:
    """parse a numeric value from the csv."""
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"CSV line {line_number}: {field_name} must be a number, got {value!r}."
        ) from error


def load_tracknet_rows(csv_path: Path) -> list[dict]:
    """read tracknetv4's documented frame, visibility, x, y csv schema."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("The predictions CSV is empty or has no header row.")

        original_names = {
            normalise_column_name(name): name for name in reader.fieldnames if name
        }
        required_names = {"frame", "visibility", "x", "y"}
        missing_names = sorted(required_names - original_names.keys())
        if missing_names:
            raise ValueError(
                "CSV must have Frame, Visibility, X, and Y columns. Missing: "
                + ", ".join(missing_names)
            )

        rows = []
        for ordinal, row in enumerate(reader):
            line_number = ordinal + 2
            visible = parse_visibility(row[original_names["visibility"]], line_number)
            rows.append(
                {
                    "ordinal": ordinal,
                    "csv_frame": parse_integer(
                        row[original_names["frame"]], "Frame", line_number
                    ),
                    "visible": visible,
                    "x": parse_float(row[original_names["x"]], "X", line_number),
                    "y": parse_float(row[original_names["y"]], "Y", line_number),
                }
            )

    if not rows:
        raise ValueError("The predictions CSV has a header but no prediction rows.")
    return rows


def choose_frame_index_mode(rows: list[dict], source_frame_count: int, requested_mode: str) -> str:
    """choose how csv rows map to source frames."""
    if requested_mode != "auto":
        return requested_mode

    frame_numbers = [row["csv_frame"] for row in rows]
    has_duplicates = len(set(frame_numbers)) != len(frame_numbers)
    # repeated ids cannot identify distinct frames, so keep csv row order.
    if has_duplicates or len(rows) == source_frame_count:
        return "row-order"
    return "csv"


def make_frame_lookup(rows: list[dict], mode: str, source_frame_count: int) -> dict[int, dict]:
    """map exactly one model row to each source frame without silent overwrites."""
    lookup: dict[int, dict] = {}
    if mode == "row-order":
        if len(rows) > source_frame_count:
            raise ValueError(
                f"CSV has {len(rows)} rows but the source video only has "
                f"{source_frame_count} frames. Refusing to shift observations."
            )
        for row in rows:
            lookup[row["ordinal"]] = row
        return lookup

    # reject duplicate frame ids instead of silently replacing an observation.
    for row in rows:
        frame_number = row["csv_frame"]
        if frame_number < 0 or frame_number >= source_frame_count:
            raise ValueError(
                f"CSV Frame {frame_number} is outside the source range "
                f"0 to {source_frame_count - 1}."
            )
        if frame_number in lookup:
            raise ValueError(
                "CSV mode requires unique Frame values. Use --frame-index-mode "
                "row-order for repeated Frame values."
            )
        lookup[frame_number] = row
    return lookup


def draw_observation(frame, point: tuple[int, int]) -> None:
    """draw the current direct observation only; there is no predicted trail."""
    cv2.circle(frame, point, 8, (0, 0, 0), -1)
    cv2.circle(frame, point, 6, (0, 0, 255), -1)
    cv2.circle(frame, point, 9, (255, 255, 255), 1)
    label_origin = (point[0] + 12, max(24, point[1] - 12))
    label = "ball: direct temporal-model observation"
    cv2.putText(
        frame,
        label,
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


def parse_arguments() -> argparse.Namespace:
    """read command-line options."""
    parser = argparse.ArgumentParser(
        description="Import TrackNetV4 observations and create CourtVision-owned outputs."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument(
        "--tracker-name",
        default="TrackNetV4",
        help="Human-readable source-model name saved in the output metadata.",
    )
    parser.add_argument("--video", type=Path, default=Path("data/raw/test_rally.mp4"))
    parser.add_argument(
        "--frame-index-mode",
        choices=("auto", "row-order", "csv"),
        default="auto",
        help="How CSV rows align with source frames. Auto safely handles repeated IDs.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("outputs/tracknetv4_ball_tracking.mp4"),
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=Path("outputs/tracknetv4_ball_tracking_data.json"),
    )
    return parser.parse_args()


def main() -> None:
    """run the script's main workflow."""
    args = parse_arguments()
    csv_path = args.predictions_csv.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    output_video_path = args.output_video.expanduser().resolve()
    output_data_path = args.output_data.expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"Could not find TrackNetV4 CSV: {csv_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Could not find source video: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the source video: {video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = capture.get(cv2.CAP_PROP_FPS)
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_width <= 0 or frame_height <= 0 or frame_rate <= 0 or source_frame_count <= 0:
        raise RuntimeError("The source video does not report usable dimensions, FPS, and frame count.")

    rows = load_tracknet_rows(csv_path)
    selected_mode = choose_frame_index_mode(rows, source_frame_count, args.frame_index_mode)
    frame_lookup = make_frame_lookup(rows, selected_mode, source_frame_count)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    output_frames = []
    visible_model_rows = 0
    accepted_observations = 0
    rejected_outside_frame = 0
    current_missing_run = 0
    longest_missing_run = 0

    for frame_number in range(source_frame_count):
        success, frame = capture.read()
        if not success:
            break

        row = frame_lookup.get(frame_number)
        accepted_point = None
        rejection_reason = None
        if row and row["visible"]:
            visible_model_rows += 1
            candidate = (round(row["x"]), round(row["y"]))
            if 0 <= candidate[0] < frame_width and 0 <= candidate[1] < frame_height:
                accepted_point = candidate
                accepted_observations += 1
                current_missing_run = 0
                draw_observation(frame, accepted_point)
            else:
                rejected_outside_frame += 1
                rejection_reason = "model_point_outside_source_frame"

        if accepted_point is None:
            current_missing_run += 1
            longest_missing_run = max(longest_missing_run, current_missing_run)

        ball_record = {
            "visible": accepted_point is not None,
            "provenance": DIRECT_PROVENANCE if accepted_point else None,
            "image_point_pixels": list(accepted_point) if accepted_point else None,
        }
        if row:
            ball_record["csv_visibility"] = row["visible"]
            ball_record["csv_frame"] = row["csv_frame"]
        if rejection_reason:
            ball_record["rejection_reason"] = rejection_reason

        output_frames.append(
            {
                "frame": frame_number,
                "time_seconds": round(frame_number / frame_rate, 6),
                "ball": ball_record,
            }
        )
        writer.write(frame)

    capture.release()
    writer.release()

    processed_frames = len(output_frames)
    tracking_data = {
        "schema_version": 1,
        "source": {
            "tracker": args.tracker_name,
            "predictions_csv": str(csv_path),
            "video": str(video_path),
            "frame_index_mode": selected_mode,
            "coordinate_space": "source_video_pixels",
            "important_limit": (
                "Only direct temporal-model observations are saved. This file contains "
                "no interpolation, occlusion prediction, or ground-plane ball coordinates."
            ),
        },
        "video": {
            "width_pixels": frame_width,
            "height_pixels": frame_height,
            "fps": frame_rate,
            "processed_frames": processed_frames,
        },
        "quality_report": {
            "csv_rows": len(rows),
            "model_reported_visible_rows": visible_model_rows,
            "accepted_direct_observations": accepted_observations,
            "accepted_observation_coverage": round(
                accepted_observations / processed_frames, 6
            ) if processed_frames else 0.0,
            "rejected_points_outside_source_frame": rejected_outside_frame,
            "longest_run_without_accepted_observation_frames": longest_missing_run,
            "longest_run_without_accepted_observation_seconds": round(
                longest_missing_run / frame_rate, 6
            ),
        },
        "frames": output_frames,
    }
    with open(output_data_path, "w", encoding="utf-8") as file:
        json.dump(tracking_data, file, indent=2)

    print(f"Frame-index mode: {selected_mode}")
    print(f"Accepted direct observations: {accepted_observations}/{processed_frames}")
    print(
        "Longest run without an accepted observation: "
        f"{longest_missing_run} frames ({longest_missing_run / frame_rate:.2f} seconds)"
    )
    print(f"Saved annotated video: {output_video_path}")
    print(f"Saved provenance-aware tracking data: {output_data_path}")


if __name__ == "__main__":
    main()
