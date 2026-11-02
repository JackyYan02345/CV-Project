"""build bounce-review candidates using the court visible in each candidate frame."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

import learn_bounce_candidates as bounce
from court_registration import CourtRegistration, LENGTH, WIDTH


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING = PROJECT / "outputs/racketvision_ball_tracking_data_validation_rally_01.json"


def calibrate_candidates(video, video_info, candidates):
    """decode in order. only run the court model when we need a court position."""
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open the video recorded in the tracking file: {video}")
        for field, prop in [("width_pixels", cv2.CAP_PROP_FRAME_WIDTH),
                            ("height_pixels", cv2.CAP_PROP_FRAME_HEIGHT),
                            ("processed_frames", cv2.CAP_PROP_FRAME_COUNT)]:
            if int(capture.get(prop)) != video_info[field]:
                raise ValueError(f"Video and tracking data disagree on {field}.")
        if abs(capture.get(cv2.CAP_PROP_FPS) - video_info["fps"]) > .01:
            raise ValueError("Video and tracking data disagree on frame rate.")

        registration = CourtRegistration() if candidates else None
        mapped, unavailable, outside = [], [], []
        next_frame = 0
        for candidate in sorted(candidates, key=lambda row: row["candidate_frame"]):
            index = candidate["candidate_frame"]
            if not next_frame <= index < video_info["processed_frames"]:
                raise ValueError(f"Invalid or duplicate candidate frame: {index}")
            while next_frame <= index:
                if not capture.grab():
                    raise ValueError(f"Video ended while reading frame {next_frame}.")
                next_frame += 1
            success, image = capture.retrieve()
            if not success:
                raise ValueError(f"Cannot decode frame {index}.")
            matrix, reason = registration.estimate(image)
            if matrix is None:
                unavailable.append({**candidate, "calibration_status": reason})
                continue
            accepted = bounce.filter_candidates_to_court([candidate], matrix, WIDTH, LENGTH)
            if accepted:
                mapped.extend(accepted)
            else:
                outside.append(index)
        return mapped, unavailable, outside
    finally:
        capture.release()


def main():
    """run the script's main workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs")
    args = parser.parse_args()
    with args.tracking.open(encoding="utf-8") as source:
        tracking = json.load(source)
    if tracking["source"]["coordinate_space"] != "source_video_pixels":
        raise ValueError("Expected ball observations in source video pixels.")
    video = Path(tracking["source"]["video"])
    if not video.is_absolute():
        video = PROJECT / video
    observations = bounce.load_direct_ball_observations(args.tracking)
    lookup = bounce.create_observation_lookup(observations)
    motions = bounce.calculate_direct_motion(observations)
    limit = bounce.calculate_motion_continuity_limit(motions)
    turns = bounce.calculate_direction_changes(motions)
    raw = bounce.build_raw_turn_candidates(turns, lookup)

    mapped, unavailable, outside = calibrate_candidates(video, tracking["video"], raw)
    continuous = bounce.filter_candidates_by_motion_continuity(mapped, limit)
    final = bounce.remove_nearby_duplicate_candidates(continuous)
    for candidate in final:
        candidate.update(manual_review_label=None, review_status="unreviewed")

    summary = {
        "direct_ball_observations": len(observations),
        "trusted_motion_records": len(motions),
        "motion_continuity_limit_pixels_per_second": limit,
        "raw_turn_candidates": len(raw),
        "frames_with_supported_court": len(raw) - len(unavailable),
        "frames_without_supported_court": len(unavailable),
        "outside_court_tolerance": len(outside),
        "bounce_review_candidate_count": len(final),
    }
    data = {
        "schema_version": 2,
        "source": {"ball_tracking_file": str(args.tracking.resolve()),
                   "source_video_file": str(video.resolve()),
                   "court_registration": "per_candidate_frame_painted_lines_v1"},
        "summary": summary,
        "important_limit": "Court coordinates assume ground contact. Review the video to establish event type and timing.",
        "candidates": final,
        "unmapped_candidates_for_review": unavailable,
        "outside_court_candidate_frames": outside,
    }

    # stage both outputs before publishing. earlier hand-labelled files use
    # different names and remain available for comparing frame numbers.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem + "_registered_bounce_review"
    with TemporaryDirectory(dir=args.output_dir) as directory:
        report = Path(directory) / (stem + ".json")
        movie = Path(directory) / (stem + ".mp4")
        report.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        bounce.create_bounce_review_video(video, movie, lookup, final + unavailable)
        movie.replace(args.output_dir / movie.name)
        report.replace(args.output_dir / report.name)
    print(json.dumps(summary, indent=2))
    print(f"Saved review in {args.output_dir.resolve()} ({stem}).")


if __name__ == "__main__":
    main()
