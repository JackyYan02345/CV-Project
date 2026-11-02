"""run racketvision balltrack in wsl and import direct observations."""

from __future__ import annotations

import argparse
from pathlib import Path, PureWindowsPath
import subprocess
import sys


def windows_path_to_wsl(path: Path) -> str:
    """
    convert an absolute windows path into the equivalent wsl mount path.
    """
    windows_path = PureWindowsPath(path.resolve())

    if not windows_path.drive:
        raise ValueError(
            f"A Windows drive path is required for WSL: {path}"
        )

    return (
        "/mnt/"
        + windows_path.drive[0].lower()
        + "/"
        + "/".join(windows_path.parts[1:])
    )


def parse_arguments() -> argparse.Namespace:
    """
    define command-line options for one independent racketvision run.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run RacketVision BallTrack and save CourtVision-owned outputs."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/raw/test_rally.mp4"),
    )
    parser.add_argument(
        "--racketvision-root",
        default="/home/dilos/RacketVision",
    )
    parser.add_argument(
        "--racketvision-python",
        default=(
            "/home/dilos/miniforge3/envs/racketvision/bin/python"
        ),
    )
    parser.add_argument("--wsl-distro", default="Ubuntu")

    # these identify the prepared input folders used inside racketvision.
    parser.add_argument("--match-id", default="courtvision_match")
    parser.add_argument("--rally-id", default="test_rally")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)

    # these two options are new. they prevent different rallies from
    # overwriting each other's courtvision tracking files.
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("outputs/racketvision_ball_tracking.mp4"),
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=Path(
            "outputs/racketvision_ball_tracking_data.json"
        ),
    )

    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    """run the script's main workflow."""
    args = parse_arguments()

    video_path = args.video.expanduser().resolve()
    output_video_path = args.output_video.expanduser().resolve()
    output_data_path = args.output_data.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Could not find input video: {video_path}"
        )

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "--threshold must be strictly between 0 and 1."
        )

    if not args.racketvision_root.startswith("/"):
        raise ValueError(
            "--racketvision-root must be a WSL path, such as "
            "/home/dilos/RacketVision."
        )

    # create output folders if they do not already exist.
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_data_path.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]

    # this directory holds racketvision's prepared video frames.
    input_root = project_root / "outputs" / "racketvision_input"

    prepare_command = [
        sys.executable,
        str(
            project_root
            / "backend"
            / "prepare_racketvision_balltrack_input.py"
        ),
        "--video",
        str(video_path),
        "--output-root",
        str(input_root),
        "--match-id",
        args.match_id,
        "--rally-id",
        args.rally_id,
    ]

    prepared_frame_dir = (
        input_root
        / "tennis"
        / "all"
        / args.match_id
        / "frame"
        / args.rally_id
    )

    prepared_median = (
        input_root
        / "tennis"
        / "all"
        / args.match_id
        / "median.npz"
    )

    # reuse frames only when this exact match/rally preparation already exists.
    if (
        prepared_frame_dir.is_dir()
        and any(prepared_frame_dir.iterdir())
        and prepared_median.is_file()
    ):
        prepare_command.append("--reuse-existing")

    results_csv = (
        input_root
        / "tennis"
        / "pred_ball"
        / args.match_id
        / args.rally_id
        / "results.csv"
    )

    remote_command = [
        args.racketvision_python,
        "inference.py",
        "--cfg",
        "configs/tracknetv3_base.py",
        "--ckpt",
        "checkpoints/balltrack_best.pth",
        "--data_root",
        windows_path_to_wsl(input_root),
        "--sport",
        "tennis",
        "--split",
        "test",
        "--device",
        "cuda",
        "--batchsize",
        str(args.batch_size),
        "--thre",
        str(args.threshold),
    ]

    wsl_command = [
        "wsl.exe",
        "-d",
        args.wsl_distro,
        "--",
        "bash",
        "-lc",
        "cd "
        + subprocess.list2cmdline(
            [args.racketvision_root + "/source/BallTrack"]
        )
        + " && "
        + subprocess.list2cmdline(remote_command),
    ]

    # this importer converts racketvision's csv predictions into courtvision's
    # visible-direct-observation video and json format.
    import_command = [
        sys.executable,
        str(
            project_root
            / "backend"
            / "import_tracknetv4_predictions.py"
        ),
        "--predictions-csv",
        str(results_csv),
        "--video",
        str(video_path),
        "--tracker-name",
        "RacketVision BallTrack (TrackNetV3)",
        "--output-video",
        str(output_video_path),
        "--output-data",
        str(output_data_path),
    ]

    print("Prepare command:")
    print(subprocess.list2cmdline(prepare_command))

    print("RacketVision command:")
    print(subprocess.list2cmdline(wsl_command))

    print("CourtVision output video:")
    print(output_video_path)

    print("CourtVision output data:")
    print(output_data_path)

    if args.dry_run:
        print(
            "Dry run only: no frames were extracted and no model was started."
        )
        return

    # step 1: extract frames into racketvision's expected folder layout.
    subprocess.run(prepare_command, check=True)

    # step 2: run the temporal ball-tracking model inside wsl.
    subprocess.run(wsl_command, check=True)

    if not results_csv.is_file():
        raise FileNotFoundError(
            "RacketVision finished but did not create:\n"
            f"{results_csv}\n\n"
            "Confirm its checkpoint and dependencies are installed, "
            "then inspect the WSL error above."
        )

    # step 3: import only direct visible observations into courtvision output.
    subprocess.run(import_command, check=True)

    print(f"Saved direct-observation video: {output_video_path}")
    print(f"Saved tracking data: {output_data_path}")


if __name__ == "__main__":
    main()