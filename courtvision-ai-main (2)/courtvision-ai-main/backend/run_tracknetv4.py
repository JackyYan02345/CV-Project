"""run tracknetv4 separately and record its inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys


def windows_path_to_wsl(path: Path) -> str:
    """convert c:\path\to\file into the usual wsl /mnt/c/path/to/file form."""
    windows_path = PureWindowsPath(path.resolve())
    if not windows_path.drive:
        raise ValueError(f"A Windows drive path is required for WSL: {path}")

    drive_letter = windows_path.drive[0].lower()
    return "/mnt/" + drive_letter + "/" + "/".join(windows_path.parts[1:])


def native_command(args: argparse.Namespace, video_path: Path, output_dir: Path) -> list[str]:
    """build a command for a normal windows or linux tracknetv4 installation."""
    tracknet_root = Path(args.tracknet_root).expanduser().resolve()
    predict_script = tracknet_root / "src" / "predict.py"
    model_weights = Path(args.model_weights).expanduser().resolve()

    missing_paths = [path for path in (predict_script, model_weights) if not path.is_file()]
    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "TrackNetV4 cannot start because these files do not exist:\n"
            f"{missing}\n\nRead backend/TRACKNETV4_SETUP.md before retrying."
        )

    return [
        str(Path(args.tracknet_python).expanduser()),
        str(predict_script),
        "--video_path",
        str(video_path),
        "--model_weights",
        str(model_weights),
        "--output_dir",
        str(output_dir),
        "--queue_length",
        str(args.queue_length),
    ]


def wsl_command(args: argparse.Namespace, video_path: Path, output_dir: Path) -> list[str]:
    """build a command for tracknetv4 installed inside a wsl linux distribution."""
    tracknet_root = args.tracknet_root.rstrip("/")
    if not tracknet_root.startswith("/"):
        raise ValueError(
            "For --runtime wsl, --tracknet-root must be a Linux path such as "
            "/home/your-linux-name/TrackNetV4."
        )
    if not args.model_weights.startswith("/"):
        raise ValueError(
            "For --runtime wsl, --model-weights must be a Linux path such as "
            "/home/your-linux-name/TrackNetV4/models/tennis.keras."
        )

    return [
        "wsl.exe",
        "-d",
        args.wsl_distro,
        "--",
        args.tracknet_python,
        f"{tracknet_root}/src/predict.py",
        "--video_path",
        windows_path_to_wsl(video_path),
        "--model_weights",
        args.model_weights,
        "--output_dir",
        windows_path_to_wsl(output_dir),
        "--queue_length",
        str(args.queue_length),
    ]


def parse_arguments() -> argparse.Namespace:
    """read command-line options."""
    parser = argparse.ArgumentParser(
        description="Run official TrackNetV4 in a separate environment."
    )
    parser.add_argument(
        "--runtime",
        choices=("native", "wsl"),
        default="native",
        help="Where TrackNetV4 is installed. WSL is recommended for its Linux-only environment file.",
    )
    parser.add_argument(
        "--tracknet-root",
        required=True,
        help="Native path to TrackNetV4, or its Linux path when --runtime wsl is used.",
    )
    parser.add_argument(
        "--tracknet-python",
        default=sys.executable,
        help="Python executable inside the TrackNetV4 environment.",
    )
    parser.add_argument(
        "--model-weights",
        required=True,
        help="TrackNetV4 .keras weights path (Linux path with --runtime wsl).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("data/raw/test_rally.mp4"),
        help="Input tennis video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tracknetv4_raw"),
        help="Where TrackNetV4's output video and CSV should be written.",
    )
    parser.add_argument(
        "--queue-length",
        type=int,
        default=0,
        help="History points TrackNetV4 draws itself; zero avoids an ambiguous trail.",
    )
    parser.add_argument(
        "--wsl-distro",
        default="Ubuntu",
        help="WSL distribution name, only used with --runtime wsl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without starting the model.",
    )
    return parser.parse_args()


def main() -> None:
    """run the script's main workflow."""
    args = parse_arguments()
    video_path = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Could not find input video: {video_path}")
    if args.queue_length < 0:
        raise ValueError("--queue-length must be zero or greater.")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = (
        native_command(args, video_path, output_dir)
        if args.runtime == "native"
        else wsl_command(args, video_path, output_dir)
    )

    print("TrackNetV4 command:")
    print(subprocess.list2cmdline(command))
    if args.dry_run:
        print("Dry run only: TrackNetV4 was not started.")
        return

    subprocess.run(command, check=True)

    manifest_path = output_dir / "courtvision_tracknetv4_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "runtime": args.runtime,
                "video": str(video_path),
                "tracknet_root": args.tracknet_root,
                "model_weights": args.model_weights,
                "queue_length": args.queue_length,
                "command": command,
                "next_step": (
                    "Run backend/import_tracknetv4_predictions.py with the CSV "
                    "produced by TrackNetV4."
                ),
            },
            file,
            indent=2,
        )

    print(f"TrackNetV4 finished. Saved run manifest: {manifest_path}")
    print("Next, import its CSV with backend/import_tracknetv4_predictions.py.")


if __name__ == "__main__":
    main()
