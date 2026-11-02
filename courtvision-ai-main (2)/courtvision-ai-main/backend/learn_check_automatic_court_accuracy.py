"""check automatic court detection against reviewed points."""

import json
import math
from pathlib import Path

import cv2
import numpy as np


VIDEO_PATH = Path("data/raw/validation_rally_01.mp4")

MAPPINGS_PATH = Path(
    "outputs/validation_rally_01_automatic_court_mappings.json"
)

OUTPUT_PATH = Path(
    "outputs/validation_rally_01_court_accuracy_check.json"
)

FRAMES_TO_CHECK = [225, 450, 746]
MAX_DISPLAY_WIDTH = 1280

COURT_WIDTH_METERS = 8.23
COURT_LENGTH_METERS = 23.77

CENTER_X = COURT_WIDTH_METERS / 2
NET_Y = COURT_LENGTH_METERS / 2

# the service lines are 6.40 meters from the net.
POINTS_TO_CLICK = [
    {
        "name": "far_center_service_intersection",
        "court_x_meters": CENTER_X,
        "court_y_meters": NET_Y - 6.40,
    },
    {
        "name": "near_center_service_intersection",
        "court_x_meters": CENTER_X,
        "court_y_meters": NET_Y + 6.40,
    },
]


def read_frame(capture, frame_number):
    """read one zero-based video frame."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = capture.read()

    if not success:
        raise RuntimeError(f"Could not read frame {frame_number}")

    return frame


def click_service_points(frame, frame_number):
    """collect service-line points from the user."""
    height, width = frame.shape[:2]

    scale = min(1.0, MAX_DISPLAY_WIDTH / width)
    display_width = round(width * scale)
    display_height = round(height * scale)

    scale_x = display_width / width
    scale_y = display_height / height

    preview = cv2.resize(frame, (display_width, display_height))
    clicked = []
    window_name = f"Frame {frame_number}: click service intersections"

    def on_mouse(event, x, y, flags, parameter):
        """handle a mouse click."""
        if event != cv2.EVENT_LBUTTONDOWN or len(clicked) >= 2:
            return

        clicked.append([x / scale_x, y / scale_y])

        cv2.circle(preview, (x, y), 7, (0, 255, 255), -1)
        cv2.putText(
            preview,
            str(len(clicked)),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imshow(window_name, preview)

    print(f"\nFrame {frame_number}:")
    print("  1. Click the FAR center-service intersection.")
    print("  2. Click the NEAR center-service intersection.")
    print("  Press Enter after both clicks, or Escape to cancel.")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, preview)

    try:
        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                raise RuntimeError("Accuracy check canceled.")

            if key in (10, 13) and len(clicked) == 2:
                break

            if cv2.getWindowProperty(
                window_name,
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                raise RuntimeError("Accuracy-check window was closed.")
    finally:
        cv2.destroyAllWindows()

    return clicked


def evaluate_clicks(frame_number, clicks, mapping_matrix):
    """compare mapped click positions against known court positions."""
    results = []

    for point_definition, click in zip(POINTS_TO_CLICK, clicks):
        image_point = np.float32([[click]])

        mapped = cv2.perspectiveTransform(
            image_point,
            mapping_matrix,
        )[0][0]

        expected_x = point_definition["court_x_meters"]
        expected_y = point_definition["court_y_meters"]

        error_meters = math.hypot(
            float(mapped[0]) - expected_x,
            float(mapped[1]) - expected_y,
        )

        results.append({
            "frame": frame_number,
            "point_name": point_definition["name"],
            "clicked_image_pixels": click,
            "expected_court_meters": [expected_x, expected_y],
            "mapped_court_meters": [
                float(mapped[0]),
                float(mapped[1]),
            ],
            "difference_meters": error_meters,
        })

    return results


def main():
    """run the script's main workflow."""
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_PATH}\n"
            "Choose a new output path before running again."
        )

    with MAPPINGS_PATH.open("r", encoding="utf-8") as file:
        mapping_data = json.load(file)

    mappings_by_frame = {
        record["frame"]: record
        for record in mapping_data["frames"]
    }

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    all_results = []

    try:
        for frame_number in FRAMES_TO_CHECK:
            mapping_record = mappings_by_frame.get(frame_number)

            if (
                mapping_record is None
                or mapping_record["image_to_court_meters_matrix"] is None
            ):
                raise RuntimeError(
                    f"No usable mapping for frame {frame_number}."
                )

            matrix = np.array(
                mapping_record["image_to_court_meters_matrix"],
                dtype=np.float64,
            )

            frame = read_frame(capture, frame_number)
            clicks = click_service_points(frame, frame_number)

            frame_results = evaluate_clicks(
                frame_number,
                clicks,
                matrix,
            )

            all_results.extend(frame_results)

            for result in frame_results:
                print(
                    f"  {result['point_name']}: "
                    f"{result['difference_meters']:.3f} m difference"
                )

    finally:
        capture.release()

    output_data = {
        "source_video": str(VIDEO_PATH),
        "source_mappings": str(MAPPINGS_PATH),
        "important_limit": (
            "Differences include manual click error and mapping error. "
            "Six clicked points are a small diagnostic sample."
        ),
        "results": all_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("x", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

    print(f"\nSaved accuracy check: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()