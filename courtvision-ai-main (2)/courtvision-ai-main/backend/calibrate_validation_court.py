"""calibrate the court using a selected validation frame."""

import json
from pathlib import Path

import cv2
import numpy as np


VIDEO_PATH = Path("data/raw/validation_rally_01.mp4")

# frame 746 shows the court clearly in the image you sent.
FRAME_NUMBER = 450

OUTPUT_DIR = Path("outputs/validation_rally_01_calibration_frame_450")
CALIBRATION_PATH = OUTPUT_DIR / "image_to_court_homography.json"
CORNERS_PREVIEW_PATH = OUTPUT_DIR / "clicked_corners.jpg"
VERIFICATION_PATH = OUTPUT_DIR / "court_line_verification.jpg"

COURT_WIDTH_METERS = 8.23
COURT_LENGTH_METERS = 23.77

# display a smaller image if the original is wider than the screen.
# click coordinates are converted back to full-resolution video pixels.
MAX_DISPLAY_WIDTH = 1280

WINDOW_NAME = "Validation court calibration"


def read_video_frame():
    """read the selected zero-based frame from the validation video."""
    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
        success, frame = capture.read()
    finally:
        capture.release()

    if not success:
        raise RuntimeError(f"Could not read frame {FRAME_NUMBER}")

    return frame


def select_court_corners(frame):
    """collect the four court corners from the user."""
    frame_height, frame_width = frame.shape[:2]

    scale = min(1.0, MAX_DISPLAY_WIDTH / frame_width)

    display_width = round(frame_width * scale)
    display_height = round(frame_height * scale)

    display_image = cv2.resize(
        frame,
        (display_width, display_height),
    )

    # use the actual resized dimensions, since rounding can make the
    # horizontal and vertical scale factors differ slightly.
    scale_x = display_width / frame_width
    scale_y = display_height / frame_height

    clicked_points = []

    def on_mouse(event, x, y, flags, parameter):
        """handle a mouse click."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(clicked_points) >= 4:
            return

        # convert the click from display pixels back to original video pixels.
        original_x = x / scale_x
        original_y = y / scale_y

        clicked_points.append([original_x, original_y])

        point_number = len(clicked_points)

        cv2.circle(
            display_image,
            (x, y),
            7,
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            display_image,
            str(point_number),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, display_image)

        print(
            f"Point {point_number}: "
            f"({original_x:.1f}, {original_y:.1f})"
        )

    print("\nClick the INNER singles-court corners in this order:")
    print("1. Far-left corner")
    print("2. Far-right corner")
    print("3. Near-right corner")
    print("4. Near-left corner")
    print("\nAfter four clicks, press Enter to save.")
    print("Press Escape to cancel and start over.")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    cv2.imshow(WINDOW_NAME, display_image)

    try:
        while True:
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                raise RuntimeError("Calibration canceled.")

            if key in (10, 13) and len(clicked_points) == 4:
                break

            # closing the window also cancels rather than hanging.
            if cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE,
            ) < 1:
                raise RuntimeError("Calibration window was closed.")

    finally:
        cv2.destroyAllWindows()

    return clicked_points


def create_corner_preview(frame, image_points):
    """draw the selected corners on a copy of the original frame."""
    preview = frame.copy()

    for index, point in enumerate(image_points, start=1):
        x, y = round(point[0]), round(point[1])

        cv2.circle(preview, (x, y), 10, (0, 255, 255), -1)
        cv2.putText(
            preview,
            str(index),
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

    return preview


def create_verification_image(frame, image_to_meters):
    """draw the selected court and its detected lines."""
    verification = frame.copy()

    meters_to_image = np.linalg.inv(image_to_meters)

    net_y = COURT_LENGTH_METERS / 2
    far_service_y = net_y - 6.40
    near_service_y = net_y + 6.40
    center_x = COURT_WIDTH_METERS / 2

    # all these lines are on the ground plane.
    court_lines = [
        ((0, 0), (COURT_WIDTH_METERS, 0)),
        ((COURT_WIDTH_METERS, 0),
         (COURT_WIDTH_METERS, COURT_LENGTH_METERS)),
        ((COURT_WIDTH_METERS, COURT_LENGTH_METERS),
         (0, COURT_LENGTH_METERS)),
        ((0, COURT_LENGTH_METERS), (0, 0)),
        ((0, far_service_y), (COURT_WIDTH_METERS, far_service_y)),
        ((0, near_service_y), (COURT_WIDTH_METERS, near_service_y)),
        ((center_x, far_service_y), (center_x, near_service_y)),
    ]

    for start, end in court_lines:
        points_meters = np.float32([[start, end]])

        points_pixels = cv2.perspectiveTransform(
            points_meters,
            meters_to_image,
        )[0]

        start_pixel = tuple(np.round(points_pixels[0]).astype(int))
        end_pixel = tuple(np.round(points_pixels[1]).astype(int))

        cv2.line(
            verification,
            start_pixel,
            end_pixel,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

    return verification


def main():
    """run the script's main workflow."""
    output_paths = [
        CALIBRATION_PATH,
        CORNERS_PREVIEW_PATH,
        VERIFICATION_PATH,
    ]

    existing_paths = [path for path in output_paths if path.exists()]

    if existing_paths:
        raise FileExistsError(
            "A validation calibration output already exists: "
            f"{existing_paths[0]}\n"
            "Choose a new output directory before running again."
        )

    frame = read_video_frame()
    image_points = select_court_corners(frame)

    # the click order corresponds to these real court positions.
    court_points_meters = np.float32([
        [0, 0],
        [COURT_WIDTH_METERS, 0],
        [COURT_WIDTH_METERS, COURT_LENGTH_METERS],
        [0, COURT_LENGTH_METERS],
    ])

    image_to_meters = cv2.getPerspectiveTransform(
        np.float32(image_points),
        court_points_meters,
    )

    if not np.isfinite(image_to_meters).all():
        raise ValueError("The calculated homography is not valid.")

    corner_preview = create_corner_preview(frame, image_points)
    verification = create_verification_image(frame, image_to_meters)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(CORNERS_PREVIEW_PATH), corner_preview):
        raise RuntimeError("Could not save the corner preview.")

    if not cv2.imwrite(str(VERIFICATION_PATH), verification):
        raise RuntimeError("Could not save the verification image.")

    calibration_data = {
        "source_video": str(VIDEO_PATH),
        "calibration_frame": FRAME_NUMBER,
        "court_width_meters": COURT_WIDTH_METERS,
        "court_length_meters": COURT_LENGTH_METERS,
        "image_corner_points_pixels": image_points,
        "image_to_court_meters_matrix": image_to_meters.tolist(),
    }

    with CALIBRATION_PATH.open("x", encoding="utf-8") as file:
        json.dump(calibration_data, file, indent=2)

    print(f"\nSaved calibration: {CALIBRATION_PATH}")
    print(f"Saved corner preview: {CORNERS_PREVIEW_PATH}")
    print(f"Saved verification image: {VERIFICATION_PATH}")


if __name__ == "__main__":
    main()