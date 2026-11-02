"""inspect near-player foot detections across selected frames."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from court_registration import CourtRegistration


VIDEO_PATH = Path("data/raw/validation_rally_01.mp4")
POSE_MODEL_PATH = Path("yolo11n-pose.pt")
OUTPUT_PATH = Path("outputs/validation_frame_746_feet.jpg")

FRAME_TO_INSPECT = 746
MINIMUM_ANKLE_CONFIDENCE = 0.30

# the pose model uses these indexes for its two ankle keypoints.
ANKLES = {
    "left ankle": 15,
    "right ankle": 16,
}


def read_video_frame(video_path, frame_number):
    """open the video and read one numbered frame."""
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()

        if not success:
            raise RuntimeError(f"Could not read frame {frame_number}")

        return frame

    finally:
        capture.release()


def select_near_player(result, frame_height):
    """choose the player closest to the selected point."""
    if result.boxes is None or result.keypoints is None:
        return None

    if len(result.boxes) == 0 or result.keypoints.conf is None:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.keypoints.conf.cpu().numpy()

    possible_players = []

    for index, box in enumerate(boxes):
        bottom_edge = box[3]

        left_ankle_confidence = confidences[index][15]
        right_ankle_confidence = confidences[index][16]

        has_reliable_ankle = (
            left_ankle_confidence >= MINIMUM_ANKLE_CONFIDENCE
            or right_ankle_confidence >= MINIMUM_ANKLE_CONFIDENCE
        )

        if bottom_edge > frame_height / 2 and has_reliable_ankle:
            possible_players.append(index)

    if not possible_players:
        return None

    # of the remaining people, select the box that reaches lowest.
    return max(possible_players, key=lambda index: boxes[index][3])


def project_to_court(point, image_to_court):
    """convert one image point to an approximate court-plane position."""
    point_array = np.float32([[point]])

    projected = cv2.perspectiveTransform(
        point_array,
        image_to_court,
    )[0][0]

    return float(projected[0]), float(projected[1])


def main():
    """run the script's main workflow."""
    frame = read_video_frame(VIDEO_PATH, FRAME_TO_INSPECT)

    # save drawings on a copy so court detection sees the original frame.
    annotated = frame.copy()
    frame_height = frame.shape[0]

    court_registration = CourtRegistration()
    image_to_court, court_status = court_registration.estimate(frame)

    print(f"Frame: {FRAME_TO_INSPECT}")
    print(f"Court registration: {court_status}")

    pose_model = YOLO(str(POSE_MODEL_PATH))
    result = pose_model.predict(frame, verbose=False)[0]

    player_index = select_near_player(result, frame_height)

    if player_index is None:
        print("No near-player candidate was found.")
    else:
        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        confidences = result.keypoints.conf.cpu().numpy()

        # draw the selected person's bounding box in blue.
        x1, y1, x2, y2 = boxes[player_index].astype(int)

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        print(f"Selected person index: {player_index}")

        for ankle_name, ankle_index in ANKLES.items():
            confidence = float(
                confidences[player_index][ankle_index]
            )

            if confidence < MINIMUM_ANKLE_CONFIDENCE:
                print(
                    f"{ankle_name}: unavailable "
                    f"(confidence {confidence:.2f})"
                )
                continue

            ankle_point = keypoints[player_index][ankle_index]
            x_pixel = float(ankle_point[0])
            y_pixel = float(ankle_point[1])

            # yellow circle marks the ankle estimated by the pose model.
            center = (round(x_pixel), round(y_pixel))

            cv2.circle(
                annotated,
                center,
                7,
                (0, 255, 255),
                -1,
            )

            cv2.putText(
                annotated,
                ankle_name,
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            print(
                f"{ankle_name}: image position "
                f"({x_pixel:.1f}, {y_pixel:.1f}), "
                f"confidence {confidence:.2f}"
            )

            if image_to_court is None:
                print("  Court position unavailable.")
                continue

            court_x, court_y = project_to_court(
                [x_pixel, y_pixel],
                image_to_court,
            )

            print(
                f"  Approximate court position: "
                f"x={court_x:.2f} m, y={court_y:.2f} m"
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_PATH), annotated):
        raise RuntimeError(f"Could not save image: {OUTPUT_PATH}")

    print(f"Saved inspection image: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()