"""track near-player feet and estimate court positions."""

import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from court_registration import CourtRegistration
from learn_inspect_player_feet import (
    ANKLES,
    MINIMUM_ANKLE_CONFIDENCE,
    POSE_MODEL_PATH,
    VIDEO_PATH,
    project_to_court,
    select_near_player,
)


# this range includes 10 frames before and after frame 746.
# both endpoints are included.
START_FRAME = 430
END_FRAME = 475

OUTPUT_VIDEO = Path("outputs/validation_shot_449_feet.mp4")
OUTPUT_DATA = Path("outputs/validation_shot_449_feet.json")


def main():
    """run the script's main workflow."""
    if END_FRAME < START_FRAME:
        raise ValueError("END_FRAME must be at least START_FRAME.")

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    writer = None
    records = []

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("The video has invalid dimensions or FPS.")

        OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(OUTPUT_VIDEO),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not create video: {OUTPUT_VIDEO}")

        pose_model = YOLO(str(POSE_MODEL_PATH))
        court_registration = CourtRegistration()

        # seek once, then read consecutive frames in order.
        capture.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

        for frame_number in range(START_FRAME, END_FRAME + 1):
            success, frame = capture.read()

            if not success:
                raise RuntimeError(
                    f"Could not read frame {frame_number}."
                )

            annotated = frame.copy()

            image_to_court, court_status = (
                court_registration.estimate(frame)
            )

            result = pose_model.predict(
                frame,
                imgsz=1280,
                verbose=False,
            )[0]
            player_index = select_near_player(result, height)

            frame_record = {
                "frame": frame_number,
                "time_seconds": frame_number / fps,
                "court_registration_status": court_status,
                "player_detected": player_index is not None,
                "ankles": {},
            }

            if player_index is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()

                x1, y1, x2, y2 = boxes[player_index].astype(int)

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2,
                )

                for ankle_name, ankle_index in ANKLES.items():
                    confidence = float(
                        confidences[player_index][ankle_index]
                    )

                    ankle_record = {
                        "confidence": confidence,
                        "image_position_pixels": None,
                        "court_position_meters": None,
                    }

                    if confidence >= MINIMUM_ANKLE_CONFIDENCE:
                        point = keypoints[player_index][ankle_index]

                        x_pixel = float(point[0])
                        y_pixel = float(point[1])

                        ankle_record["image_position_pixels"] = [
                            x_pixel,
                            y_pixel,
                        ]

                        if image_to_court is not None:
                            court_x, court_y = project_to_court(
                                [x_pixel, y_pixel],
                                image_to_court,
                            )

                            ankle_record["court_position_meters"] = {
                                "x": court_x,
                                "y": court_y,
                            }

                        center = (round(x_pixel), round(y_pixel))

                        cv2.circle(
                            annotated,
                            center,
                            7,
                            (0, 255, 255),
                            -1,
                        )

                    frame_record["ankles"][ankle_name] = ankle_record

            cv2.putText(
                annotated,
                f"Frame {frame_number} | court: {court_status}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            writer.write(annotated)
            records.append(frame_record)

            print(
                f"Processed frame {frame_number}: "
                f"court={court_status}, "
                f"player_detected={player_index is not None}"
            )

    finally:
        capture.release()

        if writer is not None:
            writer.release()

    output = {
        "source_video": str(VIDEO_PATH),
        "start_frame": START_FRAME,
        "end_frame": END_FRAME,
        "fps": fps,
        "important_limit": (
            "Ankle coordinates are estimates projected onto the court "
            "plane. Foot-ground contact and player identity require review."
        ),
        "frames": records,
    }

    with OUTPUT_DATA.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, allow_nan=False)

    print(f"\nSaved review video: {OUTPUT_VIDEO}")
    print(f"Saved ankle data: {OUTPUT_DATA}")


if __name__ == "__main__":
    main()