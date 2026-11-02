"""inspect pose keypoints for both players."""

from pathlib import Path

import cv2
import json
import math
from ultralytics import YOLO


# the pose model predicts people plus body keypoints.
MODEL_PATH = Path("yolo11n-pose.pt")

BALL_TRACKING_PATH = Path("outputs/racketvision_ball_tracking_data.json")

# our original tennis video.
VIDEO_PATH = Path("data/raw/test_rally.mp4")

# frame 346 was manually reviewed as a racket-contact event.
# we inspect it because player-wrist information matters most there.
FRAME_TO_INSPECT = 346

# save a visual result so we can check what the pose model found.
OUTPUT_IMAGE_PATH = Path(
    "outputs/pose_inspection_frame_346.jpg"
)

# yolo pose uses the standard coco 17-keypoint order.
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

def load_ball_position(frame_number: int):
    """load the ball position for the selected frame."""
    with BALL_TRACKING_PATH.open("r", encoding="utf-8") as file:
        tracking_data = json.load(file)

    # the json has one entry for every video frame.
    for frame_record in tracking_data["frames"]:
        ball = frame_record["ball"]

        # keep only the exact frame requested and a direct model observation.
        if (
            frame_record["frame"] == frame_number
            and ball["visible"]
            and ball["provenance"] == "direct_temporal_model_observation"
        ):
            # older imported records use "image_pixels"; newer ones use
            # "image_point_pixels". both mean [x_pixels, y_pixels].
            pixels = ball.get("image_point_pixels") or ball.get("image_pixels")

            if pixels is not None:
                return pixels

    # returning none makes the caller handle missing evidence safely.
    return None

def main():
    # confirm the model and video exist before doing expensive work.
    """run the script's main workflow."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find pose model: {MODEL_PATH}"
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Could not find video: {VIDEO_PATH}"
        )

    # open the video and jump directly to frame 346.
    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_TO_INSPECT,
    )

    success, frame = capture.read()
    capture.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame {FRAME_TO_INSPECT}."
        )

    # load the pose model.
    model = YOLO(str(MODEL_PATH))

    # run pose detection on the untouched video frame first.
    result = model(frame, classes=[0], conf=0.25, verbose=False)[0]

    # yolo creates a copy containing its boxes and skeleton drawing.
    annotated_frame = result.plot()

    OUTPUT_IMAGE_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
        )

    # now look up the independently tracked ball for this same frame.
    ball_position = load_ball_position(FRAME_TO_INSPECT)

    if ball_position is None:
        print("No direct ball observation exists for this frame.")
    else:
        ball_x, ball_y = ball_position

        # red circle = racketvision's directly observed ball location.
        cv2.circle(
            annotated_frame,
            (int(ball_x), int(ball_y)),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            annotated_frame,
            "tracked ball",
            (int(ball_x) + 10, int(ball_y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        print(f"Tracked ball: ({ball_x:.1f}, {ball_y:.1f})")

    # save the combined pose-and-ball inspection image.
    cv2.imwrite(OUTPUT_IMAGE_PATH, annotated_frame)

    print(
        f"Saved pose inspection image: {OUTPUT_IMAGE_PATH}"
    )

    # if no people are detected, stop with an understandable message.
    if result.boxes is None or len(result.boxes) == 0:
        print("No people were detected on this frame.")
        return

    # convert model tensors into normal python lists so we can print them.
    boxes = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()

    # keypoint x/y locations and confidence values are separate.
    keypoint_xy = result.keypoints.xy.cpu().tolist()
    keypoint_confidences = (
        result.keypoints.conf.cpu().tolist()
    )

    print(f"People detected: {len(boxes)}")

    for person_index, (
        box,
        person_confidence,
        person_keypoints,
        person_keypoint_confidences,
    ) in enumerate(
        zip(
            boxes,
            confidences,
            keypoint_xy,
            keypoint_confidences,
        ),
        start=1,
    ):
        print(f"\nPerson {person_index}")
        print(
            "Bounding box:",
            {
                "left": round(box[0], 1),
                "top": round(box[1], 1),
                "right": round(box[2], 1),
                "bottom": round(box[3], 1),
                "confidence": round(person_confidence, 3),
            },
        )

        # print the two wrist locations because they will become
        # our first racket-contact proxy.
        for keypoint_index in [9, 10]:
            keypoint_name = KEYPOINT_NAMES[
                keypoint_index
            ]

            x_pixels, y_pixels = person_keypoints[
                keypoint_index
            ]

            keypoint_confidence = (
                person_keypoint_confidences[
                    keypoint_index
                ]
            )

            print(
                keypoint_name,
                {
                    "x_pixels": round(x_pixels, 1),
                    "y_pixels": round(y_pixels, 1),
                    "confidence": round(
                        keypoint_confidence,
                        3,
                    ),
                },
            )


if __name__ == "__main__":
    main()