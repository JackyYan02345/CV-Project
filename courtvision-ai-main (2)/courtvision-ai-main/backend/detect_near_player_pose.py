from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
# detect the near player pose in the first frame
image_path = Path("outputs/first_frame.jpg")
output_path = Path("outputs/near_player_pose_preview.jpg")
# checks if the input image exists, raise an error if not
if not image_path.exists():
    raise FileNotFoundError(
        f"Could not find {image_path}. Run save_first_frame.py first."
    )

# this model detects people and estimates 17 body keypoints.
model = YOLO("yolo11n-pose.pt")
# detect people and their poses
results = model.predict(
    source=str(image_path),
    conf=0.30,
    verbose=False,
)
# count the detected people
result = results[0]
# require at least one box and keypoints
if result.boxes is None or len(result.boxes) == 0:
    raise RuntimeError("No person was detected in the image.")

if result.keypoints is None or len(result.keypoints.data) == 0:
    raise RuntimeError("A person was detected, but no body pose was found.")

# use the largest box as the near-player guess
boxes = result.boxes.xyxy.cpu().numpy()
box_widths = boxes[:, 2] - boxes[:, 0]
box_heights = boxes[:, 3] - boxes[:, 1]
box_areas = box_widths * box_heights
near_player_index = int(np.argmax(box_areas))

# get the near player's estimated body landmarks.
keypoints = result.keypoints.xy[near_player_index].cpu().numpy()
keypoint_confidences = result.keypoints.conf[near_player_index].cpu().numpy()

# start with the original image, then draw our white pose skeleton.
image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(f"Could not open {image_path}")

x1, y1, x2, y2 = boxes[near_player_index].astype(int)

# draw a white person-detection rectangle.
cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

# these pairs tell the program which joints should be connected.
skeleton_connections = [
    (5, 6),      # left shoulder to right shoulder
    (5, 11),     # left shoulder to left hip
    (6, 12),     # right shoulder to right hip
    (11, 12),    # left hip to right hip
    (11, 13),    # left hip to left knee
    (13, 15),    # left knee to left ankle
    (12, 14),    # right hip to right knee
    (14, 16),    # right knee to right ankle
]

minimum_confidence = 0.30

# draw white lines between confidently detected body landmarks.
for start_index, end_index in skeleton_connections:
    start_confidence = keypoint_confidences[start_index]
    end_confidence = keypoint_confidences[end_index]

    if start_confidence >= minimum_confidence and end_confidence >= minimum_confidence:
        start_point = tuple(keypoints[start_index].astype(int))
        end_point = tuple(keypoints[end_index].astype(int))

        cv2.line(image, start_point, end_point, (255, 255, 255), 3)

# draw a small white circle at every confident keypoint.
for point, confidence in zip(keypoints, keypoint_confidences):
    if confidence >= minimum_confidence:
        point_x, point_y = point.astype(int)
        cv2.circle(image, (point_x, point_y), 5, (255, 255, 255), -1)

cv2.putText(
    image,
    "Near player pose",
    (x1, max(30, y1 - 12)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2,
)

image_saved = cv2.imwrite(str(output_path), image)

if not image_saved:
    raise RuntimeError(f"Could not save {output_path}")

print("Near-player pose detected successfully.")
print(f"Saved pose preview to: {output_path}")