from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO

image_path = Path("outputs/first_frame.jpg")
court_corners_path = Path("outputs/court_corner_pixels.json")

preview_path = Path("outputs/ball_detection_preview.jpg")
candidate_data_path = Path("outputs/ball_detection_candidates.json")

# a low value gathers raw guesses for inspection.
# it does not mean those guesses are accepted as real tennis balls.
diagnostic_confidence = 0.01

# a candidate must reach this score before it can be considered reliable.
reliable_confidence = 0.35

# expected ball-box geometry in this 1920×1080-style broadcast frame.
minimum_ball_size_pixels = 2
maximum_ball_size_pixels = 36
minimum_aspect_ratio = 0.35
maximum_aspect_ratio = 2.8

# tennis balls are normally fluorescent yellow / yellow-green.
minimum_yellow_green_fraction = 0.04

# a ball may be near or slightly beyond a court boundary.
maximum_outside_court_distance_pixels = 80


def find_class_id(model, expected_class_name):
    """find a model class number by name instead of hard-coding it."""
    if isinstance(model.names, dict):
        class_items = model.names.items()
    else:
        class_items = enumerate(model.names)

    for class_id, class_name in class_items:
        if str(class_name).strip().lower() == expected_class_name:
            return int(class_id)

    raise RuntimeError(
        f"The model does not contain a class named '{expected_class_name}'."
    )


def yellow_green_fraction(image, x1, y1, x2, y2):
    """measure the yellow and green pixels in a patch."""
    image_height, image_width = image.shape[:2]

    left = max(0, int(np.floor(x1)))
    top = max(0, int(np.floor(y1)))
    right = min(image_width, int(np.ceil(x2)))
    bottom = min(image_height, int(np.ceil(y2)))

    candidate_patch = image[top:bottom, left:right]

    if candidate_patch.size == 0:
        return 0.0

    hsv_patch = cv2.cvtColor(candidate_patch, cv2.COLOR_BGR2HSV)

    lower_yellow_green = np.array([15, 45, 75], dtype=np.uint8)
    upper_yellow_green = np.array([70, 255, 255], dtype=np.uint8)

    color_mask = cv2.inRange(
        hsv_patch,
        lower_yellow_green,
        upper_yellow_green,
    )

    matching_pixel_count = cv2.countNonZero(color_mask)
    total_pixel_count = candidate_patch.shape[0] * candidate_patch.shape[1]

    return matching_pixel_count / total_pixel_count


if not image_path.exists():
    raise FileNotFoundError(
        f"Could not find {image_path}. Run save_first_frame.py first."
    )

if not court_corners_path.exists():
    raise FileNotFoundError(
        f"Could not find {court_corners_path}. "
        "Run click_court_corners.py first."
    )

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError(f"Could not open {image_path}")

with open(court_corners_path, "r", encoding="utf-8") as file:
    court_corners = json.load(file)

if not isinstance(court_corners, list) or len(court_corners) != 4:
    raise RuntimeError(
        "court_corner_pixels.json must contain exactly four court corners."
    )

court_polygon = np.array(court_corners, dtype=np.int32).reshape((-1, 1, 2))

# this is still a general-purpose object-detection model.
model = YOLO("yolo26n.pt")
sports_ball_class_id = find_class_id(model, "sports ball")

# preserve more original frame detail than the former 1280-pixel test.
# this is slower, but appropriate for one diagnostic image.
results = model.predict(
    source=str(image_path),
    classes=[sports_ball_class_id],
    conf=diagnostic_confidence,
    imgsz=(1088, 1920),
    rect=False,
    max_det=30,
    verbose=False,
)

result = results[0]
preview_image = image.copy()

cv2.polylines(
    preview_image,
    [court_polygon],
    True,
    (0, 255, 0),
    2,
)

all_candidates = []
reliable_candidates = []

if result.boxes is not None and len(result.boxes) > 0:
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    for box, confidence in zip(boxes, confidences):
        x1, y1, x2, y2 = box

        width_pixels = float(x2 - x1)
        height_pixels = float(y2 - y1)
        aspect_ratio = width_pixels / height_pixels if height_pixels > 0 else 0.0

        center_x = float((x1 + x2) / 2)
        center_y = float((y1 + y2) / 2)

        court_distance = float(
            cv2.pointPolygonTest(
                court_polygon,
                (center_x, center_y),
                True,
            )
        )

        color_fraction = yellow_green_fraction(image, x1, y1, x2, y2)

        passes_confidence = float(confidence) >= reliable_confidence

        passes_size = (
            minimum_ball_size_pixels <= width_pixels <= maximum_ball_size_pixels
            and minimum_ball_size_pixels <= height_pixels <= maximum_ball_size_pixels
        )

        passes_shape = (
            minimum_aspect_ratio <= aspect_ratio <= maximum_aspect_ratio
        )

        passes_color = color_fraction >= minimum_yellow_green_fraction

        passes_court_region = (
            court_distance >= -maximum_outside_court_distance_pixels
        )

        rejection_reasons = []

        if not passes_confidence:
            rejection_reasons.append("low_confidence")

        if not passes_size:
            rejection_reasons.append("implausible_size")

        if not passes_shape:
            rejection_reasons.append("implausible_shape")

        if not passes_color:
            rejection_reasons.append("not_yellow_green")

        if not passes_court_region:
            rejection_reasons.append("too_far_from_court")

        passes_all_filters = (
            passes_confidence
            and passes_size
            and passes_shape
            and passes_color
            and passes_court_region
        )

        candidate = {
            "detector_confidence": round(float(confidence), 4),
            "box_pixels": [
                round(float(x1), 1),
                round(float(y1), 1),
                round(float(x2), 1),
                round(float(y2), 1),
            ],
            "center_pixels": [
                round(center_x, 1),
                round(center_y, 1),
            ],
            "width_pixels": round(width_pixels, 2),
            "height_pixels": round(height_pixels, 2),
            "aspect_ratio": round(aspect_ratio, 3),
            "yellow_green_fraction": round(color_fraction, 4),
            "distance_from_court_pixels": round(court_distance, 1),
            "passes_confidence": passes_confidence,
            "passes_size": passes_size,
            "passes_shape": passes_shape,
            "passes_color": passes_color,
            "passes_court_region": passes_court_region,
            "passes_all_filters": passes_all_filters,
            "rejection_reasons": rejection_reasons,
        }

        all_candidates.append(candidate)

        if passes_all_filters:
            reliable_candidates.append(candidate)
            box_color = (0, 255, 0)
            label = f"needs motion test {confidence:.2f}"
        else:
            box_color = (0, 0, 255)
            reason_text = ",".join(rejection_reasons[:2])
            label = f"rejected: {reason_text}"

        cv2.rectangle(
            preview_image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            box_color,
            2,
        )

        cv2.circle(
            preview_image,
            (int(center_x), int(center_y)),
            5,
            box_color,
            -1,
        )

        cv2.putText(
            preview_image,
            label,
            (int(x1), max(25, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            box_color,
            2,
        )

# draw a readable information panel.
cv2.rectangle(
    preview_image,
    (15, 15),
    (650, 105),
    (0, 0, 0),
    -1,
)

cv2.putText(
    preview_image,
    f"Raw sports-ball guesses: {len(all_candidates)}",
    (30, 42),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2,
)

cv2.putText(
    preview_image,
    f"Reliable candidates: {len(reliable_candidates)}",
    (30, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2,
)

cv2.putText(
    preview_image,
    "Green = needs multi-frame motion confirmation",
    (30, 94),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    1,
)

preview_saved = cv2.imwrite(str(preview_path), preview_image)

if not preview_saved:
    raise RuntimeError(f"Could not save {preview_path}")

output_data = {
    "model_name": "yolo26n.pt",
    "model_class_used": "sports ball",
    "diagnostic_confidence": diagnostic_confidence,
    "reliable_confidence": reliable_confidence,
    "all_raw_candidates": all_candidates,
    "reliable_candidates_requiring_motion_confirmation": reliable_candidates,
}

with open(candidate_data_path, "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=2)

print("Stricter ball-detection diagnostic completed.")
print(f"Raw sports-ball guesses: {len(all_candidates)}")
print(f"Reliable candidates: {len(reliable_candidates)}")
print(f"Saved preview image to: {preview_path}")
print(f"Saved candidate data to: {candidate_data_path}")