from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO
# detect people and map their foot points to the court
# and saves an annotated image and a json file with the mapped positions
image_path = Path("outputs/first_frame.jpg")
homography_path = Path("outputs/image_to_court_homography.json")
preview_path = Path("outputs/player_ground_points.jpg")
positions_path = Path("outputs/player_positions_in_meters.json")
# checks if the input image exists, raise an error if not
image = cv2.imread(str(image_path))
# checks if the image was opened successfully, raise an error if not
if image is None:
    raise FileNotFoundError(f"Could not open {image_path}")
# checks if the homography json file exists, raise an error if not
with open(homography_path, "r", encoding="utf-8") as file:
    calibration_data = json.load(file)
# extracts the homography matrix from the loaded data
image_to_meters = np.array(
    calibration_data["image_to_court_meters_matrix"],
    dtype=np.float32,
)
# load the people detection model
model = YOLO("yolo26n.pt")
# runs the yolo model on the input image to detect people
# specifying the class for people (class 0) and a confidence threshold of 0.35
results = model.predict(
    source=str(image_path),
    classes=[0],
    conf=0.35,
    verbose=False,
)
# count the detected people
result = results[0]

if result.boxes is None or len(result.boxes) == 0:
    raise RuntimeError("No people were detected in the image.")
# draw the detection results
preview_image = result.plot()
mapped_people = []
# map each detected player to court coordinates
# and saves the results to a list of dictionaries
for person_number, box in enumerate(result.boxes, start=1):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    confidence = float(box.conf[0])
    # calculates the foot position of the detected person in pixel coordinates
    foot_x = (x1 + x2) / 2
    foot_y = y2
    # convert the foot point to court meters
    foot_point_pixels = np.float32([[[foot_x, foot_y]]])
    # map the foot point to court meters
    foot_point_meters = cv2.perspectiveTransform(
        foot_point_pixels,
        image_to_meters,
    )[0][0]
    # extracts the x and y coordinates of the foot position in meters
    court_x = float(foot_point_meters[0])
    court_y = float(foot_point_meters[1])
    # mark and label each court position
    cv2.circle(
        preview_image,
        (int(foot_x), int(foot_y)),
        10,
        (0, 255, 255),
        -1,
    )

    label = f"P{person_number}: ({court_x:.2f} m, {court_y:.2f} m)"

    cv2.putText(
        preview_image,
        label,
        (int(foot_x) + 15, int(foot_y) - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    mapped_people.append(
        {
            "person_number": person_number,
            "confidence": round(confidence, 3),
            "foot_pixel": [round(foot_x, 1), round(foot_y, 1)],
            "court_position_meters": {
                "x": round(court_x, 2),
                "y": round(court_y, 2),
            },
        }
    )
# save the annotated preview and mapped positions
cv2.imwrite(str(preview_path), preview_image)

with open(positions_path, "w", encoding="utf-8") as file:
    json.dump(mapped_people, file, indent=2)

print("Mapped player positions successfully.")
# prints the mapped player positions to the console for verification
for person in mapped_people:
    print(
        f"Player {person['person_number']}: "
        f"{person['court_position_meters']}"
    )

print(f"Saved preview image to: {preview_path}")
print(f"Saved position data to: {positions_path}")