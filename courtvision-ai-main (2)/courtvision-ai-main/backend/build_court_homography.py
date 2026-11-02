from pathlib import Path
import json

import cv2
# imported math library np
import numpy as np
# loads the saved first frame and the homography matrix
# draw court lines to check the mapping
frame_path = Path("outputs/first_frame.jpg")
points_path = Path("outputs/court_corner_pixels.json")
matrix_path = Path("outputs/image_to_court_homography.json")
top_down_path = Path("outputs/top_down_court.jpg")
# set court dimensions and drawing scale
court_width_meters = 8.23
court_length_meters = 23.77
pixels_per_meter = 40
# opens saved first frame (first_frame.jpg) for python to read
image = cv2.imread(str(frame_path))
# checks if the image was opened successfully, raise an error if not
if image is None:
    raise FileNotFoundError(f"Could not open {frame_path}")
# loads the court corner points from the json file
with open(points_path, "r", encoding="utf-8") as file:
    image_points = json.load(file)
# require exactly four court corners
if len(image_points) != 4:
    raise ValueError("Exactly four court-corner points are required.")

source_points = np.float32(image_points)
# define court corners in meters
court_points_meters = np.float32(
    [
        [0, 0],
        [court_width_meters, 0],
        [court_width_meters, court_length_meters],
        [0, court_length_meters],
    ]
)
# calculate the image-to-court mapping
image_to_meters = cv2.getPerspectiveTransform(
    source_points,
    court_points_meters,
)
# calculate court dimensions in pixels
court_width_pixels = int(court_width_meters * pixels_per_meter)
court_length_pixels = int(court_length_meters * pixels_per_meter)

court_points_pixels = court_points_meters * pixels_per_meter
# map the image court to a top-down view
image_to_canvas = cv2.getPerspectiveTransform(
    source_points,
    court_points_pixels,
)
# warp the image into a top-down view
top_down_court = cv2.warpPerspective(
    image,
    image_to_canvas,
    (court_width_pixels, court_length_pixels),
)
# saves the homography matrix and court dimensions to a json file for later use
with open(matrix_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "court_width_meters": court_width_meters,
            "court_length_meters": court_length_meters,
            "image_corner_points_pixels": image_points,
            "image_to_court_meters_matrix": image_to_meters.tolist(),
        },
        file,
        indent=2,
    )
# saves the top-down view of the court to a new image file
cv2.imwrite(str(top_down_path), top_down_court)
# print if successful
print("Homography created successfully.")
print(f"Saved meter-mapping data to: {matrix_path}")
print(f"Saved top-down court image to: {top_down_path}")