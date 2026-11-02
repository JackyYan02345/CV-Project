from pathlib import Path
import json

import cv2
import numpy as np
# loads the saved first frame and the homography matrix
# draw court lines to check the mapping
frame_path = Path("outputs/first_frame.jpg")
matrix_path = Path("outputs/image_to_court_homography.json")
output_path = Path("outputs/homography_verification.jpg")
# opens saved first frame (first_frame.jpg) for python to read
image = cv2.imread(str(frame_path))
# checks if the image was opened successfully, raise an error if not
if image is None:
    raise FileNotFoundError(f"Could not open {frame_path}")
# loads the homography matrix and court dimensions from the json file
with open(matrix_path, "r", encoding="utf-8") as file:
    calibration_data = json.load(file)
# extracts the court dimensions and homography matrix from the loaded data
court_width = calibration_data["court_width_meters"]
court_length = calibration_data["court_length_meters"]
# extracts the homography matrix from the loaded data
image_to_meters = np.array(
    calibration_data["image_to_court_meters_matrix"],
    dtype=np.float32,
)
# creates the inverse mapping 
# normal mapping: image pixel -> court meter position
# inverse mapping: court meter position -> image pixel
meters_to_image = np.linalg.inv(image_to_meters)
# finds the court’s exact halfway point, where the net sits
net_y = court_length / 2
# 6.40 is the distance in meters from the net to each service line
far_service_line_y = net_y - 6.40
near_service_line_y = net_y + 6.40
center_x = court_width / 2
# lists tennis court lines using meter coordinates
court_lines_in_meters = [
    ((0, 0), (court_width, 0)),
    ((court_width, 0), (court_width, court_length)),
    ((court_width, court_length), (0, court_length)),
    ((0, court_length), (0, 0)),
    ((0, net_y), (court_width, net_y)),
    ((0, far_service_line_y), (court_width, far_service_line_y)),
    ((0, near_service_line_y), (court_width, near_service_line_y)),
    ((center_x, far_service_line_y), (center_x, near_service_line_y)),
]

for start_point, end_point in court_lines_in_meters:
    line_in_meters = np.float32([[start_point, end_point]])
    # converts each real court line back onto the camera image
    line_in_pixels = cv2.perspectiveTransform(
        line_in_meters,
        meters_to_image,
    )[0]

    start_pixel = tuple(line_in_pixels[0].astype(int))
    end_pixel = tuple(line_in_pixels[1].astype(int))
    # draws each converted line in green
    cv2.line(image, start_pixel, end_pixel, (0, 255, 0), 4)
# saves the image with the drawn lines to a new file
cv2.imwrite(str(output_path), image)

print(f"Saved verification image to: {output_path}")