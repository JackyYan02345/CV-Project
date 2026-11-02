from pathlib import Path
import json

import cv2
import numpy as np

# draw mapped player positions on a top-down court
positions_path = Path("outputs/player_positions_in_meters.json")
output_path = Path("outputs/top_down_player_positions.jpg")

# real dimensions of a singles tennis court, in meters.
court_width_m = 8.23
court_length_m = 23.77
service_line_distance_from_net_m = 6.40

# this controls how large the court looks in the saved image.
pixels_per_meter = 40

# extra space lets us show players standing behind either baseline.
extra_space_m = 5.0
side_margin_pixels = 80

if not positions_path.exists():
    raise FileNotFoundError(
        f"Could not find {positions_path}. "
        "Run map_players_to_court.py first."
    )

with open(positions_path, "r", encoding="utf-8") as file:
    people = json.load(file)

if not isinstance(people, list) or len(people) == 0:
    raise RuntimeError("The player-position file does not contain any players.")

minimum_y_m = -extra_space_m
maximum_y_m = court_length_m + extra_space_m

image_width = int(court_width_m * pixels_per_meter + side_margin_pixels * 2)
image_height = int((maximum_y_m - minimum_y_m) * pixels_per_meter)

# a numpy array will be the blank image drawn onto.
canvas = np.full((image_height, image_width, 3), (35, 35, 35), dtype=np.uint8)


def meters_to_canvas(court_x_m, court_y_m):
    """convert real court coordinates in meters into image-pixel coordinates."""
    pixel_x = int(side_margin_pixels + court_x_m * pixels_per_meter)
    pixel_y = int((court_y_m - minimum_y_m) * pixels_per_meter)
    return pixel_x, pixel_y


# convert important tennis-court locations into positions on the image.
court_top_left = meters_to_canvas(0, 0)
court_bottom_right = meters_to_canvas(court_width_m, court_length_m)

net_left = meters_to_canvas(0, court_length_m / 2)
net_right = meters_to_canvas(court_width_m, court_length_m / 2)

far_service_left = meters_to_canvas(0, court_length_m / 2 - service_line_distance_from_net_m)
far_service_right = meters_to_canvas(court_width_m, court_length_m / 2 - service_line_distance_from_net_m)

near_service_left = meters_to_canvas(0, court_length_m / 2 + service_line_distance_from_net_m)
near_service_right = meters_to_canvas(court_width_m, court_length_m / 2 + service_line_distance_from_net_m)

center_service_top = meters_to_canvas(court_width_m / 2, court_length_m / 2 - service_line_distance_from_net_m)
center_service_bottom = meters_to_canvas(court_width_m / 2, court_length_m / 2 + service_line_distance_from_net_m)

# draw the blue court surface and its white boundary lines.
cv2.rectangle(canvas, court_top_left, court_bottom_right, (180, 90, 35), -1)
cv2.rectangle(canvas, court_top_left, court_bottom_right, (255, 255, 255), 3)

cv2.line(canvas, net_left, net_right, (255, 255, 255), 3)
cv2.line(canvas, far_service_left, far_service_right, (255, 255, 255), 2)
cv2.line(canvas, near_service_left, near_service_right, (255, 255, 255), 2)
cv2.line(canvas, center_service_top, center_service_bottom, (255, 255, 255), 2)

# draw every detected player's estimated position.
for person in people:
    position = person["court_position_meters"]
    court_x_m = float(position["x"])
    court_y_m = float(position["y"])

    pixel_x, pixel_y = meters_to_canvas(court_x_m, court_y_m)

    # orange means the player is outside the singles-court rectangle.
    inside_court = (
        0 <= court_x_m <= court_width_m
        and 0 <= court_y_m <= court_length_m
    )
    dot_color = (0, 255, 0) if inside_court else (0, 165, 255)

    cv2.circle(canvas, (pixel_x, pixel_y), 12, dot_color, -1)

    label = (
        f"Detection {person['person_number']}: "
        f"({court_x_m:.2f} m, {court_y_m:.2f} m)"
    )

    cv2.putText(
        canvas,
        label,
        (pixel_x + 16, pixel_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
# save the annotated court image
image_saved = cv2.imwrite(str(output_path), canvas)

if not image_saved:
    raise RuntimeError(f"Could not save {output_path}")

print(f"Saved top-down player map to: {output_path}")