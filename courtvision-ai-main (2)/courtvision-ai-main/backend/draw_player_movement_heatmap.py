from pathlib import Path
import json
# draw player movement and speed on a court map
import cv2
import numpy as np
# official singles-court measurements in meters.
tracking_path = Path("outputs/near_player_tracking_data.json")
speed_metrics_path = Path("outputs/player_speed_metrics.json")
output_path = Path("outputs/player_movement_heatmap.jpg")

# official singles-court measurements in meters.
court_width_m = 8.23
court_length_m = 23.77
service_line_distance_from_net_m = 6.40

# image-design settings.
pixels_per_meter = 45
extra_side_space_m = 2.0
extra_baseline_space_m = 5.0
maximum_display_speed_kmh = 25.0
# load and validate the input files
for required_path in [tracking_path, speed_metrics_path]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Could not find {required_path}. "
            "Run the earlier tracking and speed scripts first."
        )
# load the tracking and speed data from the json files
with open(tracking_path, "r", encoding="utf-8") as file:
    tracking_data = json.load(file)
# sort valid tracking records by frame
with open(speed_metrics_path, "r", encoding="utf-8") as file:
    speed_data = json.load(file)
# validate speed data and index it by frame
if len(tracking_data) < 2:
    raise RuntimeError("At least two tracked player positions are required.")
# sort the tracking data by frame number to ensure chronological order
tracking_data.sort(key=lambda record: record["frame_number"])

# link each video frame number to its calculated speed.
speed_by_frame = {}
# index each usable speed by frame
for metric in speed_data["frame_speed_metrics"]:
    frame_number = metric["frame_number"]
    # prefer smoothed speed when available
    if metric.get("speed_available"):
        smoothed_speed = metric.get("smoothed_speed_kmh")
        raw_speed = metric.get("raw_speed_kmh")

        # use the smoothed value when it exists; otherwise use raw speed.
        speed_by_frame[frame_number] = (
            smoothed_speed
            if smoothed_speed is not None
            else raw_speed
        )
# set the court view bounds and image size
minimum_x_m = -extra_side_space_m
maximum_x_m = court_width_m + extra_side_space_m
# set the court view bounds and image size
minimum_y_m = -extra_baseline_space_m
maximum_y_m = court_length_m + extra_baseline_space_m
# set the court view bounds and image size
image_width = int((maximum_x_m - minimum_x_m) * pixels_per_meter)
image_height = int((maximum_y_m - minimum_y_m) * pixels_per_meter)

# convert court coordinates to image pixels
def court_point_to_pixel(court_x_m, court_y_m):
    """convert a court position in meters into an image pixel location."""
    pixel_x = int((court_x_m - minimum_x_m) * pixels_per_meter)
    pixel_y = int((court_y_m - minimum_y_m) * pixels_per_meter)
    return pixel_x, pixel_y

# convert court coordinates to image pixels
def speed_to_color(speed_kmh):
    """map a speed to a heatmap color."""
    # scale speed to the display range
    normalized_speed = max(
        0.0,
        min(speed_kmh / maximum_display_speed_kmh, 1.0),
    )
    # blend cyan, yellow, and red by speed in opencv bgr order
    if normalized_speed <= 0.5:
        blend = normalized_speed * 2.0
        blue = int(255 * (1.0 - blend))
        green = 255
        red = int(255 * blend)
    else:
        blend = (normalized_speed - 0.5) * 2.0
        blue = 0
        green = int(255 * (1.0 - blend))
        red = 255
    # return the speed color in opencv bgr order
    return blue, green, red


# begin with a dark background.
canvas = np.full((image_height, image_width, 3), (35, 35, 35), dtype=np.uint8)
# draw the court surface and boundaries
court_top_left = court_point_to_pixel(0, 0)
court_bottom_right = court_point_to_pixel(court_width_m, court_length_m)

# draw the tennis-court surface.
cv2.rectangle(
    canvas,
    court_top_left,
    court_bottom_right,
    (180, 90, 35),
    -1,
)

# build a heatmap based on every reliable player position.
heatmap = np.zeros((image_height, image_width), dtype=np.float32)
# mark each tracked court position on the heatmap
for record in tracking_data:
    position = record["court_position_meters"]
    court_x_m = float(position["x"])
    court_y_m = float(position["y"])
    # map a court position to heatmap pixels
    pixel_x, pixel_y = court_point_to_pixel(court_x_m, court_y_m)
    # skip points outside the heatmap
    if 0 <= pixel_x < image_width and 0 <= pixel_y < image_height:
        cv2.circle(heatmap, (pixel_x, pixel_y), 10, 1.0, -1)

# blur the individual circles into a smooth heatmap.
heatmap = cv2.GaussianBlur(heatmap, (0, 0), 28)
# normalize the heatmap and build its transparency mask
if heatmap.max() > 0:
    normalized_heatmap = cv2.normalize(
        heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    # color the heatmap by movement intensity
    heatmap_colors = cv2.applyColorMap(
        normalized_heatmap,
        cv2.COLORMAP_JET,
    )
    # hide low-intensity areas and fade the heatmap
    heat_alpha = normalized_heatmap.astype(np.float32) / 255.0
    heat_alpha *= 0.55
    heat_alpha[normalized_heatmap < 8] = 0.0
    # overlay the heatmap while keeping the court visible
    canvas = (
        canvas.astype(np.float32) * (1.0 - heat_alpha[:, :, None])
        + heatmap_colors.astype(np.float32) * heat_alpha[:, :, None]
    ).astype(np.uint8)

# draw court lines after the heatmap so they remain visible.
white = (255, 255, 255)
# redraw court boundaries over the heatmap
cv2.rectangle(canvas, court_top_left, court_bottom_right, white, 3)

net_left = court_point_to_pixel(0, court_length_m / 2)
net_right = court_point_to_pixel(court_width_m, court_length_m / 2)

far_service_left = court_point_to_pixel(
    0,
    court_length_m / 2 - service_line_distance_from_net_m,
)
far_service_right = court_point_to_pixel(
    court_width_m,
    court_length_m / 2 - service_line_distance_from_net_m,
)

near_service_left = court_point_to_pixel(
    0,
    court_length_m / 2 + service_line_distance_from_net_m,
)
near_service_right = court_point_to_pixel(
    court_width_m,
    court_length_m / 2 + service_line_distance_from_net_m,
)

center_service_top = court_point_to_pixel(
    court_width_m / 2,
    court_length_m / 2 - service_line_distance_from_net_m,
)

center_service_bottom = court_point_to_pixel(
    court_width_m / 2,
    court_length_m / 2 + service_line_distance_from_net_m,
)
# redraw court lines over the heatmap
cv2.line(canvas, net_left, net_right, white, 3)
cv2.line(canvas, far_service_left, far_service_right, white, 2)
cv2.line(canvas, near_service_left, near_service_right, white, 2)
cv2.line(canvas, center_service_top, center_service_bottom, white, 2)

# draw the player path. do not connect lines across tracking gaps.
for index in range(1, len(tracking_data)):
    previous = tracking_data[index - 1]
    current = tracking_data[index]
    # draw movement only between consecutive frames
    previous_frame = int(previous["frame_number"])
    current_frame = int(current["frame_number"])
    
    if current_frame - previous_frame != 1:
        continue
    # draw the segment between consecutive positions
    previous_position = previous["court_position_meters"]
    current_position = current["court_position_meters"]
    # skip segments outside the canvas
    previous_pixel = court_point_to_pixel(
        float(previous_position["x"]),
        float(previous_position["y"]),
    )
    # skip segments outside the canvas
    current_pixel = court_point_to_pixel(
        float(current_position["x"]),
        float(current_position["y"]),
    )
    # color each segment by its recorded speed
    speed_kmh = speed_by_frame.get(current_frame, 0.0)
    line_color = speed_to_color(speed_kmh)
    # draw the speed-colored movement segment
    cv2.line(
        canvas,
        previous_pixel,
        current_pixel,
        line_color,
        3,
    )

# mark the start and final tracked positions.
first_position = tracking_data[0]["court_position_meters"]
last_position = tracking_data[-1]["court_position_meters"]
# add route markers, speed details, and the color legend
start_pixel = court_point_to_pixel(
    float(first_position["x"]),
    float(first_position["y"]),
)
# add route markers, speed details, and the color legend
end_pixel = court_point_to_pixel(
    float(last_position["x"]),
    float(last_position["y"]),
)
# add route markers, speed details, and the color legend
cv2.circle(canvas, start_pixel, 10, (0, 255, 0), -1)
cv2.circle(canvas, end_pixel, 10, (0, 0, 255), -1)
# add route markers, speed details, and the color legend
cv2.putText(
    canvas,
    "START",
    (start_pixel[0] + 12, start_pixel[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (0, 255, 0),
    1,
)
# add route markers, speed details, and the color legend
cv2.putText(
    canvas,
    "END",
    (end_pixel[0] + 12, end_pixel[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (0, 0, 255),
    1,
)
# add the speed summary and color legend
summary = speed_data["summary"]

cv2.putText(
    canvas,
    "CourtVision AI: Player Movement Trail",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    white,
    2,
)

cv2.putText(
    canvas,
    f"Average speed: {summary['average_speed_kmh']:.2f} km/h",
    (20, 62),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.48,
    white,
    1,
)

cv2.putText(
    canvas,
    f"Max smoothed speed: {summary['smoothed_speed_max_kmh']:.2f} km/h",
    (20, 84),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.48,
    white,
    1,
)

cv2.putText(
    canvas,
    "Trail color: cyan = slower, yellow = medium, red = faster",
    (20, image_height - 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.42,
    white,
    1,
)
# save the completed heatmap image
image_saved = cv2.imwrite(str(output_path), canvas)
# save the completed heatmap image
if not image_saved:
    raise RuntimeError(f"Could not save {output_path}")
# save the completed heatmap image
print("Player movement heatmap completed.")
print(f"Saved image to: {output_path}")