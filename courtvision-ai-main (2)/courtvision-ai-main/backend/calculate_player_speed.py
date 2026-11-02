from pathlib import Path
import json
import math
from statistics import fmean
# calculate player speed from the tracking data
input_path = Path("outputs/near_player_tracking_data.json")
output_path = Path("outputs/player_speed_metrics.json")
# larger windows smooth speed but add lag
moving_average_window = 5
# check if the input tracking data file exists, raise an error if not
if not input_path.exists():
    raise FileNotFoundError(
        f"Could not find {input_path}. "
        "Run track_near_player_in_video.py first."
    )
# load frame positions and timestamps
with open(input_path, "r", encoding="utf-8") as file:
    tracking_data = json.load(file)
# require at least two tracking records
if not isinstance(tracking_data, list) or len(tracking_data) < 2:
    raise RuntimeError(
        "The tracking data must contain at least two player positions."
    )

# put records in video-frame order, even if a file was saved out of order.
tracking_data.sort(key=lambda record: record["frame_number"])
# prepare per-frame speed and gap records
speed_frames = []
tracking_gaps = []
# store raw speeds, smoothed speeds, distance, and time
raw_speeds_kmh = []
smoothed_speeds_kmh = []
# track valid distance, time, and recent speeds
total_valid_distance_m = 0.0
total_valid_time_s = 0.0

# this list is cleared whenever tracking has a gap
# that prevents us from smoothing across missing player positions.
recent_raw_speeds_kmh = []
# compare consecutive records to measure speed
for index in range(1, len(tracking_data)):
    previous = tracking_data[index - 1]
    current = tracking_data[index]
    # read frame numbers and timestamps for both records
    previous_frame = int(previous["frame_number"])
    current_frame = int(current["frame_number"])
    # read timestamps for both records
    previous_time = float(previous["time_seconds"])
    current_time = float(current["time_seconds"])
    # check frame and time gaps before measuring speed
    frame_gap = current_frame - previous_frame
    time_delta_s = current_time - previous_time
    # read both court positions
    previous_position = previous["court_position_meters"]
    current_position = current["court_position_meters"]
    # read coordinates for both court positions
    previous_x_m = float(previous_position["x"])
    previous_y_m = float(previous_position["y"])
    # read the current court coordinates
    current_x_m = float(current_position["x"])
    current_y_m = float(current_position["y"])

    # a gap means the player was not tracked in one or more video frames.
    # we preserve the gap rather than inventing motion through it.
    if frame_gap != 1 or time_delta_s <= 0:
        tracking_gaps.append(
            {
                "after_frame": previous_frame,
                "before_frame": current_frame,
                "missing_frame_count": max(0, frame_gap - 1),
                "time_gap_seconds": round(time_delta_s, 3),
            }
        )
        # clear the recent speeds list so that we do not smooth across a tracking gap
        recent_raw_speeds_kmh = []
        # record unavailable speed after a tracking gap
        speed_frames.append(
            {
                "frame_number": current_frame,
                "time_seconds": round(current_time, 3),
                "speed_available": False,
                "reason": "tracking_gap",
            }
        )

        continue
    # measure distance between consecutive positions
    delta_x_m = current_x_m - previous_x_m
    delta_y_m = current_y_m - previous_y_m
    # use straight-line distance between positions
    distance_m = math.hypot(delta_x_m, delta_y_m)

    # meters per second multiplied by 3.6 becomes kilometers per hour.
    raw_speed_kmh = (distance_m / time_delta_s) * 3.6
    # add valid distance, time, and raw speed
    total_valid_distance_m += distance_m
    total_valid_time_s += time_delta_s
    # add the new speed to the smoothing window
    raw_speeds_kmh.append(raw_speed_kmh)
    recent_raw_speeds_kmh.append(raw_speed_kmh)
    # keep only the most recent speeds in the window
    if len(recent_raw_speeds_kmh) > moving_average_window:
        recent_raw_speeds_kmh.pop(0)
    # calculate a moving average when enough speeds are available
    smoothed_speed_kmh = None
    # save a smoothed speed when the window is full
    if len(recent_raw_speeds_kmh) == moving_average_window:
        smoothed_speed_kmh = fmean(recent_raw_speeds_kmh)
        smoothed_speeds_kmh.append(smoothed_speed_kmh)
    # record the measured speed for this frame
    speed_frames.append(
        {
            "frame_number": current_frame,
            "time_seconds": round(current_time, 3),
            "speed_available": True,
            "distance_from_previous_frame_m": round(distance_m, 6),
            "raw_speed_kmh": round(raw_speed_kmh, 6),
            "smoothed_speed_kmh": (
                round(smoothed_speed_kmh, 6)
                if smoothed_speed_kmh is not None
                else None
            ),
        }
    )
# require valid intervals before calculating the summary
if total_valid_time_s <= 0:
    raise RuntimeError("No valid consecutive tracking intervals were found.")
# measure straight-line displacement from first to last point
first_position = tracking_data[0]["court_position_meters"]
last_position = tracking_data[-1]["court_position_meters"]
# calculate net displacement across the full sequence
net_displacement_m = math.hypot(
    float(last_position["x"]) - float(first_position["x"]),
    float(last_position["y"]) - float(first_position["y"]),
)
# summarize speed metrics and tracking gaps
summary = {
    "saved_position_count": len(tracking_data),
    "first_timestamp_seconds": float(tracking_data[0]["time_seconds"]),
    "last_timestamp_seconds": float(tracking_data[-1]["time_seconds"]),
    "valid_tracked_duration_seconds": round(total_valid_time_s, 6),
    "valid_tracked_distance_meters": round(total_valid_distance_m, 6),
    "net_displacement_meters": round(net_displacement_m, 6),
    "average_speed_kmh": round(
        (total_valid_distance_m / total_valid_time_s) * 3.6,
        6,
    ),
    "raw_speed_min_kmh": round(min(raw_speeds_kmh), 6),
    "raw_speed_max_kmh": round(max(raw_speeds_kmh), 6),
    "raw_speed_mean_kmh": round(fmean(raw_speeds_kmh), 6),
    "moving_average_window_frames": moving_average_window,
    "smoothed_speed_min_kmh": round(min(smoothed_speeds_kmh), 6),
    "smoothed_speed_max_kmh": round(max(smoothed_speeds_kmh), 6),
    "smoothed_speed_mean_kmh": round(fmean(smoothed_speeds_kmh), 6),
    "tracking_gap_count": len(tracking_gaps),
}
# save and print the speed summary
output_data = {
    "summary": summary,
    "tracking_gaps": tracking_gaps,
    "frame_speed_metrics": speed_frames,
}

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=2)

print("Player speed calculation completed.")
print(f"Average speed: {summary['average_speed_kmh']:.2f} km/h")
print(f"Maximum smoothed speed: {summary['smoothed_speed_max_kmh']:.2f} km/h")
print(f"Valid distance: {summary['valid_tracked_distance_meters']:.2f} m")
print(f"Tracking gaps: {summary['tracking_gap_count']}")
print(f"Saved speed data to: {output_path}")