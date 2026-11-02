from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO
# track the near player and map foot positions to the court
# and saves an annotated video and a json file with the tracked positions
video_path = Path("data/raw/test_rally.mp4")
homography_path = Path("outputs/image_to_court_homography.json")
tracker_settings_path = Path("backend/tennis_bytetrack.yaml")
# set output paths for video and tracking data
output_video_path = Path("outputs/near_player_pose_tracking.mp4")
output_data_path = Path("outputs/near_player_tracking_data.json")
diagnostics_path = Path("outputs/near_player_tracking_diagnostics.json")

# keep the first experiment short
test_duration_seconds = 8.0

# we will not automatically switch to a different player after this long
maximum_reacquisition_gap_seconds = 0.75

# approximate maximum realistic tennis movement:
# 10 m/s plus 1 meter of safety margin.
maximum_player_speed_meters_per_second = 10.0
reacquisition_safety_margin_meters = 1.0
# the minimum confidence for a keypoint to be considered valid.
minimum_keypoint_confidence = 0.30
# connections between keypoints to draw the skeleton
skeleton_connections = [
    (5, 6), # left shoulder to right shoulder
    (5, 7), # left shoulder to left elbow
    (7, 9), # left elbow to left wrist
    (6, 8), # right shoulder to right elbow
    (8, 10), # right elbow to right wrist
    (5, 11), # left shoulder to left hip
    (6, 12), # right shoulder to right hip
    (11, 12), # left hip to right hip
    (11, 13), # left hip to left knee
    (13, 15), # left knee to left ankle
    (12, 14), # right hip to right knee
    (14, 16), # right knee to right ankle
]

# draw the detected pose skeleton
def draw_pose(image, keypoints, keypoint_confidences):
    """draw a white body skeleton onto one video frame."""
    # draw white lines between confidently detected body landmarks
    for start_index, end_index in skeleton_connections:
        if (
            keypoint_confidences[start_index] >= minimum_keypoint_confidence
            and keypoint_confidences[end_index] >= minimum_keypoint_confidence
        ):
            start_point = tuple(keypoints[start_index].astype(int))
            end_point = tuple(keypoints[end_index].astype(int))
            cv2.line(image, start_point, end_point, (255, 255, 255), 3)
    # draw a small white circle at every confident keypoint
    for point, confidence in zip(keypoints, keypoint_confidences):
        if confidence >= minimum_keypoint_confidence:
            point_x, point_y = point.astype(int)
            cv2.circle(image, (point_x, point_y), 5, (255, 255, 255), -1)

# draw the foot point and return its tracking data
def get_foot_point(box, keypoints, keypoint_confidences):
    """return the player foot point from a detection."""
    # check both ankles for confident keypoints
    ankle_points = []
    # check both ankles for confident keypoints
    for ankle_index in [15, 16]:
        if keypoint_confidences[ankle_index] >= minimum_keypoint_confidence:
            ankle_points.append(keypoints[ankle_index])
    # if we have at least one confident ankle point, return the average of them
    if len(ankle_points) > 0:
        return np.mean(ankle_points, axis=0)
    # use the box bottom when both ankles are uncertain
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, y2], dtype=np.float32)

# use the homography matrix to convert image pixels into court meters
def pixel_point_to_court_meters(foot_point_pixels, image_to_meters):
    """use the homography matrix to convert image pixels into court meters."""
    point_for_transform = np.float32(
        [[[foot_point_pixels[0], foot_point_pixels[1]]]]
    )
    # map the foot point to court meters
    transformed_point = cv2.perspectiveTransform(
        point_for_transform,
        image_to_meters,
    )[0][0]
    # extract the x and y coordinates of the foot position in meters
    return float(transformed_point[0]), float(transformed_point[1])

# builds a list of candidate dictionaries for each tracked person
def build_candidates(result, image_to_meters):
    """
    create one easy-to-use dictionary for every currently tracked person.
    """
    candidates = []
    # if no people were detected, return an empty list of candidates
    if (
        result.boxes is None
        or len(result.boxes) == 0
        or result.boxes.id is None
        or result.keypoints is None
        or result.keypoints.conf is None
    ):
        return candidates
    # read boxes, ids, keypoints, and scores
    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().numpy()
    all_keypoints = result.keypoints.xy.cpu().numpy()
    all_keypoint_confidences = result.keypoints.conf.cpu().numpy()
    # build a candidate for each detection
    for index, box in enumerate(boxes):
        keypoints = all_keypoints[index]
        keypoint_confidences = all_keypoint_confidences[index]
        # estimate the foot position in image pixels using the get_foot_point function
        foot_point_pixels = get_foot_point(
            box,
            keypoints,
            keypoint_confidences,
        )
        # map the foot point to court meters
        court_x_m, court_y_m = pixel_point_to_court_meters(
            foot_point_pixels,
            image_to_meters,
        )
        # use box area to choose the largest person
        x1, y1, x2, y2 = box
        box_area = float((x2 - x1) * (y2 - y1))
        # save this detection as a candidate
        candidates.append(
            {
                "track_id": int(track_ids[index]),
                "box": box,
                "box_area": box_area,
                "keypoints": keypoints,
                "keypoint_confidences": keypoint_confidences,
                "foot_point_pixels": foot_point_pixels,
                "court_x_m": court_x_m,
                "court_y_m": court_y_m,
            }
        )
    # return the list of candidate dictionaries for all currently tracked people
    return candidates

# check that all input files exist
for required_path in [video_path, homography_path, tracker_settings_path]:
    if not required_path.exists():
        raise FileNotFoundError(f"Could not find {required_path}")
# load the image-to-court mapping
with open(homography_path, "r", encoding="utf-8") as file:
    calibration_data = json.load(file)
# extract the homography matrix from the loaded data
image_to_meters = np.array(
    calibration_data["image_to_court_meters_matrix"],
    dtype=np.float32,
)
# load the pose model
model = YOLO("yolo11n-pose.pt")
# open and validate the source video
capture = cv2.VideoCapture(str(video_path))
# validate that the video was opened successfully, raise an error if not
if not capture.isOpened():
    raise RuntimeError(f"Could not open {video_path}")
# read the video size and frame rate
frames_per_second = capture.get(cv2.CAP_PROP_FPS)
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# validate that the video reported a valid frame rate, raise an error if not
if frames_per_second <= 0:
    capture.release()
    raise RuntimeError("The video did not report a valid frame rate.")
# limit processing to the requested duration
maximum_frames = int(frames_per_second * test_duration_seconds)
# create the output video writer
video_codec = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(output_video_path),
    video_codec,
    frames_per_second,
    (frame_width, frame_height),
)
# check that the video writer opened
if not writer.isOpened():
    capture.release()
    raise RuntimeError(f"Could not create {output_video_path}")
# track the selected id and last known position
near_player_track_id = None
last_court_position = None
last_seen_time_seconds = None
# prepare tracking and diagnostic records
tracking_data = []
diagnostics = []
# count processed frames, tracked frames, and safe reacquisitions
processed_frames = 0
frames_with_player = 0
reacquisition_count = 0
# process each frame of the video up to the maximum number of frames specified
for frame_number in range(maximum_frames):
    # stop when the next frame cannot be read
    frame_read, frame = capture.read()
    # validate that the frame was read successfully, break the loop if not
    if not frame_read:
        break

    time_seconds = frame_number / frames_per_second

    # a lower confidence here gives the tracker more chances to recover.
    results = model.track(
        frame,
        persist=True,
        tracker=str(tracker_settings_path),
        conf=0.15,
        verbose=False,
    )
    # require at least one tracked person
    result = results[0]
    candidates = build_candidates(result, image_to_meters)
    visible_track_ids = [candidate["track_id"] for candidate in candidates]
    # prepare tracking state for this frame
    annotated_frame = frame.copy()
    target_candidate = None
    tracking_status = "not_started"

    # on the first usable frame, the largest visible player is the near player.
    if near_player_track_id is None and last_seen_time_seconds is None:
        if len(candidates) > 0:
            target_candidate = max(
                candidates,
                key=lambda candidate: candidate["box_area"],
            )
            near_player_track_id = target_candidate["track_id"]
            tracking_status = "initial_player_selected"
    # if we have a previous track id, try to find it in the current candidates
    elif near_player_track_id is not None:
        # best outcome: the old tracker id is still available.
        for candidate in candidates:
            if candidate["track_id"] == near_player_track_id:
                target_candidate = candidate
                tracking_status = "same_tracker_id"
                break

        # if the old id is gone, attempt a short, safe reacquisition.
        if target_candidate is None and last_seen_time_seconds is not None:
            seconds_since_last_seen = time_seconds - last_seen_time_seconds
            # allow id recovery only after a short gap
            if seconds_since_last_seen <= maximum_reacquisition_gap_seconds:
                allowed_distance_m = (
                    maximum_player_speed_meters_per_second
                    * seconds_since_last_seen
                    + reacquisition_safety_margin_meters
                )
                # find candidates within the safe distance
                safe_candidates = []
                # check each candidate against the safe distance
                for candidate in candidates:
                    candidate_position = np.array(
                        [candidate["court_x_m"], candidate["court_y_m"]]
                    )
                    # measure distance from the last position
                    distance_from_last_position_m = float(
                        np.linalg.norm(
                            candidate_position - last_court_position
                        )
                    )
                    # keep candidates within the safe distance
                    if distance_from_last_position_m <= allowed_distance_m:
                        safe_candidates.append(
                            (
                                distance_from_last_position_m,
                                candidate,
                            )
                        )
                # choose the safe candidate nearest the last position
                if len(safe_candidates) > 0:
                    safe_candidates.sort(key=lambda item: item[0])
                    target_candidate = safe_candidates[0][1]
                    # count when the tracker id changes
                    old_track_id = near_player_track_id
                    near_player_track_id = target_candidate["track_id"]
                    reacquisition_count += 1
                    # mark the player as safely reacquired
                    tracking_status = (
                        f"reacquired_ID_{old_track_id}_as_"
                        f"ID_{near_player_track_id}"
                    )

            else:
                tracking_status = "long_gap_not_reidentified"
        # mark an unconfirmed id as missing
        if target_candidate is None and tracking_status == "not_started":
            tracking_status = "ID_missing_not_safely_reidentified"
    # draw the selected player and update its last position
    if target_candidate is not None:
        keypoints = target_candidate["keypoints"]
        keypoint_confidences = target_candidate["keypoint_confidences"]
        box = target_candidate["box"]
        # draw the near player's pose and bounding box on the annotated frame
        draw_pose(annotated_frame, keypoints, keypoint_confidences)
        # draw and label the selected player box
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # draw the foot point and update its last position
        foot_point_pixels = target_candidate["foot_point_pixels"]
        court_x_m = target_candidate["court_x_m"]
        court_y_m = target_candidate["court_y_m"]
        
        foot_x_pixel = int(foot_point_pixels[0])
        foot_y_pixel = int(foot_point_pixels[1])
        # mark the foot point and show its court coordinates
        cv2.circle(
            annotated_frame,
            (foot_x_pixel, foot_y_pixel),
            10,
            (0, 255, 255),
            -1,
        )
        # draw the near player's track id and court coordinates on the annotated frame
        label = (
            f"Near player | ID {near_player_track_id} | "
            f"({court_x_m:.2f} m, {court_y_m:.2f} m)"
        )
        # keep the label inside the frame
        cv2.putText(
            annotated_frame,
            label,
            (max(10, x1), max(30, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        # update the last known court position and last seen time for the near player
        last_court_position = np.array([court_x_m, court_y_m])
        last_seen_time_seconds = time_seconds
        frames_with_player += 1
        # record this frame's selected-player data
        tracking_data.append(
            {
                "frame_number": frame_number,
                "time_seconds": round(time_seconds, 3),
                "tracker_id": near_player_track_id,
                "tracking_status": tracking_status,
                "foot_pixel": [
                    round(float(foot_point_pixels[0]), 1),
                    round(float(foot_point_pixels[1]), 1),
                ],
                "court_position_meters": {
                    "x": round(court_x_m, 3),
                    "y": round(court_y_m, 3),
                },
            }
        )
    # show visible ids when tracking is unavailable
    else:
        reason = (
            f"Tracking unavailable: {tracking_status}. "
            f"Visible IDs: {visible_track_ids}"
        )
        # place the tracking status inside the frame
        cv2.putText(
            annotated_frame,
            reason,
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    # record frame-level tracking diagnostics
    diagnostics.append(
        {
            "frame_number": frame_number,
            "time_seconds": round(time_seconds, 3),
            "status": tracking_status,
            "visible_track_ids": visible_track_ids,
            "selected_tracker_id": near_player_track_id,
            "player_was_tracked": target_candidate is not None,
        }
    )
    # write the frame and update the count
    writer.write(annotated_frame)
    processed_frames += 1
    # print progress every 30 frames to provide feedback on the processing status
    if frame_number % 30 == 0:
        print(f"Processed frame {frame_number} of about {maximum_frames}.")
# release video files and save tracking results
capture.release()
writer.release()
# validate that at least one frame was processed, raise an error if not
if processed_frames == 0:
    raise RuntimeError("No video frames were processed.")

with open(output_data_path, "w", encoding="utf-8") as file:
    json.dump(tracking_data, file, indent=2)

with open(diagnostics_path, "w", encoding="utf-8") as file:
    json.dump(diagnostics, file, indent=2)
# print a summary of processed and tracked frames
# frames with the player tracked, safe id reacquisitions,
# and the paths to the saved video and data files
print()
print("Improved near-player tracking test completed.")
print(f"Processed frames: {processed_frames}")
print(f"Frames with player tracked: {frames_with_player}")
print(f"Safe ID reacquisitions: {reacquisition_count}")
print(f"Saved video: {output_video_path}")
print(f"Saved player data: {output_data_path}")
print(f"Saved diagnostic data: {diagnostics_path}")
