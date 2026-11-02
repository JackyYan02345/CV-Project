"""find plausible ball paths in a fixed broadcast view."""

from pathlib import Path
import json

import cv2
import numpy as np
from ultralytics import YOLO


video_path = Path("data/raw/test_rally.mp4")
court_corners_path = Path("outputs/court_corner_pixels.json")
pose_model_path = Path("yolo11n-pose.pt")

output_video_path = Path("outputs/moving_ball_candidates.mp4")
output_data_path = Path("outputs/moving_ball_candidate_tracks.json")

# keep the first experiment short so it is easy to inspect and adjust.
test_duration_seconds = 8.0

# a tennis ball is normally fluorescent yellow. during fast motion and video
# compression it can also look pale or nearly white, so we use a second,
# carefully limited bright-pixel test.
yellow_lower_hsv = np.array([18, 70, 75], dtype=np.uint8)
yellow_upper_hsv = np.array([42, 255, 255], dtype=np.uint8)
bright_lower_hsv = np.array([0, 0, 175], dtype=np.uint8)
bright_upper_hsv = np.array([179, 105, 255], dtype=np.uint8)

# a true broadcast ball is small. these are image-pixel limits, not metres.
minimum_contour_area = 2.0
maximum_contour_area = 150.0
minimum_width_pixels = 2
maximum_width_pixels = 30
minimum_height_pixels = 2
maximum_height_pixels = 24
minimum_aspect_ratio = 0.28
maximum_aspect_ratio = 3.6
minimum_compactness = 0.08

# a pixel must change enough from the prior frame to count as moving.
motion_difference_threshold = 20
minimum_motion_fraction = 0.16

# every third frame, estimate player ankles so shoe/foot details cannot become
# ball candidates. this is deliberately a small exclusion zone, not a whole
# player box: the ball is allowed to appear near a player's body.
pose_detection_interval_frames = 3
minimum_ankle_confidence = 0.25
ankle_exclusion_radius_pixels = 48

# a candidate well inside a torso is much more likely to be clothing or a body
# edge than a visible tennis ball.  we intentionally do not mask the arms or
# racket area because a real ball can be close to a player's racquet at impact.
minimum_body_core_keypoint_confidence = 0.30
body_core_padding_pixels = 8

# rules for joining observations. the earlier version permitted a very large
# jump after a gap. these rules use a prediction from recent motion instead.
maximum_missing_frames = 2
maximum_first_match_distance_pixels = 95.0
maximum_predicted_residual_pixels = 58.0
maximum_step_pixels_per_frame = 105.0
minimum_median_step_pixels_per_frame = 10.0
minimum_points_for_reliable_track = 4
minimum_track_coverage = 0.45
minimum_track_displacement_pixels = 45.0
maximum_drawn_history_points = 8

# search a little outside court lines, but not in scoreboards or spectators.
court_search_expansion_pixels = 55


def point_is_near_an_ankle(point, ankle_points):
    """return true when a candidate probably comes from a shoe or foot."""
    point_x, point_y = point

    for ankle_x, ankle_y in ankle_points:
        distance = float(np.hypot(point_x - ankle_x, point_y - ankle_y))

        if distance <= ankle_exclusion_radius_pixels:
            return True

    return False


def point_is_inside_body_core(point, body_core_polygons):
    """return true only when a point is in or immediately beside a torso."""
    point_x, point_y = point

    for polygon in body_core_polygons:
        signed_distance = cv2.pointPolygonTest(
            polygon,
            (float(point_x), float(point_y)),
            True,
        )

        if signed_distance >= -body_core_padding_pixels:
            return True

    return False


def find_player_protected_regions(frame, pose_model):
    """return feet and torso-core regions that should not become ball pixels."""
    results = pose_model.predict(
        frame,
        classes=[0],  # coco class 0 means person.
        conf=0.25,
        imgsz=960,
        verbose=False,
    )
    result = results[0]

    if result.keypoints is None or result.keypoints.conf is None:
        return [], []

    all_keypoints = result.keypoints.xy.cpu().numpy()
    all_confidences = result.keypoints.conf.cpu().numpy()
    ankle_points = []
    body_core_polygons = []

    for keypoints, confidences in zip(all_keypoints, all_confidences):
        # coco pose keypoints 15 and 16 represent left and right ankles.
        for ankle_index in (15, 16):
            if confidences[ankle_index] >= minimum_ankle_confidence:
                ankle_x, ankle_y = keypoints[ankle_index]
                ankle_points.append((float(ankle_x), float(ankle_y)))

        # coco pose points 5, 6, 11, and 12 are shoulders and hips.  together
        # they form a torso quadrilateral.  arms and the racquet are not part
        # of this polygon, so contact-side ball candidates remain possible.
        torso_indices = (5, 6, 12, 11)

        if all(
            confidences[index] >= minimum_body_core_keypoint_confidence
            for index in torso_indices
        ):
            torso_polygon = np.array(
                [keypoints[index] for index in torso_indices],
                dtype=np.float32,
            ).reshape((-1, 1, 2))
            body_core_polygons.append(torso_polygon)

    return ankle_points, body_core_polygons


def extract_ball_like_candidates(
    frame,
    previous_gray_frame,
    search_mask,
    ankle_points,
    body_core_polygons,
):
    """find small, moving, ball-shaped yellow or pale image regions."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    yellow_mask = cv2.inRange(
        hsv_frame,
        yellow_lower_hsv,
        yellow_upper_hsv,
    )
    bright_mask = cv2.inRange(
        hsv_frame,
        bright_lower_hsv,
        bright_upper_hsv,
    )
    color_mask = cv2.bitwise_or(yellow_mask, bright_mask)
    color_mask = cv2.bitwise_and(color_mask, search_mask)

    if previous_gray_frame is None:
        return [], gray_frame, 0, 0

    frame_difference = cv2.absdiff(gray_frame, previous_gray_frame)
    _, motion_mask = cv2.threshold(
        frame_difference,
        motion_difference_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    motion_mask = cv2.dilate(
        motion_mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    combined_mask = cv2.bitwise_and(color_mask, motion_mask)
    contours, _ = cv2.findContours(
        combined_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []
    candidates_rejected_near_ankle = 0
    candidates_rejected_in_body_core = 0

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))

        if not minimum_contour_area <= contour_area <= maximum_contour_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        if not (
            minimum_width_pixels <= width <= maximum_width_pixels
            and minimum_height_pixels <= height <= maximum_height_pixels
        ):
            continue

        aspect_ratio = width / height

        if not minimum_aspect_ratio <= aspect_ratio <= maximum_aspect_ratio:
            continue

        box_area = width * height
        compactness = contour_area / box_area

        if compactness < minimum_compactness:
            continue

        motion_patch = motion_mask[y:y + height, x:x + width]

        if motion_patch.size == 0:
            continue

        motion_fraction = cv2.countNonZero(motion_patch) / box_area

        if motion_fraction < minimum_motion_fraction:
            continue

        center_x = float(x + width / 2)
        center_y = float(y + height / 2)

        if point_is_near_an_ankle((center_x, center_y), ankle_points):
            candidates_rejected_near_ankle += 1
            continue

        if point_is_inside_body_core(
            (center_x, center_y),
            body_core_polygons,
        ):
            candidates_rejected_in_body_core += 1
            continue

        yellow_fraction = (
            cv2.countNonZero(yellow_mask[y:y + height, x:x + width])
            / box_area
        )
        bright_fraction = (
            cv2.countNonZero(bright_mask[y:y + height, x:x + width])
            / box_area
        )

        candidates.append(
            {
                "center_x": center_x,
                "center_y": center_y,
                "box": [int(x), int(y), int(width), int(height)],
                "contour_area": round(contour_area, 3),
                "aspect_ratio": round(float(aspect_ratio), 3),
                "compactness": round(float(compactness), 3),
                "motion_fraction": round(float(motion_fraction), 3),
                "yellow_fraction": round(float(yellow_fraction), 3),
                "bright_fraction": round(float(bright_fraction), 3),
            }
        )

    return (
        candidates,
        gray_frame,
        candidates_rejected_near_ankle,
        candidates_rejected_in_body_core,
    )


def per_frame_distance(first_point, second_point):
    """measure image-pixel distance after accounting for skipped frames."""
    frame_gap = second_point["frame_number"] - first_point["frame_number"]

    if frame_gap <= 0:
        return 0.0

    distance = float(
        np.hypot(
            second_point["center_x"] - first_point["center_x"],
            second_point["center_y"] - first_point["center_y"],
        )
    )
    return distance / frame_gap


def match_cost(track, candidate, frame_number):
    """return a finite cost only if a candidate plausibly continues a track."""
    last_point = track["points"][-1]
    frame_gap = frame_number - last_point["frame_number"]

    if frame_gap < 1 or frame_gap > maximum_missing_frames + 1:
        return None

    candidate_position = np.array(
        [candidate["center_x"], candidate["center_y"]],
        dtype=np.float32,
    )
    last_position = np.array(
        [last_point["center_x"], last_point["center_y"]],
        dtype=np.float32,
    )

    if len(track["points"]) == 1:
        distance = float(np.linalg.norm(candidate_position - last_position))

        if distance > maximum_first_match_distance_pixels * frame_gap:
            return None

        return distance

    previous_point = track["points"][-2]
    previous_position = np.array(
        [previous_point["center_x"], previous_point["center_y"]],
        dtype=np.float32,
    )
    previous_gap = (
        last_point["frame_number"] - previous_point["frame_number"]
    )

    if previous_gap <= 0:
        return None

    velocity = (last_position - previous_position) / previous_gap
    predicted_position = last_position + velocity * frame_gap
    residual = float(np.linalg.norm(candidate_position - predicted_position))
    step_per_frame = float(
        np.linalg.norm(candidate_position - last_position) / frame_gap
    )

    if residual > maximum_predicted_residual_pixels:
        return None

    if step_per_frame > maximum_step_pixels_per_frame:
        return None

    return residual + 0.15 * step_per_frame


def update_tracks(tracks, candidates, frame_number, time_seconds, next_track_id):
    """associate current candidates with recent tracks using motion prediction."""
    possible_matches = []

    for track_index, track in enumerate(tracks):
        if not track["active"]:
            continue

        last_frame = track["points"][-1]["frame_number"]

        if frame_number - last_frame > maximum_missing_frames + 1:
            track["active"] = False
            continue

        for candidate_index, candidate in enumerate(candidates):
            cost = match_cost(track, candidate, frame_number)

            if cost is not None:
                possible_matches.append((cost, track_index, candidate_index))

    possible_matches.sort(key=lambda match: match[0])
    used_track_indices = set()
    used_candidate_indices = set()

    for _, track_index, candidate_index in possible_matches:
        if (
            track_index in used_track_indices
            or candidate_index in used_candidate_indices
        ):
            continue

        point = {
            "frame_number": frame_number,
            "time_seconds": round(time_seconds, 3),
            **candidates[candidate_index],
        }
        tracks[track_index]["points"].append(point)
        tracks[track_index]["active"] = True
        used_track_indices.add(track_index)
        used_candidate_indices.add(candidate_index)

    for candidate_index, candidate in enumerate(candidates):
        if candidate_index in used_candidate_indices:
            continue

        tracks.append(
            {
                "track_id": next_track_id,
                "active": True,
                "points": [
                    {
                        "frame_number": frame_number,
                        "time_seconds": round(time_seconds, 3),
                        **candidate,
                    }
                ],
            }
        )
        next_track_id += 1

    return tracks, next_track_id


def score_track(track):
    """return quality details, or none when a path is too weak to display."""
    points = track["points"]

    if len(points) < minimum_points_for_reliable_track:
        return None

    frame_span = points[-1]["frame_number"] - points[0]["frame_number"] + 1
    coverage = len(points) / frame_span

    if coverage < minimum_track_coverage:
        return None

    step_sizes = [
        per_frame_distance(points[index - 1], points[index])
        for index in range(1, len(points))
    ]
    median_step = float(np.median(step_sizes))
    total_displacement = float(
        np.hypot(
            points[-1]["center_x"] - points[0]["center_x"],
            points[-1]["center_y"] - points[0]["center_y"],
        )
    )

    if median_step < minimum_median_step_pixels_per_frame:
        return None

    if total_displacement < minimum_track_displacement_pixels:
        return None

    # longer, denser paths with sensible image movement earn a better score.
    score = (
        5.0 * len(points)
        + 18.0 * coverage
        + min(median_step, 60.0) / 4.0
        + min(total_displacement, 300.0) / 60.0
    )

    return {
        "score": round(score, 3),
        "point_count": len(points),
        "coverage": round(coverage, 3),
        "median_step_pixels_per_frame": round(median_step, 3),
        "total_displacement_pixels": round(total_displacement, 3),
    }


def choose_current_best_track(tracks, frame_number):
    """select at most one recent, reliable path for the current frame."""
    options = []

    for track in tracks:
        if not track["active"]:
            continue

        if frame_number - track["points"][-1]["frame_number"] > 1:
            continue

        quality = score_track(track)

        if quality is not None:
            options.append((quality["score"], track, quality))

    if not options:
        return None, None

    options.sort(key=lambda item: item[0], reverse=True)
    _, selected_track, quality = options[0]
    return selected_track, quality


def draw_selected_track(frame, track, quality):
    """draw only direct, recent observations from the selected track."""
    points = track["points"][-maximum_drawn_history_points:]
    path_color = (0, 165, 255)  # orange, deliberately not court green.

    for index in range(1, len(points)):
        earlier_point = points[index - 1]
        later_point = points[index]

        # never draw across a frame gap: that would imply an invented path.
        if later_point["frame_number"] - earlier_point["frame_number"] != 1:
            continue

        cv2.line(
            frame,
            (int(earlier_point["center_x"]), int(earlier_point["center_y"])),
            (int(later_point["center_x"]), int(later_point["center_y"])),
            path_color,
            2,
        )

    current_point = points[-1]
    x, y, width, height = current_point["box"]
    cv2.rectangle(frame, (x, y), (x + width, y + height), path_color, 2)
    cv2.circle(
        frame,
        (int(current_point["center_x"]), int(current_point["center_y"])),
        5,
        (0, 255, 255),
        -1,
    )
    cv2.putText(
        frame,
        f"temporary ball candidate | path {track['track_id']}",
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        path_color,
        2,
    )
    cv2.putText(
        frame,
        f"quality {quality['score']:.1f}",
        (x, min(frame.shape[0] - 12, y + height + 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        path_color,
        1,
    )


def load_court_polygon(path):
    """load and validate the four clicked singles-court corners."""
    with open(path, "r", encoding="utf-8") as file:
        corners = json.load(file)

    if not isinstance(corners, list) or len(corners) != 4:
        raise RuntimeError(
            "court_corner_pixels.json must contain exactly four corners."
        )

    return np.array(corners, dtype=np.int32).reshape((-1, 1, 2))


for required_path in [video_path, court_corners_path, pose_model_path]:
    if not required_path.exists():
        raise FileNotFoundError(f"Could not find {required_path}")

capture = cv2.VideoCapture(str(video_path))

if not capture.isOpened():
    raise RuntimeError(f"Could not open {video_path}")

frames_per_second = capture.get(cv2.CAP_PROP_FPS)
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

if frames_per_second <= 0:
    capture.release()
    raise RuntimeError("The video has no valid frame rate.")

maximum_frames = int(frames_per_second * test_duration_seconds)
court_polygon = load_court_polygon(court_corners_path)

court_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.fillConvexPoly(court_mask, court_polygon, 255)
expansion_size = court_search_expansion_pixels * 2 + 1
search_mask = cv2.dilate(
    court_mask,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (expansion_size, expansion_size),
    ),
)

writer = cv2.VideoWriter(
    str(output_video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    frames_per_second,
    (frame_width, frame_height),
)

if not writer.isOpened():
    capture.release()
    raise RuntimeError(f"Could not create {output_video_path}")

pose_model = YOLO(str(pose_model_path))
previous_gray_frame = None
latest_ankle_points = []
latest_body_core_polygons = []
tracks = []
next_track_id = 1
processed_frames = 0
total_candidates_rejected_near_ankles = 0
total_candidates_rejected_in_body_core = 0
frames_with_reliable_candidate = 0
selected_track_history = []

for frame_number in range(maximum_frames):
    frame_read, frame = capture.read()

    if not frame_read:
        break

    time_seconds = frame_number / frames_per_second

    if frame_number % pose_detection_interval_frames == 0:
        (
            latest_ankle_points,
            latest_body_core_polygons,
        ) = find_player_protected_regions(frame, pose_model)

    (
        candidates,
        current_gray_frame,
        rejected_near_ankle,
        rejected_in_body_core,
    ) = (
        extract_ball_like_candidates(
            frame,
            previous_gray_frame,
            search_mask,
            latest_ankle_points,
            latest_body_core_polygons,
        )
    )
    total_candidates_rejected_near_ankles += rejected_near_ankle
    total_candidates_rejected_in_body_core += rejected_in_body_core

    tracks, next_track_id = update_tracks(
        tracks,
        candidates,
        frame_number,
        time_seconds,
        next_track_id,
    )
    selected_track, selected_quality = choose_current_best_track(
        tracks,
        frame_number,
    )

    annotated_frame = frame.copy()
    cv2.polylines(annotated_frame, [court_polygon], True, (0, 255, 0), 2)

    if selected_track is not None:
        draw_selected_track(annotated_frame, selected_track, selected_quality)
        frames_with_reliable_candidate += 1
        selected_track_history.append(
            {
                "frame_number": frame_number,
                "time_seconds": round(time_seconds, 3),
                "track_id": selected_track["track_id"],
                "quality": selected_quality,
            }
        )

    status_text = (
        "Reliable temporary ball candidate"
        if selected_track is not None
        else "No reliable ball candidate in this frame"
    )
    status_color = (0, 255, 255) if selected_track is not None else (180, 180, 180)

    cv2.rectangle(annotated_frame, (15, 15), (760, 150), (0, 0, 0), -1)
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_number}",
        (30, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        status_text,
        (30, 71),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        status_color,
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Small moving candidates: {len(candidates)} | "
        f"rejected near ankles: {rejected_near_ankle}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        annotated_frame,
        f"Rejected inside player torso: {rejected_in_body_core} | "
        "orange = direct observations only.",
        (30, 123),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    writer.write(annotated_frame)
    previous_gray_frame = current_gray_frame
    processed_frames += 1

    if frame_number % 30 == 0:
        print(f"Processed frame {frame_number} of about {maximum_frames}.")

capture.release()
writer.release()

reliable_tracks = []

for track in tracks:
    quality = score_track(track)

    if quality is not None:
        reliable_tracks.append(
            {
                "track_id": track["track_id"],
                **quality,
                "points": track["points"],
            }
        )

output_data = {
    "purpose": (
        "Temporary ball-candidate diagnostics. A listed path is not a "
        "guaranteed tennis ball."
    ),
    "test_duration_seconds": test_duration_seconds,
    "processed_frames": processed_frames,
    "frames_with_reliable_candidate": frames_with_reliable_candidate,
    "candidates_rejected_near_ankles": total_candidates_rejected_near_ankles,
    "candidates_rejected_in_body_core": total_candidates_rejected_in_body_core,
    "reliable_track_count": len(reliable_tracks),
    "reliable_tracks": reliable_tracks,
    "selected_track_history": selected_track_history,
}

with open(output_data_path, "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=2)

print()
print("Stricter temporal ball-candidate analysis completed.")
print(f"Processed frames: {processed_frames}")
print(f"Frames with a reliable temporary candidate: {frames_with_reliable_candidate}")
print(f"Candidates rejected near player ankles: {total_candidates_rejected_near_ankles}")
print(f"Candidates rejected inside player torsos: {total_candidates_rejected_in_body_core}")
print(f"Reliable temporary tracks saved: {len(reliable_tracks)}")
print(f"Saved diagnostic video to: {output_video_path}")
print(f"Saved track data to: {output_data_path}")
