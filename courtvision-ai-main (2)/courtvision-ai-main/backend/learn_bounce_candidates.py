"""find possible bounces from direct ball observations."""
import cv2
import numpy as np

import json
import math
from pathlib import Path


# path points to the json created after racketvision correctly tracked the ball
INPUT_PATH = Path("outputs/racketvision_ball_tracking_data.json")

# this json file contains the image-to-court transformation matrix
# created when calibrating the court
HOMOGRAPHY_PATH = Path(
    "outputs/image_to_court_homography.json"
)

# this is where our bounce-review results will be saved.
OUTPUT_PATH = Path(
    "outputs/learned_bounce_review_candidates.json"
)

# original video used for the review overlay.
VIDEO_PATH = Path("data/raw/test_rally.mp4")

# new video that will show direct ball observations and possible bounces.
REVIEW_VIDEO_OUTPUT_PATH = Path(
    "outputs/learned_bounce_review.mp4"
)

# manual review results in chronological candidate order.
#
# these labels came from watching the review video.
#
# do not change this order unless the chronological candidate order changes.
MANUAL_REVIEW_LABELS = [
    "confirmed_bounce",
    "rejected_bad_detection",
    "confirmed_bounce",
    "confirmed_bounce",
    "rejected_bad_detection",
    "confirmed_bounce",
    "confirmed_racket_contact",
    "confirmed_bounce",
    "confirmed_bounce",
    "confirmed_racket_contact",
    "confirmed_racket_contact",
    "confirmed_bounce",
    "confirmed_racket_contact",
    "confirmed_bounce",
    "confirmed_bounce",
]

# this label tells us the point came directly from the model
# we don't want to use guessed or interpolated points later
DIRECT_PROVENANCE = "direct_temporal_model_observation"


def load_direct_ball_observations(path):
    """load direct ball observations from the input file."""

    # stop early with error message if the tracking step has not been run
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. "
            "Run the RacketVision ball-tracking script first."
        )

    # open the json file and turn it into normal python dictionaries and lists
    with open(path, "r", encoding="utf-8") as file:
        tracking_data = json.load(file)

    # empty list will gradually hold the good observations
    observations = []

    # the json has one record for each video frame.
    for frame_record in tracking_data["frames"]:

        # get the nested ball information for this frame
        ball = frame_record["ball"]

        # a ball record can be missing, invisible, or direct
        # we keep it only if the model says it is visible and direct
        is_direct_observation = (
            ball["visible"]
            and ball["provenance"] == DIRECT_PROVENANCE
            and ball["image_point_pixels"] is not None
        )

        if not is_direct_observation:
            continue

        # image_point_pixels is stored as [x, y]
        x_pixels, y_pixels = ball["image_point_pixels"]

        # save a smaller, easier-to-use record
        observations.append(
            {
                "frame": frame_record["frame"],
                "time_seconds": frame_record["time_seconds"],
                "x_pixels": x_pixels,
                "y_pixels": y_pixels,
            }
        )

    return observations

def create_observation_lookup(observations):
    """map each observation to its frame."""

    # the frame number becomes the dictionary key
    # the full observation becomes the dictionary value
    return {
        observation["frame"]: observation
        for observation in observations
    }


def load_court_homography(path):
    """load the court image-to-ground mapping."""

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. "
            "Build the court homography before filtering bounces."
        )

    with open(path, "r", encoding="utf-8") as file:
        homography_data = json.load(file)

    # convert the saved nested list into a numpy matrix that opencv can use.
    image_to_court_matrix = np.array(
        homography_data["image_to_court_meters_matrix"],
        dtype=np.float32,
    )

    # a homography must always be a 3 by 3 matrix.
    if image_to_court_matrix.shape != (3, 3):
        raise ValueError(
            "The court homography matrix must have shape 3 x 3."
        )

    court_width_meters = homography_data[
        "court_width_meters"
    ]

    court_length_meters = homography_data[
        "court_length_meters"
    ]

    return (
        image_to_court_matrix,
        court_width_meters,
        court_length_meters,
    )

def pixel_to_court_meters(
    x_pixels,
    y_pixels,
    image_to_court_matrix,
):
    """convert image coordinates to court meters."""

    # create one opencv-compatible point.
    image_point = np.array(
        [[[x_pixels, y_pixels]]],
        dtype=np.float32,
    )

    # apply the saved perspective transformation.
    transformed_point = cv2.perspectiveTransform(
        image_point,
        image_to_court_matrix,
    )

    # opencv returns a nested array, so extract the actual x and y values.
    court_x_meters, court_y_meters = transformed_point[0][0]

    return (
        float(court_x_meters),
        float(court_y_meters),
    )

def calculate_direct_motion(observations):
    """measure motion between observed ball positions."""

    # this list will hold one motion measurement per valid frame pair
    motion_records = []

    # zip pairs each observation with the next one:
    # first loop:  observation 1 and observation 2
    # second loop: observation 2 and observation 3
    # etc...
    for previous, current in zip(observations, observations[1:]):

        # find how many video frames separate the two observations
        frame_gap = current["frame"] - previous["frame"]
        # find the time between the observations in seconds
        time_delta_seconds = (
            current["time_seconds"] - previous["time_seconds"]
        )

        # we only trust direct, consecutive frame pairs
        # example of accepted pair:
        # frame 12 -> frame 13
        # example of rejected pair:
        # frame 8 -> frame 11
        if frame_gap != 1 or time_delta_seconds <= 0:
            continue

        # calculate how far the ball moved horizontally in pixels
        delta_x_pixels = current["x_pixels"] - previous["x_pixels"]

        # calculate how far the ball moved vertically in pixels
        delta_y_pixels = current["y_pixels"] - previous["y_pixels"]

        # math.hypot uses the pythagorean theorem:
        # distance = square root of (delta x squared + delta y squared)
        # this gives the straight-line distance the ball moved in the image
        distance_pixels = math.hypot(
            delta_x_pixels,
            delta_y_pixels,
        )

        # pixels per second is useful for comparing movement in the video
        # important: this is not real tennis-ball speed in km/h
        # the ball may be above the court, and one camera cannot reliably
        # calculate its 3-d physical position yet
        image_speed_pixels_per_second = (
            distance_pixels / time_delta_seconds
        )

        # save everything we may need later for bounce analysis
        motion_records.append(
            {
                "from_frame": previous["frame"],
                "to_frame": current["frame"],
                "time_seconds": current["time_seconds"],
                "delta_x_pixels": delta_x_pixels,
                "delta_y_pixels": delta_y_pixels,
                "distance_pixels": distance_pixels,
                "image_speed_pixels_per_second": (
                    image_speed_pixels_per_second
                ),
            }
        )

    return motion_records

def calculate_motion_continuity_limit(motion_records):
    """set the largest motion allowed between observations."""

    # pull only the image-speed number from every trusted motion record.
    image_speeds = [
        motion["image_speed_pixels_per_second"]
        for motion in motion_records
    ]

    # we need at least one speed measurement to calculate a threshold.
    if not image_speeds:
        raise RuntimeError(
            "Cannot calculate a motion limit without motion records."
        )

    # the 99th percentile means:
    #
    # 99 percent of the recorded speeds are at or below this value.
    #
    # we use numpy because it can calculate percentiles reliably.
    speed_at_99th_percentile = np.percentile(
        image_speeds,
        99,
    )

    # add a safety buffer so a genuine fast shot near the top of the
    # normal motion range is not rejected too aggressively.
    safety_multiplier = 1.25

    motion_limit_pixels_per_second = (
        speed_at_99th_percentile * safety_multiplier
    )

    return float(motion_limit_pixels_per_second)

def calculate_direction_changes(motion_records):
    """find changes in the ball direction."""

    # this list will store direction-change measurements.
    direction_changes = []

    # compare each motion record with the one immediately after it.
    for previous_motion, current_motion in zip(
        motion_records,
        motion_records[1:],
    ):

        # the two movements must connect.
        #
        # valid:
        # previous: 12 → 13
        # current:  13 → 14
        #
        # invalid:
        # previous: 8 → 11
        # current:  11 → 12
        #
        # we do not compare motion across a tracking gap.
        if previous_motion["to_frame"] != current_motion["from_frame"]:
            continue

        # treat each movement as a 2-d direction vector.
        previous_x = previous_motion["delta_x_pixels"]
        previous_y = previous_motion["delta_y_pixels"]

        current_x = current_motion["delta_x_pixels"]
        current_y = current_motion["delta_y_pixels"]

        # the dot product tells us whether two vectors point similarly.
        #
        # positive dot product: similar general direction
        # zero dot product: roughly a right-angle turn
        # negative dot product: moving back in the opposite direction
        dot_product = (
            previous_x * current_x
            + previous_y * current_y
        )

        # find the length of both movement vectors.
        previous_length = math.hypot(previous_x, previous_y)
        current_length = math.hypot(current_x, current_y)

        # avoid dividing by zero if a movement was effectively zero pixels.
        if previous_length == 0 or current_length == 0:
            continue

        # this formula converts the dot product into the cosine of the angle.
        cosine_of_angle = dot_product / (
            previous_length * current_length
        )

        # small floating-point rounding errors can create values slightly
        # above 1 or below -1. acos only accepts values from -1 through 1.
        cosine_of_angle = max(
            -1.0,
            min(1.0, cosine_of_angle),
        )

        # acos gives radians. degrees converts radians into an easier number
        # for humans to read, such as 35 degrees or 145 degrees.
        turn_angle_degrees = math.degrees(
            math.acos(cosine_of_angle)
        )

        direction_changes.append(
            {
                # this is the shared middle frame where the turn occurred.
                "turn_frame": previous_motion["to_frame"],

                # keep both movements so we can inspect them later.
                "previous_motion": previous_motion,
                "current_motion": current_motion,

                # the key measurement for future bounce candidates.
                "turn_angle_degrees": turn_angle_degrees,
            }
        )

    return direction_changes

def build_raw_turn_candidates(
    direction_changes,
    observation_lookup,
):
    """find possible turns in the ball path."""

    # this number is intentionally modest.
    #
    # tiny one- or two-pixel movements can come from model jitter rather
    # than real ball movement, so we ignore them for bounce analysis.
    minimum_movement_pixels = 3.0

    # we are looking for a clearly noticeable turn.
    #
    # 0 degrees = same direction
    # 90 degrees = right-angle direction change
    # 180 degrees = opposite direction
    #
    # we will adjust this later only after seeing real review results.
    minimum_turn_angle_degrees = 55.0

    raw_candidates = []

    for direction_change in direction_changes:

        # get the angle calculated in the previous step.
        turn_angle = direction_change["turn_angle_degrees"]

        # get the movement immediately before and after the turn.
        previous_motion = direction_change["previous_motion"]
        current_motion = direction_change["current_motion"]

        # both sides of a real event should involve meaningful movement.
        incoming_distance = previous_motion["distance_pixels"]
        outgoing_distance = current_motion["distance_pixels"]

        # reject weak or nearly-stationary motion.
        if min(incoming_distance, outgoing_distance) < minimum_movement_pixels:
            continue

        # reject gentle turns. they are normal flight movement, not a
        # useful starting point for bounce review.
        if turn_angle < minimum_turn_angle_degrees:
            continue

        # find the direct ball observation at the exact middle frame.
        turn_frame = direction_change["turn_frame"]
        ball_observation = observation_lookup.get(turn_frame)

        # this should normally exist, but this safety check prevents a
        # confusing crash if the data format changes later.
        if ball_observation is None:
            continue

        raw_candidates.append(
            {
                # "candidate" is deliberate. this is not a confirmed bounce.
                "candidate_frame": turn_frame,
                "candidate_time_seconds": (
                    ball_observation["time_seconds"]
                ),

                # this is the ball's location in the original video image.
                "x_pixels": ball_observation["x_pixels"],
                "y_pixels": ball_observation["y_pixels"],

                # evidence explaining why the frame was selected.
                "turn_angle_degrees": turn_angle,
                "incoming_distance_pixels": incoming_distance,
                "outgoing_distance_pixels": outgoing_distance,
                "incoming_image_speed_pixels_per_second": (
                    previous_motion[
                        "image_speed_pixels_per_second"
                    ]
                ),
                "outgoing_image_speed_pixels_per_second": (
                    current_motion[
                        "image_speed_pixels_per_second"
                    ]
                ),

                # this makes the uncertainty visible in the data itself.
                "status": "raw_turn_candidate_not_confirmed_bounce",
            }
        )

    return raw_candidates

def filter_candidates_to_court(
    raw_candidates,
    image_to_court_matrix,
    court_width_meters,
    court_length_meters,
):
    """keep candidates inside the court."""

    # allow a small amount beyond each court boundary.
    #
    # we do not want to reject a real bounce just because the model was
    # a few pixels off near a sideline or baseline.
    court_tolerance_meters = 0.35

    court_candidates = []

    for candidate in raw_candidates:

        # convert this candidate's image location to estimated court metres.
        court_x_meters, court_y_meters = pixel_to_court_meters(
            candidate["x_pixels"],
            candidate["y_pixels"],
            image_to_court_matrix,
        )

        # check whether the candidate is inside the court, allowing
        # a small tolerance around the boundaries.
        inside_court = (
            -court_tolerance_meters
            <= court_x_meters
            <= court_width_meters + court_tolerance_meters
            and
            -court_tolerance_meters
            <= court_y_meters
            <= court_length_meters + court_tolerance_meters
        )

        if not inside_court:
            continue

        # copy() makes a new dictionary, so we preserve the original
        # raw candidate and add court-specific information separately.
        court_candidate = candidate.copy()

        court_candidate["court_x_meters"] = court_x_meters
        court_candidate["court_y_meters"] = court_y_meters

        court_candidate["status"] = (
            "court_filtered_turn_candidate_not_confirmed_bounce"
        )

        court_candidates.append(court_candidate)

    return court_candidates

def filter_candidates_by_motion_continuity(
    court_candidates,
    motion_limit_pixels_per_second,
):
    """remove candidates with implausible motion."""

    continuity_filtered_candidates = []

    for candidate in court_candidates:

        # a possible bounce has motion entering it and motion leaving it.
        incoming_speed = candidate[
            "incoming_image_speed_pixels_per_second"
        ]

        outgoing_speed = candidate[
            "outgoing_image_speed_pixels_per_second"
        ]

        # if either side of the turn exceeds the rally-specific limit,
        # do not use this candidate for bounce analysis.
        #
        # we do not delete the original tracking data. we only prevent
        # this suspicious movement from becoming bounce evidence.
        if (
            incoming_speed > motion_limit_pixels_per_second
            or outgoing_speed > motion_limit_pixels_per_second
        ):
            continue

        continuity_candidate = candidate.copy()

        continuity_candidate["motion_limit_pixels_per_second"] = (
            motion_limit_pixels_per_second
        )

        continuity_candidate["status"] = (
            "motion_and_court_filtered_turn_candidate_not_confirmed_bounce"
        )

        continuity_filtered_candidates.append(
            continuity_candidate
        )

    return continuity_filtered_candidates

def remove_nearby_duplicate_candidates(
    continuity_candidates,
):
    """keep one candidate from each nearby group."""

    # at 30 fps, six frames is about 0.2 seconds.
    #
    # this is not a bounce-duration claim. it is simply a small window
    # for avoiding several labels for the same visual event.
    minimum_separation_frames = 6

    # start with the strongest turns first.
    #
    # if frames 92 and 93 are near each other, whichever one has the
    # larger angle gets considered first and is therefore kept.
    candidates_sorted_by_strength = sorted(
        continuity_candidates,
        key=lambda candidate: candidate["turn_angle_degrees"],
        reverse=True,
    )

    final_candidates = []

    for candidate in candidates_sorted_by_strength:

        candidate_frame = candidate["candidate_frame"]

        # check whether this candidate is too close to any event we
        # already kept.
        is_too_close_to_existing_event = any(
            abs(
                candidate_frame
                - kept_candidate["candidate_frame"]
            )
            <= minimum_separation_frames
            for kept_candidate in final_candidates
        )

        # if a stronger candidate already represents this short time
        # window, skip this weaker duplicate.
        if is_too_close_to_existing_event:
            continue

        final_candidate = candidate.copy()

        final_candidate["duplicate_window_frames"] = (
            minimum_separation_frames
        )

        final_candidate["status"] = (
            "deduplicated_bounce_review_candidate_not_confirmed"
        )

        final_candidates.append(final_candidate)

    # sort back into chronological order before returning.
    #
    # the earlier sorting was only for selecting the strongest candidate.
    return sorted(
        final_candidates,
        key=lambda candidate: candidate["candidate_frame"],
    )

def attach_manual_review_labels(
    final_candidates,
    manual_review_labels,
):
    """add manual review labels to candidates."""

    # make sure the candidate list is in chronological order.
    chronological_candidates = sorted(
        final_candidates,
        key=lambda candidate: candidate["candidate_frame"],
    )

    # we need exactly one human label for every final candidate.
    #
    # if this check fails, stop instead of silently assigning labels to
    # the wrong events.
    if len(chronological_candidates) != len(
        manual_review_labels
    ):
        raise ValueError(
            "The number of manual labels does not match the "
            "number of final review candidates."
        )

    reviewed_candidates = []

    # zip pairs the first candidate with the first label,
    # the second candidate with the second label, and so on.
    for candidate, manual_label in zip(
        chronological_candidates,
        manual_review_labels,
    ):
        reviewed_candidate = candidate.copy()

        # preserve the automatic status so we never lose information
        # about how this candidate was originally generated.
        reviewed_candidate["automatic_status"] = (
            candidate["status"]
        )

        # add the human decision.
        reviewed_candidate["manual_review_label"] = (
            manual_label
        )

        # this status now means a person has reviewed the event.
        reviewed_candidate["status"] = "manually_reviewed"

        reviewed_candidates.append(reviewed_candidate)

    return reviewed_candidates

def save_bounce_review_candidates(
    output_path,
    final_candidates,
    observation_count,
    motion_record_count,
    motion_limit_pixels_per_second,
):
    """save bounce candidates for manual review."""

    # create the outputs folder if it does not already exist.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # keep metadata with the candidates so future code understands:
    # - where the candidates came from,
    # - what filtering rules were used,
    # - and what the important limitations are.
    output_data = {
        "schema_version": 1,

        "source": {
            "ball_tracking_file": str(INPUT_PATH),
            "court_homography_file": str(HOMOGRAPHY_PATH),
        },

        "summary": {
            "direct_ball_observations": observation_count,
            "trusted_motion_records": motion_record_count,
            "motion_continuity_limit_pixels_per_second": (
                motion_limit_pixels_per_second
            ),
            "bounce_review_candidate_count": len(
                final_candidates
            ),
        },

        # this text protects future code from accidentally treating a
        # candidate as a confirmed bounce.
        "important_limit": (
            "These are automatic review candidates, not confirmed "
            "bounces. Each candidate needs video review before it is "
            "used as a factual bounce event."
        ),

        # store the final list itself.
        "candidates": final_candidates,
    }

    # indent=2 makes the json readable when you open it manually.
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            indent=2,
        )

def create_bounce_review_video(
    video_path,
    output_path,
    observation_lookup,
    final_candidates,
):
    """create a video showing the bounce candidates."""

    # open the original source video.
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open source video: {video_path}"
        )

    # read video details needed to create a matching output video.
    frame_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frames_per_second = capture.get(
        cv2.CAP_PROP_FPS
    )

    if (
        frame_width <= 0
        or frame_height <= 0
        or frames_per_second <= 0
    ):
        raise RuntimeError(
            "The source video does not report usable dimensions "
            "and frame rate."
        )

    # make sure the output folder exists.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # mp4v is a broadly compatible video codec for mp4 output.
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frames_per_second,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    # turn the candidate list into another fast lookup dictionary.
    #
    # example:
    # candidate_lookup[93]
    #
    # returns the bounce-review candidate for frame 93.
    candidate_lookup = {
        candidate["candidate_frame"]: candidate
        for candidate in final_candidates
    }

    frame_number = 0

    while True:
        success, frame = capture.read()

        # stop when opencv reaches the end of the source video.
        if not success:
            break

        # draw the red dot only when there is a direct observation
        # at this exact video frame.
        ball_observation = observation_lookup.get(
            frame_number
        )

        if ball_observation is not None:
            ball_x = round(
                ball_observation["x_pixels"]
            )

            ball_y = round(
                ball_observation["y_pixels"]
            )

            # opencv colors use bgr, not rgb:
            #
            # (0, 0, 255) means red.
            cv2.circle(
                frame,
                (ball_x, ball_y),
                5,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

        # check whether this frame is one of our 15 review candidates.
        candidate = candidate_lookup.get(frame_number)

        if candidate is not None:
            candidate_x = round(candidate["x_pixels"])
            candidate_y = round(candidate["y_pixels"])

            # in opencv bgr format, this is orange.
            orange = (0, 165, 255)

            # draw an orange ring around the direct ball point.
            cv2.circle(
                frame,
                (candidate_x, candidate_y),
                14,
                orange,
                2,
                cv2.LINE_AA,
            )

            # keep the label inside the video frame.
            label_x = min(
                candidate_x + 18,
                frame_width - 220,
            )

            label_y = max(
                candidate_y - 16,
                30,
            )

            cv2.putText(
                frame,
                "BOUNCE? review",
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                orange,
                2,
                cv2.LINE_AA,
            )

        # write this annotated frame into the new video.
        writer.write(frame)

        # move to the next frame number.
        frame_number += 1

    # always release video resources when finished.
    capture.release()
    writer.release()

    print(
        f"Saved bounce review video to: {output_path}"
    )

def main():
    # step 1: load only direct, trustworthy ball observations.
    """run the script's main workflow."""
    observations = load_direct_ball_observations(INPUT_PATH)

    print(f"Direct ball observations: {len(observations)}")

    # step 2: create a fast frame-number lookup.
    observation_lookup = create_observation_lookup(observations)

    # step 3: load the court calibration once.
    (
        image_to_court_matrix,
        court_width_meters,
        court_length_meters,
    ) = load_court_homography(HOMOGRAPHY_PATH)

    # step 4: measure only direct motion between consecutive frames.
    motion_records = calculate_direct_motion(observations)

    print(
        f"Trusted consecutive motion records: "
        f"{len(motion_records)}"
    )

    # step 5: learn the unusually high one-frame speed limit
    # from this specific rally.
    motion_limit_pixels_per_second = (
        calculate_motion_continuity_limit(motion_records)
    )

    print(
        "Motion continuity limit: "
        f"{motion_limit_pixels_per_second:.2f} "
        "pixels per second"
    )

    # step 6: compare connected motion vectors for sharp turns.
    direction_changes = calculate_direction_changes(
        motion_records
    )

    print(
        f"Connected direction-change records: "
        f"{len(direction_changes)}"
    )

    # step 7: keep large turns and attach each one to the
    # directly observed ball location.
    raw_candidates = build_raw_turn_candidates(
        direction_changes,
        observation_lookup,
    )

    print(
        f"\nRaw turn candidates before court filtering: "
        f"{len(raw_candidates)}"
    )

    # step 8: reject candidates outside the calibrated court.
    court_candidates = filter_candidates_to_court(
        raw_candidates,
        image_to_court_matrix,
        court_width_meters,
        court_length_meters,
    )

    print(
        f"Court-filtered turn candidates: "
        f"{len(court_candidates)}"
    )

    # step 9: reject candidates that rely on an extreme
    # one-frame ball-position jump.
    continuity_candidates = (
        filter_candidates_by_motion_continuity(
            court_candidates,
            motion_limit_pixels_per_second,
        )
    )

    print(
        "Motion and court filtered candidates: "
        f"{len(continuity_candidates)}"
    )

    # merge nearby sharp turns so a single visual event does not appear
    # as several separate bounce candidates.
    final_candidates = remove_nearby_duplicate_candidates(
        continuity_candidates
    )

    # attach the decisions you made while watching the review video
    reviewed_candidates = attach_manual_review_labels(
        final_candidates,
        MANUAL_REVIEW_LABELS,
    )

    print(
        "Deduplicated review candidates: "
        f"{len(reviewed_candidates)}"
    )

    print("\nFinal bounce-review candidates:")

    for candidate in reviewed_candidates:
        print(
            {
                "frame": candidate["candidate_frame"],
                "time_seconds": candidate[
                    "candidate_time_seconds"
                ],
                "court_x_meters": round(
                    candidate["court_x_meters"],
                    2,
                ),
                "court_y_meters": round(
                    candidate["court_y_meters"],
                    2,
                ),
                "turn_angle_degrees": round(
                    candidate["turn_angle_degrees"],
                    2,
                ),
                "status": candidate["status"],
            }
        )

    # save the candidates so another script can create a review video
    # or let a person confirm/reject each possible bounce.
    save_bounce_review_candidates(
        OUTPUT_PATH,
        reviewed_candidates,
        len(observations),
        len(motion_records),
        motion_limit_pixels_per_second,
    )

    print(
        f"\nSaved bounce-review candidates to: "
        f"{OUTPUT_PATH}"
    )

    save_bounce_review_candidates(
        OUTPUT_PATH,
        reviewed_candidates,
        len(observations),
        len(motion_records),
        motion_limit_pixels_per_second,
    )

    print(
        f"\nSaved bounce-review candidates to: "
        f"{OUTPUT_PATH}"
    )

    create_bounce_review_video(
        VIDEO_PATH,
        REVIEW_VIDEO_OUTPUT_PATH,
        observation_lookup,
        reviewed_candidates,
    )

if __name__ == "__main__":
    main()