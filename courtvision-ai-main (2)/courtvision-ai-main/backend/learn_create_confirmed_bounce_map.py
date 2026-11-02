"""draw confirmed bounce locations on the court."""

import json
from pathlib import Path

import cv2
import numpy as np


CONFIRMED_BOUNCES_PATH = Path("outputs/confirmed_bounces.json")
OUTPUT_IMAGE_PATH = Path("outputs/confirmed_bounce_map.png")

# standard singles-court dimensions in meters.
SINGLES_COURT_WIDTH_METERS = 8.23
COURT_LENGTH_METERS = 23.77
NET_Y_METERS = COURT_LENGTH_METERS / 2

# the service line is 6.40 m from the net on each side.
SERVICE_LINE_DISTANCE_FROM_NET_METERS = 6.40

# controls the size of the generated image.
PIXELS_PER_METER = 42
PADDING_PIXELS = 120


def load_confirmed_bounces():
    """
    load bounce events that already confirmed by manual review.
    """
    with CONFIRMED_BOUNCES_PATH.open("r", encoding="utf-8") as file:
        bounce_data = json.load(file)

    return bounce_data["confirmed_bounces"]


def court_to_image_pixels(court_x_meters, court_y_meters):
    """convert court coordinates to image pixels."""
    image_x = int(
        PADDING_PIXELS + court_x_meters * PIXELS_PER_METER
    )
    image_y = int(
        PADDING_PIXELS + court_y_meters * PIXELS_PER_METER
    )

    return image_x, image_y


def create_court_canvas():
    """
    create a blue court and draw the main singles-court markings.
    """
    court_width_pixels = int(
        SINGLES_COURT_WIDTH_METERS * PIXELS_PER_METER
    )
    court_length_pixels = int(
        COURT_LENGTH_METERS * PIXELS_PER_METER
    )

    canvas_width = court_width_pixels + 2 * PADDING_PIXELS
    canvas_height = court_length_pixels + 2 * PADDING_PIXELS

    # opencv colors use bgr order, not rgb.
    canvas = np.full(
        (canvas_height, canvas_width, 3),
        (180, 95, 35),
        dtype=np.uint8,
    )

    left_x = PADDING_PIXELS
    right_x = PADDING_PIXELS + court_width_pixels
    top_y = PADDING_PIXELS
    bottom_y = PADDING_PIXELS + court_length_pixels

    white = (255, 255, 255)
    line_thickness = 3

    # singles-court outer boundary.
    cv2.rectangle(
        canvas,
        (left_x, top_y),
        (right_x, bottom_y),
        white,
        line_thickness,
    )

    # net.
    _, net_y = court_to_image_pixels(0, NET_Y_METERS)
    cv2.line(
        canvas,
        (left_x, net_y),
        (right_x, net_y),
        white,
        line_thickness,
    )

    # service lines on both sides of the net.
    far_service_y_meters = (
        NET_Y_METERS - SERVICE_LINE_DISTANCE_FROM_NET_METERS
    )
    near_service_y_meters = (
        NET_Y_METERS + SERVICE_LINE_DISTANCE_FROM_NET_METERS
    )

    _, far_service_y = court_to_image_pixels(0, far_service_y_meters)
    _, near_service_y = court_to_image_pixels(0, near_service_y_meters)

    cv2.line(
        canvas,
        (left_x, far_service_y),
        (right_x, far_service_y),
        white,
        line_thickness,
    )
    cv2.line(
        canvas,
        (left_x, near_service_y),
        (right_x, near_service_y),
        white,
        line_thickness,
    )

    # center service line runs only b/w the two service lines.
    center_x, _ = court_to_image_pixels(
        SINGLES_COURT_WIDTH_METERS / 2,
        0,
    )
    cv2.line(
        canvas,
        (center_x, far_service_y),
        (center_x, near_service_y),
        white,
        line_thickness,
    )

    # title and orientation labels.
    cv2.putText(
        canvas,
        "Manually Confirmed Tennis Bounces",
        (PADDING_PIXELS, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        white,
        2,
    )

    cv2.putText(
        canvas,
        "court y = 0 m",
        (PADDING_PIXELS, canvas_height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        white,
        2,
    )

    return canvas


def draw_confirmed_bounces(canvas, confirmed_bounces):
    """draw confirmed bounce locations on the court."""
    drawn_count = 0

    for bounce in confirmed_bounces:
        court_position = bounce["court_position_meters"]
        court_x = court_position["x"]
        court_y = court_position["y"]

        # a confirmed bounce should be within the singles court. still
        # validate here so an unexpected bad coordinate does not distort the map.
        if not (
            0 <= court_x <= SINGLES_COURT_WIDTH_METERS
            and 0 <= court_y <= COURT_LENGTH_METERS
        ):
            print(
                f"Skipping {bounce['bounce_id']}: "
                f"outside court ({court_x:.2f}, {court_y:.2f})"
            )
            continue

        image_x, image_y = court_to_image_pixels(court_x, court_y)

        # green circle with black outline: visible on the blue court.
        cv2.circle(
            canvas,
            (image_x, image_y),
            12,
            (0, 0, 0),
            -1,
        )
        cv2.circle(
            canvas,
            (image_x, image_y),
            9,
            (0, 255, 0),
            -1,
        )

        bounce_number = bounce["bounce_id"].replace("bounce_", "")

        cv2.putText(
            canvas,
            bounce_number,
            (image_x + 13, image_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        drawn_count += 1

    return drawn_count


def main():
    """run the script's main workflow."""
    confirmed_bounces = load_confirmed_bounces()
    court_map = create_court_canvas()

    drawn_count = draw_confirmed_bounces(
        court_map,
        confirmed_bounces,
    )

    saved = cv2.imwrite(str(OUTPUT_IMAGE_PATH), court_map)

    if not saved:
        raise RuntimeError(
            f"Could not save court map: {OUTPUT_IMAGE_PATH}"
        )

    print(f"Confirmed bounce events loaded: {len(confirmed_bounces)}")
    print(f"Bounce markers drawn inside singles court: {drawn_count}")
    print(f"Saved bounce court map: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()