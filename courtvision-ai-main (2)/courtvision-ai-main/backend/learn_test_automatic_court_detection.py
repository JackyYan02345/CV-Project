"""test automatic court detection on a validation image."""

from pathlib import Path

import cv2
import torch
from torchvision import models, transforms


VIDEO_PATH = Path("data/raw/validation_rally_01.mp4")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "keypoints_model_geoaug.pth"

OUTPUT_VIDEO_PATH = Path(
    "outputs/validation_rally_01_automatic_court_test.mp4"
)

# the model expects a 224-by-224 rgb image and predicts positions in that
# resized image. we will convert those positions back to the video size.
MODEL_IMAGE_SIZE = 224

# each pair identifies two predicted landmarks that form a court line.
# these indexes come from the model's documented landmark order.
COURT_LINES = [
    (0, 1),    # far baseline: doubles corners
    (2, 3),    # near baseline: doubles corners
    (0, 2),    # left doubles sideline
    (1, 3),    # right doubles sideline
    (4, 5),    # left singles sideline
    (6, 7),    # right singles sideline
    (8, 9),    # far service line
    (10, 11),  # near service line
    (12, 13),  # center service line
]


def load_court_model():
    """load the automatic court detection model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find the court model: {MODEL_PATH}"
        )

    model = models.resnet50(weights=None)

    # replace the usual image-classification output with 14 (x, y) pairs.
    model.fc = torch.nn.Linear(model.fc.in_features, 14 * 2)

    # load only the checkpoint's model weights.
    weights = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(weights)
    model.eval()

    return model


def create_image_preprocessor():
    """prepare images for the court detection model."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def predict_landmarks(frame, model, preprocess):
    """
    predict 14 court landmarks and return them in original video pixels.
    """
    frame_height, frame_width = frame.shape[:2]

    # opencv reads frames in bgr order; the model expects rgb.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # add a batch dimension: the model expects a group of images, even
    # when that group contains only one image.
    model_input = preprocess(rgb_frame).unsqueeze(0)

    # inference mode avoids storing information needed only during training.
    with torch.inference_mode():
        prediction = model(model_input)

    landmarks = prediction.reshape(14, 2).cpu().numpy()

    # convert x and y coordinates from 224-by-224 model space into the
    # original video's width and height.
    landmarks[:, 0] *= frame_width / MODEL_IMAGE_SIZE
    landmarks[:, 1] *= frame_height / MODEL_IMAGE_SIZE

    return landmarks


def draw_predicted_court(frame, landmarks, frame_number):
    """draw the predicted court on the image."""
    annotated = frame.copy()

    # skip a line if either predicted endpoint is outside the image.
    # this prevents misleading lines from being drawn across the frame.
    height, width = annotated.shape[:2]

    def is_inside_image(point):
        """check whether a point falls inside the image."""
        x, y = point
        return 0 <= x < width and 0 <= y < height

    for start_index, end_index in COURT_LINES:
        start = landmarks[start_index]
        end = landmarks[end_index]

        if not (is_inside_image(start) and is_inside_image(end)):
            continue

        start_pixel = (round(float(start[0])), round(float(start[1])))
        end_pixel = (round(float(end[0])), round(float(end[1])))

        cv2.line(
            annotated,
            start_pixel,
            end_pixel,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

    for index, point in enumerate(landmarks):
        if not is_inside_image(point):
            continue

        x = round(float(point[0]))
        y = round(float(point[1]))

        cv2.circle(annotated, (x, y), 5, (0, 255, 255), -1)

        cv2.putText(
            annotated,
            str(index),
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"Frame {frame_number} - predicted court, review alignment",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def main():
    """run the script's main workflow."""
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Could not find video: {VIDEO_PATH}")

    # protect any earlier output from accidental replacement.
    if OUTPUT_VIDEO_PATH.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_VIDEO_PATH}\n"
            "Choose a new OUTPUT_VIDEO_PATH before running again."
        )

    model = load_court_model()
    preprocess = create_image_preprocessor()

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    writer = None

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError("The video has invalid dimensions or FPS.")

        OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(OUTPUT_VIDEO_PATH),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Could not create output video: {OUTPUT_VIDEO_PATH}"
            )

        print(f"Video: {width} x {height}, {fps:.2f} FPS")
        print(f"Reported frames: {total_frames}")
        print("Running court detection on CPU. This may take a few minutes.")

        frame_number = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            landmarks = predict_landmarks(
                frame,
                model,
                preprocess,
            )

            annotated = draw_predicted_court(
                frame,
                landmarks,
                frame_number,
            )

            writer.write(annotated)
            frame_number += 1

            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames")

    finally:
        capture.release()

        if writer is not None:
            writer.release()

    if frame_number == 0:
        raise RuntimeError("No video frames were processed.")

    print(f"\nProcessed {frame_number} frames.")
    print(f"Saved visual test: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()