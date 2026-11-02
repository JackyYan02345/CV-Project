from pathlib import Path
import cv2
# stores the address of tennis video
input_video_path = Path("data/raw/test_rally.mp4")
# stores where to save the image
output_image_path = Path("outputs/first_frame.jpg")
# create the output folder if needed
output_image_path.parent.mkdir(parents=True, exist_ok=True)
# opens video for python
capture = cv2.VideoCapture(str(input_video_path))

if not capture.isOpened():
    raise FileNotFoundError(f"Could not open {input_video_path}")
# read the first frame
success, frame = capture.read()
# close video to free up resources
capture.release()
# check if python recieved an image not an  opened video file, else raise error
if not success:
    raise RuntimeError("The video opened, but Python could not read its first frame.")
# save the first frame as a jpeg
image_saved = cv2.imwrite(str(output_image_path), frame)

if not image_saved:
    raise RuntimeError(f"Could not save {output_image_path}")

print(f"First frame saved to: {output_image_path}")