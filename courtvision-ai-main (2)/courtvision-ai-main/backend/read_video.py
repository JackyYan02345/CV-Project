from pathlib import Path
import cv2

# show basic video properties
video_path = Path("data/raw/test_rally.mp4")
# open the video for frame-by-frame reading
capture = cv2.VideoCapture(str(video_path))
# safety check
if not capture.isOpened():
    raise FileNotFoundError(
        f"Could not open {video_path}. "
        "Check that the video exists and its name is test_rally.mp4."
    )
# read video properties with opencv
fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_seconds = frame_count / fps if fps else 0
# close video to free up resources
capture.release()
# print it all out
print("Video opened successfully.")
print(f"Size: {width} x {height} pixels")
print(f"Frame rate: {fps:.2f} FPS")
print(f"Frames: {frame_count}")
print(f"Duration: {duration_seconds:.2f} seconds")