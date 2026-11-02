from pathlib import Path

import cv2
# import yolo model from ultralytics library
from ultralytics import YOLO
# detect people in the first frame and save a preview
image_path = Path("outputs/first_frame.jpg")
output_path = Path("outputs/people_detection_preview.jpg")
# checks if the input image exists, raise an error if not
if not image_path.exists():
    raise FileNotFoundError(f"Could not find {image_path}")
# load the people detection model
model = YOLO("yolo26n.pt")
# detect people in the image
# results are stored in the results variable
results = model.predict(
    source=str(image_path),
    classes=[0],
    conf=0.35,
    verbose=False,
)
# count the detected people
result = results[0]
# set the count to zero when no boxes are found
# count the detected boxes
if result.boxes is None:
    person_count = 0
else:
    person_count = len(result.boxes)
# draw the detection results
annotated_image = result.plot()
# save the annotated preview
image_saved = cv2.imwrite(str(output_path), annotated_image)

if not image_saved:
    raise RuntimeError(f"Could not save {output_path}")

print(f"Detected {person_count} person(s).")
print(f"Saved preview image to: {output_path}")