from pathlib import Path
# loads a python tool for saving data in a simple text format
import json
import cv2

image_path = Path("outputs/first_frame.jpg")
points_path = Path("outputs/court_corner_pixels.json")
preview_path = Path("outputs/court_corner_preview.jpg")
# opens saved first frame (first_frame.jpg) for python to read
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not open {image_path}")
# draw clicks on a copy of the original image
display_image = image.copy()
# store each clicked point as an x/y pair
# store image points in court-corner order
# later processing, so we will store them in the order they are clicked.
# clicked point is the exact spot where you click your mouse on the court image
clicked_points = []

window_name = "Click court corners"
# defines a function that will be called whenever the user clicks on the image
# handle mouse click coordinates
# record left-click coordinates
# mark each click with its order
# the function then updates the displayed image to show the new point
def record_click(event, x, y, flags, parameter):
    """record a selected court corner."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(clicked_points) >= 4:
        return

    clicked_points.append([x, y])

    point_number = len(clicked_points)

    cv2.circle(display_image, (x, y), 10, (0, 255, 255), -1)
    cv2.putText(
        display_image,
        str(point_number),
        (x + 12, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    cv2.imshow(window_name, display_image)

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, record_click)

print("Click these FOUR points in this exact order:")
print("1. Far-left singles-court corner")
print("2. Far-right singles-court corner")
print("3. Near-right singles-court corner")
print("4. Near-left singles-court corner")
print("Use the INNER singles sidelines, not the outer doubles sidelines.")
print("Press any key after all four points are marked.")
# show the image and collect four court corners
cv2.imshow(window_name, display_image)

while len(clicked_points) < 4:
    cv2.waitKey(20)
# close the window after four clicks
cv2.waitKey(0)
cv2.destroyAllWindows()
# save clicked corners and the preview image
with open(points_path, "w", encoding="utf-8") as file:
    json.dump(clicked_points, file, indent=2)

cv2.imwrite(str(preview_path), display_image)

print(f"Saved pixel coordinates to: {points_path}")
print(f"Saved marked preview image to: {preview_path}")
print(f"Your clicked points: {clicked_points}")