import cv2
import numpy as np

points = []
temp_img = None

def select_points(event, x, y, flags, param):
    global points, temp_img
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)} selected: ({x}, {y})")
        cv2.circle(temp_img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(temp_img, str(len(points)), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Select 4 points for perspective transform", temp_img)


def get_perspective_points(image):
    global points, temp_img
    temp_img = image.copy()

    cv2.imshow("Select 4 points for perspective transform", temp_img)
    cv2.setMouseCallback("Select 4 points for perspective transform", select_points)

    print("Click 4 points on the image in this order:")
    print("1. Top-left of the region")
    print("2. Top-right of the region")
    print("3. Bottom-right of the region")
    print("4. Bottom-left of the region")
    print("Press any key when done...")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 4:
        return np.float32(points)
    else:
        print(f"Warning: Only {len(points)} points selected, need 4")
        return None
