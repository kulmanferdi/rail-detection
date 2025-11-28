import numpy as np
import cv2
from point import get_perspective_points

def birdseye_view(image, use_perspective=True):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        display_image = image.copy()
    else:
        gray_image = image
        display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if use_perspective:
        src_points = get_perspective_points(display_image)

        if src_points is not None:
            height, width = gray_image.shape
            dst_points = np.float32([
                [0, 0],  # Top-left
                [width - 1, 0],  # Top-right
                [width - 1, height - 1],  # Bottom-right
                [0, height - 1]  # Bottom-left
            ])

            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            birdseye_image = cv2.warpPerspective(gray_image, matrix, (width, height))

            return birdseye_image
        else:
            print("Perspective points not selected. ")

    return gray_image