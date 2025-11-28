import cv2
import numpy as np


def warp_trapezoids_to_birds_eye(image, num_strips=20):
    """
    Transform railway perspective to bird's eye view using trapezoid strips.
    """
    h, w = image.shape[:2]

    # Define ROI (middle portion of the image)
    roi_top = int(h * 0.20/0.40)
    roi_bottom = h
    roi_left = int(w * 0.15)
    roi_right = int(w * 0.85)

    roi = image[roi_top:roi_bottom, roi_left:roi_right]
    roi_h, roi_w = roi.shape[:2]

    warped = np.zeros_like(roi)

    # Parameters controlling trapezoid narrowing
    top_shrink = 0.4  # fraction shrink at top (40%)

    y_start = 0
    for i in range(num_strips):
        # Vertical bounds in ROI
        src_y1 = int(i * roi_h / num_strips)
        src_y2 = int((i + 1) * roi_h / num_strips)

        # Shrink factor increases towards top
        shrink_factor1 = top_shrink * (1 - src_y1 / roi_h)
        shrink_factor2 = top_shrink * (1 - src_y2 / roi_h)

        # Define trapezoid corners in ROI
        src_rect = np.float32([
            [roi_w * shrink_factor1, src_y1],
            [roi_w * (1 - shrink_factor1), src_y1],
            [roi_w * (1 - shrink_factor2), src_y2],
            [roi_w * shrink_factor2, src_y2]
        ])

        # Destination rectangle (uniform strip)
        dst_y1 = y_start
        dst_y2 = y_start + (src_y2 - src_y1)
        dst_rect = np.float32([
            [0, dst_y1],
            [roi_w, dst_y1],
            [roi_w, dst_y2],
            [0, dst_y2]
        ])

        # Warp trapezoid to rectangle
        M = cv2.getPerspectiveTransform(src_rect, dst_rect)
        strip = cv2.warpPerspective(roi, M, (roi_w, roi_h))

        warped[dst_y1:dst_y2, :] = strip[dst_y1:dst_y2, :]
        y_start = dst_y2

    return warped


# Example usage
if __name__ == "__main__":
    img = cv2.imread("dataset/web/rail-test1.jpg")
    birds_eye = warp_trapezoids_to_birds_eye(img, num_strips=30)

    cv2.imshow("Original", img)
    cv2.imshow("Birds Eye View", birds_eye)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
