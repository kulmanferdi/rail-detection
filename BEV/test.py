import cv2
import numpy as np

def birds_eye_from_masks(original, rail_mask, centerline_mask, num_strips=5, offset=50):
    h, w = rail_mask.shape[:2]
    warped = np.zeros_like(original)

    # Precompute rail boundaries
    left_rail = []
    right_rail = []
    centerline = []

    for y in range(h):
        row = rail_mask[y, :]
        xs = np.where(row > 0)[0]
        if len(xs) > 0:
            left_rail.append(xs[0])
            right_rail.append(xs[-1])
        else:
            left_rail.append(None)
            right_rail.append(None)

        # centerline
        cx = np.where(centerline_mask[y, :] > 0)[0]
        centerline.append(cx[0] if len(cx) > 0 else None)

    roi_h = h
    roi_w = w

    y_start = 0
    for i in range(num_strips):
        y1 = int(i * roi_h / num_strips)
        y2 = min(int((i+1) * roi_h / num_strips), h-1)

        if left_rail[y1] is None or right_rail[y1] is None: continue
        if left_rail[y2] is None or right_rail[y2] is None: continue

        # trapezoid corners in source
        src_pts = np.float32([
            [left_rail[y1] - offset, y1],
            [right_rail[y1] + offset, y1],
            [right_rail[y2] + offset, y2],
            [left_rail[y2] - offset, y2]
        ])

        # destination rectangle (centered on centerline)
        rect_width = (right_rail[y1] - left_rail[y1]) + 2*offset
        dst_pts = np.float32([
            [centerline[y1] - rect_width//2, y_start],
            [centerline[y1] + rect_width//2, y_start],
            [centerline[y1] + rect_width//2, y_start + (y2-y1)],
            [centerline[y1] - rect_width//2, y_start + (y2-y1)]
        ])

        # warp trapezoid
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        strip = cv2.warpPerspective(original, M, (roi_w, roi_h))

        warped[y_start:y_start+(y2-y1), :] = strip[y_start:y_start+(y2-y1), :]
        y_start += (y2-y1)

    return warped


# Example usage
if __name__ == "__main__":
    original = cv2.imread("dataset/osdar/rgb_highres_center/012_1631441453.300000030.png")
    rail_mask = cv2.imread("dataset/osdar/rgb_highres_center/012_1631441453.300000030.png.mask_segment.png", cv2.IMREAD_GRAYSCALE)
    centerline_mask = cv2.imread("dataset/osdar/rgb_highres_center/012_1631441453.300000030.png.mask_center.png", cv2.IMREAD_GRAYSCALE)

    birds_eye = birds_eye_from_masks(original, rail_mask, centerline_mask)

    cv2.imshow("Birds Eye View", birds_eye)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
