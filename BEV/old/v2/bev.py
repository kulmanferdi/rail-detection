import cv2
import numpy as np
import os
import glob

def birds_eye_from_masks(original, rail_mask, centerline_mask, num_strips=4, offset=-10):
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


# ---------- Batch Processing ----------
if __name__ == "__main__":
    input_folder = "dataset/osdar/fire/rgb_highres_center"
    output_folder = "output_birds_eye"
    os.makedirs(output_folder, exist_ok=True)

    # Find all original images (assuming they end with .png and not .mask_*.png)
    originals = glob.glob(os.path.join(input_folder, "*.png"))
    originals = [f for f in originals if ".mask_" not in f]

    for orig_path in originals:
        base = os.path.splitext(orig_path)[0]

        rail_path = base + ".mask_segment.png"
        center_path = base + ".mask_center.png"
        track_path = base + ".mask_track.png"

        # Load images
        original = cv2.imread(orig_path)
        rail_mask = cv2.imread(rail_path, cv2.IMREAD_GRAYSCALE)
        centerline_mask = cv2.imread(center_path, cv2.IMREAD_GRAYSCALE)
        track_mask = cv2.imread(track_path, cv2.IMREAD_GRAYSCALE)

        if None in (original, rail_mask, centerline_mask, track_mask):
            print(f"Skipping {orig_path}: missing one or more masks")
            continue

        # Compute bird’s eye
        birds_eye = birds_eye_from_masks(original, rail_mask, centerline_mask)

        # Save result
        out_path = os.path.join(output_folder, os.path.basename(base) + "_birds_eye.png")
        cv2.imwrite(out_path, birds_eye)
        print(f"Processed {orig_path} → {out_path}")

