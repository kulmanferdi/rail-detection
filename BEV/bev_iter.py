import os
import glob
import cv2
import numpy as np

def extract_rail_edges(rail_mask):
    h, w = rail_mask.shape
    left_rail, right_rail = [], []
    for y in range(h):
        xs = np.where(rail_mask[y] > 0)[0]
        if len(xs) >= 2:
            left_rail.append(xs[0])
            right_rail.append(xs[-1])
        else:
            left_rail.append(None)
            right_rail.append(None)
    return left_rail, right_rail

def extract_centerline(centerline_mask):
    h, w = centerline_mask.shape
    centerline = []
    for y in range(h):
        xs = np.where(centerline_mask[y] > 0)[0]
        centerline.append(xs[0] if len(xs) > 0 else None)
    return centerline

def birds_eye_from_masks(original, rail_mask, centerline_mask, num_strips=4, offset=-4):
    h, w = rail_mask.shape[:2]
    warped = np.zeros_like(original)

    left_rail, right_rail = extract_rail_edges(rail_mask)
    centerline = extract_centerline(centerline_mask)

    y_start = 0
    for i in range(num_strips):
        y1 = int(i * h / num_strips)
        y2 = min(int((i + 1) * h / num_strips), h - 1)

        if None in (left_rail[y1], right_rail[y1], left_rail[y2], right_rail[y2]):
            continue
        if centerline[y1] is None:
            continue

        src_pts = np.float32([
            [left_rail[y1] - offset, y1],
            [right_rail[y1] + offset, y1],
            [right_rail[y2] + offset, y2],
            [left_rail[y2] - offset, y2]
        ])

        rect_width = (right_rail[y1] - left_rail[y1]) + 2 * offset
        dst_pts = np.float32([
            [centerline[y1] - rect_width // 2, y_start],
            [centerline[y1] + rect_width // 2, y_start],
            [centerline[y1] + rect_width // 2, y_start + (y2 - y1)],
            [centerline[y1] - rect_width // 2, y_start + (y2 - y1)]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        strip = cv2.warpPerspective(original, M, (w, h))

        warped[y_start:y_start + (y2 - y1), :] = strip[y_start:y_start + (y2 - y1), :]
        y_start += (y2 - y1)

    return warped


if __name__ == "__main__":
    input_folder = "dataset/osdar/fire/rgb_highres_center"
    output_folder = "output_birds_eye"
    os.makedirs(output_folder, exist_ok=True)

    originals = glob.glob(os.path.join(input_folder, "*.png"))
    originals = [f for f in originals if ".mask_" not in f]

    for orig_path in originals:
        base = os.path.splitext(orig_path)[0]
        print(base)
        rail_path = base + ".png.mask_segment.png"
        center_path = base + ".png.mask_center.png"
        track_path = base + ".png.mask_track.png"
        concave = base + ".png.segment_concave.png"

        original = cv2.imread(orig_path)
        rail_mask = cv2.imread(rail_path, cv2.IMREAD_GRAYSCALE)
        centerline_mask = cv2.imread(center_path, cv2.IMREAD_GRAYSCALE)
        track_mask = cv2.imread(track_path, cv2.IMREAD_GRAYSCALE)

        if original is None or rail_mask is None or centerline_mask is None:
            print(f"Skipping {orig_path}: missing one or more masks")
            continue

        birds_eye = birds_eye_from_masks(original, rail_mask, centerline_mask)
        out_path = os.path.join(output_folder, os.path.basename(base) + "_birds_eye.png")
        cv2.imwrite(out_path, birds_eye)
        print(f"Processed {orig_path} → {out_path}")
