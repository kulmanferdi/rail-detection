import cv2
import numpy as np
import matplotlib.pyplot as plt

bev = cv2.imread('output/bev_binary.png', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((3, 3), np.uint8)
bev_clean = cv2.morphologyEx(bev, cv2.MORPH_CLOSE, kernel)
bev_clean = cv2.morphologyEx(bev_clean, cv2.MORPH_OPEN, kernel)
h, w = bev_clean.shape

vertical_projection = np.sum(bev_clean, axis=0) / 255

from scipy.ndimage import gaussian_filter1d

smoothed = gaussian_filter1d(vertical_projection, sigma=5)

from scipy.signal import find_peaks

peaks, properties = find_peaks(smoothed, height=h * 0.3, distance=w // 4)

if len(peaks) < 2:
    raise ValueError(f"Found only {len(peaks)} rail peaks, need 2")

# Take the two highest peaks
peak_heights = [(p, smoothed[p]) for p in peaks]
peak_heights.sort(key=lambda x: x[1], reverse=True)
rail_x_positions = sorted([peak_heights[0][0], peak_heights[1][0]])

def extract_rail_from_band(image, center_x, band_width=30):
    """Extract rail points from vertical band around center_x"""
    x_start = max(0, center_x - band_width)
    x_end = min(image.shape[1], center_x + band_width)

    band = image[:, x_start:x_end]

    # Find white pixels in each row
    rail_points = []
    for y in range(band.shape[0]):
        row = band[y, :]
        white_indices = np.where(row > 127)[0]
        if len(white_indices) > 0:
            # Take median x position in this row
            x = x_start + int(np.median(white_indices))
            rail_points.append([x, y])

    return np.array(rail_points)


left_rail = extract_rail_from_band(bev_clean, rail_x_positions[0])
right_rail = extract_rail_from_band(bev_clean, rail_x_positions[1])


def smooth_rail(points, poly_degree=3):
    """Fit polynomial to rail points for smoothing"""
    if len(points) < 10:
        return points

    y = points[:, 1]
    x = points[:, 0]

    coeffs = np.polyfit(y, x, poly_degree)
    poly = np.poly1d(coeffs)

    y_smooth = np.linspace(y.min(), y.max(), 200)
    x_smooth = poly(y_smooth)

    return np.column_stack((x_smooth, y_smooth)).astype(np.int32)


left_rail_smooth = smooth_rail(left_rail)
right_rail_smooth = smooth_rail(right_rail)

y_min = max(left_rail_smooth[:, 1].min(), right_rail_smooth[:, 1].min())
y_max = min(left_rail_smooth[:, 1].max(), right_rail_smooth[:, 1].max())
y_common = np.linspace(y_min, y_max, 200)

left_x = np.interp(y_common, left_rail_smooth[:, 1], left_rail_smooth[:, 0])
right_x = np.interp(y_common, right_rail_smooth[:, 1], right_rail_smooth[:, 0])

center_x = (left_x + right_x) / 2
centerline_points = np.column_stack((center_x, y_common)).astype(np.int32)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Show vertical projection
axes[0].plot(smoothed)
axes[0].plot(peaks, smoothed[peaks], 'ro', markersize=10)
axes[0].set_title('Vertical Projection (Rail Detection)')
axes[0].set_xlabel('X Position')
axes[0].set_ylabel('White Pixel Count')
axes[0].grid(True)

# Show rails and centerline
bev_color = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
cv2.polylines(bev_color, [left_rail_smooth], False, (0, 255, 0), 2)
cv2.polylines(bev_color, [right_rail_smooth], False, (0, 255, 0), 2)
cv2.polylines(bev_color, [centerline_points], False, (0, 0, 255), 3)

axes[1].imshow(cv2.cvtColor(bev_color, cv2.COLOR_BGR2RGB))
axes[1].set_title('Rails (Green) and Centerline (Red)')
axes[1].axis('off')

plt.savefig('centerline.png', dpi=150)
plt.tight_layout()
plt.show()
