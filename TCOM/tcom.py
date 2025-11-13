import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def calculate_tcom(binary_image_path, reference_x=None):
    """
    Calculate Track Center Offset Measurement (TCOM) for each pixel row.

    Args:
        binary_image_path: Path to binary rail image
        reference_x: Reference center position (None = use image center)

    Returns:
        tcom_values: Array of offset values (mm or pixels) for each row
        y_positions: Array of y-coordinates
        centerline_x: Array of centerline x-positions
    """
    bev = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((3, 3), np.uint8)
    bev_clean = cv2.morphologyEx(bev, cv2.MORPH_CLOSE, kernel)
    bev_clean = cv2.morphologyEx(bev_clean, cv2.MORPH_OPEN, kernel)

    h, w = bev_clean.shape

    vertical_projection = np.sum(bev_clean, axis=0) / 255
    smoothed = gaussian_filter1d(vertical_projection, sigma=5)
    peaks, _ = find_peaks(smoothed, height=h * 0.3, distance=w // 4)

    if len(peaks) < 2:
        raise ValueError(f"Found only {len(peaks)} rail peaks, need 2")

    peak_heights = [(p, smoothed[p]) for p in peaks]
    peak_heights.sort(key=lambda x: x[1], reverse=True)
    rail_x_positions = sorted([peak_heights[0][0], peak_heights[1][0]])

    def extract_rail_from_band(image, center_x, band_width=30):
        x_start = max(0, center_x - band_width)
        x_end = min(image.shape[1], center_x + band_width)
        band = image[:, x_start:x_end]

        rail_points = []
        for y in range(band.shape[0]):
            row = band[y, :]
            white_indices = np.where(row > 127)[0]
            if len(white_indices) > 0:
                x = x_start + int(np.median(white_indices))
                rail_points.append([x, y])

        return np.array(rail_points)

    left_rail = extract_rail_from_band(bev_clean, rail_x_positions[0])
    right_rail = extract_rail_from_band(bev_clean, rail_x_positions[1])

    def smooth_rail(points, poly_degree=3):
        if len(points) < 10:
            return points

        y = points[:, 1]
        x = points[:, 0]
        coeffs = np.polyfit(y, x, poly_degree)
        poly = np.poly1d(coeffs)

        y_smooth = np.linspace(y.min(), y.max(), h)
        x_smooth = poly(y_smooth)

        return np.column_stack((x_smooth, y_smooth))

    left_rail_smooth = smooth_rail(left_rail)
    right_rail_smooth = smooth_rail(right_rail)

    y_min = max(left_rail_smooth[:, 1].min(), right_rail_smooth[:, 1].min())
    y_max = min(left_rail_smooth[:, 1].max(), right_rail_smooth[:, 1].max())

    y_positions = np.arange(int(y_min), int(y_max) + 1)

    left_x = np.interp(y_positions, left_rail_smooth[:, 1], left_rail_smooth[:, 0])
    right_x = np.interp(y_positions, right_rail_smooth[:, 1], right_rail_smooth[:, 0])

    centerline_x = (left_x + right_x) / 2

    if reference_x is None:
        reference_x = w / 2

    tcom_values = centerline_x - reference_x

    return tcom_values, y_positions, centerline_x


def visualize_tcom(binary_image_path, tcom_values, y_positions, centerline_x,
                   save_path='tcom_analysis.png'):
    """Visualize TCOM results"""
    bev = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    h, w = bev.shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    bev_color = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    centerline_points = np.column_stack((centerline_x, y_positions)).astype(np.int32)
    cv2.polylines(bev_color, [centerline_points], False, (0, 0, 255), 2)

    cv2.line(bev_color, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)

    axes[0].imshow(cv2.cvtColor(bev_color, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Centerline (Red) vs Reference (Blue)')
    axes[0].axis('off')

    axes[1].plot(tcom_values, y_positions, linewidth=2)
    axes[1].axvline(x=0, color='r', linestyle='--', label='Reference')
    axes[1].set_xlabel('TCOM Offset (pixels)')
    axes[1].set_ylabel('Y Position (pixels)')
    axes[1].set_title('Track Center Offset Measurement')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].invert_yaxis()

    tcom_normalized = (tcom_values - tcom_values.min()) / (tcom_values.max() - tcom_values.min())
    colormap = plt.cm.RdYlGn_r(tcom_normalized)

    overlay = bev_color.copy()
    for i, y in enumerate(y_positions):
        color = (colormap[i, 2] * 255, colormap[i, 1] * 255, colormap[i, 0] * 255)
        cv2.line(overlay, (0, int(y)), (w, int(y)), color, 2)

    blended = cv2.addWeighted(bev_color, 0.6, overlay, 0.4, 0)

    axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[2].set_title('TCOM Heatmap (Red=Max Offset, Green=Min Offset)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print("\n=== TCOM Statistics ===")
    print(f"Mean offset: {np.mean(tcom_values):.2f} pixels")
    print(f"Std deviation: {np.std(tcom_values):.2f} pixels")
    print(f"Max offset: {np.max(np.abs(tcom_values)):.2f} pixels")
    print(f"Min offset: {np.min(np.abs(tcom_values)):.2f} pixels")


if __name__ == "__main__":
    tcom, y_pos, centerline = calculate_tcom('data/web/bev_binary.png')

    visualize_tcom('data/web/bev_binary.png', tcom, y_pos, centerline)

    np.savez('tcom_data.npz',
             tcom=tcom,
             y_positions=y_pos,
             centerline_x=centerline)

    print("\nTCOM data saved to tcom_data.npz")
