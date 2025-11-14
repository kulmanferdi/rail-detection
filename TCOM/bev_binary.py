import cv2
import numpy as np
import matplotlib.pyplot as plt

bev = cv2.imread('output/birdseye_view_output.png')

gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
sizes = stats[:, -1]

if len(sizes) > 2:
    largest_idx = np.argsort(sizes[1:])[::-1][:2] + 1
else:
    largest_idx = np.arange(1, len(sizes))

clean = np.zeros_like(binary)
for idx in largest_idx:
    clean[labels == idx] = 255

clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

cv2.imwrite('output/bev_binary.png', clean)
plt.imshow(clean, cmap='gray')
plt.title("Binary Rails")
plt.axis('off')
plt.show()