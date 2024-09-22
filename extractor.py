import cv2
import numpy as np
from skimage.morphology import skeletonize


# Load fingerprint image
fingerprint = cv2.imread('Real/2__F_Right_little_finger.BMP', cv2.IMREAD_GRAYSCALE)
fingerprint = cv2.resize(fingerprint, fx=2, fy=2, dsize=None)

# Apply Otsu's thresholding for binarization
_, binary_fingerprint = cv2.threshold(fingerprint, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Skeletonization (thinning)
skeleton_fingerprint = skeletonize(binary_fingerprint // 255)

minutiae_endings = []
minutiae_bifurcations = []

# Scan the image
for y in range(1, skeleton_fingerprint.shape[0] - 1):
    for x in range(1, skeleton_fingerprint.shape[1] - 1):
        # Get the 3x3 neighborhood
        neighborhood = skeleton_fingerprint[y-1:y+2, x-1:x+2]

        # Count the number of ridge pixels (value of 1 in the neighborhood)
        ridge_count = np.sum(neighborhood)

        # Ridge ending: Exactly 2 pixels are 1 (including the center)
        if skeleton_fingerprint[y, x] == 1 and ridge_count == 2:
            minutiae_endings.append((x, y))

        # Bifurcation: Exactly 4 pixels are 1 (including the center)
        elif skeleton_fingerprint[y, x] == 1 and ridge_count == 4:
            minutiae_bifurcations.append((x, y))


# Plot the ridge endings and bifurcations on the original fingerprint
import matplotlib.pyplot as plt

plt.imshow(fingerprint, cmap='gray')

# Plot ridge endings
for (x, y) in minutiae_endings:
    plt.plot(x, y, 'ro')  # Red dot for ridge endings

# Plot bifurcations
for (x, y) in minutiae_bifurcations:
    plt.plot(x, y, 'bo')  # Blue dot for bifurcations

print(len(minutiae_endings))
print(len(minutiae_bifurcations))

plt.show()