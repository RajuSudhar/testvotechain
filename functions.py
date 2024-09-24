import os
import cv2
import numpy as np
from skimage.morphology import skeletonize

def match_fingerprint(key_fingerprint) -> str:    # Returns the best fingerprint match
    best_score = 0
    matched_image = None
    matched_fileName = None
    kp1, kp2, mp = None, None, None

    for file in os.listdir("Real"):
        fingerprint_image = cv2.imread("Real/" + file)
        sift = cv2.SIFT.create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(key_fingerprint, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10},
                                        {}).knnMatch(descriptors_1, descriptors_2, k=2)

        match_point = []
        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_point.append(p)

        keypoints = min(len(keypoints_2),len(keypoints_1))


        if len(match_point) / keypoints * 100 > best_score:
            best_score = len(match_point) / keypoints * 100
            matched_fileName = file
            matched_image = fingerprint_image
            kp1, kp2, mp = keypoints_1, keypoints_2, match_point

    print("BEST MATCH : ", matched_fileName)
    print("BEST SCORE : ", str(best_score))

    return matched_fileName

def extract_feature_val(fingerprint_img):   #Returns Termination and Bifurcation count in a Fingerprint
    fingerprint = cv2.imread(fingerprint_img, cv2.IMREAD_GRAYSCALE)
    fingerprint = cv2.resize(fingerprint, fx=2, fy=2, dsize=None)

    # Apply Otsu's thresholding for binarization
    _, binary_fingerprint = cv2.threshold(fingerprint, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Skeletonization (thinning)
    skeleton_fingerprint = skeletonize(binary_fingerprint // 255)

    # minutiae_endings = []
    minutiae_endings_count = 0
    # minutiae_bifurcations = []
    minutiae_bifurcations_count = 0

    # Scan the image
    for y in range(1, skeleton_fingerprint.shape[0] - 1):
        for x in range(1, skeleton_fingerprint.shape[1] - 1):
            # Get the 3x3 neighborhood
            neighborhood = skeleton_fingerprint[y - 1:y + 2, x - 1:x + 2]

            # Count the number of ridge pixels (value of 1 in the neighborhood)
            ridge_count = np.sum(neighborhood)

            # Ridge ending: Exactly 2 pixels are 1 (including the center)
            if skeleton_fingerprint[y, x] == 1 and ridge_count == 2:
                #minutiae_endings.append((x, y))
                minutiae_endings_count += 1

            # Bifurcation: Exactly 4 pixels are 1 (including the center)
            elif skeleton_fingerprint[y, x] == 1 and ridge_count == 4:
                #minutiae_bifurcations.append((x, y))
                minutiae_bifurcations_count += 1

    '''
    # Plot the ridge endings and bifurcations on the original fingerprint
    import matplotlib.pyplot as plt

    plt.imshow(fingerprint, cmap='gray')

    # Plot ridge endings
    for (x, y) in minutiae_endings:
        plt.plot(x, y, 'ro')  # Red dot for ridge endings

    # Plot bifurcations
    for (x, y) in minutiae_bifurcations:
        plt.plot(x, y, 'bo')  # Blue dot for bifurcations
    
    plt.show()
    #print(minutiae_endings_count)
    #print(minutiae_bifurcations_count)
    '''

    return minutiae_endings_count, minutiae_bifurcations_count

