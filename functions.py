import mysql.connector
import os
import cv2
import database
import numpy as np
from skimage.morphology import skeletonize

def match_fingerprint(key_fingerprint) -> str:
    best_score = 0
    matched_image = None
    matched_fileName = None
    kp1, kp2, mp = None, None, None

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(key_fingerprint, None)

    for file in os.listdir("Real"):
        fingerprint_image = cv2.imread(os.path.join("Real", file))
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        matches = cv2.FlannBasedMatcher().knnMatch(descriptors_1, descriptors_2, k=2)

        match_point = [p for p, q in matches if p.distance < 0.1 * q.distance]

        keypoints = min(len(keypoints_2), len(keypoints_1))

        match_score = len(match_point) / keypoints * 100
        if match_score > best_score:
            best_score = match_score
            matched_fileName = file
            matched_image = fingerprint_image
            kp1, kp2, mp = keypoints_1, keypoints_2, match_point

    print("BEST MATCH : ", matched_fileName)
    print("BEST SCORE : ", str(best_score))

    return matched_fileName

def extract_feature_val(fingerprint_img):   
    fingerprint = cv2.imread(fingerprint_img, cv2.IMREAD_GRAYSCALE)
    fingerprint = cv2.resize(fingerprint, (0, 0), fx=2, fy=2)

    _, binary_fingerprint = cv2.threshold(fingerprint, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    skeleton_fingerprint = skeletonize(binary_fingerprint // 255)

    minutiae_endings_count = 0
    minutiae_bifurcations_count = 0

    for y in range(1, skeleton_fingerprint.shape[0] - 1):
        for x in range(1, skeleton_fingerprint.shape[1] - 1):
            neighborhood = skeleton_fingerprint[y - 1:y + 2, x - 1:x + 2]
            ridge_count = np.sum(neighborhood)

            if skeleton_fingerprint[y, x] == 1:
                if ridge_count == 2:
                    minutiae_endings_count += 1
                elif ridge_count == 4:
                    minutiae_bifurcations_count += 1

    return minutiae_endings_count, minutiae_bifurcations_count

def fetch_and_compare_fingerprints(key_image_path: str, termination_count: int, bifurcation_count: int) -> int:
    TERMINATION_THRESHOLD = 30
    BIFURCATION_THRESHOLD = 30

    try:
        # Connect to MySQL
        connection = database.connect_mydb()
        if connection.is_connected():
            cursor = connection.cursor()
            # Query to fetch fingerprints within the specified range
            select_query = """
            SELECT fingerprint_image, id
            FROM fingerprint_data 
            WHERE bifurcation_count BETWEEN %s AND %s
            OR termination_count BETWEEN %s AND %s
            ORDER BY
            ABS(bifurcation_count - %s) ASC,
            ABS(termination_count - %s) DESC;
            """
            cursor.execute(select_query,
                           (bifurcation_count - BIFURCATION_THRESHOLD, bifurcation_count + BIFURCATION_THRESHOLD,
                            termination_count - TERMINATION_THRESHOLD, termination_count + TERMINATION_THRESHOLD,
                            bifurcation_count, termination_count))

            # Fetch all matching rows
            fingerprint_set = cursor.fetchall()

            # Load the key image
            key_fingerprint = cv2.imread(key_image_path)
            sift = cv2.SIFT_create()
            keypoints_1, descriptors_1 = sift.detectAndCompute(key_fingerprint, None)

            best_score = float('-inf')
            matched_image = None
            fingerprint_id = None

            # Iterate through the retrieved images
            for fingerprint in fingerprint_set:

                # Convert the binary data to an image
                fingerprint_image = np.frombuffer(fingerprint[0], np.uint8)
                fingerprint_image = cv2.imdecode(fingerprint_image, cv2.IMREAD_UNCHANGED)

                fingerprint_id = fingerprint[1]

                keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
                matches = cv2.FlannBasedMatcher().knnMatch(descriptors_1, descriptors_2, k=2)
                match_point = [p for p, q in matches if p.distance < 0.1 * q.distance]

                keypoints = min(len(keypoints_2), len(keypoints_1))

                # Compare key_image with the current fingerprint_image
                match_score = len(match_point) / keypoints * 100
                if match_score > best_score:
                    best_score = match_score
                    matched_image = fingerprint_image
                    kp1, kp2, mp = keypoints_1, keypoints_2, match_point
            return fingerprint_id

    except mysql.connector.Error as e:
        print(f"Error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()