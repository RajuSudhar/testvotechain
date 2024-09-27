from typing import Union, BinaryIO
import os
import cv2
import mysql.connector
import numpy as np
from numpy import ndarray
from skimage.morphology import skeletonize
import database

class Fingerprint:
    def __init__(self, f_id, f_image):
        self.fingerprint_id = f_id
        self.fingerprint_image = f_image
    def get_image(self):
        # Convert the binary data (BLOB) into an image
        return self.fingerprint_image

def match_fingerprint(key_fingerprint : ndarray) -> str:
    best_score = 0
    matched_fileName = None
    #matched_image,kp1, kp2, mp = None,None, None, None

    sift = cv2.SIFT_create()
    key_fingerprint_keypoints, key_fingerprint_descriptors = sift.detectAndCompute(key_fingerprint, None)

    for file in os.listdir("Real"):
        fingerprint = cv2.imread(os.path.join("Real", file))
        fingerprint_keypoints, fingerprint_descriptors = sift.detectAndCompute(fingerprint, None)

        matches = cv2.FlannBasedMatcher().knnMatch(key_fingerprint_descriptors, fingerprint_descriptors, k=2)

        match_point = [p for p, q in matches if p.distance < 0.1 * q.distance]

        keypoints = min(len(fingerprint_keypoints), len(key_fingerprint_keypoints))

        match_score = len(match_point) / keypoints * 100
        if match_score > best_score:
            best_score = match_score
            matched_fileName = file
            #matched_image = fingerprint_image
            #kp1, kp2, mp = key_fingerprint_keypoints, fingerprint_keypoints_2, match_point

    #print("BEST MATCH : ", matched_fileName)
    #print("BEST SCORE : ", str(best_score))

    return matched_fileName

def extract_feature_val(fingerprint_img) -> tuple[int, int]:
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

def fetch_and_compare_fingerprints(key_image_path: str, termination_count: int, bifurcation_count: int) -> Fingerprint:
    try:
        # Connect to MySQL
        connection = database.connect_mydb()
        if connection.is_connected():
            # Fetch all matching rows
            fingerprint_set = fetch_matchable_fingerprints(connection, termination_count, bifurcation_count)

            # Load the key image
            key_fingerprint = cv2.imread(key_image_path)
            sift = cv2.SIFT_create()
            key_fingerprint_keypoints, key_fingerprint_descriptors = sift.detectAndCompute(key_fingerprint, None)

            best_score = float('-inf')
            #matched_image, kp1, kp2, mp = None, None, None, None
            best_match_fingerprint = None

            # Iterate through the retrieved images
            for fingerprint in fingerprint_set:
                matched_fingerprint = Fingerprint(fingerprint[1], convert_binary_to_image(fingerprint[0]))
                # Convert the binary data to an image

                fingerprint_keypoints, fingerprint_descriptors = sift.detectAndCompute(matched_fingerprint.fingerprint_image, None)
                matches = cv2.FlannBasedMatcher().knnMatch(key_fingerprint_descriptors, fingerprint_descriptors, k=2)
                match_point = [p for p, q in matches if p.distance < 0.1 * q.distance]

                keypoints = min(len(fingerprint_keypoints), len(key_fingerprint_keypoints))

                # Compare key_image with the current fingerprint_image
                match_score = len(match_point) / keypoints * 100
                if match_score > best_score:
                    best_score = match_score
                    best_match_fingerprint = matched_fingerprint
                    #matched_image = fingerprint_image
                    #kp1, kp2, mp = key_fingerprint_keypoints, fingerprint_keypoints_2, match_point

            '''Debugging
            result = cv2.drawMatches(key_fingerprint, kp1, matched_image, kp2, mp, None)
            result = cv2.resize(result, (0, 0), fx=4, fy=4)
            cv2.imshow("Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()             
            '''
            return best_match_fingerprint

    except mysql.connector.Error as e:
        print(f"Error: {e}")

    finally:
        if connection.is_connected():
            connection.close()

def fetch_matchable_fingerprints(connection, termination_count : int, bifurcation_count : int):
    TERMINATION_THRESHOLD = 30
    BIFURCATION_THRESHOLD = 30
    try:
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
    finally:
        dataset = cursor.fetchall()
        cursor.close()
        return dataset

def convert_image_to_binary(image: Union[str, BinaryIO]) -> bytes:
    if isinstance(image, str):  # Check if the input is a file path
        with open(image, 'rb') as file:
            return file.read()
    else:  # Assume it's a file-like object
        return image.read()

def convert_binary_to_image(image_binary: bytes) -> ndarray:
    # Convert the binary data to an image
    image = np.frombuffer(image_binary, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image