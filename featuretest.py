import cv2
import numpy as np
import mysql.connector
import time
from functions import extract_feature_val

TERMINATION_THRESHOLD = 30
BIFURCATION_THRESHOLD = 30 # Threshold +-30

def fetch_and_compare_fingerprints(key_image_path: str, termination_count: int, bifurcation_count: int):
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host='localhost',
            database='testvotechain',
            user='python',
            password='Impython312'
        )

        if connection.is_connected():
            cursor = connection.cursor()
            # Query to fetch fingerprints within the specified range
            select_query = """
            SELECT fingerprint_image, bifurcation_count, termination_count
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
            print(len(fingerprint_set))

            # Load the key image
            key_fingerprint = cv2.imread(key_image_path)
            sift = cv2.SIFT.create()
            keypoints_1, descriptors_1 = sift.detectAndCompute(key_fingerprint, None)

            best_score = float('-inf')
            matched_image = None

            # Iterate through the retrieved images
            for fingerprint in fingerprint_set:

                # Convert the binary data to an image
                fingerprint_image = np.frombuffer(fingerprint[0], np.uint8)
                fingerprint_image = cv2.imdecode(fingerprint_image, cv2.IMREAD_UNCHANGED)

                keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
                matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10},
                                                {}).knnMatch(descriptors_1, descriptors_2, k=2)
                match_point = []
                for p, q in matches:
                    if p.distance < 0.1 * q.distance:
                        match_point.append(p)

                keypoints = min(len(keypoints_2), len(keypoints_1))


                # Compare key_image with the current fingerprint_image

                if len(match_point) / keypoints * 100 > best_score:
                    best_score = len(match_point) / keypoints * 100
                    matched_image = fingerprint_image
                    kp1, kp2, mp = keypoints_1, keypoints_2, match_point

            print(f"Best match score: {best_score}")
            return matched_image

    except mysql.connector.Error as e:
        print(f"Error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

#img_path = "Real/300__F_Left_ring_finger.BMP"
img_path = "Altered/Altered-HARD/300__F_Left_ring_finger_Obl.BMP"
terminal_count,bi_count = extract_feature_val(img_path)
print(terminal_count, bi_count)
a = time.time()
best_image = fetch_and_compare_fingerprints(img_path,terminal_count,bi_count)
b = time.time()
print(b - a)
cv2.imshow("key", cv2.imread(img_path,None))
cv2.waitKey(0)
cv2.imshow("match", best_image)
cv2.waitKey(0)
cv2.destroyAllWindows()