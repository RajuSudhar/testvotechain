import os
import cv2

def match_fingerprint(key_fingerprint):
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