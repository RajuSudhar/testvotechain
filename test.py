import os
import cv2

sample = cv2.imread("Altered/Altered-Easy/1__M_Right_little_finger_Zcut.BMP")

best_score = 0
image = None
fileName = None
kp1, kp2, mp = None, None, None


for file in [file for file in os.listdir("Real")]:
    fingerprint_image = cv2.imread("Real/" + file)
    sift = cv2.SIFT.create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10},
                                    {}).knnMatch(descriptors_1, descriptors_2, k=2)

    match_point = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_point.append(p)

    keypoints = 0
    if len(keypoints_1) < len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)

    if len(match_point) / keypoints * 100 > best_score:
        best_score = len(match_point) / keypoints * 100
        fileName = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_point


print("BEST MATCH : ", fileName)
print("BEST SCORE : ", str(best_score))
print(kp1)

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, fx=4, fy=4, dsize=None)
cv2.imshow("Result", result)
cv2.waitKey(0)