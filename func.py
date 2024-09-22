import hashlib
import cv2
import numpy as np


def generate_minutiae_fingerprint_hash(img_file) -> str:


    sift = cv2.SIFT.create()
    keypoints, descriptors = sift.detectAndCompute(img_file, None)

    if not all(isinstance(kp, cv2.KeyPoint) for kp in keypoints):
        raise ValueError("keypoints must be a list of cv2.KeyPoint objects")
    if not isinstance(descriptors, np.ndarray):
        raise ValueError("descriptors must be a numpy array")
    
    def hash_data(data) -> str:
        combined_string = data.tobytes()
        hash_object = hashlib.sha256(combined_string)
        return hash_object.hexdigest()

    keypoints_data = [(kp.pt[0], kp.pt[1], kp.angle) for kp in keypoints]
    descriptors_data = descriptors.flatten()
    combined_data = np.concatenate([np.array(keypoints_data).flatten(), descriptors_data])
    fingerprint_hash = hash_data(combined_data)

    return fingerprint_hash


File1 = cv2.imread("Real/1__M_Left_ring_finger.BMP")
testfile = cv2.imread("Altered/Altered-Easy/1__M_Left_ring_finger_Zcut.BMP")

hash_is = generate_minutiae_fingerprint_hash(File1)
newhash = generate_minutiae_fingerprint_hash(testfile)


print(hash_is)
print(newhash)

cv2.imshow("og", File1)
cv2.imshow("Test",testfile)
cv2.waitKey(0)
