import cv2
import time
import database
import functions
from functions import extract_feature_val, fetch_and_compare_fingerprints, Fingerprint

#img_path = "Real/300__F_Left_ring_finger.BMP"
img_path = "Altered/Altered-HARD/300__F_Left_ring_finger_Obl.BMP"
terminal_count,bi_count = extract_feature_val(img_path)
print(terminal_count, bi_count)
a = time.time()
best_match = fetch_and_compare_fingerprints(img_path,terminal_count,bi_count)
best_image = best_match.get_image()
b = time.time()
print(b - a)
cv2.imshow("key", cv2.imread(img_path,None))
cv2.waitKey(0)
cv2.imshow("match", best_image)
cv2.waitKey(0)
cv2.destroyAllWindows()