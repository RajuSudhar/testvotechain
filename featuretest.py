import cv2
import numpy as np
import mysql.connector
import time
from functions import extract_feature_val

TERMINATION_THRESHOLD = 30
BIFURCATION_THRESHOLD = 30 # Threshold +-30


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