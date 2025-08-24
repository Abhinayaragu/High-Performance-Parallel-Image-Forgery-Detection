import cv2
import numpy as np
import os

def detect_copymove(img, img_path, output_dir):
    try:
        filename = os.path.basename(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors, descriptors)

        match_img = cv2.drawMatches(img, keypoints, img, keypoints, matches[:50], None, flags=2)
        output_path = os.path.join(output_dir, f"copymove_{filename}")
        cv2.imwrite(output_path, match_img)

        return f"Copy-Move saved: {output_path}"
    except Exception as e:
        return f"Copy-Move failed on {img_path}: {e}"
