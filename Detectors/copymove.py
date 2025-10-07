import cv2
import os
import numpy as np

def detect_copy_move(image_path, output_dir):
   
    try:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(image_path)
        out_path = os.path.join(output_dir, f"CopyMove_{base}")

        img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_color is None:
            return f"Copy-move failed: cannot read {image_path}"

        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # ORB detector
        orb = cv2.ORB_create(nfeatures=1500)
        kps, des = orb.detectAndCompute(img, None)
        if des is None or len(kps) < 5:
            # not enough features
            cv2.imwrite(out_path, img_color)
            return f"Copy-move: not enough features ({len(kps) if kps else 0}) - saved blank vis: {out_path}"

        # match descriptors with themselves
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des, des)

        # filter out trivial self-match (same keypoint)
        filtered = []
        for m in matches:
            if m.queryIdx == m.trainIdx:
                continue
            # spatial separation check
            p1 = np.array(kps[m.queryIdx].pt)
            p2 = np.array(kps[m.trainIdx].pt)
            if np.linalg.norm(p1 - p2) < 10:  # too close -> ignore
                continue
            filtered.append(m)

        # create visualization: draw top matches
        filtered_sorted = sorted(filtered, key=lambda x: x.distance)
        top = filtered_sorted[:150]  # limit for visualization
        vis = cv2.drawMatches(img_color, kps, img_color, kps, top, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imwrite(out_path, vis)
        return f"Copy-move saved: {out_path} (matches={len(filtered)})"
    except Exception as e:
        return f"Copy-move failed for {image_path}: {e}"
