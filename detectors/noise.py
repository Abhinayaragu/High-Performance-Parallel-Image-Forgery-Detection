import cv2
import os
import numpy as np

def detect_noise(img, img_path, output_dir):
    try:
        filename = os.path.basename(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        output_path = os.path.join(output_dir, f"noise_{filename}.txt")
        with open(output_path, "w") as f:
            f.write(f"Laplacian variance (blur/noise measure): {laplacian_var}\n")

        return f"Noise analysis saved: {output_path}"
    except Exception as e:
        return f"Noise detection failed on {img_path}: {e}"
