import cv2
import os
import numpy as np

def detect_noise(image_path, output_dir):
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(image_path)
        out_path = os.path.join(output_dir, f"Noise_{base}")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return f"Noise failed: cannot read {image_path}"

        # Laplacian variance (focus/sharpness measure)
        lap = cv2.Laplacian(img, cv2.CV_64F)
        lap_var = float(lap.var())

        # Residual noise (image - blurred_image)
        blur = cv2.GaussianBlur(img, (9,9), 0)
        residual = cv2.absdiff(img, blur)

        # Save residual heatmap
        # Normalize to 0-255
        if residual.max() > 0:
            residual_norm = (255.0 * (residual / (residual.max()))).astype("uint8")
        else:
            residual_norm = residual

        cv2.imwrite(out_path, residual_norm)

        return f"Noise saved: {out_path} (lap_var={lap_var:.2f})"
    except Exception as e:
        return f"Noise failed for {image_path}: {e}"
