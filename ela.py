from PIL import Image, ImageChops, ImageEnhance
import os
import numpy as np

def detect_ela(image_path, output_dir):
   
    try:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.basename(image_path)
        ela_out = os.path.join(output_dir, f"ELA_{base}")

        img = Image.open(image_path).convert("RGB")
        tmp = os.path.join(output_dir, f"tmp_resaved_{base}")
        img.save(tmp, "JPEG", quality=90)

        recompressed = Image.open(tmp)
        diff = ImageChops.difference(img, recompressed)

        # enhance visibility
        diff = ImageEnhance.Brightness(diff).enhance(10.0)

        # compute a simple numeric score (variance)
        arr = np.asarray(diff).astype("float32")
        score = float(arr.var())

        diff.save(ela_out)
        try:
            os.remove(tmp)
        except Exception:
            pass

        return f"ELA saved: {ela_out} (score={score:.2f})"
    except Exception as e:
        return f"ELA failed for {image_path}: {e}"
