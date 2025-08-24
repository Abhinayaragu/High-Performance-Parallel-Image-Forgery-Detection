from PIL import Image, ImageChops, ImageEnhance
import os

def detect_ela(img, img_path, output_dir):
    try:
        filename = os.path.basename(img_path)
        temp_path = os.path.join(output_dir, f"ela_{filename}")

        # Save image at 90% quality
        im = Image.open(img_path).convert('RGB')
        im.save(temp_path, 'JPEG', quality=90)
        resaved = Image.open(temp_path)

        # Compute difference
        ela = ImageChops.difference(im, resaved)
        extrema = ela.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1
        ela = ImageEnhance.Brightness(ela).enhance(scale)
        ela.save(temp_path)

        return f"ELA saved: {temp_path}"
    except Exception as e:
        return f"ELA failed on {img_path}: {e}"
