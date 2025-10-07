import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

# Which detectors to run (you can edit)
DETECTORS = ["ela", "noise", "copymove"]

# Image extensions to process
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
