from mpi4py import MPI
import os
import cv2
from config import INPUT_DIR, OUTPUT_DIR
from detectors.ela import detect_ela
from detectors.noise import detect_noise
from detectors.copymove import detect_copymove

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return f"{img_path}: Failed to read image"

    results = {}
    results['ela'] = detect_ela(img, img_path, OUTPUT_DIR)
    results['noise'] = detect_noise(img, img_path, OUTPUT_DIR)
    results['copymove'] = detect_copymove(img, img_path, OUTPUT_DIR)
    return results

if rank == 0:
    # Master process
    images = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    chunks = [images[i::size] for i in range(size)]
else:
    chunks = None

my_images = comm.scatter(chunks, root=0)
my_results = [process_image(img) for img in my_images]

results = comm.gather(my_results, root=0)

if rank == 0:
    print("Forgery Detection Completed ✅")
    for worker_res in results:
        for res in worker_res:
            print(res)
