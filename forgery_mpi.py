from mpi4py import MPI
import os
import re
import time
from config import INPUT_DIR, OUTPUT_DIR, IMAGE_EXTENSIONS, DETECTORS
from detectors import detect_ela, detect_noise, detect_copy_move

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time()

# Helper functions
def list_images(input_dir):
    files = []
    for f in sorted(os.listdir(input_dir)):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
            files.append(os.path.join(input_dir, f))
    return files

def parse_ela(msg: str):
    m = re.search(r"score=([0-9]*\.?[0-9]+)", msg)
    return float(m.group(1)) if m else None

def parse_noise(msg: str):
    m = re.search(r"lap_var=([0-9]*\.?[0-9]+)", msg)
    return float(m.group(1)) if m else None

def parse_copymove(msg: str):
    m = re.search(r"matches=([0-9]+)", msg)
    return int(m.group(1)) if m else None

# Decision thresholds
ELA_THRESHOLD = 15.0       # variance above => suspicious
LAPLACIAN_THRESHOLD = 100.0 # variance below => suspicious
COPY_MATCH_THRESHOLD = 10   # >= matches => suspicious

# Master rank prepares images
if rank == 0:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = list_images(INPUT_DIR)
    if not images:
        raise SystemExit(f"No images found in {INPUT_DIR}. Please add some to run the pipeline.")
    # Split images for all ranks
    chunks = [images[i::size] for i in range(size)]
    print(f"[Rank 0] Found {len(images)} images. Running with {size} MPI ranks.")
else:
    chunks = None

# Scatter image chunks to all ranks
my_chunk = comm.scatter(chunks, root=0)

# Process images
for img_path in my_chunk:
    try:
        print(f"[Rank {rank}] Processing {os.path.basename(img_path)}")

        # run detectors
        res_ela = detect_ela(img_path, OUTPUT_DIR) if "ela" in DETECTORS else "ELA skipped"
        res_noise = detect_noise(img_path, OUTPUT_DIR) if "noise" in DETECTORS else "Noise skipped"
        res_cm = detect_copy_move(img_path, OUTPUT_DIR) if "copymove" in DETECTORS else "CopyMove skipped"

        # parse numeric evidence
        ela_score = parse_ela(res_ela)
        lap_var = parse_noise(res_noise)
        cm_matches = parse_copymove(res_cm)

        # detector-level flags
        ela_flag = (ela_score is not None and ela_score > ELA_THRESHOLD)
        noise_flag = (lap_var is not None and lap_var < LAPLACIAN_THRESHOLD)
        copymove_flag = (cm_matches is not None and cm_matches >= COPY_MATCH_THRESHOLD)

        # voting fusion
        votes = int(bool(ela_flag)) + int(bool(noise_flag)) + int(bool(copymove_flag))
        final_status = "Forgery Likely" if votes >= 2 else "Authentic/Unclear"

        # write per-image summary file
        summary_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path) + ".txt")
        with open(summary_path, "w") as fh:
            fh.write(f"file: {os.path.basename(img_path)}\n")
            fh.write(f"ela_msg: {res_ela}\n")
            fh.write(f"noise_msg: {res_noise}\n")
            fh.write(f"copymove_msg: {res_cm}\n\n")
            fh.write(f"ela_score: {ela_score}\n")
            fh.write(f"lap_var: {lap_var}\n")
            fh.write(f"copy_matches: {cm_matches}\n\n")
            fh.write(f"ela_flag: {int(ela_flag)}\n")
            fh.write(f"noise_flag: {int(noise_flag)}\n")
            fh.write(f"copymove_flag: {int(copymove_flag)}\n")
            fh.write(f"votes: {votes}\n")
            fh.write(f"final_status: {final_status}\n")

        print(f"[Rank {rank}] Done {os.path.basename(img_path)} -> {final_status}")

    except Exception as e:
        print(f"[Rank {rank}] Error processing {img_path}: {e}")

comm.Barrier()

if rank == 0:
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[Rank 0] All ranks completed. Total elapsed time: {elapsed:.2f} seconds")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "timing_log.txt")
    with open(log_file, "a") as f:
        f.write(f"{size} processes -> {elapsed:.2f} seconds\n")
