1. Put images (jpg/png) into data/input/
2. Run: ./run_mpi.sh
3. Results (ELA, noise residual, copy-move visual, per-image .txt summary) will be in data/output/

Run fewer/more ranks by editing run_mpi.sh or run mpirun directly:
mpirun -np 2 python3 forgery_mpi.py
