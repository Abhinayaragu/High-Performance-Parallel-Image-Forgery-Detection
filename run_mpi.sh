#!/bin/bash
# adjust -np to number of MPI ranks you want (use 4 for your i3 with 4 threads)
mpirun --oversubscribe -np 4 python3 forgery_mpi.py

