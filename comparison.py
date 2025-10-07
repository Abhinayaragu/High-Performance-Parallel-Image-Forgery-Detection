import plotext as plt

log_file = "data/output/timing_log.txt"

processes = []
times = []

with open(log_file, "r") as f:
    for line in f:
        line = line.strip()
        if "processes ->" in line:
            proc_part, time_part = line.split("processes ->")
            proc = int(proc_part.strip())
            time_val = float(time_part.replace("seconds","").strip())
            processes.append(proc)
            times.append(time_val)

if not processes:
    print("No data found in timing log!")
else:
    plt.plot(processes, times, marker="x")
    plt.title("HPC Execution Time")
    plt.xlabel("Number of MPI Processes")
    plt.ylabel("Execution Time (s)")
    
    plt.ylim(5, 0)
    
    plt.show()
