from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n = comm.Get_size()

if rank == 0:
    comm.send(0, dest=1)
    start_time = time.perf_counter()

for i in range(50):
    data = comm.recv(source=(rank-1) % n)
    print(f"{data} -> {rank}", flush=True)
    comm.send(rank, dest=(rank+1) % n)

rank == 0 and print(f"elapsed time: {time.perf_counter()-start_time}")
