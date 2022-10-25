import numpy as np
from numba import njit, prange, set_num_threads
import time 


@njit(parallel=True)
def heat_divergence(T, dx):
    nx, ny = T.shape
    divT = np.zeros_like(T)
    for ix in prange(1, nx-1):
        for iy in prange(1, ny-1):
            divT[ix, iy] = (T[ix-1, iy]+T[ix+1, iy] +
                            T[ix, iy+1]+T[ix, iy-1]-4*T[ix, iy])/(dx**2)
    return divT


@njit(parallel=True)
def pde_step(T, K, dt, dx):
    divT = heat_divergence(T, dx)
    newT = np.copy(T)
    nx, ny = T.shape
    for ix in prange(1, nx-1):
        for iy in prange(1, ny-1):
            newT[ix, iy] += K*divT[ix, iy]*dt
    return newT


class PDESimu:
    def __init__(self, T0=None, Tend=1, K=.1, dt=.01, dx=.1,nprocess=4):
        self.T0 = T0
        self.ts = np.arange(0, Tend, dt)
        self.its = np.arange(0, len(self.ts))
        self.Ts = [T0]
        T = T0

        set_num_threads(nprocess)
        start_time = time.perf_counter()
        for _ in self.its[1:]:
            T = pde_step(T, K, dt, dx)
            self.Ts.append(T)
        print(f"Target PDE simulation finished in {time.perf_counter()-start_time}s with {nprocess} processes.")


if __name__ == "__main__":
    nx, ny = 12, 12
    T0 = np.zeros((nx, ny))
    T0[:, 0] = 20
    T0[:, ny-1] = 40
    T0[0, :] = 40
    T0[nx-1, :] = 40

    simu = PDESimu(T0, 1, 1, .1, .1)
    print(simu.Ts[-1])
