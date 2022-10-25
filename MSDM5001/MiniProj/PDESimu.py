import numpy as np
from numba import njit, prange, set_num_threads
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


@njit(parallel=True)
def pde_step(T, newT ,K, dt, h):
    nx, ny = T.shape
    for ix in prange(1, nx-1):
        for iy in prange(1, ny-1):
            newT[ix, iy] += K*(T[ix-1, iy]+T[ix+1, iy] +
                               T[ix, iy+1]+T[ix, iy-1]-4*T[ix, iy])/(h**2)*dt
    return newT


class PDESimu:
    def __init__(self, T0=None, t_end=1, K=.1, dt=.01, h=.1, nprocess=4):
        self.T0 = T0
        self.ts = np.arange(0, t_end+dt, dt)
        self.its = np.arange(0, len(self.ts))
        self.Ts = [T0]
        self.h = h
        self.nprocess = nprocess
        T = T0

        set_num_threads(self.nprocess)
        self.used_time = 0
        start_time = time.perf_counter()
        for _ in self.its[1:]:
            newT = np.copy(T)
            start_time = time.perf_counter()
            T = pde_step(T, newT, K, dt, h)
            self.used_time += time.perf_counter()-start_time
            self.Ts.append(T)

    def anim(self, step=1):
        fig, ax = plt.subplots()
        img = ax.imshow(self.Ts[0], cmap="coolwarm")
        cbar = fig.colorbar(img)
        title = ax.set_title("T(t=0)")

        def draw_init():
            ax.set_xlabel("x")
            ax.set_xticklabels([x*self.h for x in ax.get_xticks()])
            ax.set_ylabel("y")
            ax.set_yticklabels([y*self.h for y in ax.get_yticks()])

        def draw_T(f):
            title.set_text(f"T(t={self.ts[f]:.1f}s)")
            img.set_data(self.Ts[f])

        anim = FuncAnimation(fig, draw_T,
                             init_func=draw_init, frames=self.its[::step])
        return anim
