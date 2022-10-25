import numpy as np
from matplotlib import pyplot as plt


class ODESimu:
    def __init__(self, T0=1, t_end=1, h=.01):
        self.T0 = T0
        self.t_end = t_end
        self.h = h
        self.ts = np.arange(0, t_end, step=h)
        self.Ts = np.zeros_like(self.ts)
        self.Ts[0] = T0

        for i in range(1, len(self.ts)):
            self.Ts[i] = self.Ts[i-1]*(1-h)

    def plot(self, *args, plotter=plt, **kwargs):
        plotter.plot(self.ts, self.Ts, *args, **kwargs)


class ODETheory:
    def __init__(self, T0=1, t_end=1, h=.01):
        self.T0 = T0
        self.t_end = t_end
        self.h = h
        self.ts = np.arange(0, t_end, step=h)
        self.Ts = T0*np.exp(-self.ts)

    def plot(self, *args, plotter=plt, **kwargs):
        plotter.plot(self.ts, self.Ts, *args, **kwargs)
