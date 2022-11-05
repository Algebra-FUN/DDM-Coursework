# %% 0
# Import all dependency
import math
from skimage.measure import marching_cubes
from matplotlib import pyplot as plt
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

# %% 1
# A sphere on a cube


def bind(cls):
    def decor(func):
        setattr(cls, func.__name__, func)
        return func
    return decor


@bind(Axes3D)
def plot_ball(ax, x0, y0, z0, r, **kwargs):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = x0 + r * np.outer(np.cos(u), np.sin(v))
    y = y0 + r * np.outer(np.sin(u), np.sin(v))
    z = z0 + r * np.outer(np.ones_like(u), np.cos(v))
    return ax.plot_surface(x, y, z, **kwargs)


@bind(Axes3D)
def plot_box(ax, x0, y0, z0, l, **kwargs):
    d = l / 2
    x = x0 + np.array([-d, -d, d, d, -d]*2).reshape(2, 5)
    y = y0 + np.array([-d, d, d, -d, -d]*2).reshape(2, 5)
    z = z0 + d * np.vstack((np.ones(5), -np.ones(5)))
    return ax.plot_surface(x, y, z, **kwargs)


@bind(Axes3D)
def plot_zrect(ax, x0, y0, z0, l, **kwargs):
    d = l / 2
    x = x0 + np.array([-d, -d, d, d]).reshape(2, 2)
    y = y0 + np.array([-d, d, -d, d]).reshape(2, 2)
    z = z0 * np.ones((2, 2))
    return ax.plot_surface(x, y, z, **kwargs)


fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')

ax.plot_box(10, 20, 30, 10, alpha=0, linewidth=.5, edgecolors='blue')
ax.plot_zrect(10, 20, 35, 10, color="yellow",
              linewidth=.5, edgecolors='blue')
ax.plot_ball(10, 20, 40, 5, color='red',
             linewidth=.5, edgecolors='black')
ax.scatter(10, 20, 30, marker="*", color="red")

ax.set_box_aspect([1, 1, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()

# %% 2
# Taubin’s heart equation


def heart(x, y, z):
    """ The Taubin's heart function. """
    return ((x**2 + 9/4*y**2 + z**2 - 1)**3
            - x**2*z**3 - 9/80*y**2*z**3)


n = 100
lims = [-2, 2]
ps = np.linspace(*lims, n)
x, y, z = np.meshgrid(*[ps]*3)
H = heart(x, y, z)

verts, faces = marching_cubes(H, level=0)[:2]


def get_coord(x, v):
    frac, _ = math.modf(v)
    return x[math.floor(v)] * (1-frac) + x[math.ceil(v)] * frac


get_coords = np.vectorize(partial(get_coord, ps))

coords = get_coords(verts)

fig = plt.figure()
ax = fig.add_subplot(projection='3d', proj_type='ortho')
ax.set_title("\u2661 in Python")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_zlim(lims)

ax.plot_trisurf(coords[:, 1], coords[:, 0],
                faces, coords[:, 2], cmap='Spectral_r')

x, z = np.meshgrid(*[ps]*2)
cross = heart(x, 0, z)
ax.contour(x, cross, z, zdir="y", levels=0, offset=2, color="black")

plt.show()
# %% 3
