def heart(x, y, z):
    """ The Taubin's heart function. """
    return ( (x**2 + 9/4*y**2 + z**2 - 1)**3
            - x**2*z**3 - 9/80*y**2*z**3 )

""" Consider a region: lims[0] <= x, y, z <= # lims[1]
    Divide x, y, z into n points each, then
    calculate H = heart(x, y, z) in the region. """
n = 100
lims = [-2, 2]
H = heart(None, None, None)

""" Use the marching cubes algorithm to solve
    the surface of H = heart(x, y, z) = level. """
from skimage.measure import marching_cubes
verts, faces = marching_cubes(H, level=0)[:2]

""" verts = a  V*3 array
    faces = an F*3 array
    The algorithm forms F triangles by joining V vertices.
    
    For example, if faces[f] = [a, b, c],
    the f-th triangle is formed by joining
    verts[a], verts[b], and verts[c].
        
    If verts[v] = [i, j, k], the v-th vertex's
    coordinates are (x[j], y[i], z[k]).
    Note that the x-coordinate uses j
    and the y-coordinate uses i.
    
    However, [i, j, k] may not be integers,
    so you need to scale the them linearly.
    For example, if verts[v, 2] = 12.3,
    the v-the vertex's actual z-coordinate is
        z[floor(12.3)] * (1-0.3) + z[ceil(12.3)] * 0.3
    
    Now, calculate the vertices' coordinates.
    Follow the index of convetion of verts, so
    coords[:, 1] = x-coordinates and coords[:, 0] = y-coordinates """
coords = verts

""" Plot the data on a 3D Axes with ax.plot_trisurf().
    For the title, the Unicode of the heart symbol is U+2661 """
ax = None
ax.set_title('')
# You do not need to edit this line.
ax.plot_trisurf(coords[:, 1], coords[:, 0], 
                faces, coords[:, 2], cmap='Spectral_r')

""" Finally, compute the heart's cross-section at y=0
    and plot it with ax.contour().
    You will probably need to use the listed parameters. """
cross = heart(None, 0, None) 
ax.contour(zdir=None, levels=None, offset=None)