"""
This implements the paper 'Distance Transforms of Sampled Functions'
by Felzenszwalb and Huttenlocher.

Distance fields are useful in a variety of applications, including
image segmentation, antialiasing algorithms, and texture synthesis.
"""

from numba import jit
import numpy as np
from . import io

INF = 1e20

def generate_sdf(image: np.ndarray, wrapx=False, wrapy=False):
    """Create a signed distance field from a boolean field."""
    a = generate_udf(image, wrapx, wrapy)
    b = generate_udf(image == 0.0, wrapx, wrapy)
    return a - b

def generate_udf(image: np.ndarray, wrapx=False, wrapy=False):
    """Create an unsigned distance field from a boolean field."""
    assert image.dtype == 'bool', 'Pixel values must be boolean'
    assert len(image.shape) == 3, 'Shape is not rows x cols x channels'
    assert image.shape[2] == 1, 'Image must be grayscale'
    return _generate_edt(image, wrapx, wrapy)

def generate_gdf(image: np.ndarray, wrapx=False, wrapy=False):
    "Create an generalized squared distance field from a scalar field."
    assert image.dtype == 'float64', 'Pixel values must be real'
    assert len(image.shape) == 3, 'Shape is not rows x cols x channels'
    assert image.shape[2] == 1, 'Image must be grayscale'
    return _generate_gdt(image, wrapx, wrapy)

def generate_cpcf(image: np.ndarray):
    """Create closest point coordinate field from a boolean field."""
    assert image.dtype == 'bool', 'Pixel values must be boolean'
    assert len(image.shape) == 3, 'Shape is not rows x cols x channels'
    assert image.shape[2] == 1, 'Image must be grayscale'
    return _generate_cpcf(image)

def dereference_cpcf(source: np.ndarray, cpcf: np.ndarray):
    """
    For each coordinate in the cpcf, make a lookup in th source.
    This is useful for creating generalized voronoi diagrams.
    """
    assert len(cpcf.shape) == 3, 'Shape is not rows x cols x channels'
    assert len(source.shape) == 3, 'Shape is not rows x cols x channels'
    assert cpcf.shape[2] == 2, 'Coordinate must be 2-tuples'
    voronoi = source.copy()
    _deref_cpcf(voronoi, source, cpcf)
    return voronoi

def _deref_cpcf(voronoi, source, cpcf):
    for y in range(cpcf.shape[0]):
        for x in range(cpcf.shape[1]):
            i, j = cpcf[y][x]
            voronoi[y][x] = source[j][i]

def _generate_gdt(image, wrapx, wrapy):
    image = io.unshape(image)
    result = image.copy()
    _generate_udf(result, wrapx, wrapy)
    return io.reshape(result)

def _generate_edt(image, wrapx, wrapy):
    image = io.unshape(image)
    result = np.where(image, 0.0, INF)
    _generate_udf(result, wrapx, wrapy)
    return np.sqrt(io.reshape(result))

def _generate_cpcf(image):
    image = io.unshape(image)
    result = np.where(image, 0.0, INF)
    i, j = _generate_udf(result, False, False)
    cpcf = np.dstack([i, j])
    result = cpcf.copy()
    _process_cpcf(cpcf, result)
    np.copyto(cpcf, result)
    return cpcf

@jit(nopython=True, fastmath=True, cache=True)
def _process_cpcf(cpcf, result):
    for y in range(cpcf.shape[0]):
        for x in range(cpcf.shape[1]):
            i, j = cpcf[y][x]
            result[y][x][0] = i
            result[y][x][1] = cpcf[y][i][1]

def _generate_udf(result, wrapx, wrapy):

    scratch = result
    if wrapx: scratch = np.hstack([scratch, scratch, scratch])
    if wrapy: scratch = np.vstack([scratch, scratch, scratch])

    height, width = scratch.shape
    capacity = max(width, height)
    i = np.empty(scratch.shape, dtype='u2')
    j = np.empty(scratch.shape, dtype='u2')
    d = np.zeros([capacity])
    z = np.zeros([capacity + 1])
    v = np.zeros([capacity], dtype='u2')
    _generate_udf_native(width, height, d, z, v, i, j, scratch)

    x0, x1 = width // 3, 2 * width // 3
    y0, y1 = height // 3, 2 * height // 3
    if wrapx: scratch = scratch[:,x0:x1]
    if wrapy: scratch = scratch[y0:y1,:]
    if wrapx or wrapy: np.copyto(result, scratch)

    return i, j

@jit(nopython=True, fastmath=True, cache=True)
def _generate_udf_native(width, height, d, z, v, i, j, result):
    for x in range(width):
        f = result[:,x]
        edt(f, d, z, v, j[:,x], height)
        result[:,x] = d[:height]
    for y in range(height):
        f = result[y,:]
        edt(f, d, z, v, i[y,:], width)
        result[y,:] = d[:width]

@jit(nopython=True, fastmath=True, cache=True)
def edt(f, d, z, v, i, n):
    # Find the lower envelope of a sequence of parabolas.
    #   f...source data (returns the Y of the parabola vertex at X)
    #   d...destination data (final distance values are written here)
    #   z...temporary used to store X coords of parabola intersections
    #   v...temporary used to store X coords of parabola vertices
    #   i...resulting X coords of parabola vertices
    #   n...number of pixels in "f" to process

    # Always add the first pixel to the enveloping set since it is
    # obviously lower than all parabolas processed so far.
    k: int = 0
    v[0] = 0
    z[0] = -INF
    z[1] = +INF

    for q in range(1, n):

        # If the new parabola is lower than the right-most parabola in
        # the envelope, remove it from the envelope. To make this
        # determination, find the X coordinate of the intersection (s)
        # between the parabolas with vertices at (q,f[q]) and (p,f[p]).
        p = v[k]
        s = ((f[q] + q*q) - (f[p] + p*p)) / (2.0*q - 2.0*p)
        while s <= z[k]:
            k = k - 1
            p = v[k]
            s = ((f[q] + q*q) - (f[p] + p*p)) / (2.0*q - 2.0*p)

        # Add the new parabola to the envelope.
        k = k + 1
        v[k] = q
        z[k] = s
        z[k + 1] = +INF

    # Go back through the parabolas in the envelope and evaluate them
    # in order to populate the distance values at each X coordinate.
    k = 0
    for q in range(n):
        while z[k + 1] < float(q):
            k = k + 1
        dx = q - v[k]
        d[q] = dx * dx + f[v[k]]
        i[q] = v[k]
