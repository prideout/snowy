"""
This implements the paper 'Distance Transforms of Sampled Functions'
by Felzenszwalb and Huttenlocher.

Distance fields are useful in a variety of applications, including
image segmentation, antialiasing algorithms, and texture synthesis.
"""

from numba import jit
import numba
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

def _generate_gdt(image, wrapx, wrapy):
    image = io.unshape(image)
    height, width = image.shape
    capacity = max(width, height)
    if wrapx or wrapy: capacity *= 2
    d = np.zeros([capacity])
    z = np.zeros([capacity + 1])
    v = np.zeros([capacity], dtype='i4')
    result = image.copy()
    _generate_udf(width, height, d, z, v, result, wrapx, wrapy)
    return io.reshape(result)

def _generate_edt(image, wrapx, wrapy):
    image = io.unshape(image)
    height, width = image.shape
    capacity = max(width, height)
    if wrapx or wrapy: capacity *= 2
    d = np.zeros([capacity])
    z = np.zeros([capacity + 1])
    v = np.zeros([capacity], dtype='i4')
    result = np.where(image, 0.0, INF)
    _generate_udf(width, height, d, z, v, result, wrapx, wrapy)
    return np.sqrt(io.reshape(result))

@jit(nopython=True, fastmath=True)
def _generate_udf(width, height, d, z, v, result, wrapx, wrapy):
    # Compute 1D distance fields for columns, then for rows.
    for x in range(width):
        f = result[:,x]
        edt(f, d, z, v, height, wrapy)
        result[:,x] = d[:height]
    for y in range(height):
        f = result[y,:]
        edt(f, d, z, v, width, wrapx)
        result[y,:] = d[:width]

@jit(nopython=True, fastmath=True)
def edt(f, d, z, v, n, wrap):
    # Find the lower envelope of a sequence of parabolas.
    #   f...source data (returns the Y of the parabola rooted at X)
    #   d...destination data (final distance values are written here)
    #   z...temporary used to store X coords of parabola intersections
    #   v...temporary used to store X coords of parabola roots
    #   n...number of pixels in "f" to process

    # Always add the first pixel to the enveloping set since it is
    # obviously lower than all parabolas processed so far.
    k: int = 0
    v[0] = 0
    z[0] = -INF
    z[1] = +INF
    upper = 2*n if wrap else n

    for q in range(1, upper):

        # If the new parabola is lower than the right-most parabola in
        # the envelope, remove it from the envelope. To make this
        # determination, find the X coordinate of the intersection (s)
        # between the parabolas rooted at (q,f[q]) and (p,f[p]).
        p = v[k]
        s = ((f[q%n] + q*q) - (f[p%n] + p*p)) / (2.0*q - 2.0*p)
        while s <= z[k]:
            k = k - 1
            p = v[k]
            s = ((f[q%n] + q*q) - (f[p%n] + p*p)) / (2.0*q - 2.0*p)

        # Add the new parabola to the envelope.
        k = k + 1
        v[k] = q
        z[k] = s
        z[k + 1] = +INF

    # Go back through the parabolas in the envelope and evaluate them
    # in order to populate the distance values at each X coordinate.
    k = 0
    for q in range(0, upper):
        while z[k + 1] < float(q):
            k = k + 1
        d[q % n] = (q - v[k]) * (q - v[k]) + f[v[k] % n]
