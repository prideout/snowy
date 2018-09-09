import numpy as np
import math
from . import io

def generate_noise(width, height, frequency, seed=1, wrapx=False,
                   wrapy=False, offset=[0,0]):
    """Generate a single-channel gradient noise image.
    
    A frequency of 1.0 creates a single surflet across the width of the
    image, while a frequency of 4.0 creates a 4x4 grid such that the
    (2,2) surflet is centered. Noise values live within the [-1,+1]
    range.
    """
    return _noise(width, height, frequency, seed, wrapx, wrapy, offset)

def _noise(width, height, frequency, seed, wrapx, wrapy, offset):
    nrows, ncols = int(height), int(width)
    table = Noise(seed)

    # Compute the span of U texcoords in [0,+1] such that 0 is at the
    # left edge of the left-most texel, and +1 is at the right edge of
    # the right-most pixel.
    maxx = frequency
    hw = 0.5 * maxx / width
    u = np.linspace(hw, maxx - hw, ncols) + offset[0]

    # Compute the span of V texcoords according to the aspect ratio.
    maxy = frequency * float(height) / width
    hh = 0.5 * maxy / height
    v = np.linspace(hh, maxy - hh, nrows) + offset[1]

    # Generate floating point texture coordinates, then split them into
    # integer and fractional components.
    u, v = np.meshgrid(u, v, sparse=True)
    i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
    i1, j1 = i0 + 1, j0 + 1
    x0, y0 = u - i0, v - j0
    x1, y1 = x0 - 1, y0 - 1

    # Find the 2D vectors at the nearest grid cell corners.
    if wrapx:
        assert math.modf(frequency)[0] == 0.0, \
            "wrapx requires an integer frequency"
        i0 = i0 % int(frequency)
        i1 = i1 % int(frequency)
    if wrapy:
        assert math.modf(maxy)[0] == 0.0, \
            "wrapy requires frequency*width/height to be an integer"
        j0 = j0 % int(maxy)
        j1 = j1 % int(maxy)
    grad00 = _gradient(table, i0, j0)
    grad01 = _gradient(table, i0, j1)
    grad10 = _gradient(table, i1, j0)
    grad11 = _gradient(table, i1, j1)

    va = dot(x0, y0, grad00[0], grad00[1])
    vb = dot(x1, y0, grad10[0], grad10[1])
    vc = dot(x0, y1, grad01[0], grad01[1])
    vd = dot(x1, y1, grad11[0], grad11[1])

    # Lerp the neighboring 4 surflets
    t0 = x0*x0*x0*(x0*(x0*6.0 - 15.0) + 10.0)
    t1 = y0*y0*y0*(y0*(y0*6.0 - 15.0) + 10.0)
    result = va + t0 * (vb-va) + t1 * (vc-va) + t0 * t1 * (va-vb-vc+vd)
    return io.reshape(result)

class Noise:
    def __init__(self, seed):
        self.rnd = np.random.RandomState(seed)
        self.size = 256
        self.mask = int(self.size - 1)
        self.indices = np.arange(self.size, dtype = np.int16)
        self.rnd.shuffle(self.indices)
        theta = np.linspace(0, math.tau, self.size, endpoint=False)
        self.gradients = [np.cos(theta), np.sin(theta)]

def _gradient(table: Noise, i, j):
    perm, mask = table.indices, table.mask
    u, v = table.gradients
    hash = perm[np.bitwise_and(perm[np.bitwise_and(i, mask)] + j, mask)]
    return u[hash], v[hash]

def dot(x, y, gradx, grady):
    return gradx * x + grady * y
