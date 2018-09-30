from . import io
from numba import prange, jit
import math
import numpy as np

SWEEP_DIRECTIONS = np.int16([
    (1, 0), (0, 1), (-1, 0), (0, -1), # Rook
    (1, 1), (-1, -1), (1, -1), (-1, 1), # Bishop
    (2, 1), (2, -1), (-2, 1), (-2, -1), # Knight
    (1, 2), (1, -2), (-1, 2), (-1, -2) # Knight
])

def compute_skylight(elevation):
    """Compute ambient occlusion from a height map."""
    height, width, nchan = elevation.shape
    assert nchan == 1
    elevation = elevation[:,:,0]
    result = np.zeros([height, width])
    _compute_skylight(result, elevation)
    result = 1.0 - result * 2.0 / np.pi
    return io.reshape(result)

def _compute_skylight(dst, src):
    height, width = src.shape

    # TODO Fix allocation or explain the "3"
    seedpoints = np.empty([3 * max(width, height), 2], dtype='i2')
    maxpathlen = max(width, height) + 1

    for direction in SWEEP_DIRECTIONS:
        nsweeps = _generate_seedpoints(src, direction, seedpoints)
        print('Horizon direction: ', direction)
        sweeps = np.empty([nsweeps, maxpathlen, 3])
        pts = np.empty([nsweeps, 3])
        _horizon_scan(src, dst, direction, seedpoints, sweeps, pts)

# TODO This function needs to be rewritten or documented.
def _generate_seedpoints(field, direction, seedpoints):
    h, w = field.shape[:2]
    s = 0
    sx, sy = np.sign(direction)
    ax, ay = np.abs(direction)
    nsweeps = ay * w + ax * h - (ax + ay - 1)
    for x in range(-ax, w - ax):
        for y in range(-ay, h - ay):
            if x >= 0 and x < w and y >= 0 and y < h: continue
            px, py = x, y
            if sx < 0: px = w - x - 1
            if sy < 0: py = h - y - 1
            seedpoints[s][0] = px
            seedpoints[s][1] = py
            s += 1
    assert nsweeps == s
    return nsweeps

SIG0 = "void(f8[:,:], f8[:,:], i2[:], i2[:,:], f8[:,:,:], f8[:,:])"
@jit([SIG0], nopython=True, fastmath=True, parallel=True)
def _horizon_scan(heights, occlusion, direction, seedpoints, sweeps, pts):
    h, w = heights.shape[:2]
    cellw = 1 / max(w, h)
    cellh = 1 / max(w, h)
    nsweeps = len(sweeps)
    for sweep in prange(nsweeps):
        thispt = pts[sweep]
        stack = sweeps[sweep]
        startpt = seedpoints[sweep]
        pathlen = 0
        i, j = startpt
        ii, jj = min(max(0, i), w-1), min(max(0, j), h-1)

        thispt[0] = i * cellw
        thispt[1] = j * cellh
        thispt[2] = heights[jj][ii]

        stack_top = 0

        stack[stack_top] = thispt

        i += direction[0]
        j += direction[1]
        while i >= 0 and i < w and j >= 0 and j < h:

            thispt[0] = i * cellw
            thispt[1] = j * cellh
            thispt[2] = heights[j][i]

            while stack_top > 0:

                a, b = thispt, stack[stack_top]
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                y = b[2] - a[2]
                x = math.sqrt(dx * dx + dy * dy)
                s1 = y / x

                a, b = thispt, stack[stack_top - 1]
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                y = b[2] - a[2]
                x = math.sqrt(dx * dx + dy * dy)
                s2 = y / x

                if s1 >= s2: break
                stack_top -= 1

            horizonpt = stack[stack_top]
            stack_top += 1
            stack[stack_top] = thispt

            d = horizonpt - thispt
            dx = d[2] / np.linalg.norm(d)
            occlusion[j][i] += math.atan(max(dx, 0))

            i += direction[0]
            j += direction[1]
