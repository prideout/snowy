from . import io
from numba import jit
import math
import numpy as np

SWEEP_DIRECTIONS = np.int16([
    (1, 0), (0, 1), (-1, 0), (0, -1), # Rook
    (1, 1), (-1, -1), (1, -1), (-1, 1), # Bishop
    (2, 1), (2, -1), (-2, 1), (-2, -1), # Knight
    (1, 2), (1, -2), (-1, 2), (-1, -2) # Knight
])

def compute_skylight(elevation):
    height, width, nchan = elevation.shape
    assert nchan == 1
    result = np.zeros([height, width])

    # TODO Fix allocation or explain the "3"
    seedpoints = np.empty([3 * max(width, height), 2], dtype='i2')
    maxpathlen = max(width, height) + 1

    for direction in SWEEP_DIRECTIONS:
        nsweeps = _generate_seedpoints(elevation, direction, seedpoints)
        print('Sweep: ', direction, nsweeps)

        # Allocate stacks of 3D points for the horizon, one for each
        # sweep. In a serial implementation we wouldn't need to allocate
        # this much memory, but we're trying to make life easy for
        # multithreading.
        sweeps = np.empty([nsweeps, maxpathlen, 3])

        _horizon_scan(elevation, result, direction, seedpoints, sweeps)

    result = 1.0 - result * 2.0 / np.pi
    return io.reshape(result)

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

def _horizon_scan(heights, occlusion, direction, seedpoints, sweeps):
    h, w = heights.shape[:2]
    cellw = 1 / max(w, h)
    cellh = 1 / max(w, h)
    nsweeps = len(sweeps)

    for sweep in range(nsweeps):
        stack = sweeps[sweep]
        startpt = seedpoints[sweep]
        pathlen = 0
        i, j = startpt
        ii, jj = np.clip(i, 0, w-1), np.clip(j, 0, h-1)
        thispt = (i * cellw, j * cellh, heights[jj][ii])
        stack_top = 0
        np.copyto(stack[stack_top], thispt)
        i += direction[0]
        j += direction[1]
        nsteps = 0
        while i >= 0 and i < w and j >= 0 and j < h:
            thispt = (i * cellw, j * cellh, heights[j][i])
            while stack_top > 0:
                s1 = _azimuth_slope(thispt, stack[stack_top])
                s2 = _azimuth_slope(thispt, stack[stack_top - 1])
                if s1 >= s2: break
                stack_top -= 1
            horizonpt = stack[stack_top]
            stack_top += 1
            np.copyto(stack[stack_top], thispt)
            occlusion[j][i] += _compute_occlusion(thispt, horizonpt)
            i += direction[0]
            j += direction[1]
            nsteps += 1

def _azimuth_slope(a, b):
    d = a - b
    x = math.sqrt(d[0]**2 + d[1]**2)
    y = b[2] - a[2]
    return y / x

def _compute_occlusion(thispt, horizonpt):
    d = horizonpt - thispt
    dx = d[2] / np.linalg.norm(d)
    return math.atan(max(dx, 0))
