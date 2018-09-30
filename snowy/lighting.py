import numpy as np
from . import io

SWEEP_DIRECTIONS = np.int16([
    (1, 0), (0, 1), (-1, 0), (0, -1), # Rook
    (1, 1), (-1, -1), (1, -1), (-1, 1), # Bishop
    (2, 1), (2, -1), (-2, 1), (-2, -1), # Knight
    (1, 2), (1, -2), (-1, 2), (-1, -2) # Knight
])

def compute_skylight(elevation):
    height, width = elevation.shape[:2]
    result = np.zeros([height, width])

    # TODO Fix allocation or explain the "3"
    seedpoints = np.empty([3 * max(width, height), 2], dtype='i2')

    for direction in SWEEP_DIRECTIONS:
        nsweeps = _generate_seedpoints(elevation, direction, seedpoints)
        _horizon_scan(elevation, result, direction, seedpoints, nsweeps)

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
            if sy < 0: px = h - y - 1
            seedpoints[s][0] = x
            seedpoints[s][1] = y
            s += 1
    assert nsweeps == s
    return nsweeps

def _horizon_scan(heights, occlusion, direction, seedpoints, nsweeps):
    h, w = heights.shape[:2]
    maxpathlen = max(w, h) + 1
    cellw = 1 / max(w, h)
    cellh = 1 / max(w, h)

    # Initialize a stack of candidate horizon points, one for each
    # sweep. In a serial implementation we wouldn't need to allocate
    # this much memory, but we're trying to make life easy for
    # multithreading.
    hull_buffer = np.float64([nsweeps, maxpathlen, 3])

    for sweep in range(nsweeps):
        startpt = seedpoints[sweep]
        pathlen = 0
        i, j = startpt
        thispt = (i * cellw, j * cellh, heights[j][i])
