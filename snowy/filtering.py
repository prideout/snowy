"""Define resize, blur, and related constants."""

from . import io
from collections import namedtuple
from numba import guvectorize
import math
import numpy as np

RowOps = namedtuple('RowOps', 'tindices sindices fweights'.split())

GAUSSIAN_SCALE = 1.0 / np.sqrt(0.5 * np.pi)

def hermite(x):
    x = np.clip(x, 0, 1)
    return 2 * x * x * x - 3 * x * x + 1

def triangle(x):
    x = np.clip(x, 0, 1)
    return 1.0 - x

def gaussian(x):
    x = np.clip(x, 0, 2)
    return np.exp(-2 * x * x) * GAUSSIAN_SCALE

def nearest(x):
    return np.less_equal(x, 0.5) * 1.0

def sinc(x):
    if x <= 0.00001: return 1.0
    return np.sin(np.pi * x) / (np.pi * x)

def lanczos(x):
    x = np.clip(x, 0, 1)
    return sinc(x) * sinc(x)

def mitchell(x):
    B = 1.0 / 3.0
    C = 1.0 / 3.0
    P0 = (6 - 2*B) / 6.0
    P1 = 0
    P2 = (-18 +12*B + 6*C) / 6.0
    P3 = (12 - 9*B - 6*C) / 6.0
    Q0 = (8*B +24*C) / 6.0
    Q1 = (-12*B -48*C) / 6.0
    Q2 = (6*B +30*C) / 6.0
    Q3 = (-1*B - 6*C) / 6.0
    if x >= 2.0: return 0.0
    if x >= 1.0: return Q0 + Q1*x + Q2*x*x + Q3*x*x*x
    return P0 + P1*x + P2*x*x + P3*x*x*x

class Filter:
    def __init__(self, fn, radius):
        self.radius = radius
        self.function = fn

HERMITE  = Filter(hermite, 1)
TRIANGLE = Filter(triangle, 1)
GAUSSIAN = Filter(gaussian, 2)
NEAREST  = Filter(nearest, 0)
LANCZOS  = Filter(lanczos, 1)
MITCHELL = Filter(mitchell, 2)

def resize(source, width=None, height=None, filter=None, radius=1,
           wrapx=False, wrapy=False):
    """Create a new numpy image with the desired size.

    Either width or height can be null, in which case its value
    is inferred from the aspect ratio of the source image.

    Filter can be HERMITE, TRIANGLE, GAUSSIAN, NEAREST, LANCZOS, or
    MITCHELL.
    """
    assert len(source.shape) == 3, 'Shape is not rows x cols x channels'
    assert width != None or height != None,  'Missing target size'
    aspect = source.shape[1] / source.shape[0]
    if width == None: width = height * aspect
    if height == None: height = width / aspect
    magnifying = width > source.shape[1]
    if filter == None: filter = MITCHELL if magnifying else LANCZOS
    return resample(source, width, height, filter, radius, wrapx, wrapy)

def resample(source, width, height, filter, radius, wrapx, wrapy):
    nchans = source.shape[2]
    def fn(t): return filter.function(t / radius)
    scaled_filter = Filter(fn, radius * filter.radius)
    srows, scols = source.shape[0], source.shape[1]
    trows, tcols = int(height), int(width)
    vresult = np.zeros([srows, tcols, nchans])
    rowops = create_ops(tcols, scols, scaled_filter, wrapx)
    convolve(vresult, source, rowops)
    vresult = transpose(vresult)
    hresult = np.zeros([tcols, trows, nchans])
    rowops = create_ops(trows, srows, scaled_filter, wrapy)
    convolve(hresult, vresult, rowops)
    return transpose(hresult)

def blur(image, filter=GAUSSIAN, radius=4, wrapx=False, wrapy=False):
    """Resample an image and produce a new image with the same size.
    
    For a list of available filters, see <a href="#resize">resize</a>.
    """
    width, height = image.shape[1], image.shape[0]
    return resize(image, width, height, filter, radius, wrapx, wrapy)

def transpose(source: np.ndarray):
    return np.swapaxes(source, 0, 1)

def create_ops(ntarget, nsource, filter: Filter, wrap) -> RowOps:
    # Generate a sequence of operations to perform a 1D convolution
    # where each operation is represented by 3-tuple of: target index,
    # source index, weight.
    tindices, sindices, fweights = [], [], []
    dtarget = 1.0 / ntarget
    dsource = 1.0 / nsource
    minifying = ntarget < nsource
    fextent = dtarget if minifying else dsource
    fdomain = float(ntarget if minifying else nsource)
    x = dtarget / 2
    for tindex in range(ntarget):
        minx = x - filter.radius * fextent
        maxx = x + filter.radius * fextent
        minsi = int(minx * float(nsource))
        maxsi = int(math.ceil(maxx * float(nsource)))
        localops = []
        weightsum = 0.0
        for sindex in range(minsi, maxsi+1):
            wrapped = sindex
            if sindex < 0 or sindex >= nsource:
                if wrap:
                    wrapped = sindex % nsource
                else:
                    continue
            sx = (0.5 + sindex) * dsource
            t = fdomain * abs(sx - x)
            weight = filter.function(t)
            if weight != 0:
                localops.append((tindex, wrapped, weight))
                weightsum += weight
        if weightsum > 0.0:
            for op in localops:
                tindices.append(op[0])
                sindices.append(op[1])
                fweights.append(op[2] / weightsum)
        x += dtarget
    return RowOps(tindices, sindices, fweights)

SIG0 = "void(f8[:,:,:], f8[:,:,:], i4[:], i4[:], f8[:])"
SIG1 = "(r0,c0,d),(r0,c1,d),(i),(i),(i)"
@guvectorize([SIG0], SIG1, target='parallel')
def jit_convolve(target, source, tinds, sinds, weights):
    nrows, nchan, nops = target.shape[0], target.shape[2], len(tinds)
    for c in range(nchan):
        for row in range(nrows):
            for op in range(nops):
                tind, sind, weight = tinds[op], sinds[op], weights[op]
                target[row][tind][c] += source[row][sind][c] * weight

def convolve(target, source, rowops: RowOps):
    # Perform highly generalized 1D convolution. This is almost
    # equivalent to:
    #
    # for row in range(len(target)):
    #     target[row][tindices] += source[row][sindices] * fweights
    #
    # ...but with the crucial feature of allowing the same index to
    # appear multiple times in tindices.
    #
    # Note that standard numpy convolution assumes a stationary kernel,
    # whereas this function could possibly be used to apply a varying
    # kernel.
    tindices, sindices, fweights = rowops
    assert len(tindices) == len(sindices) == len(fweights)
    assert len(target) == len(source)
    jit_convolve(target, source,
                 np.int32(tindices), np.int32(sindices),
                 np.double(fweights))
