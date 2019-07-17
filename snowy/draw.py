from numba import guvectorize

import numpy as np
import math

def draw_polygon(target: np.ndarray, source: np.ndarray,
              vertices: np.ndarray):
    """Draw a textured convex polygon into the target image.
    
    The vertices are specified with a nx5 array where each row is XYWUV.
    The UV coordinates address the source image in [0,+1] with +V going
    downward. The XY coordinates are in the range [-1,+1] and their
    domain is the entire target image with +Y going upward. The W
    coordinate is to allow for perspective-correct interpolation. If you
    don't know what that means, then set W to 1.
    """
    assert len(target.shape) == 3, 'Target shape must be 3D.'
    assert target.shape[2] == 4, 'Target must be RGBA.'
    assert len(source.shape) == 3, 'Source shape must be 3D.'
    assert source.shape[2] == 4, 'Source must be RGBA.'
    assert vertices.shape[1] == 5, 'Vertices must be nx5.'

    n = vertices.shape[0]
    for tri in range(2, n):
        indices = np.array([0, tri - 1, tri])
        draw_triangle(target, source, vertices[indices])

def draw_triangle(target: np.ndarray, source: np.ndarray,
                  vertices: np.ndarray):
    """Draw a textured triangle into the target image.
    
    The vertices are specified with a 3x5 array where each row is XYWUV.
    The UV coordinates address the source image in [0,+1] with +V going
    downward. The XY coordinates are in the range [-1,+1] and their
    domain is the entire target image with +Y going upward. The W
    coordinate is to allow for perspective-correct interpolation. If you
    don't know what that means, then set W to 1.
    """
    assert len(target.shape) == 3, 'Target shape must be 3D.'
    assert target.shape[2] == 4, 'Target must be RGBA.'
    assert len(source.shape) == 3, 'Source shape must be 3D.'
    assert source.shape[2] == 4, 'Source must be RGBA.'
    assert vertices.shape == (3, 5), 'Vertices must be 3x5.'

    vertices = np.copy(vertices)
    xy = vertices[:, :2]
    w = vertices[:, 2:3]
    uv = vertices[:, 3:]

    w = 1.0 / w
    xy *= w
    uv *= w

    xy[:, 0] = (xy[:, 0] + 1.0) * 0.5 * target.shape[1]
    xy[:, 1] = (xy[:, 1] + 1.0) * 0.5 * target.shape[0]

    v0, v1, v2 = xy
    area = 1 / edge(v0, v1, v2)

    source = source.astype(target.dtype, copy=False)
    v0 = v0.astype(np.float32, copy=False)
    v1 = v1.astype(np.float32, copy=False)
    v2 = v2.astype(np.float32, copy=False)
    uv = uv.astype(np.float32, copy=False)
    w = w.astype(np.float32, copy=False)

    maxx = math.ceil(max([v0[0], v1[0], v2[0]]))
    maxy = math.ceil(max([v0[1], v1[1], v2[1]]))
    minx = math.floor(min([v0[0], v1[0], v2[0]]))
    miny = math.floor(min([v0[1], v1[1], v2[1]]))
    target = target[miny:maxy+1,minx:maxx+1,:]
    v0[0] -= minx
    v1[0] -= minx
    v2[0] -= minx
    v0[1] -= miny
    v1[1] -= miny
    v2[1] -= miny

    _rasterize(target, source, area, v0, v1, v2, uv, w)

SIG0 = "void(f4[:,:,:],f4[:,:,:],f8,f4[:],f4[:],f4[:],f4[:,:],f4[:,:])"
SIG1 = "(r0,c0,D4),(r1,c1,D4),(),(D2),(D2),(D2),(D3,D2),(D3,D1)"
@guvectorize([SIG0], SIG1, target='parallel', cache=True)
def _rasterize(target, source, area, v0, v1, v2, uv, w):
    height, width, _ = target.shape
    sheight, swidth, _ = source.shape
    ya0 = v2[1] - v1[1]
    ya1 = v0[1] - v2[1]
    ya2 = v1[1] - v0[1]
    yb0 = v2[0] - v1[0]
    yb1 = v0[0] - v2[0]
    yb2 = v1[0] - v0[0]
    for row in range(height):
        for col in range(width):
            px = col + .5
            py = height - row + .5
            w0 = (px - v1[0]) * ya0 - (py - v1[1]) * yb0
            w1 = (px - v2[0]) * ya1 - (py - v2[1]) * yb1
            w2 = (px - v0[0]) * ya2 - (py - v0[1]) * yb2
            if w0 < 0 or w1 < 0 or w2 < 0:
                continue
            w0 *= area
            w1 *= area
            w2 *= area
            s = w0 * uv[0][0] + w1 * uv[1][0] + w2 * uv[2][0]
            t = w0 * uv[0][1] + w1 * uv[1][1] + w2 * uv[2][1]
            s /= w0 * w[0][0] + w1 * w[1][0] + w2 * w[2][0]
            t /= w0 * w[0][1] + w1 * w[1][1] + w2 * w[2][1]
            scol = int(s * swidth) % swidth
            srow = int(t * sheight) % sheight
            target[row][col] = source[srow][scol]

def edge(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
 