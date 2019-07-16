from numba import jit
import numpy as np
from . import io

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

    triverts = np.zeros((3, 5), dtype=np.float32)
    np.copyto(triverts[0], vertices[0])

    n = vertices.shape[0]
    for tri in range(2, n):
        np.copyto(triverts[1], vertices[tri - 1])
        np.copyto(triverts[2], vertices[tri])
        draw_triangle(target, source, triverts)

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
    area = 1 / _edge(v0, v1, v2)
    p = np.array([0, 0])

    source = source.astype(target.dtype, copy=False)
    v0 = v0.astype(np.float32, copy=False)
    v1 = v1.astype(np.float32, copy=False)
    v2 = v2.astype(np.float32, copy=False)
    uv = uv.astype(np.float32, copy=False)
    w = w.astype(np.float32, copy=False)
    p = p.astype(np.float32, copy=False)

    _rasterize(target, source, area, v0, v1, v2, uv, w, p)

@jit(nopython=True, fastmath=True, cache=True)
def _rasterize(target, source, area, v0, v1, v2, uv, w, p):
    height, width, comp = target.shape
    for row in range(height):
        for col in range(width):
            p[0] = col + .5
            p[1] = height - row + .5
            w0 = _edge(v1, v2, p)
            w1 = _edge(v2, v0, p)
            w2 = _edge(v0, v1, p)
            if w0 < 0 or w1 < 0 or w2 < 0:
                continue
            w0 *= area
            w1 *= area
            w2 *= area
            st = w0 * uv[0] + w1 * uv[1] + w2 * uv[2]
            st /= w0 * w[0] + w1 * w[1] + w2 * w[2]
            target[row][col] = _sample(source, st)

@jit(nopython=True, fastmath=True, cache=True)
def _sample(source, uv):
    height, width, comp = source.shape
    col = int(uv[0] * width)
    col = max(0, min(col, width - 1))
    row = int(uv[1] * height)
    row = max(0, min(row, height - 1))
    return source[row][col]

@jit(nopython=True, fastmath=True, cache=True)
def _edge(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
 