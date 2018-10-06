from numba import jit
import numpy as np
from . import io

def rgb_to_luminance(image: np.ndarray):
    "Read the first three color planes and return a grayscale image."
    assert image.shape[2] >= 3
    r, g, b = np.dsplit(image[:,:,:3], image.shape[2])
    return io.reshape(0.2125 * r + 0.7154 * g + 0.0721 * b)

def compute_sobel(image: np.ndarray):
    "Apply Sobel operator for edge detection."
    assert len(image.shape) == 3, 'Shape is not rows x cols x channels'
    assert image.shape[2] == 1, 'Image must be grayscale'
    result = np.empty(image.shape)
    _compute_sobel(result, image)
    return result

@jit(nopython=True, fastmath=True, cache=True)
def _compute_sobel(target, source):
    height, width = source.shape[:2]
    for row in range(height):
        for col in range(width):
            xm1 = max(0, col - 1)
            ym1 = max(0, row - 1)
            xp1 = min(width - 1, col + 1)
            yp1 = min(height - 1, row + 1)
            t00 = source[ym1][xm1]
            t10 = source[ym1][col]
            t20 = source[ym1][xp1]
            t01 = source[row][xm1]
            t21 = source[row][xp1]
            t02 = source[yp1][xm1]
            t12 = source[yp1][col]
            t22 = source[yp1][xp1]
            gx = t00 + 2.0 * t01 + t02 - t20 - 2.0 * t21 - t22
            gy = t00 + 2.0 * t10 + t20 - t02 - 2.0 * t12 - t22
            target[row][col] = np.sqrt(gx * gx + gy * gy)
