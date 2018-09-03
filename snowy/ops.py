"""Define add_border etc."""

from snowy.io import *
from numba import jit, guvectorize
import numpy as np

def add_left(image: np.ndarray, T=2, V=0) -> np.ndarray:
    height, width, nchan = image.shape
    newshape = height, width + T, nchan
    result = np.full(newshape, np.float64(V))
    np.copyto(result[:,T:], image)
    return result

def add_right(image: np.ndarray, T=2, V=0) -> np.ndarray:
    height, width, nchan = image.shape
    newshape = height, width + T, nchan
    result = np.full(newshape, np.float64(V))
    np.copyto(result[:,:-T], image)
    return result

def add_top(image: np.ndarray, T=2, V=0) -> np.ndarray:
    height, width, nchan = image.shape
    newshape = height + T, width, nchan
    result = np.full(newshape, np.float64(V))
    np.copyto(result[T:,:], image)
    return result

def add_bottom(image: np.ndarray, T=2, V=0) -> np.ndarray:
    height, width, nchan = image.shape
    newshape = height + T, width, nchan
    result = np.full(newshape, np.float64(V))
    np.copyto(result[:-T,:], image)
    return result

def add_border(image: np.ndarray, width=2, value=0, sides='ltrb'):
    """Extend the size of an image by adding borders.

    <p>
    The <code>sides</code> argument defaults to
    <code>"LTRB"</code>, which enables borders for all four sides: Left,
    Top, Right, and Bottom. This can be used to select which borders you
    wish to add.
    </p>

    """
    result = image
    sides = sides.upper()
    if 'L' in sides: result = add_left(result, width, value)
    if 'T' in sides: result = add_top(result, width, value)
    if 'R' in sides: result = add_right(result, width, value)
    if 'B' in sides: result = add_bottom(result, width, value)
    return result

def hstack(images, border_width=2, border_value=0):
    """Horizontally concatenate a list of images with a border.
    
    This is similar to numpy's <code>hstack</code> except that it adds
    a border around each image. The borders can be controlled
    with the optional <code>border_width</code> and
    <code>border_value</code> arguments. See also <a href="vstack">
    vstack</a>.
    """
    if border_width == 0: return np.hstack(images)
    T, V = border_width, border_value
    result = []
    for image in images[:-1]:
        result.append(add_border(image, T, V, 'LTB'))
    result.append(add_border(images[-1], T, V))
    return np.hstack(result)

def vstack(images, border_width=2, border_value=0):
    """Vertically concatenate a list of images with a border.
    
    This is similar to numpy's <code>vstack</code> except that it adds
    a border around each image. The borders can be controlled
    with the optional <code>border_width</code> and
    <code>border_value</code> arguments. See also <a href="hstack">
    hstack</a>.
    """
    if border_width == 0: return np.vstack(images)
    T, V = border_width, border_value
    result = []
    for image in images[:-1]:
        result.append(add_border(image, T, V, 'LTR'))
    result.append(add_border(images[-1], T, V))
    return np.vstack(result)

def unitize(img):
    """Remap the values so that they span the range from 0 to +1."""
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))

def gradient(img):
    """Compute X derivatives and Y derivatives."""
    nx, ny = np.gradient(unshape(img))
    return reshape(nx), reshape(ny)

def rotate(source: np.ndarray, degrees) -> np.ndarray:
    """Rotate image counter-clockwise by a multiple of 90 degrees."""
    assert len(source.shape) == 3, 'Shape is not rows x cols x channels'
    assert source.dtype == np.float, 'Images must be doubles.'
    h, w, c = source.shape
    degrees %= 360
    if degrees == 90:
        result = np.empty([w, h, c])
        rotate90(result, source)
    elif degrees == 180:
        result = np.empty([h, w, c])
        rotate180(result, source)
    elif degrees == 270:
        result = np.empty([w, h, c])
        rotate270(result, source)
    else:
        assert False, 'Angle must be a multiple of 90.'
    return result

def hflip(source: np.ndarray) -> np.ndarray:
    """Horizontally mirror the given image."""
    assert len(source.shape) == 3, 'Shape is not rows x cols x channels'
    assert source.dtype == np.float, 'Images must be doubles.'
    h, w, c = source.shape
    result = np.empty([h, w, c])
    jit_hflip(result, source)
    return result

def vflip(source: np.ndarray) -> np.ndarray:
    """Vertically mirror the given image."""
    assert len(source.shape) == 3, 'Shape is not rows x cols x channels'
    assert source.dtype == np.float, 'Images must be doubles.'
    h, w, c = source.shape
    result = np.empty([h, w, c])
    jit_vflip(result, source)
    return result

def compose(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Compose a source image with alpha onto a destination image."""
    a, b = ensure_alpha(src), ensure_alpha(dst)
    alpha = extract_alpha(a)
    result = b * (1.0 - alpha) + a * alpha
    if dst.shape[2] == 3:
        return extract_rgb(result)
    return result

def compose_premultiplied(dst: np.ndarray, src: np.ndarray):
    """Draw an image with premultiplied alpha over the destination."""
    a, b = ensure_alpha(src), ensure_alpha(dst)
    alpha = extract_alpha(a)
    result = b * (1.0 - alpha) + a
    if dst.shape[2] == 3:
        return extract_rgb(result)
    return result

SIG0 = "void(f8[:,:,:], f8[:,:,:])"
SIG1 = "(r,c,d),(c,r,d)"
@guvectorize([SIG0], SIG1, target='parallel')
def rotate90(result, source):
    nrows, ncols, nchan = source.shape
    for row in range(nrows):
        for col in range(ncols):
            for chan in range(nchan):
                v = source[row][col][chan]
                result[-col-1][row][chan] = v

SIG0 = "void(f8[:,:,:], f8[:,:,:])"
SIG1 = "(r,c,d),(r,c,d)"
@guvectorize([SIG0], SIG1, target='parallel')
def rotate180(result, source):
    nrows, ncols, nchan = source.shape
    for row in range(nrows):
        for col in range(ncols):
            for chan in range(nchan):
                v = source[row][col][chan]
                result[-row-1][-col-1][chan] = v

SIG0 = "void(f8[:,:,:], f8[:,:,:])"
SIG1 = "(r,c,d),(c,r,d)"
@guvectorize([SIG0], SIG1, target='parallel')
def rotate270(result, source):
    nrows, ncols, nchan = source.shape
    for row in range(nrows):
        for col in range(ncols):
            for chan in range(nchan):
                v = source[row][col][chan]
                result[col][-row-1][chan] = v

SIG0 = "void(f8[:,:,:], f8[:,:,:])"
SIG1 = "(r,c,d),(r,c,d)"
@guvectorize([SIG0], SIG1, target='parallel')
def jit_hflip(result, source):
    nrows, ncols, nchan = source.shape
    for row in range(nrows):
        for col in range(ncols):
            for chan in range(nchan):
                v = source[row][col][chan]
                result[row][-col-1][chan] = v

SIG0 = "void(f8[:,:,:], f8[:,:,:])"
SIG1 = "(r,c,d),(r,c,d)"
@guvectorize([SIG0], SIG1, target='parallel')
def jit_vflip(result, source):
    nrows, ncols, nchan = source.shape
    for row in range(nrows):
        for col in range(ncols):
            for chan in range(nchan):
                v = source[row][col][chan]
                result[-row-1][col][chan] = v
