"""Define add_border etc."""

import numpy as np
from . import io

def add_left(image: np.ndarray, T=2, V=0) -> np.ndarray:
    height, width, nchan = image.shape
    newshape = height, width + T, nchan
    result = np.full(newshape, float(V))
    np.copyto(result[:,T:], image)
    return result

def add_right(image: np.ndarray, T=2, V=0) -> np.ndarray:
    height, width, nchan = image.shape
    newshape = height, width + T, nchan
    result = np.full(newshape, float(V))
    np.copyto(result[:,:-T], image)
    return result

def add_top(image: np.ndarray, T=2, V=0) -> np.ndarray:
    height, width, nchan = image.shape
    newshape = height + T, width, nchan
    result = np.full(newshape, float(V))
    np.copyto(result[T:,:], image)
    return result

def add_bottom(image: np.ndarray, T=2, V=0) -> np.ndarray:
    height, width, nchan = image.shape
    newshape = height + T, width, nchan
    result = np.full(newshape, float(V))
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
    
    <p>
    Also takes a <code>width</code> argument which defaults to 2. This
    can control the thickness of the border.
    </p>

    <p>
    The <code>value</code> argument (defaults to 0) controls the fill
    color used for the border. For example:
    </p>

    <pre class="highlight">
    image = snowy.add_border(image, width=3, value=1, sides='LTR')
    </pre>

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
    <code>border_value</code> arguments.
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
    <code>border_value</code> arguments.
    """
    if border_width == 0: return np.vstack(images)
    T, V = border_width, border_value
    result = []
    for image in images[:-1]:
        result.append(add_border(image, T, V, 'LTR'))
    result.append(add_border(images[-1], T, V))
    return np.vstack(result)

def unitize(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))

def gradient(img):
    nx, ny = np.gradient(io.unshape(img))
    return io.reshape(nx), io.reshape(ny)
