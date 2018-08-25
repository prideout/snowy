"""Define save, load, show, reshape and unshape."""

import imageio
import numpy as np
import os
import tempfile

def show(image):
    """Display an image in a platform-specific way."""
    if isinstance(image, np.ndarray):
        show_array(image)
    elif isinstance(image, str):
        show_filename(image)
    else:
        raise ValueError('Unsupported type')

def reshape(image):
    """Add a trailing dimension to single-channel 2D images.

    See also <a href="#unshape">unshape</a>.
    """
    if len(image.shape) == 2:
        image = np.reshape(image, image.shape + (1,))
    return image

def unshape(image):
    """Remove the trailing dimension from single-channel 3D images.
    
    See also <a href="#reshape">reshape</a>.
    """
    if len(image.shape) == 3 and image.shape[2] == 1:
        return np.reshape(image, image.shape[:2])
    return image

def _load(filename: str):
    # PNG files are always loaded as RGBA because paletted byte-based
    # images with transparency are problematic. Moreover with Snowy we
    # decided that it is fine for the in-memory representation (floats)
    # to not match the on-disk representation (bytes).
    if filename.endswith('.png'):
        return imageio.imread(filename, 'PNG-PIL',
                pilmode='RGBA') / 255.0
    elif filename.endswith('.exr'):
        imageio.plugins.freeimage.download()
    return imageio.imread(filename)

def load(filename: str):
    """Create a numpy array from the given image file.

    Regardless of the pixel format on disk, PNG pixels are always
    divided by 255.0 and extended to 4 color channels before being
    returned to the caller.

    See also <a href="#reshape">reshape</a> (which this calls) and
    <a href="#save">save</a>.
    """
    return reshape(np.float64(_load(filename)))

def save(image: np.ndarray, filename: str, image_format: str=None):
    """Save a numpy array as an image file at the given path.

    See also <a href="#unshape">unshape</a> (which this calls) and
    <a href="#load">load</a>.
    """
    if filename.endswith('.exr'):
        imageio.plugins.freeimage.download()
        image_format = 'EXR-FI'
        image = np.float32(image)
    imageio.imwrite(filename, unshape(image), image_format)

def show_array(image: np.ndarray):
    with tempfile.NamedTemporaryFile() as fp:
        imageio.imwrite(fp.name, image, format='PNG-PIL')
        show(fp.name)

def show_filename(image: str):
    if 0 == os.system('which -s imgcat'):
        print("\n")
        os.system('imgcat ' + image)
    elif platform.system() == 'Darwin':
        os.system('open ' + image)
    elif platform.system() == 'Linux':
        os.system('xdg-open ' + image)
    else:
        print('Generated ' + image)

def compose(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Compose a source image with alpha onto a destination image."""
    alpha = extract_alpha(src)
    a = extract_rgb(dst) * (1.0 - alpha)
    b = extract_rgb(src) * alpha
    return a + b

def compose_premultiplied(dst: np.ndarray, src: np.ndarray):
    """Draw an image with premultiplied alpha over the destination."""
    alpha = extract_alpha(src)
    a = extract_rgb(dst) * (1.0 - alpha)
    b = extract_rgb(src)
    return a + b

def extract_alpha(image: np.ndarray) -> np.ndarray:
    """Extract the alpha plane from an RGBA image."""
    assert len(image.shape) == 3 and image.shape[2] == 4
    return np.dsplit(image, 4)[3]

def extract_rgb(image: np.ndarray) -> np.ndarray:
    """Extract the RGB planes from an RGBA image."""
    assert len(image.shape) == 3 and image.shape[2] >= 3
    planes = np.dsplit(image, image.shape[2])
    return np.dstack(planes[:3])

def to_planar(image: np.ndarray) -> np.ndarray:
    """Convert a row-major image into a channel-major image."""
    assert len(image.shape) == 3
    return np.dsplit(image, image.shape[2])

def from_planar(image: np.ndarray) -> np.ndarray:
    """Convert a channel-major image into row-major image."""
    assert len(image.shape) == 3
    return np.dstack(image)
