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

def ensure_alpha(src: np.ndarray) -> np.ndarray:
    """If the incoming image is 3-channel, adds a 4th channel."""
    assert len(src.shape) == 3
    if src.shape[2] != 3:
        return src
    alpha = np.ones(src.shape[:2])
    r, g, b = to_planar(src)
    return from_planar(np.array([r, g, b, alpha]))

def extract_alpha(image: np.ndarray) -> np.ndarray:
    """Extract the alpha plane from an RGBA image.
    
    Note that this returns a copy, not a view. To manipulate the pixels
    in a <i>view</i> of the alpha plane, simply make a numpy slice, as
    in: <code>alpha_view = myimage[:,:,3]</code>.
    """
    assert len(image.shape) == 3 and image.shape[2] == 4
    return np.dsplit(image, 4)[3].copy()

def extract_rgb(image: np.ndarray) -> np.ndarray:
    """Extract the RGB planes from an RGBA image.
    
    Note that this returns a copy. If you wish to obtain a view that
    allows mutating pixels, simply use slicing instead. For
    example, to invert the colors of an image while leaving alpha
    intact, you can do:
    <code>myimage[:,:,:3] = 1.0 - myimage[:,:,:3]</code>.
    """
    assert len(image.shape) == 3 and image.shape[2] >= 3
    planes = np.dsplit(image, image.shape[2])
    return np.dstack(planes[:3])

def to_planar(image: np.ndarray) -> np.ndarray:
    """Convert a row-major image into a channel-major image.
    
    This creates a copy, not a view.
    """
    assert len(image.shape) == 3
    result = np.array(np.dsplit(image, image.shape[2]))
    return np.reshape(result, result.shape[:-1])

def from_planar(image: np.ndarray) -> np.ndarray:
    """Create a channel-major image into row-major image.
    
    This creates a copy, not a view.
    """
    assert len(image.shape) == 3
    return np.dstack(image)
