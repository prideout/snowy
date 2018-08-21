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
        return np.reshape(image, image.shape + (1,))
    return image

def unshape(image):
    """Remove the trailing dimension from single-channel 3D images.
    
    See also <a href="#reshape">reshape</a>.
    """
    if len(image.shape) == 3 and image.shape[2] == 1:
        return np.reshape(image, image.shape[:2])
    return image

def load(filename: str):
    """Create a numpy array from the given image file."""
    if filename.endswith('.exr'):
        imageio.plugins.freeimage.download()
    imgarray = imageio.imread(filename)
    return reshape(np.float64(imgarray))

def save(image: np.ndarray, filename: str, image_format: str=None):
    """Save a numpy array as an image file at the given path."""
    if filename.endswith('.exr'):
        imageio.plugins.freeimage.download()
        image_format = 'EXR-FI'
        image = np.float32(image)
    result = imageio.imwrite(filename, unshape(image), image_format)
    return result

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
