#!/usr/bin/env python3

import sys
sys.path.append('../snowy')

import timeit
import snowy
from PIL import Image
import numpy as np

global imgarray
global pilimage

ZOOM = 16

def minify_with_pil():
    global imgarray
    global pilimage
    height, width = imgarray.shape[:2]
    print(imgarray.shape, imgarray.dtype)
    pilimage = pilimage.resize((width//ZOOM, height//ZOOM))
    # pilimage.show()

def minify_with_filo():
    global imgarray
    global pilimage
    print(imgarray.shape, imgarray.dtype)
    height, width = imgarray.shape[:2]
    imgarray = snowy.resize(imgarray, width//ZOOM, height//ZOOM)
    snowy.show(imgarray)

def setup(grayscale=False, imgfile='~/Desktop/SaltLakes.jpg'):
    print('Loading image...')
    global imgarray
    global pilimage
    imgarray = snowy.load(imgfile)
    if grayscale:
        assert imgarray.shape[2] == 3, "Not an RGB image."
        r,g,b = np.split(imgarray, 3, axis=2)
        imgarray = r
    pilimage = Image.fromarray(np.uint8(snowy.unshape(imgarray)))

seconds = timeit.timeit('minify_with_pil()', setup='setup()',
      globals=globals(), number=1)
print(f"PIL minification took {seconds:6.3} seconds")

seconds = timeit.timeit('minify_with_filo()', setup='setup()',
      globals=globals(), number=1)
print(f"Snowy minification took {seconds:6.3} seconds")
