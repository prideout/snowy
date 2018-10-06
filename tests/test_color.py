#!/usr/bin/env python3 -m pytest -s

# The shebang runs the test with stdout enabled and must be invoked from
# the repo root.

import snowy as sn
import numpy as np
import pytest
import timeit

from skimage.color.adapt_rgb import adapt_rgb
from skimage import filters
from skimage.color import rgb2gray

def as_gray(image_filter, image, *args, **kwargs):
    gray_image = rgb2gray(image)
    return image_filter(gray_image, *args, **kwargs)

@adapt_rgb(as_gray)
def skimage_sobel(image):
    return filters.sobel(image)

def test_luminance():
    source = sn.load('tests/sobel_input.png')[:,:,:3]

    L = rgb2gray(source)
    skresult = np.dstack([L, L, L])
    small_skresult = sn.resize(skresult, width=256)

    L = sn.rgb_to_luminance(source)
    snresult = np.dstack([L, L, L])
    small_snresult = sn.resize(snresult, width=256)

    L = skimage_sobel(source)
    sksobel = np.dstack([L, L, L])
    small_sksobel = sn.resize(sksobel, width=256)

    L = sn.rgb_to_luminance(source)
    L = sn.compute_sobel(L)
    snsobel = np.dstack([L, L, L])
    small_snsobel = sn.resize(snsobel, width=256)

    sn.show(np.hstack([
        small_skresult,
        small_snresult,
        small_sksobel,
        small_snsobel]))

def test_thick():
    source = sn.load('tests/sobel_input.png')[:,:,:3]
    small_source = sn.resize(source, width=256)
    blurred = sn.blur(source, radius=2)
    small_blurred = sn.resize(blurred, width=256)

    L = skimage_sobel(blurred)
    sksobel = np.dstack([L, L, L])
    small_sksobel = sn.resize(sksobel, width=256)

    L = sn.rgb_to_luminance(blurred)
    L = sn.compute_sobel(L)
    snsobel = np.dstack([L, L, L])
    small_snsobel = sn.resize(snsobel, width=256)

    small_sksobel  = np.clip(1 - small_sksobel * 40, 0, 1)
    small_snsobel  = np.clip(1 - small_snsobel * 40, 0, 1)

    strip = np.hstack([
        small_blurred,
        small_source * small_sksobel,
        small_source * small_snsobel])
    sn.show(strip)
