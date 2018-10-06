#!/usr/bin/env python3 -m pytest -s

# The shebang runs the test with stdout enabled and must be invoked from
# the repo root.

import snowy as sn
import numpy as np
import math
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

def create_circle(w, h, radius, center=[0,5.0,5]):
    cx, cy = center
    hw, hh = 0.5 / w, 0.5 / h
    dp = max(hw, hh)
    x = np.linspace(hw, 1 - hw, w)
    y = np.linspace(hh, 1 - hh, h)
    u, v = np.meshgrid(x, y, sparse=True)
    d2, r2 = (u-cx)**2 + (v-cy)**2, radius**2
    result = np.where(d2 < r2, 1.0, 0.0)
    return sn.reshape(result)

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

def test_coords():

    h, w = 800, 800
    height, width = h, w

    # Draw seed image

    cyan   = np.full([h, w, 3], np.float64([(27, 56,  80)]) / 200)
    pink   = np.full([h, w, 3], np.float64([175, 111, 127]) / 255)
    orange = np.full([h, w, 3], np.float64([239, 159,  95]) / 255)
    yellow = np.full([h, w, 3], np.float64([239, 207,  95]) / 255)

    colors = np.zeros([h, w, 3])
    def max_color(v): return np.maximum(colors, v)
    def sub_color(v): return colors * (1 - v)

    colors = max_color(create_circle(w, h, 0.37, [0.4, 0.5]) * cyan)
    colors = max_color(create_circle(w, h, 0.37, [0.6, 0.4]) * cyan)
    colors = max_color(create_circle(w, h, 0.27, [0.7, 0.6]) * cyan)
    colors = sub_color(create_circle(w, h, 0.35, [0.4, 0.5]))
    colors = sub_color(create_circle(w, h, 0.35, [0.6, 0.4]))
    colors = sub_color(create_circle(w, h, 0.25, [0.7, 0.6]))
    colors = max_color(create_circle(w, h, 0.01, [0.4, 0.5]) * orange)
    colors = max_color(create_circle(w, h, 0.01, [0.6, 0.4]) * pink)
    colors = max_color(create_circle(w, h, 0.01, [0.7, 0.6]) * yellow)

    colors = sn.linearize(colors)

    # Create generalized voronoi

    luma = sn.reshape(np.sum(colors, 2))
    coords = sn.generate_cpcf(luma != 0)
    voronoi = sn.dereference_cpcf(colors, coords)

    # Warp the voronoi

    warpx, warpy = width / 15, height / 15
    noise = sn.generate_fBm(width, height, 4, 4, 3)

    i, j = np.arange(width, dtype='i2'), np.arange(height, dtype='i2')
    coords = np.dstack(np.meshgrid(i, j, sparse=False))

    warpx = warpx * np.cos(noise * math.pi * 2)
    warpy = warpy * np.sin(noise * math.pi * 2)
    coords += np.int16(np.dstack([warpx, warpy]))

    coords[:,:,0] = np.clip(coords[:,:,0], 0, width - 1)
    coords[:,:,1] = np.clip(coords[:,:,1], 0, height - 1)
    warped = sn.dereference_cpcf(voronoi, coords)

    strip = [sn.resize(i, height=256) for i in (colors, voronoi, warped)]
    sn.show(sn.hstack(strip))
