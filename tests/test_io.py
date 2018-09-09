#!/usr/bin/env python3 -m pytest -s

import snowy
import numpy as np
import os
import pytest
import tempfile

from snowy.io import show_filename
from snowy.io import show_array

def path(filename: str):
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(scriptdir, filename)

def test_io():

    # Ensure that to_planar and from_planar do the right thing to shape.
    a = np.array([[[1,2,3,4],[5,6,7,8]]], dtype='f8')
    assert a.shape == (1,2,4)
    b = snowy.to_planar(a)
    assert b.shape == (4,1,2)
    c = snowy.from_planar(b)
    assert np.array_equal(a, c)

    # Ensure that to_planar creates a copy, not a view.
    b[0,0,0] = 100
    assert np.amax(a) == 8

    # Ensure that from_planar creates a copy, not a view.
    c[0,0,0] == 200
    assert np.amax(b) == 100

    # Ensure that extract_rgb does the right thing with shape and makes
    # a copy rather than a view.
    color = snowy.extract_rgb(a)
    assert color.shape == (1, 2, 3)
    color[0,0,0] = 100
    assert np.amax(a) == 8

    # Ensure that extract_alpha does the right thing with shape and
    # makes a copy rather than a view.
    alpha = snowy.extract_alpha(a)
    assert alpha.shape == (1, 2, 1)
    alpha[0,0,0] = 100
    assert np.amax(a) == 8

    # This next snippet doesn't test Snowy but shows how to make a view
    # of the alpha plane.
    alpha_view = a[:,:,3]
    assert alpha_view[0,0] == 4
    assert alpha_view[0,1] == 8
    alpha_view[0,0] = 100
    assert np.amax(a) == 100

def test_range():

    source = path('../docs/ground.jpg')
    ground = snowy.load(source)
    assert np.amin(ground) >= 0 and np.amax(ground) <= 1

    with tempfile.NamedTemporaryFile() as fp:
        target = fp.name + '.png'
        snowy.export(ground, target)
        show_filename(target)

    show_filename(source)
    show_array(ground, True)

    blurred = snowy.blur(ground, radius=10)
    snowy.show(blurred)

def test_solid():
    gray = np.ones([100, 100, 4]) / 2
    snowy.show(gray)

def test_gamma():

    source = path('gamma_dalai_lama_gray.jpg')
    dalai_lama = snowy.load(source)
    snowy.show(dalai_lama)

    small = snowy.resize(dalai_lama, height=32)
    snowy.export(small, path('small_dalai_lama.png'))
    snowy.show(small)
