#!/usr/bin/env python3 -m pytest -s

# The shebang runs the test with stdout enabled and must be invoked from
# the repo root.

import snowy
import numpy as np
import pytest

w, h = 1920 / 4, 1080 / 4

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def create_circle(w, h, radius=0.4, cx=0.5, cy=0.5):
    hw, hh = 0.5 / w, 0.5 / h
    dp = max(hw, hh)
    x = np.linspace(hw, 1 - hw, w)
    y = np.linspace(hh, 1 - hh, h)
    u, v = np.meshgrid(x, y, sparse=True)
    d2, r2 = (u-cx)**2 + (v-cy)**2, radius**2
    result = 1 - smoothstep(radius-dp, radius+dp, np.sqrt(d2))
    return snowy.reshape(result)

def test_draw_triangle():

    w, h = 100, 100
    def show(im):
        snowy.show(snowy.resize(im, height=100, filter=None))

    yellow = np.full((w, h, 4), (1, 1, 0, 1))
    red = np.full((w, h, 4), (1, 0, 0, 1))
    trans_border = np.full((w, h, 4), (0, 0, 1, 0.2))
    t = 5; trans_border[t:h-t,t:w-t] *= 0
    c0 = create_circle(w, h, 0.3) * yellow * 100000
    c1 = create_circle(w, h, 0.07, 0.8, 0.8) * red * 10000
    circles = np.clip(c0 + c1 + trans_border, 0, 1)
    r, g, b, a = circles.swapaxes(0, 2)
    luma = snowy.reshape(r + g + b)
    mask = luma != 0.0
    sdf = snowy.unitize(np.abs(snowy.generate_sdf(mask)))
    cpcf = snowy.generate_cpcf(mask)

    voronoi = snowy.dereference_coords(circles, cpcf)
    show(voronoi)

    target = np.full((200, 400, 4), (0, 0, 0, 1), dtype=np.float32)
    snowy.draw_triangle(target, voronoi, np.array([
        (-.9, -.2, 1., 0., 0.),
        (+.1, +.9, 1., 1., 0.),
        (+.4, -.5, 1., 0., 1.) ]))
    show(target)

def test_draw_quad():

    w, h = 100, 100
    def show(im):
        snowy.show(snowy.resize(im, height=100, filter=None))

    yellow = np.full((w, h, 4), (1, 1, 0, 1))
    red = np.full((w, h, 4), (1, 0, 0, 1))
    trans_border = np.full((w, h, 4), (0, 0, 1, 0.2))
    t = 5; trans_border[t:h-t,t:w-t] *= 0
    c0 = create_circle(w, h, 0.3) * yellow * 100000
    c1 = create_circle(w, h, 0.07, 0.8, 0.8) * red * 10000
    circles = np.clip(c0 + c1 + trans_border, 0, 1)
    r, g, b, a = circles.swapaxes(0, 2)
    luma = snowy.reshape(r + g + b)
    mask = luma != 0.0
    sdf = snowy.unitize(np.abs(snowy.generate_sdf(mask)))
    cpcf = snowy.generate_cpcf(mask)

    voronoi = snowy.dereference_coords(circles, cpcf)
    show(voronoi)

    target = np.full((200, 400, 4), (0, 0, 0, 1), dtype=np.float32)
    snowy.draw_polygon(target, voronoi, np.array([
        (-1., -1, 1., 0., 1.),
        (-.5, +1, 1., 0., 0.),
        (+.5, +1, 1., 1., 0.),
        (+1., -1, 1., 1., 1.) ]))
    show(target)
