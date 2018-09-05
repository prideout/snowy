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

def test_sdf():
    c0 = create_circle(200, 200, 0.3)
    c1 = create_circle(200, 200, 0.08, 0.8, 0.8)
    c0 = np.clip(c0 + c1, 0, 1)
    circles = snowy.add_border(c0, value=1)
    mask = circles != 0.0
    sdf = snowy.unitize(snowy.generate_sdf(mask))
    nx, ny = snowy.gradient(sdf)
    grad = snowy.unitize(nx + ny)
    snowy.show(snowy.hstack([circles, sdf, grad]))

def test_udf():
    c0 = create_circle(200, 200, 0.3)
    c1 = create_circle(200, 200, 0.08, 0.8, 0.8)
    c0 = np.clip(c0 + c1, 0, 1)
    circles = snowy.add_border(c0, value=1)
    mask = circles != 0.0
    udf = snowy.unitize(snowy.generate_udf(mask))
    nx, ny = snowy.gradient(udf)
    grad = snowy.unitize(nx + ny)
    snowy.show(snowy.hstack([circles, udf, grad]))

def test_gdf():
    "This is a (failed) effort to create a smoother distance field."
    c0 = create_circle(200, 200, 0.3)
    c1 = create_circle(200, 200, 0.08, 0.8, 0.8)
    c0 = np.clip(c0 + c1, 0, 1)
    circles = snowy.add_border(c0, value=1)
    circles = np.clip(snowy.blur(circles, radius=2), 0, 1)
    circles = np.clip(snowy.blur(circles, radius=2), 0, 1)
    source = (1.0 - circles) * 2000.0
    gdf = np.sqrt(snowy.generate_gdf(source))
    gdf = snowy.unitize(gdf)
    nx, ny = snowy.gradient(gdf)
    grad = snowy.unitize(nx + ny)
    snowy.show(snowy.hstack([circles, gdf, grad]))

def test_tweet():
    import snowy as sn, numpy as np
    img = sn.generate_noise(2000, 500, seed=2, frequency=5, wrapx=True)
    sdf = snowy.generate_sdf(img < 0.0, wrapx=True, wrapy=False)
    img = 0.5 + 0.5 * np.sign(img) - img
    get_mask = lambda L, U: np.logical_and(sdf > L, sdf < U)
    get_contour = lambda L, U: np.where(get_mask(L, U), img, 0)
    sn.show(sn.resize(img, height=100, wrapx=True, wrapy=False))
    img -= get_contour(20, 30)
    img -= get_contour(60, 70)
    img -= get_contour(100, 110)
    sn.show(sn.resize(img, height=100, wrapx=True, wrapy=False))
    sn.show(sn.resize(np.hstack([img, img]), height=200, wrapx=True))
    # sn.save(sn.resize(img, height=200), "~/Desktop/tweet.png")
