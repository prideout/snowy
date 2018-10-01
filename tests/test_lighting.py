#!/usr/bin/env python3 -m pytest -s

# The shebang runs the test with stdout enabled and must be invoked from
# the repo root.

import os
import snowy as sn
import numpy as np

def path(filename: str):
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(scriptdir, filename)

def create_falloff(w, h, radius=0.4, cx=0.5, cy=0.5):
    hw, hh = 0.5 / w, 0.5 / h
    x = np.linspace(hw, 1 - hw, w)
    y = np.linspace(hh, 1 - hh, h)
    u, v = np.meshgrid(x, y, sparse=True)
    d2 = (u-cx)**2 + (v-cy)**2
    return 1-sn.unitize(sn.reshape(d2))

def create_island(seed, freq=3.5):
    w, h = 750, 512
    falloff = create_falloff(w, h)
    n1 = 1.000 * sn.generate_noise(w, h, freq*1, seed+0)
    n2 = 0.500 * sn.generate_noise(w, h, freq*2, seed+1)
    n3 = 0.250 * sn.generate_noise(w, h, freq*4, seed+2)
    n4 = 0.125 * sn.generate_noise(w, h, freq*8, seed+3)
    elevation = falloff * (falloff / 2 + n1 + n2 + n3 + n4)
    mask = elevation < 0.4
    elevation = sn.unitize(sn.generate_udf(mask))
    return np.power(elevation, 3.0)

def test_ao():

    ref = sn.load(path('islands.png'))
    sz, sz2 = 128, 256
    ref = sn.resize(ref, height=sz)
    heightmap = ref[:,:sz,:]
    occlusion = ref[:,sz:sz2,:]
    viz = np.hstack([heightmap, occlusion])
    sn.show(viz)

    heightmap = heightmap[:,:,0:1]
    occlusion = sn.compute_skylight(heightmap)
    viz = np.hstack([occlusion])
    sn.show(viz)

    isle = create_island(10)
    occlusion = sn.compute_skylight(isle)
    normals = 0.5 * (sn.compute_normals(isle) + 1.0)
    normals = sn.resize(normals, 750, 512)
    isle = np.dstack([isle, isle, isle])
    occlusion = np.dstack([occlusion, occlusion, occlusion])
    sn.show(sn.resize(sn.hstack([isle, occlusion, normals]), height=256))
