#!/usr/bin/env python3 -m pytest -s

# The shebang runs the test with stdout enabled and must be invoked from
# the repo root.

import os
import snowy as sn
import numpy as np

def path(filename: str):
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(scriptdir, filename)

def test_ao():

    ref = sn.load(path('islands.png'))
    sz, sz2 = 512, 1024
    ref = sn.resize(ref, height=sz)
    heightmap = ref[:,:sz,:]
    occlusion = ref[:,sz:sz2,:]
    occlusion = sn.unitize(occlusion)
    viz = sn.resize(np.hstack([heightmap, occlusion]), height=128)
    sn.show(viz)

    heightmap = heightmap[:,:,0:1]
    occlusion = sn.compute_skylight(heightmap)
    occlusion = sn.unitize(occlusion)
    viz = sn.resize(np.hstack([occlusion]), height=128)
    sn.show(viz)
