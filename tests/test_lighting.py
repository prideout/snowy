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
    heightmap = ref[:,:512,:]
    occlusion = ref[:,512:1024,:]
    occlusion *= occlusion # <== make it more visually pronounced
    viz = sn.resize(np.hstack([heightmap, occlusion]), height=128)
    sn.show(viz)

    occlusion = sn.compute_skylight(heightmap)
    viz = sn.resize(np.hstack([occlusion]), height=128)
    sn.show(viz)
