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
    viz = np.hstack([heightmap, occlusion])
    sn.show(viz)

    heightmap = heightmap[:,:,0:1]
    occlusion = sn.compute_skylight(heightmap)
    viz = np.hstack([occlusion])
    sn.show(viz)
