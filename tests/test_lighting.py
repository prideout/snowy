#!/usr/bin/env python3 -m pytest -s

# The shebang runs the test with stdout enabled and must be invoked from
# the repo root.

import os
import snowy as sn
import numpy as np
from scipy import interpolate

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

def test_normals():
    isle = create_island(10)
    occlusion = sn.compute_skylight(isle)

    import timeit
    height, width, nchan = isle.shape
    normals = np.empty([height - 1, width - 1, 3])
    seconds = timeit.timeit(lambda: np.copyto(normals,
            sn.compute_normals(isle)), number=1)
    print(f'\ncompute_normals took {seconds} seconds')

    normals = sn.resize(normals, 750, 512)

    # Flatten the normals just a bit
    normals += np.float64([0,0,1])
    normals /= sn.reshape(np.sqrt(np.sum(normals * normals, 2)))

    # Compute the lambertian diffuse factor
    lightdir = np.float64([0.2, -0.2, 1])
    lightdir /= np.linalg.norm(lightdir)
    df = np.clip(np.sum(normals * lightdir, 2), 0, 1)
    df = sn.reshape(df)
    df *= occlusion

    # Apply color LUT
    gradient_image = sn.resize(sn.load(path('gradient.png')), width=1024)[:,:,:3]
    def applyColorGradient(elevation):
        xvals = np.arange(1024)
        yvals = gradient_image[0]
        apply_lut = interpolate.interp1d(xvals, yvals, axis=0)
        el = np.clip(1023 * elevation, 0, 1023)
        return apply_lut(sn.unshape(el))
    albedo = applyColorGradient(isle * 0.5 + 0.55)
    albedo *= df

    # Visualize the lighting layers
    normals = 0.5 * (normals + 1.0)
    isle = np.dstack([isle, isle, isle])
    occlusion = np.dstack([occlusion, occlusion, occlusion])
    df = np.dstack([df, df, df])
    sn.show(sn.resize(sn.hstack([isle, occlusion, normals, df, albedo]), height=256))
