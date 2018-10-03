# 1. Create falloff shape.

import snowy
import numpy as np
from functools import reduce
from scipy import interpolate

width, height = 768, 256
x, y = np.linspace(-1, 1, width), np.linspace(-1, 1, height)
u, v = np.meshgrid(x, y, sparse=True)
falloff = np.clip(1 - (u * u + v * v), 0, 1)
falloff = snowy.reshape(falloff / 2)
snowy.show(falloff)

# 2. Add layers of gradient noise and scale with falloff.

noise = snowy.generate_noise
noise = [noise(width, height, 6 * 2**f, int(f)) * 1/2**f for f in range(4)]
noise = reduce(lambda x, y: x+y, noise) 
elevation = falloff * (falloff / 2 + noise)
elevation = snowy.generate_udf(elevation < 0.1)
elevation /= np.amax(elevation)
snowy.show(elevation)

# 3. Compute ambient occlusion.

occlusion = snowy.compute_skylight(elevation)
snowy.show(occlusion)

# 4. Generate normal map.

normals = snowy.resize(snowy.compute_normals(elevation), width, height)
snowy.show(0.5 + 0.5 * normals)

# 5. Apply harsh diffuse lighting.

lightdir = np.float64([0.2, -0.2, 1])
lightdir /= np.linalg.norm(lightdir)
lambert = np.sum(normals * lightdir, 2)
snowy.show(snowy.reshape(lambert) * occlusion)

# 6. Lighten the occlusion, flatten the normals, and re-light.

occlusion = 0.5 + 0.5 * occlusion
normals += np.float64([0,0,0.5])
normals /= snowy.reshape(np.sqrt(np.sum(normals * normals, 2)))
lambert = np.sum(normals * lightdir, 2)
lighting = snowy.reshape(lambert) * occlusion
snowy.show(lighting)

# 7. Apply color gradient.

xvals = np.arange(256)
yvals = snowy.load('tests/terrain.png')[0,:,:3]
apply_lut = interpolate.interp1d(xvals, yvals, axis=0)
el = elevation * 0.2 + 0.49
el = np.clip(255 * el, 0, 255)
albedo = apply_lut(snowy.unshape(el))
snowy.show(albedo * lighting)
