"""Tiny module for manipulating and generating floating-point images.

The Snowy API is intentionally flat and only defines functions, no
classes. Users should simply import the top-level package, not any of
its submodules.
"""

from .filtering import *
from .io import *
from .ops import *
from .distance import *
from .noise import *
from .lighting import *

__all__ = '''
GAUSSIAN HERMITE LANCZOS MITCHELL NEAREST TRIANGLE
blur resize

LINEAR SRGB GAMMA
delinearize
ensure_alpha
extract_alpha
extract_rgb
from_planar
linearize
load
reshape
export
show
to_planar
unshape
ColorSpace

add_border
compose
compose_premultiplied
gradient
hflip
hstack
rotate
unitize
vflip
vstack

generate_gdf
generate_sdf
generate_udf
generate_cpcf
dereference_cpcf

generate_noise

compute_skylight
compute_normals
'''.split()

# deprecated functions:
save = export
