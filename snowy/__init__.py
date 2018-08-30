"""Tiny module for manipulating and generating floating-point images."""

from . import filtering
blur = filtering.blur
resize = filtering.resize
GAUSSIAN = filtering.GAUSSIAN
HERMITE = filtering.HERMITE
LANCZOS = filtering.LANCZOS
MITCHELL = filtering.MITCHELL
NEAREST = filtering.NEAREST
TRIANGLE = filtering.TRIANGLE

from . import io
load = io.load
reshape = io.reshape
save = io.save
show = io.show
unshape = io.unshape
extract_alpha = io.extract_alpha
extract_rgb = io.extract_rgb
to_planar = io.to_planar
from_planar = io.from_planar
ensure_alpha = io.ensure_alpha

from . import ops
add_border = ops.add_border
gradient = ops.gradient
hstack = ops.hstack
unitize = ops.unitize
vstack = ops.vstack
rotate = ops.rotate
hflip = ops.hflip
vflip = ops.vflip
compose = ops.compose
compose_premultiplied = ops.compose_premultiplied

from . import distance
generate_sdf = distance.generate_sdf
generate_udf = distance.generate_udf
generate_gdf = distance.generate_gdf

from . import noise
generate_noise = noise.generate_noise
