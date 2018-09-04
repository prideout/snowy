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
delinearize = io.delinearize
ensure_alpha = io.ensure_alpha
extract_alpha = io.extract_alpha
extract_rgb = io.extract_rgb
from_planar = io.from_planar
linearize = io.linearize
load = io.load
reshape = io.reshape
save = io.save
show = io.show
to_planar = io.to_planar
unshape = io.unshape
ColorSpace = io.ColorSpace

from . import ops
add_border = ops.add_border
compose = ops.compose
compose_premultiplied = ops.compose_premultiplied
gradient = ops.gradient
hflip = ops.hflip
hstack = ops.hstack
rotate = ops.rotate
unitize = ops.unitize
vflip = ops.vflip
vstack = ops.vstack

from . import distance
generate_gdf = distance.generate_gdf
generate_sdf = distance.generate_sdf
generate_udf = distance.generate_udf

from . import noise
generate_noise = noise.generate_noise
