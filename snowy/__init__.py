"""Tiny module for manipulating and generating floating-point images."""

from . import distance
from . import filtering
from . import io
from . import noise
from . import ops

generate_sdf = distance.generate_sdf
generate_udf = distance.generate_udf

blur = filtering.blur
resize = filtering.resize
GAUSSIAN = filtering.GAUSSIAN
HERMITE = filtering.HERMITE
LANCZOS = filtering.LANCZOS
MITCHELL = filtering.MITCHELL
NEAREST = filtering.NEAREST
TRIANGLE = filtering.TRIANGLE

load = io.load
reshape = io.reshape
save = io.save
show = io.show
unshape = io.unshape

generate_noise = noise.generate_noise

add_border = ops.add_border
gradient = ops.gradient
hstack = ops.hstack
unitize = ops.unitize
vstack = ops.vstack
rotate = ops.rotate
hflip = ops.hflip
vflip = ops.vflip
