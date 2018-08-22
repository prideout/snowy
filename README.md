## Snowy

This is a tiny Python 3 module for manipulating and generating images.
- Simple and flat API. See the [documentation](https://github.prideout.net/snowy/).
- Supports resize and blur with a variety of filters.
- Honors a specified wrap mode for proper boundary behavior.
- Supports simple OpenEXR images (never clamps colors).
- Written purely in Python 3 and accelerated using [numba](https://numba.pydata.org/).
- Efficiently generates gradient noise and distance fields.

Snowy is somewhat similar to [pillow](https://python-pillow.org/) and
[scikit-image](https://scikit-image.org/), but has a small feature set with some unique abilities.
Painting and 2D paths are outside the scope of the library. See
  [pycairo](https://pycairo.readthedocs.io/en/latest/) or
  [skia-pathops](https://github.com/fonttools/skia-pathops).

<!--

Examples
- The EXR demo image sucks
- Load the flake, create drop shadow, composite it over noise
- Islands
- to_planar and from_planar

Write a section on wrap modes

I think we need some modicum of color space handling, at least in load / save -- isn't the blurry
  poodle slightly dark?  Create a page in "test" and use chrome. (or implement generate_gallery)

travis
  should run generate as well as test_snowy

BUG:
  when saving a solid color image, I think this exception can be thrown:
    "Max value == min value, ambiguous given dtype"
  also, "Lossy conversion from float64 to uint8." warnings are annoying

open graph tags and thumbnail

TODO items after open source release

  io can have generate_gallery for making HTML, and optional forced width/height/filter
  arbitrary rotation
  height field AO
  CPCF's
  variable radius blur (radius multiplier is a fn not a constant)
    test with a distance field + gradient
  prefiltering as seen in docs/hoppe
  pixel art scaling algorithm(s)
  add fractal.py
      mandelbrot example from numba
      also this: https://en.wikipedia.org/wiki/Buddhabrot

-->
