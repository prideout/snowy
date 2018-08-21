## Snowy

This is a tiny Python 3 module for manipulating and generating images.
- Simple and flat API. See the [documentation](https://github.prideout.net/snowy/).
- Supports resize and blur with a variety of filters.
- Honors the wrap mode] for proper boundary behavior.
- Supports simple OpenEXR images (never clamps colors).
- Images are numpy arrays with shape `[height,width,bands]`.
- Written purely in Python 3 and accelerated using [numba](https://numba.pydata.org/).
- Efficiently generates gradient noise and distance fields.

Snowy is somewhat similar to [pillow](https://python-pillow.org/) and
[scikit-image](https://scikit-image.org/), but has a small feature set with some unique abilities.
Painting and 2D paths are outside the scope of the library. See
  [pycairo](https://pycairo.readthedocs.io/en/latest/) or
  [skia-pathops](https://github.com/fonttools/skia-pathops).

<!--

canva.com/color-palette

Examples
- Rotation, hflip, vflip
- Load the flake, create drop shadow, composite it over noise
- Islands
- to_planar and from_planar
- The EXR demo image sucks

I think we need some modicum of color space handling, at least in load / save -- isn't the blurry
  poodle slightly dark?  Create a page in "test" and use chrome. (or implement generate_gallery)

travis

open graph tags and thumbnail

post-blog entry

  io can have generate_gallery for making HTML, and optional forced width/height/filter
  arbitrary rotation
  height field AO
  CPCF's
  variable radius blur (radius multiplier is a fn not a constant)
    use numba to help speed this
    NOTE this function is very slow; if possible, it's better to
    blur an entire image and compose it with a mask
    (maybe give an example too)
  prefiltering as seen in docs/hoppe
  pixel art scaling algorithm(s)
  add fractal.py
      mandelbrot example from numba
      also this: https://en.wikipedia.org/wiki/Buddhabrot

-->
