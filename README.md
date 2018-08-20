<!--

Can inspect print the source for a func?  Make the docs literate!

Why do periods have preceding space?  Why are bullets so spaced out?

Add docstrings for reshape, unshape, dstack, dsplit that show the entire src.

add README examples of:
  adding alpha to cut out a circle
  combining R G B planes
  splitting R G B planes
    r,g,b = np.split(imgarray, 3, axis=2)
  removing white background and adding drop shadow
  terse island gen

ops
  add_border should be more efficient
  blit
  snowy dstack dsplit (join / split aliases)
  docstrings for all funcs
  add tests for color images
  hstack, vstack
  hflip, vflip
  blit
  add_border
  split, join

travis

open graph tags and thumbnail

blog entry
  Use filament to draw reflective sphere with and without HDR

post-blog entry

  add fractal.py with the mandelbrot example from numba

  variable radius blur (radius multiplier is a fn not a constant)
    use numba to help speed this
    NOTE this function is very slow; if possible, it's better to
    blur an entire image and compose it with a mask
    (maybe give an example too)

  prefiltering as seen in docs/hoppe
  pixel art scaling algorithm(s)

-->

# Snowy

This is a tiny Python 3 module for manipulating and generating images.
- Simple and flat API. See the [documentation](https://github.prideout.net/snowy/).
- Supports [resize and blur](#resize-and-blur) with a variety of filters.
- Honors the [wrap mode]() for proper boundary behavior.
- Supports simple OpenEXR images (never clamps colors).
- Images are numpy arrays with shape `[height,width,bands]`.
- Written purely in Python 3 and accelerated using [numba](https://numba.pydata.org/).
- Efficiently generates [gradient noise](#gradient-noise) and [distance fields](#distance-fields).

Snowy is somewhat similar to [pillow](https://python-pillow.org/) and
[scikit-image](https://scikit-image.org/), but has a small feature set with some unique abilities.
Painting and 2D paths are outside the scope of the library. See
  [pycairo](https://pycairo.readthedocs.io/en/latest/) or
  [skia-pathops](https://github.com/fonttools/skia-pathops).

With snowy, all images are three-dimensional numpy arrays. For example, RGB images have shape
`[height,width,3]` and grayscale images have shape `[height,width,1]`. Snowy provides some
convenient unary functions to make it easy to interop with other libraries:
- To add or remove the trailing "1" for grayscale images, use [shape](#shape) and
[unshape](#unshape).
- To switch an array in or out of planar format (color channels are in the leading dimension), use
[dsplit](#dsplit) and [dstack](#dstack).

## Installing

To install and update snowy, do this:

`pip3 install -U snowy`

## Examples

### Resize and blur

This snippet downsamples an image, blurs it, and stacks it up against the original.

```python
import snowy

source = snowy.open('source.png')
source = snowy.resize(source, 200, 200)
blurry = snowy.blur(source, radius=4.0)
snowy.save(snowy.hstack([source, blurry]), 'diptych.png')
```

<img src="https://github.com/prideout/snowy/raw/master/docs/diptych.png" height="128px">

The next snippet first magnifies an image using a nearest-neighbor filter, then using the default
Mitchell filter.

```python
parrot = snowy.load('parrot.png')
height, width = parrot.shape[:2]
nearest = snowy.resize(parrot, width * 6, filter=snowy.NEAREST) 
mitchell = snowy.resize(parrot, width * 6)
snowy.show(snowy.hstack([nearest, mitchell]))
```

<img src="https://github.com/prideout/snowy/raw/master/docs/diptych-parrot.png" height="128px">

### Crop

This loads an OpenEXR image, then crops out the top half by slicing the numpy array.

```python
sunset = snowy.load('sunset.exr')
cropped_sunset = sunset[:100,:,:]
snowy.show(cropped_sunset / 50.0) # darken the image
```

<img src="https://github.com/prideout/snowy/raw/master/docs/cropped-sunset.png" height="128px">

### Gradient Noise

Snowy's `noise` function generates a single-channel image whose values are in [-1,&nbsp;+1].

```python
n = snowy.noise(100, 100, frequency=4, seed=42, wrapx=True)
n = np.hstack([n, n])
snowy.show(0.5 + 0.5 * n)
```

<img src="https://github.com/prideout/snowy/raw/master/docs/noise.png" height="128px">

### Distance Fields

TBD
