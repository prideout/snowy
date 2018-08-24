# Snowy <img src="snowy2.png" height="64px" style="vertical-align:text-bottom;margin-left:10px">

This is a tiny module for manipulating and generating images, written purely in Python 3 and
accelerated with [numba](https://numba.pydata.org/). It has a small, flat API with some interesting
features, like the ability to specify boundary behavior during filtering.

With snowy, all images are three-dimensional numpy arrays. For example, RGB images have shape
`[height,width,3]` and grayscale images have shape `[height,width,1]`. Snowy provides some
utility functions that make it easy to work with other modules (see [interop](#interop)).

## Installing

To install and update snowy, do this:

`pip3 install -U snowy`

## Examples

Not interested in example code? Skip to the [quick reference](#quick_reference).

### Resize and blur

This snippet does a [resize](#resize), then a [blur](#blur), then horizontally concatenates the two
images.

```python
import snowy

source = snowy.open('poodle.png')
source = snowy.resize(source, height=200)
blurry = snowy.blur(source, radius=4.0)
snowy.save(snowy.hstack([source, blurry]), 'diptych.png')
```

<img src="diptych.png" height="150px">

The next snippet first magnifies an image using a nearest-neighbor filter, then using the default
Mitchell filter.

```python
parrot = snowy.load('parrot.png')
height, width = parrot.shape[:2]
nearest = snowy.resize(parrot, width * 6, filter=snowy.NEAREST) 
mitchell = snowy.resize(parrot, width * 6)
snowy.show(snowy.hstack([nearest, mitchell]))
```

<img src="diptych-parrot.png" height="128px">

### Rotate and flip

```python
gibbons = snowy.load('gibbons.jpg')
rotated = snowy.rotate(gibbons, 180)
flipped = snowy.vflip(gibbons)
triptych = snowy.hstack([gibbons, rotated, flipped],
    border_width=4, border_value=[128,0,0])
```

<img src="xforms.jpg" height="150px">

### Cropping and composing

If you need to crop an image, just use Python slicing.

For example, this loads an OpenEXR image, then crops out the top half by slicing the numpy array.

```python
sunset = snowy.load('sunset.exr')
cropped_sunset = sunset[:100,:,:]
snowy.show(cropped_sunset / 50.0) # darken the image
```

<aside class="notice">
By the way, if you're interested in tone mapping and other HDR operations, be sure to check
out the [hydra](https://github.com/tatsy/hydra) module. And, if you wish to simply load / store
raw double-precision data, consider using `numpy.save(filename, array)` and `numpy.load(filename)`
for `npy` files.
</aside>

<img src="cropped-sunset.png" height="100px">

Next we'll draw an icon over the sunset image using [#compose](compose):

```python
icon = snowy.load('snowflake.png')
icon = snow.resize(snowflake, 128, 128)
snowy.show(icon)

sunset = snowy.compose(sunset[:128,:128], icon)
snowy.show(sunset)
```

<img src="snowflake.png" height="128px">
<img src="composed.png" height="128px">

We can also create a drop shadow for our icon:

```python
# TBD
translate()
```

### Gradient noise

Snowy's [generate_noise](#generate_noise) function generates a single-channel image whose values are
in [-1,&nbsp;+1].

```python
n = snowy.generate_noise(100, 100, frequency=4, seed=42, wrapx=True)
n = np.hstack([n, n])
snowy.show(0.5 + 0.5 * n)
```

<img src="noise.png" height="128px">

### Distance fields

This uses [generate_sdf](#generate_sdf) to create a signed distance field from a monochrome
picture of two circles enclosed by a square. Note the usage of [unitize](#unitize) to adjust the
values into the `[0,1]` range.

```python
circles = snowy.load('circles.png')
sdf = snowy.unitize(snowy.generate_sdf(circles != 0.0))
snowy.show(snowy.hstack([circles, sdf]))
```

<img src="sdf.png" height="128px">

### Image generation

TBD

<img src="gradient.png" width="150px">

<img src="island.png" height="150px">

## Wrap modes

TBD

## Interop

Snowy's algorithms require images to be row-major three-dimensional `float64` numpy arrays, with
color channels living in the trailing dimension. If you're working with another module that does not
follow this convention, consider using one of the following interop functions.

- To add or remove the trailing 1 from the shape of grayscale images, use [reshape](#reshape) and
[unshape](#unshape).
- To swap color channels in or out of the leading dimension, use [to_planar](#to_planar) and
[from_planar](#from_planar).
- To cast between `float64` and other types, just use numpy. For example,
`np.uint8(myimg * 255)` or `np.float64(myimg) / 255`.
- To swap rows with columns, use numpy's
[swapaxes function](https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html).

## Quick Reference

<table>
$quickref$
</table>

## Reference

This section includes the source code of each entry point. This allows
you to clearly see the default parameters and expectations of each function.
