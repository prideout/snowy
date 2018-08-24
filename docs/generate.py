#!/usr/bin/env python3

import os
import sys
import subprocess
sys.path.append('../snowy')

result = subprocess.run('git rev-parse HEAD'.split(), stdout=subprocess.PIPE)
sha = result.stdout.strip().decode("utf-8")[:7]
version = f'<small>v0.0.1 ~ {sha}</small>'

def qualify(filename: str):
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(scriptdir, filename)

import snowy
import numpy as np
import CommonMark

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import pygments.styles

header = '''
<!DOCTYPE html>
<head>
<title>Snowy</title>
<link rel="icon" href="snowflake64.png" type="image/x-icon">
<meta name=viewport content='width=device-width,initial-scale=1'>
<meta charset='utf-8'>
<link href="https://fonts.googleapis.com/css?family=Alegreya"
    rel="stylesheet">
<link href="https://fonts.googleapis.com/css?family=Inconsolata"
    rel="stylesheet">
<style>
body {
    margin: 0;
    font-size: 15px;
    font-family: "Avenir Next", "HelveticaNeue", "Helvetica Neue",
        Helvetica, Arial, "Lucida Grande", sans-serif;
    text-rendering: optimizeLegibility;
    font-weight: 400;
    -webkit-font-smoothing:auto;
    background-color: #e2e2e2;
}
small {
    color: #a0a0a0;
    margin-top: 26px;
}
td:first-child {
    padding-right: 15px;
}
p:first-child {
    clear: left;
}
h1 {
    margin-top: 0;
    margin-bottom: 0;
    font-family: 'Alegreya', serif;
    font-size: 45px;
}
main {
    overflow: auto;
    margin: 0 auto;
    padding: 0 80px 20px 80px;
    max-width: 800px;
    background-color: #ffffff;
    position: relative;
    color: #404040;
    border-left: solid 2px black;
    border-right: solid 2px black;
}
@media (max-width: 960px){
    body{ background-color: #f2f2f2; }
    main{ padding: 0 20px 100px 20px; }
}
pre {
    padding: 10px;
    background-color: #f8f8f8;
    white-space: pre-wrap;
    font-family: 'Inconsolata', monospace;
}
code {
    font-family: 'Inconsolata', monospace;
}
a {
    text-decoration: none;
    color: #2962ad;
}
p.aside {
    background: white;
    font-size: small;
    border-left: solid 5px gray;
    padding: 10px;
}
h2 a, h3 a, h4 a { color: black }
h2 a:hover, h3 a:hover, h4 a:hover { color: #19529d }
</style>
'''

forkme = '''
<!-- GITHUB FORK ME LOGO -->
<a href="https://github.com/prideout/snowy"
class="github-corner" aria-label="View source on Github">
<svg width="80" height="80" viewBox="0 0 250 250"
style="color:#fff; position: absolute; top: 0; border: 0;
right: 0;" aria-hidden="true"> <path d="M0,0 L115,115 L130,115 L142,142
L250,250 L250,0 Z"></path> <path d="M128.3,109.0 C113.8,99.7 119.0,89.6
119.0,89.6 C122.0,82.7 120.5,78.6 120.5,78.6 C119.2,72.0 123.4,76.3
123.4,76.3 C127.3,80.9 125.5,87.3 125.5,87.3 C122.9,97.6 130.6,101.9
134.4,103.2" fill="currentColor" style="transform-origin: 130px 106px;"
class="octo-arm"></path><path d="M115.0,115.0 C114.9,115.1 118.7,116.5
119.8,115.4 L133.7,101.6 C136.9,99.2 139.9,98.4 142.2,98.6 C133.8,88.0
127.5,74.4 143.8,58.0 C148.5,53.4 154.0,51.2 159.7,51.0 C160.3,49.4
163.2,43.6 171.4,40.1 C171.4,40.1 176.1,42.5 178.8,56.2 C183.1,58.6
187.2,61.8 190.9,65.4 C194.5,69.0 197.7,73.2 200.1,77.6 C213.8,80.2
216.3,84.9 216.3,84.9 C212.7,93.1 206.9,96.0 205.4,96.6 C205.1,102.4
203.0,107.8 198.3,112.5 C181.9,128.9 168.3,122.5 157.7,114.1
C157.9,116.9 156.7,120.9 152.7,124.9 L141.0,136.5 C139.8,137.7
141.6,141.9 141.8,141.8 Z" fill="currentColor"
class="octo-body"></path></svg></a>
<style>.github-corner:hover svg { fill: #19529d }</style>

'''

markdown = open(qualify('intro.md')).read()
htmldoc = CommonMark.commonmark(markdown)

# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
from bs4 import BeautifulSoup, Comment
soup = BeautifulSoup(htmldoc, 'html.parser')
comments = soup.find_all(string=lambda text:isinstance(text,Comment))
for comment in comments:
    comment.extract()

# All h4 sections are actually asides.
admonitions = soup.findAll("h4")
for admonition in admonitions:
    p = admonition.find_next_sibling("p")
    p['class'] = 'aside'
    admonition.extract()

formatter = HtmlFormatter(style='tango')
snippets = soup.findAll("code", {"class": "language-python"})
for snippet in snippets:
    code = snippet.contents[0]
    highlighted = highlight(code, PythonLexer(), formatter)
    newcode = BeautifulSoup(highlighted, 'html.parser')
    snippet.parent.replace_with(newcode)

htmlfile = open(qualify('index.html'), 'w')
htmlfile.write(header)

# for s in pygments.styles.get_all_styles(): print(s)
htmlfile.write('<style>')
htmlfile.write(formatter.get_style_defs('.highlight'))
htmlfile.write('''
.highlight .mb, .highlight .mf, .highlight .mh, .highlight .mi,
.highlight .mo { color: #0063cf; }
''')
htmlfile.write('</style>')
htmlfile.write('<main>\n')
htmlfile.write(forkme)
htmlfile.write(str(soup))

import inspect
quickref = ''
for member in inspect.getmembers(snowy):
    name, value = member
    if name.startswith('__'):
        continue
    if not inspect.isfunction(value):
        continue
    doc = inspect.getdoc(value)
    src = inspect.getsource(value)

    dsbegin = src.find(r'"""')
    dsend = src.rfind(r'"""') + 4
    dsbegin = src[:dsbegin].rfind('\n') + 1
    src = src[:dsbegin] + src[dsend:]
    nlines = len(src.split('\n'))

    highlighted_src = highlight(src, PythonLexer(), formatter)
    if doc:
        doclines = doc.split('\n')
        quickref += '<tr>\n'
        quickref += f'<td><a href="#{name}">{name}</a></td>\n'
        quickref += f'<td>{doclines[0]}</td>\n'
        quickref += '<tr>\n'
        htmlfile.write(f'<h3>{name}</h3>\n<p>\n')
        htmlfile.write(' '.join(doclines))
        htmlfile.write('\n</p>\n')
        htmlfile.write(highlighted_src)

htmlfile.write('''
<a href="https://github.prideout.net/">
<img src="https://github.prideout.net/assets/PublishedLogo.svg"
width="175px">
</a>
</main>
''')

htmlfile.close()

# Post process HTML (add anchors)
htmldoc = open(qualify('index.html')).read()
htmldoc = htmldoc.replace('$quickref$', quickref)
htmldoc = htmldoc.replace('<h1>', version + '\n<h1>')
soup = BeautifulSoup(htmldoc, 'html.parser')
for tag in 'h2 h3 h4'.split():
    headings = soup.find_all(tag)
    for heading in headings:
        content = heading.contents[0].strip()
        id = content.replace(' ', '_').lower()
        heading["id"] = id
        anchor = soup.new_tag('a', href='#' + id)
        anchor.string = content
        heading.contents[0].replace_with(anchor)
open(qualify('index.html'), 'w').write(str(soup))

# Test rotations and flips

gibbons = snowy.load(qualify('gibbons.jpg'))
gibbons = snowy.resize(gibbons, width=gibbons.shape[1] // 5)
gibbons90 = snowy.rotate(gibbons, 90)
gibbons180 = snowy.rotate(gibbons, 180)
gibbons270 = snowy.rotate(gibbons, 270)
hflipped = snowy.hflip(gibbons)
vflipped = snowy.vflip(gibbons)
snowy.save(snowy.hstack([gibbons, gibbons180, vflipped],
    border_width=4, border_value=[128,0,0]), qualify("xforms.png"))

# Test noise generation

n = snowy.generate_noise(100, 100, frequency=4, seed=42, wrapx=True)
n = np.hstack([n, n])
n = 0.5 + 0.5 * n
snowy.show(n)
snowy.save(n, qualify('noise.png'))

# First try minifying grayscale

gibbons = snowy.load(qualify('snowy.jpg'))
gibbons = np.swapaxes(gibbons, 0, 2)
gibbons = np.swapaxes(gibbons[0], 0, 1)
gibbons = snowy.reshape(gibbons)
source = snowy.resize(gibbons, height=200)
blurry = snowy.blur(source, radius=4.0)
diptych_filename = qualify('diptych.png')
snowy.save(snowy.hstack([source, blurry]), diptych_filename)
os.system('optipng ' + diptych_filename)
snowy.show(diptych_filename)

# Next try color

gibbons = snowy.load(qualify('snowy.jpg'))
source = snowy.resize(gibbons, height=200)
blurry = snowy.blur(source, radius=4.0)
diptych_filename = qualify('diptych.png')
snowy.save(snowy.hstack([source, blurry]), diptych_filename)
os.system('optipng ' + diptych_filename)
snowy.show(diptych_filename)

# Moving on to magnification...

parrot = snowy.load(qualify('parrot.png'))
scale = 6
nearest = snowy.resize(parrot, width=32*scale, filter=snowy.NEAREST)
mitchell = snowy.resize(parrot, height=26*scale)
diptych_filename = qualify('diptych-parrot.png')
snowy.save(snowy.hstack([nearest, mitchell]), diptych_filename)
os.system('optipng ' + diptych_filename)
snowy.show(diptych_filename)

# EXR cropping

sunset = snowy.load(qualify('small.exr'))
print(sunset.shape)
cropped_sunset = sunset[:100,:,:]
cropped_filename = qualify('cropped-sunset.png')
snowy.save(cropped_sunset / 50.0, cropped_filename)
os.system('optipng ' + cropped_filename)
snowy.show(cropped_filename)

quit()

STEPPED_PALETTE = [
    000, 0x203060 ,
    64,  0x2C316F ,
    125, 0x2C316F ,
    125, 0x46769D ,
    126, 0x46769D ,
    127, 0x324060 ,
    131, 0x324060 ,
    132, 0x9C907D ,
    137, 0x9C907D ,
    137, 0x719457 ,
    170, 0x719457 , # Light green
    170, 0x50735A ,
    180, 0x50735A ,
    180, 0x9FA881 ,
    200, 0x9FA881 ,
    250, 0xFFFFFF ,
    255, 0xFFFFFF
    ]

SMOOTH_PALETTE = [
    000, 0x203060 , # Dark Blue
    126, 0x2C316F , # Light Blue
    127, 0xE0F0A0 , # Yellow
    128, 0x719457 , # Dark Green
    200, 0xFFFFFF , # White
    255, 0xFFFFFF ] # White

from scipy import interpolate

def applyColorGradient(elevation_image, gradient_image):
    xvals = np.arange(256)
    yvals = gradient_image[0]
    apply_lut = interpolate.interp1d(xvals, yvals, axis=0)
    return apply_lut(snowy.unshape(np.clip(elevation_image, 0, 255)))

def create_falloff(w, h, radius=0.4, cx=0.5, cy=0.5):
    hw, hh = 0.5 / w, 0.5 / h
    dp = max(hw, hh)
    x = np.linspace(hw, 1 - hw, w)
    y = np.linspace(hh, 1 - hh, h)
    u, v = np.meshgrid(x, y, sparse=True)
    d2 = (u-cx)**2 + (v-cy)**2
    return 1-snowy.unitize(snowy.reshape(d2))

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def create_circle(w, h, radius=0.4, cx=0.5, cy=0.5):
    hw, hh = 0.5 / w, 0.5 / h
    dp = max(hw, hh)
    x = np.linspace(hw, 1 - hw, w)
    y = np.linspace(hh, 1 - hh, h)
    u, v = np.meshgrid(x, y, sparse=True)
    d2, r2 = (u-cx)**2 + (v-cy)**2, radius**2
    result = 1 - smoothstep(radius-dp, radius+dp, np.sqrt(d2))
    return snowy.reshape(result)

c0 = create_circle(200, 200, 0.3)
c1 = create_circle(200, 200, 0.08, 0.8, 0.8)
c0 = np.clip(c0 + c1, 0, 1)
circles = snowy.add_border(c0, value=1)
sdf = snowy.unitize(snowy.generate_sdf(circles != 0.0))
stack = snowy.hstack([circles, sdf])
snowy.save(stack, qualify('sdf.png'))
snowy.show(stack)

def create_island(seed, gradient, freq=3):
    w, h = 750, 512
    falloff = create_falloff(w, h)
    n1 = 1.000 * snowy.generate_noise(w, h, freq*1, seed+0)
    n2 = 0.500 * snowy.generate_noise(w, h, freq*2, seed+1)
    n3 = 0.250 * snowy.generate_noise(w, h, freq*4, seed+2)
    n4 = 0.125 * snowy.generate_noise(w, h, freq*8, seed+3)
    elevation = falloff * (falloff / 2 + n1 + n2 + n3 + n4)
    mask = elevation < 0.4
    elevation = snowy.generate_sdf(mask) - 100 * n4
    mask = np.where(elevation < 0, 1, 0)
    el = 128 + 127 * elevation / np.amax(elevation)
    return applyColorGradient(el, gradient)

def createColorGradient(pal):
    inds = pal[0::2]
    cols = np.array(pal[1::2])
    red, grn, blu = cols >> 16, cols >> 8, cols
    cols = [c & 0xff for c in [red, grn, blu]]
    cols = [interpolate.interp1d(inds, c) for c in cols]
    img = np.arange(0, 255)
    img = np.dstack([fn(img) for fn in cols])
    return snowy.resize(img, 256, 32)

gradient = createColorGradient(STEPPED_PALETTE)
snowy.save(snowy.add_border(gradient), qualify('gradient.png'))
snowy.show(gradient)

isles = []
for i in range(6):
    isle = create_island(i * 5, gradient)
    isles.append(isle)
snowy.save(isles[2], qualify('island.png'))
isles = snowy.hstack(isles)
snowy.show(isles)

# flake = snowy.load(qualify('snowflake64.png'))
# snowy.show(flake)
