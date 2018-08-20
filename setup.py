import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snowy",
    version="0.0.1",
    author="Philip Rideout",
    description="Small Image Library for Python 3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    install_requires=[
        'imageio>=2.3',
        'numpy>=1.14',
        'numba>=0.39',
    ],
    url="https://github.com/prideout/snowy",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
