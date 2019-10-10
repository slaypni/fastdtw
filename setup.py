import os.path
import sys

from Cython.Build import cythonize
import numpy
from setuptools import Extension, find_packages, setup

classifiers = [
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering",
]

extensions = cythonize([Extension("fastdtw._fastdtw",
                                  [os.path.join("fastdtw", "_fastdtw.pyx")],
                                  language="c++",
                                  include_dirs=[numpy.get_include()],
                                  libraries=["stdc++"],)])

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.rst")) as f:
    long_description = f.read()

needs_pytest = set(["pytest", "test", "ptr"]).intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

kwargs = {
    "name": "fastdtw",
    "version": "0.3.4",
    "author": "Kazuaki Tanida",
    "url": "https://github.com/slaypni/fastdtw",
    "description": "Dynamic Time Warping (DTW) algorithm with an O(N) time and memory complexity.",
    "long_description": long_description,
    "license": "MIT",
    "keywords": ["dtw"],
    "install_requires": ["numpy"],
    "packages": find_packages(),
    "ext_modules": extensions,
    "test_suite": "tests",
    "setup_requires": pytest_runner,
    "tests_require": ["pytest"],
    "classifiers": classifiers,
}

setup(**kwargs)
