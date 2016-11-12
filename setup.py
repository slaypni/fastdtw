import os.path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import warnings

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

classifiers = [
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Topic :: Scientific/Engineering'
]


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


ext = '.pyx' if USE_CYTHON else '.cpp'

extensions = [Extension(
        'fastdtw._fastdtw',
        [os.path.join('fastdtw', '_fastdtw' + ext)],
        language="c++",
        include_dirs=[],
        libraries=["stdc++"]
    )]

if USE_CYTHON:
    extensions = cythonize(extensions)

kwargs = {
    'name': 'fastdtw',
    'version': '0.3.0',
    'author': 'Kazuaki Tanida',
    'url': 'https://github.com/slaypni/fastdtw',
    'description': 'Dynamic Time Warping (DTW) algorithm with an O(N) time and memory complexity.',
    'license': 'MIT',
    'keywords': ['dtw'],
    'install_requires': ['numpy'],
    'packages': find_packages(),
    'ext_modules':  extensions,
    'test_suite': 'tests',
    'setup_requires': ['pytest-runner'],
    'tests_require': ['pytest'],
    'classifiers': classifiers
}

try:
    setup(**kwargs)
except SystemExit:
    del kwargs['ext_modules']
    warnings.warn('compilation failed. Installing pure python package')
    setup(**kwargs)
