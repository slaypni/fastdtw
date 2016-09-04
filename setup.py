from setuptools import setup, find_packages, Extension
import os.path
import warnings

classifiers = [
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Topic :: Scientific/Engineering'
]

extensions = [Extension(
        'fastdtw._fastdtw',
        [os.path.join('fastdtw', "_fastdtw.pyx")],
        language="c++",
        include_dirs=[],
        libraries=["stdc++"]
    )]

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
