from setuptools import setup, find_packages

classifiers = [
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Topic :: Scientific/Engineering'
]

setup(
    name='fastdtw',
    version='0.2.2',
    author='Kazuaki Tanida',
    url='https://github.com/slaypni/fastdtw',
    description='Dynamic Time Warping (DTW) algorithm with an O(N) time and memory complexity.',
    license='MIT',
    keywords=['dtw'],
    py_modules=['fastdtw'],
    install_requires=['six'],
    classifiers=classifiers
)
