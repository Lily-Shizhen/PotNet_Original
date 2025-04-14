import os
import sys
import os
import sys
#!/usr/bin/env python
"""Setup script for the theta package
"""
import os
import sys
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy


# Get conda prefix path
conda_prefix = os.environ.get('CONDA_PREFIX')

# Include paths
include_dirs = [
    numpy.get_include(),
    os.path.join(conda_prefix, 'include')
]

# Library paths
library_dirs = [
    os.path.join(conda_prefix, 'lib')
]

extensions = [
    Extension('series',
              sources=['series.pyx',
                       'bessel.c'],
              include_dirs=[numpy.get_include(), 'gsl/include'],
              library_dirs=['gsl/lib'],
              libraries=['gsl', 'gslcblas'],
              extra_compile_args=['-std=c99', '-I./gsl/include'],
              )
]

setup(
    name='functions',
    author='kruskallin',
    author_email='kruskallin@tamu.edu',
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions),
    install_requires=[
        "numpy >= 1.13",
    ],
    zip_safe=False,
)
