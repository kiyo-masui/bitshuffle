from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

import numpy as np

ext_bshuf = Extension("bitshuffle.ext",
                   ["src/bitshuffle.c", "bitshuffle/ext.pyx"],
                   libraries = ['hdf5'],
                   include_dirs=[np.get_include()],
                   extra_compile_args=['-Ofast', '-march=native',]
                   )


h5filter = Extension("bitshuffle.h5",
                   ["src/h5filter.c",],
                   libraries = ['hdf5'],
                   include_dirs=[np.get_include()],
                   extra_compile_args=['-Ofast', '-march=native',]
                   )


filter_plugin = Extension("plugins.libh5bshuf",
                   ["src/bshuf_h5plugin.c", "src/bshuf_h5filter.c",
                    "src/bitshuffle.c"],
                   #["src/bshuf_h5plugin.c"],
                   libraries = ['hdf5'],
                   include_dirs=['src/'],
                   extra_compile_args=['-fPIC', '-g', '-Ofast',
                                       '-march=native']
                   )



setup(
    name = 'bitshuffle',
    version = "0.1",

    packages = find_packages(),
    scripts=[],
    ext_modules = [ext_bshuf, filter_plugin],
    cmdclass = {'build_ext': build_ext},
    requires = ['numpy', 'h5py'],

    # metadata for upload to PyPI
    author = "Kiyoshi Wesley Masui",
    author_email = "kiyo@physics.ubc.ca",
    description = "Bit shuffle filter for typed data compression.",
    license = "GPL v2.0",
    url = "http://github.com/kiyo-masui/bitshuffle"
)
