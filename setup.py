from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

import numpy as np

ext_bt = Extension("bitshuffle.ext",
                   ["src/bitshuffle.c", "bitshuffle/ext.pyx"],
                   libraries = [],
                   include_dirs=[np.get_include()],
                   # '-Wa,-q' is max specific and only there because
                   # soemthing is wrong with my gcc. It switches to the
                   # clang assembler.
                   extra_compile_args=['-Ofast', '-march=native',
                   '-Wa,-q'],
                   #extra_compile_args=['-fopenmp', '-march=native'],
                   )


setup(
    name = 'bitshuffle',
    version = "0.1",

    packages = find_packages(),
    scripts=[],
    ext_modules = [ext_bt],
    cmdclass = {'build_ext': build_ext},
    requires = ['numpy', 'h5py'],

    # metadata for upload to PyPI
    author = "Kiyoshi Wesley Masui",
    author_email = "kiyo@physics.ubc.ca",
    description = "Bit shuffle filter for typed data compression.",
    license = "GPL v2.0",
    url = "http://github.com/kiyo-masui/bitshuffle"
)
