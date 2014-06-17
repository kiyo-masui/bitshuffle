from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

import numpy as np

ext_bshuf = Extension("bitshuffle.ext",
                   ["src/bitshuffle.c", "bitshuffle/ext.pyx"],
                   libraries = ['hdf5'],
                   include_dirs=[np.get_include()],
                   # '-Wa,-q' is Mac specific and only there because
                   # soemthing is wrong with my gcc. It switches to the
                   # clang assembler.
                   extra_compile_args=['-Ofast', '-march=native',
                   '-Wa,-q'],
                   #extra_compile_args=['-fopenmp', '-march=native'],
                   )


bshuf_h5 = Extension("bitshuffle.h5",
                   ["src/h5filter.c",],
                   libraries = ['hdf5'],
                   include_dirs=[np.get_include()],
                   # '-Wa,-q' is Mac specific and only there because
                   # soemthing is wrong with my gcc. It switches to the
                   # clang assembler.
                   extra_compile_args=['-Ofast', '-march=native',
                   '-Wa,-q'],
                   #extra_compile_args=['-fopenmp', '-march=native'],
                   )


#filter_plugin = Extension("plugins.libh5filters.so",
filter_plugin = Extension("src.libh5filters",
                   ["src/h5filter.c", ],
                   libraries = ['hdf5'],
                   include_dirs=['src/'],
                   # '-Wa,-q' is Mac specific and only there because
                   # soemthing is wrong with my gcc. It switches to the
                   # clang assembler.
                   #extra_compile_args=['-Ofast', '-march=native',
                   #'-Wa,-q'],
                   #extra_compile_args=['-fopenmp', '-march=native'],
                   extra_compile_args=['-fPIC', '-g', '-Ofast', '-march=native',
                                       '-Wa,-q'],
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
