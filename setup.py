from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

import numpy as np



COMPILE_FLAGS = ['-Ofast', '-march=native',]


ext_bshuf = Extension("bitshuffle.ext",
                   ["src/bitshuffle.c", "bitshuffle/ext.pyx"],
                   include_dirs=[np.get_include()],
                   depends=["src/bitshuffle.h"],
                   extra_compile_args=COMPILE_FLAGS,
                   )


h5filter = Extension("bitshuffle.h5",
                   ["bitshuffle/h5.pyx", "src/bshuf_h5filter.c",
                    "src/bitshuffle.c"],
                   depends=["src/bitshuffle.h", 'src/bshuf_h5filter.h'],
                   libraries = ['hdf5'],
                   extra_compile_args=COMPILE_FLAGS,
                   )


filter_plugin = Extension("plugins.libh5bshuf",
                   ["src/bshuf_h5plugin.c", "src/bshuf_h5filter.c",
                    "src/bitshuffle.c"],
                   depends=["src/bitshuffle.h", 'src/bshuf_h5filter.h'],
                   libraries = ['hdf5'],
                   extra_compile_args=['-fPIC', '-g'] + COMPILE_FLAGS,
                   )


lzf_plugin = Extension("plugins.libh5LZF",
                   ["src/lzf_h5plugin.c", "lzf/lzf_filter.c",
                    "lzf/lzf/lzf_c.c", "lzf/lzf/lzf_d.c"],
                   depends=["src/bitshuffle.h", "lzf/lzf_filter.h",
                            "lzf/lzf/lzf.h", "lzf/lzf/lzfP.h"],
                   include_dirs = ["lzf/", "lzf/lzf/"],
                   libraries = ['hdf5'],
                   extra_compile_args=['-fPIC', '-g'] + COMPILE_FLAGS,
                   )


# TODO hdf5 support should be an "extra". Figure out how to set this up.

err = setup(
    name = 'bitshuffle',
    version = "0.1",

    packages = find_packages(),
    scripts=[],
    ext_modules = [ext_bshuf, h5filter, filter_plugin, lzf_plugin],
    cmdclass = {'build_ext': build_ext},
    requires = ['numpy', 'h5py'],
    #extras_require = {'H5':  ["h5py"]},

    # metadata for upload to PyPI
    author = "Kiyoshi Wesley Masui",
    author_email = "kiyo@physics.ubc.ca",
    description = "Bit shuffle filter for typed data compression.",
    license = "GPL v2.0",
    url = "http://github.com/kiyo-masui/bitshuffle"
)


# TODO: after success installation, print message suggesting setting
# "HDF5_PLUGIN_PATH"

#try:
#    import bitshuffle
#except:
#    pass

