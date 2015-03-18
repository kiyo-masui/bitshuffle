from setuptools import setup, Extension
from setuptools.command.install import install as install_
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
import numpy as np
# XXX h5py needs to be present to run setup.py. Can't be installed
# automatically?
import h5py
import os
import sys
from os import path
import shutil
import glob

VERSION_MAJOR = 0
VERSION_MINOR = 2
VERSION_POINT = 1

# Only unset in the 'release' branch and in tags.
VERSION_DEV = 1

VERSION = "%d.%d.%d" % (VERSION_MAJOR, VERSION_MINOR, VERSION_POINT)
if VERSION_DEV:
    VERSION = VERSION + ".dev%d" % VERSION_DEV


COMPILE_FLAGS = ['-Ofast', '-march=native', '-std=c99', '-fopenmp']
# Cython breaks strict aliasing rules.
COMPILE_FLAGS += ["-fno-strict-aliasing"]
#COMPILE_FLAGS = ['-Ofast', '-march=core2', '-std=c99', '-fopenmp']

MACROS = [
          ('BSHUF_VERSION_MAJOR', VERSION_MAJOR),
          ('BSHUF_VERSION_MINOR', VERSION_MINOR),
          ('BSHUF_VERSION_POINT', VERSION_POINT),
          ]


H5PLUGINS_DEFAULT = '/usr/local/hdf5/lib/plugin'

# Copied from h5py.
# TODO, figure out what the canonacal way to do this should be.
INCLUDE_DIRS = []
LIBRARY_DIRS = []
if sys.platform == 'darwin':
    # putting here both macports and homebrew paths will generate
    # "ld: warning: dir not found" at the linking phase 
    INCLUDE_DIRS += ['/opt/local/include'] # macports
    LIBRARY_DIRS += ['/opt/local/lib']     # macports
    INCLUDE_DIRS += ['/usr/local/include'] # homebrew
    LIBRARY_DIRS += ['/usr/local/lib']     # homebrew
elif sys.platform.startswith('freebsd'):
    INCLUDE_DIRS += ['/usr/local/include'] # homebrew
    LIBRARY_DIRS += ['/usr/local/lib']     # homebrew


ext_bshuf = Extension("bitshuffle.ext",
                   ["bitshuffle/ext.pyx", "src/bitshuffle.c", "lz4/lz4.c"],
                   include_dirs=INCLUDE_DIRS + [np.get_include(), "src/",
                                                "lz4/"],
                   library_dirs = LIBRARY_DIRS,
                   depends=["src/bitshuffle.h", "src/iochain.h", "lz4/lz4.h"],
                   libraries = ['gomp'],
                   extra_compile_args=COMPILE_FLAGS,
                   define_macros=MACROS,
                   )


h5filter = Extension("bitshuffle.h5",
                   ["bitshuffle/h5.pyx", "src/bshuf_h5filter.c",
                    "src/bitshuffle.c", "lz4/lz4.c"],
                   include_dirs=INCLUDE_DIRS + ["src/", "lz4/"],
                   library_dirs = LIBRARY_DIRS,
                   depends=["src/bitshuffle.h", "src/iochain.h",
                            'src/bshuf_h5filter.h', "lz4/lz4.h"],
                   libraries = ['hdf5', 'gomp'],
                   extra_compile_args=COMPILE_FLAGS,
                   define_macros=MACROS,
                   )


filter_plugin = Extension("plugin.libh5bshuf",
                   ["src/bshuf_h5plugin.c", "src/bshuf_h5filter.c",
                    "src/bitshuffle.c", "lz4/lz4.c"],
                   include_dirs=INCLUDE_DIRS +["src/", "lz4/"] ,
                   library_dirs = LIBRARY_DIRS,
                   depends=["src/bitshuffle.h", "src/iochain.h",
                            'src/bshuf_h5filter.h', "lz4/lz4.h"],
                   libraries = ['hdf5', 'gomp'],
                   extra_compile_args=['-fPIC', '-g'] + COMPILE_FLAGS,
                   define_macros=MACROS,
                   )


lzf_plugin = Extension("plugin.libh5LZF",
                   ["src/lzf_h5plugin.c", "lzf/lzf_filter.c",
                    "lzf/lzf/lzf_c.c", "lzf/lzf/lzf_d.c"],
                   depends=["lzf/lzf_filter.h",
                            "lzf/lzf/lzf.h", "lzf/lzf/lzfP.h"],
                   include_dirs = INCLUDE_DIRS + ["lzf/", "lzf/lzf/"],
                   library_dirs = LIBRARY_DIRS,
                   libraries = ['hdf5'],
                   extra_compile_args=['-fPIC', '-g'] + COMPILE_FLAGS,
                   )


H5VERSION = h5py.h5.get_libversion()
if (H5VERSION[0] < 1 or (H5VERSION[0] == 1
    and (H5VERSION[1] < 8 or (H5VERSION[1] == 8 and H5VERSION[2] < 11)))):
    H51811P = False
    EXTENSIONS = [ext_bshuf, h5filter]
else:
    H51811P = True
    EXTENSIONS = [ext_bshuf, h5filter, filter_plugin, lzf_plugin]

#EXTENSIONS = cythonize(EXTENSIONS)


# Custom installation to include installing dynamic filters.
class install(install_):
    user_options = install_.user_options + [
        ('h5plugin', None, 'Install HDF5 filter plugins for use outside of python.'),
        ('h5plugin-dir=', None,
         'Where to install filter plugins. Default %s.' % H5PLUGINS_DEFAULT),
    ]
    def initialize_options(self):
        install_.initialize_options(self)
        self.h5plugin = False
        self.h5plugin_dir = H5PLUGINS_DEFAULT
    def finalize_options(self):
        install_.finalize_options(self)
        assert self.h5plugin or not self.h5plugin, "Invalid h5plugin argument."
        self.h5plugin_dir = path.abspath(self.h5plugin_dir)
    def run(self):
        install_.run(self)
        if self.h5plugin:
            if H51811P:
                pass
            else:
                print "HDF5 < 1.8.11, not installing filter plugins."
                return
            #from h5py import h5
            #h5version = h5.get_libversion()
            plugin_build = path.join(self.build_lib, "plugin")
            try:
                os.makedirs(self.h5plugin_dir)
            except OSError as e:
                if e.args[0] == 17:
                    # Directory already exists, this is fine.
                    pass
                else:
                    raise
            plugin_libs = glob.glob(path.join(plugin_build, "*"))
            for plugin_lib in plugin_libs:
                plugin_name = path.split(plugin_lib)[1]
                shutil.copy2(plugin_lib, path.join(self.h5plugin_dir, plugin_name))
            print "Installed HDF5 filter plugins to %s" % self.h5plugin_dir


# TODO hdf5 support should be an "extra". Figure out how to set this up.

setup(
    name = 'bitshuffle',
    version = VERSION,

    packages = ['bitshuffle'],
    scripts=[],
    ext_modules = EXTENSIONS,
    cmdclass = {'build_ext': build_ext, 'install': install},
    #cmdclass = {'install': install},
    #install_requires = ['numpy', 'h5py', 'Cython', 'setuptools>=0.7'],
    install_requires = ['numpy', 'h5py', 'Cython'],
    #extras_require = {'H5':  ["h5py"]},
    package_data={'': ['bitshuffle/tests/data/*']},

    # metadata for upload to PyPI
    author = "Kiyoshi Wesley Masui",
    author_email = "kiyo@physics.ubc.ca",
    description = "Bitshuffle filter for improving typed data compression.",
    license = "MIT",
    url = "http://github.com/kiyo-masui/bitshuffle"
)

