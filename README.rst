==========
Bitshuffle
==========

Filter for improving compression of typed binary data.

Bitshuffle is an algorithm for rearranging typed, binary data for improving
for improving compression, as well as a python/C package that implements this
algorithm within the Numpy framework.

The library can be used along side HDF5 to compress and decompress datasets and
is integrated through the `dynamically loaded filters`_ framework. Bitshuffle
is HDF5 filter number ``32008``.

Algorithmically, Bitshuffle is closely related to HDF5's shuffle filter
except it operates at the bit level instead of the byte level. Arranging a
typed data array in to a matrix with the elements as the rows and the bits
within the elements as the columns, Bitshuffle "transposes" the matrix,
such that all the least-significant-bits are in a row, etc.  This transpose
is performed within blocks of data of length 2048 elements long [1]_.

This does not in itself compress data, only rearranges it for more efficient
compression. To perform the actual compression you will need a compression
library.  Bitshuffle has been designed to be well matched Marc Lehmann's
LZF_. Note that because Bitshuffle modifies the data at the bit level,
sophisticated entropy reducing compression libraries such as GZIP and BZIP are
unlikely to achieve significantly better compression than a simpler and faster
duplicate-string-elimination algorithms such as LZF.

The Bitshuffle algorithm relies on neighbouring elements of a dataset being
highly correlated to improve data compression. Any correlations that span at
least 24 elements of the dataset may be exploited to improve compression.

Bitshuffle was designed with performance in mind. On most machines the
time required for Bitshuffle+LZF well below the time required to read or write
the compressed data to disk. Because it is able to exploit the SSE and AVX
instruction sets present on modern Intel and AMD processors, on these machines
compression is only marginally slower than an out-of-cache memory copy.

As a bonus, Bitshuffle ships with a dynamically loaded version of
`h5py`'s LZF compression filter, such that the filter can be transparently
used outside of python and in command line utilities such as ``h5dump``.

.. _[1]: Chosen to be well matched to the 8kB window of the LZF compression library.

.. _`dynamically loaded filters`: http://www.hdfgroup.org/HDF5/doc/Advanced/DynamicallyLoadedFilters/HDF5DynamicallyLoadedFilters.pdf

.. _LZF: http://oldhome.schmorp.de/marc/liblzf.html


Installation
------------

Installation requires HDF5 1.8 or later, HDF5 for python, Numpy and Cython.
To use the dynamically loaded HDF5 filter requires HDF5 1.8.11 or later.

To install::

    python setup.py install [--h5plugin [--h5plugin-dir=spam]]

If using the dynamically loaded HDF5 filter, set the environment variable
``HDF5_PLUGIN_PATH`` to the value of ``--h5plugin-dir`` or use HDF5's default
search location of ``/usr/local/hdf5/lib/plugin``.


Usage
-----

The `bitshuffle` module contains routines for shuffling and unshuffling
Numpy arrays.

If installed with the dynamically loaded filter plugins, Bitshuffle can be used
in conjunction with HDF5 both inside and outside of python, in the same way as
any other filter; simply by specifying the filter number ``32008``. Otherwise
the filter will be available only within python and only after importing
`bitshuffle.h5`. Reading Bitshuffle encoded datasets will be transparent.
The filter can added to new datasets either through the `h5py` low level
interface or through the convenience functions provided in
`bitshuffle.h5`. See the tests for examples.


Usage from C
------------

If you wish to use Bitshuffle in your C program and would prefer not to use the
HDF5 dynamically loaded filter, the C library in the ``src/`` directory is self
contained and complete.


