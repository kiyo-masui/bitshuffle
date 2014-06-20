==========
Bitshuffle
==========

Filter for improving compression of typed binary data.

Bitshuffle is an algorithm for rearranging typed, binary data for improving
for improving compression as well as a C/python package that implements this
algorithm within the Numpy framework. The library can be used along side
HDF5 to compress and decompress datasets and write them to disk, and is
integrated into HDF5 through the `dynamically loaded filters`_ framework.
Bitshuffle is HDF5 filter number ``32008``.

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
To us the HDF5 filter outside of python requires HDF5 1.8.11 or later and
setting the environment variable ``HDF5_PLUGIN_PATH`` to the ``plugins/``
directory within the Bitshuffle installation.


Usage
-----

If the ``HDF5_PLUGIN_PATH`` environment variable is set as described above, the
Bitshuffle filter should be accessible both inside and outside of python. The
filter can be used in the normal way by specifying filter number ``32008``.  If
the variable is not set, the filter will be available only within python and
only after importing :mod:`bitshuffle.h5`. Reading Bitshuffle encoded datasets
will be transparent The filter can added to new datasets either through
the :mod:`h5py` low level interface or through the convenience functions
provided in :mod:`bitshuffle.h5`. See the tests for examples.


