==========
bitshuffle
==========

Filter for improving compression of typed binary data.

``bitshuffle`` is an algorithm for rearranging typed, binary data for improving
for improving compression as well as a C/python package that implements this
algorithm within the ``numpy`` framework. The library can be used along side
HDF5 to compress and decompress datasets and write them to disk, and is
integrated into HDF5 through the `dynamically loaded filters`_ framework.

Algorithmically, ``bitshuffle`` is closely related to HDF5's ``shuffle`` filter
except it operates at the bit level instead of the byte level. Arranging a
typed data array in to a matrix with the elements as the rows and the bits
within the elements as the columns, ``bitshuffle`` "transposes" the matrix,
such that all the least-significant-bits are in a row, etc.  This transpose
is performed within blocks of data of length 2048 elements long [1]_.

This does not in itself compress data, only rearranges it for more efficient
compression. To perform the actual compression you will need a compression
library.  ``bitshuffle`` has been designed to be well matched Marc Lehmann's
LZF_. Note that because ``bitshuffle`` modifies the data at the bit level,
sophisticated entropy reducing compression libraries such as GZIP and BZIP are
unlikely to achieve significantly better compression than a simpler and faster
duplicate-string-elimination algorithms such as LZF.

The ``bitshuffle`` algorithm relies on neighbouring elements of a dataset being
highly correlated to improve data compression. Any correlations that span at
least 24 elements of the dataset may be exploited to improve compression.

``bitshuffle`` was designed with performance in mind. On most machines the
time required for ``bitshuffle``+``LZF`` well below the time required to write
the compressed data to disk. Because it is able to exploit the SSE and AVX
instruction sets present on modern Intel and AMD processors, on these machines
compression is only marginally slower than an out-of-cache memory copy.

As a bonus, ``bitshuffle`` ships with a dynamically loaded version of
``h5py``'s LZF compression filter, such that the filter can be transparently
used outside of python and in command line utilities such as ``h5dump``.

.. _[1]: Chosen to be well matched to the 8kB window of the LZF compression library.

.. _`dynamically loaded filters`: http://www.hdfgroup.org/HDF5/doc/Advanced/DynamicallyLoadedFilters/HDF5DynamicallyLoadedFilters.pdf

.. _LZF: http://oldhome.schmorp.de/marc/liblzf.html


Installation
------------

Installation requires ``HDF5>=1.8.11``, ``h5py``, ``numpy`` and ``Cython``.  To
use the HDF5 filter outside of python set the environment variable
``HDF5_PLUGIN_PATH`` to the ``plugins/`` directory withing the ``bitshuffle``
installation.


