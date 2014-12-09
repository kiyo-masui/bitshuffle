"""
Filter for improving compression of typed binary data.

Functions
=========

    using_SSE2
    using_AVX2
    bitshuffle
    bitunshuffle
    compress_lz4
    decompress_lz4

"""

from ext import (__version__, bitshuffle, bitunshuffle, using_SSE2, using_AVX2,
                 compress_lz4, decompress_lz4)
