import h5py
import numpy

cimport cython


cdef extern int bshuf_register_h5filter()


def register_h5_filter():
    ret = bshuf_register_h5filter()
    if ret < 0:
        raise RuntimeError("Failed to register bitshuffle HDF5 filter.", ret)


register_h5_filter()
