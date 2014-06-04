import numpy as np

cimport numpy as np
cimport cython


np.import_array()


# Prototypes from bittranspose.c
cdef extern int no_transpose_copy(void *A, void *B, int size, int elem_size)
cdef extern int byte_transpose_simple(void *A, void *B, int size, int elem_size)


def _setup_arr(arr):
    shape = arr.shape
    if not arr.flags['C_CONTIGUOUS']:
        msg = "Input array must be C-contiguouse."
        raise ValueError(msg)
    size = arr.size
    dtype = arr.dtype
    itemsize = dtype.itemsize
    out = np.empty(size * itemsize, dtype=np.uint8)
    return out, size, itemsize


def memory_copy(np.ndarray arr not None):
    """Just copies the data.

    For testing and profiling purposes.

    """

    cdef np.ndarray out
    out, size, itemsize = _setup_arr(arr)
    cdef void* arr_ptr = <void*> arr.data
    cdef void* out_ptr = <void*> out.data
    err = no_transpose_copy(arr_ptr, out_ptr, size, itemsize)
    if err:
        msg = "Failed. Error code %d."
        raise ValueError(msg % err)
    return out


def byte_simple(np.ndarray arr not None):
    """Transpose bytes within words but not bits.

    """

    cdef np.ndarray out
    out, size, itemsize = _setup_arr(arr)
    cdef void* arr_ptr = <void*> arr.data
    cdef void* out_ptr = <void*> out.data
    err = byte_transpose_simple(arr_ptr, out_ptr, size, itemsize)
    if err:
        msg = "Failed. Error code %d."
        raise ValueError(msg % err)
    return out



