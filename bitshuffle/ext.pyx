import numpy as np

cimport numpy as np
cimport cython


np.import_array()


# Prototypes from bittranspose.c
cdef extern int shuff_just_copy(void *A, void *B, int size, int elem_size)
cdef extern int shuff_byte_T_elem_simple(void *A, void *B, int size, int elem_size)
cdef extern int shuff_byte_T_elem_fast(void *A, void *B, int size, int elem_size)
cdef extern int shuff_bit_T_byte(void *A, void *B, int size, int elem_size)
cdef extern int shuff_bit_T_byte_avx(void *A, void *B, int size, int elem_size)
cdef extern int shuff_bit_T_byte_avx1(void *A, void *B, int size, int elem_size)


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


def just_copy(np.ndarray arr not None):
    """Copies the data.

    For testing and profiling purposes.

    """

    cdef np.ndarray out
    out, size, itemsize = _setup_arr(arr)
    cdef void* arr_ptr = <void*> arr.data
    cdef void* out_ptr = <void*> out.data
    err = shuff_just_copy(arr_ptr, out_ptr, size, itemsize)
    if err:
        msg = "Failed. Error code %d."
        raise ValueError(msg % err)
    return out


def byte_T_elem_simple(np.ndarray arr not None):
    """Transpose bytes within words but not bits.

    """

    cdef np.ndarray out
    out, size, itemsize = _setup_arr(arr)
    cdef void* arr_ptr = <void*> arr.data
    cdef void* out_ptr = <void*> out.data
    err = shuff_byte_T_elem_simple(arr_ptr, out_ptr, size, itemsize)
    if err:
        msg = "Failed. Error code %d."
        raise ValueError(msg % err)
    return out


def byte_T_elem_fast(np.ndarray arr not None):
    """Transpose bytes within words but not bits.

    """

    cdef np.ndarray out
    out, size, itemsize = _setup_arr(arr)
    cdef void* arr_ptr = <void*> arr.data
    cdef void* out_ptr = <void*> out.data
    err = shuff_byte_T_elem_fast(arr_ptr, out_ptr, size, itemsize)
    if err:
        msg = "Failed. Error code %d."
        raise ValueError(msg % err)
    return out



def bit_T_byte(np.ndarray arr not None):
    """Transpose bits within each byte of an array.

    """

    cdef np.ndarray out
    out, size, itemsize = _setup_arr(arr)
    cdef void* arr_ptr = <void*> arr.data
    cdef void* out_ptr = <void*> out.data
    err = shuff_bit_T_byte(arr_ptr, out_ptr, size, itemsize)
    if err:
        msg = "Failed. Error code %d."
        raise ValueError(msg % err)
    return out


def bit_T_byte_avx(np.ndarray arr not None):
    """Transpose bits within each byte of an array.

    """

    cdef np.ndarray out
    out, size, itemsize = _setup_arr(arr)
    cdef void* arr_ptr = <void*> arr.data
    cdef void* out_ptr = <void*> out.data
    err = shuff_bit_T_byte_avx(arr_ptr, out_ptr, size, itemsize)
    if err:
        msg = "Failed. Error code %d."
        raise ValueError(msg % err)
    return out

def bit_T_byte_avx1(np.ndarray arr not None):
    """Transpose bits within each byte of an array.

    """

    cdef np.ndarray out
    out, size, itemsize = _setup_arr(arr)
    cdef void* arr_ptr = <void*> arr.data
    cdef void* out_ptr = <void*> out.data
    err = shuff_bit_T_byte_avx1(arr_ptr, out_ptr, size, itemsize)
    if err:
        msg = "Failed. Error code %d."
        raise ValueError(msg % err)
    return out



