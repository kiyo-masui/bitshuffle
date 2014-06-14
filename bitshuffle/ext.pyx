import numpy as np

cimport numpy as np
cimport cython


np.import_array()


# Repeat each calcualtion this many times. For timeing.
cdef int REPEATC = 64
REPEAT = REPEATC


# Prototypes from bitshuffle.c
cdef extern int bshuf_copy(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_byte_elem_scal(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_byte_elem(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_byte_elem_SSE(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_bit_byte(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_bit_byte_unrolled(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_bit_byte_SSE(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_bit_byte_AVX(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_bit_byte_AVX_unrolled(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_bitrow_eight(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_bit_elem_AVX(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_bit_elem_scal(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_byte_bitrow_SSE(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_byte_bitrow(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_shuffle_bit_eightelem_SSE(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_untrans_bit_elem_SSE(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_untrans_bit_elem_scal(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_trans_bit_elem(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_untrans_bit_elem(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_bitshuffle(void *A, void *B, int size, int elem_size)
cdef extern int bshuf_bitunshuffle(void *A, void *B, int size, int elem_size)


ctypedef int (*Cfptr) (void *A, void *B, int size, int elem_size)


def _setup_arr(arr):
    shape = arr.shape
    if not arr.flags['C_CONTIGUOUS']:
        msg = "Input array must be C-contiguouse."
        raise ValueError(msg)
    size = arr.size
    dtype = arr.dtype
    itemsize = dtype.itemsize
    # XXX For dev only.  Use empty!.
    out = np.zeros(size, dtype=dtype)
    return out, size, itemsize


cdef _wrap_C_fun(Cfptr fun, np.ndarray arr):
    """Wrap a C function with standard call signature."""

    cdef int ii, size, itemsize, err
    cdef np.ndarray out
    out, size, itemsize = _setup_arr(arr)
    cdef void* arr_ptr = <void*> arr.data
    cdef void* out_ptr = <void*> out.data
    for ii in range(REPEATC):
        err = fun(arr_ptr, out_ptr, size, itemsize)
    if err:
        msg = "Failed. Error code %d."
        raise ValueError(msg % err)
    return out


def copy(np.ndarray arr not None):
    """Copies the data.

    For testing and profiling purposes.

    """
    return _wrap_C_fun(&bshuf_copy, arr)


def trans_byte_elem_scal(np.ndarray arr not None):
    """Transpose bytes within words but not bits.

    """
    return _wrap_C_fun(&bshuf_trans_byte_elem_scal, arr)


def trans_byte_elem_SSE(np.ndarray arr not None):
    """Transpose bytes within array elements.

    """
    return _wrap_C_fun(&bshuf_trans_byte_elem_SSE, arr)


def trans_bit_byte(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_trans_bit_byte, arr)


def trans_bit_byte_unrolled(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_trans_bit_byte_unrolled, arr)


def trans_bit_byte_SSE(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_trans_bit_byte_SSE, arr)


def trans_bit_byte_AVX(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_trans_bit_byte_AVX, arr)


def trans_bit_byte_AVX_unrolled(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_trans_bit_byte_AVX_unrolled, arr)


def trans_bitrow_eight(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_trans_bitrow_eight, arr)


def trans_bit_elem_AVX(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_trans_bit_elem_AVX, arr)


def trans_bit_elem_scal(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_trans_bit_elem_scal, arr)


def trans_byte_bitrow_SSE(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_trans_byte_bitrow_SSE, arr)


def trans_byte_bitrow(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_trans_byte_bitrow, arr)


def shuffle_bit_eightelem_SSE(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_shuffle_bit_eightelem_SSE, arr)


def untrans_bit_elem_SSE(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_untrans_bit_elem_SSE, arr)


def untrans_bit_elem_scal(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_untrans_bit_elem_scal, arr)


def trans_bit_elem(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_trans_bit_elem, arr)


def untrans_bit_elem(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_untrans_bit_elem, arr)


def bitshuffle(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_bitshuffle, arr)


def bitunshuffle(np.ndarray arr not None):
    return _wrap_C_fun(&bshuf_bitunshuffle, arr)


