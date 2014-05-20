"""Python mockup of the bitshuffle core algorithm used for devellopment."""

import numpy as np

INTMAX = 64
INTDTYPE = np.int64

def mm_movemask_epi8(m128):
    bits = np.unpackbits(m128)
    dst = np.zeros(INTMAX, dtype=np.uint8)
    for ii in range(16):
        dst[INTMAX - 1 - ii] = bits[128 - 1 - (8*ii + 7)]
    out = np.packbits(dst)
    # Swap endianness.
    out = out[::-1].copy()
    return out.view(INTDTYPE)[0]


def mm_slli_epi16(m128, shift):
    dst = m128.copy()
    dst = np.unpackbits(m128)
    # Shift the bits.
    dst = np.roll(dst, -shift).copy()
    # Zero pad the right of each word.
    for ii in range(8):
        dst[(ii + 1) * 16 - shift:(ii + 1) * 16] = 0
    return np.packbits(dst)


def bt16x8(m128):
    """Transpose 16x8 bit array."""

    m128_tmp = np.empty_like(m128)
    m128_tmp[:8] = m128[8:]
    m128_tmp[8:] = m128[:8]
    m128 = m128_tmp

    out = np.empty(8, dtype=np.uint16)
    for ii in range(8):
        out[ii] = mm_movemask_epi8(m128)
        m128 = mm_slli_epi16(m128, 1)
    return out.view(dtype=np.uint8)


def bit_transpose(arr):
    nelem = arr.size
    itemsize = arr.dtype.itemsize
    in_buf = arr.flat[:].view(np.uint8)
    nbyte = nelem * itemsize
    if nbyte % 16:
        raise ValueError("Input array length must be multiple of 16 bytes.")
    in_buf.shape = (nelem, itemsize)
    # Byte transpose.
    buf0 = np.empty((itemsize, nelem), dtype=np.uint8)
    for jj in range(nelem):
        for ii in range(itemsize):
            buf0[ii,jj] = in_buf[jj,ii]
    # Transpose the bits and unpack.
    buf1 = np.empty((itemsize, 8, nelem // 8), dtype=np.uint8)
    tmp_m128 = np.empty(16, dtype=np.uint8)
    for ii in range(itemsize):
        for jj in range(nelem // 16):
            for kk in range(16):
                tmp_m128[kk] = in_buf[16 * jj + kk,ii]
            tmp_m128 = bt16x8(tmp_m128)
            for kk in range(16):
                buf1[ii,kk // 2,2 * jj + (kk % 2)] = tmp_m128[kk]

    return buf1.flat[:]
