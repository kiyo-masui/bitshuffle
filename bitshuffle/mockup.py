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


# Not 100% sure that I got the mm_unpack intrinsics right.  Wrote them on a
# plane with no internet access.
def mm_unpacklo_epi8(m128a, m128b):
    dst = m128a.copy()
    for ii in range(8):
        dst[2 * ii] = m128a[ii]
        dst[2 * ii + 1] = m128b[ii]
    return dst


def mm_unpackhi_epi8(m128a, m128b):
    dst = m128a.copy()
    for ii in range(8):
        dst[2 * ii] = m128a[8 + ii]
        dst[2 * ii + 1] = m128b[8 + ii]
    return dst


def mm_unpacklo_epi16(m128a, m128b):
    dst = m128a.copy()
    for ii in range(4):
        dst[4*ii:4*ii + 2] = m128a[2*ii:2*ii + 2]
        dst[4*ii + 2:4*ii + 4] = m128b[2*ii:2*ii + 2]
    return dst


def mm_unpackhi_epi16(m128a, m128b):
    dst = m128a.copy()
    for ii in range(4):
        dst[4*ii:4*ii + 2] = m128a[8 + 2*ii:8 + 2*ii + 2]
        dst[4*ii + 2:4*ii + 4] = m128b[8 + 2*ii:8 + 2*ii + 2]
    return dst

def mm_unpacklo_epi32(m128a, m128b):
    dst = m128a.copy()
    for ii in range(2):
        dst[8*ii:8*ii + 4] = m128a[4*ii:4*ii + 4]
        dst[8*ii + 4:8*ii + 8] = m128b[4*ii:4*ii + 4]
    return dst


def mm_unpackhi_epi32(m128a, m128b):
    dst = m128a.copy()
    for ii in range(2):
        dst[8*ii:8*ii + 4] = m128a[8 + 4*ii:8 + 4*ii + 4]
        dst[8*ii + 4:8*ii + 8] = m128b[8 + 4*ii:8 + 4*ii + 4]
    return dst

def mm_unpacklo_epi64(m128a, m128b):
    dst = m128a.copy()
    for ii in range(1):
        dst[16*ii:16*ii + 8] = m128a[8*ii:8*ii + 8]
        dst[16*ii + 8:8*ii + 16] = m128b[8*ii:8*ii + 8]
    return dst


def mm_unpackhi_epi64(m128a, m128b):
    dst = m128a.copy()
    for ii in range(1):
        dst[16*ii:16*ii + 8] = m128a[8 + 8*ii:8 + 8*ii + 8]
        dst[16*ii + 8:8*ii + 16] = m128b[8 + 8*ii:8 + 8*ii + 8]
    return dst


def mm_load_si128(arr, i):
    out = np.empty(16, dtype=np.uint8)
    arr = arr.view(np.uint8)
    arr.shape = (arr.size,)
    for ii in range(16):
        out[ii] = arr[i + ii]
    return out


def mm_store_si128(m128, arr, i):
    #print arr.shape, m128.shape, i
    arr = arr.view(np.uint8)
    arr.shape = (arr.size,)
    #print arr.shape
    for ii in range(16):
        arr[i + ii] = m128[ii]


def mm_shufflehi_epi16(m128, imm):
    dst = m128.copy()
    for ii in range(4):
        w = imm % 4
        dst[6 - 2*ii:8 - 2*ii] = m128[6 - 2*w:8 - 2*w]
        imm //= 4
    return dst


def mm_shufflelo_epi16(m128, imm):
    dst = m128.copy()
    for ii in range(4):
        w = imm % 4
        dst[14 - 2*ii:16 - 2*ii] = m128[14 - 2*w:16 - 2*w]
        imm //= 4
    return dst



# I think I can solve the general problem by starting with a 64 bit interleave
# in recursing down to 8.  This might be as low of 6 instructions per 16 bytes
# always.


def bt16x8(m128):
    """Transpose 16x8 bit array."""

    # XXX Can be replaced by a single call to mm_permute_epi
    m128_tmp = np.empty_like(m128)
    m128_tmp[:8] = m128[8:]
    m128_tmp[8:] = m128[:8]
    m128 = m128_tmp

    out = np.empty(8, dtype=np.uint16)
    for ii in range(8):
        out[ii] = mm_movemask_epi8(m128)
        m128 = mm_slli_epi16(m128, 1)
    return out.view(dtype=np.uint8)


# I think we want to use combination of bitwise and, bit shifts by 1 byte and
# mm_packus_epi16 to interleave bytes.  This is (2ands + 1shift + 1pack) * 2 +
# 2loads + 2stores = 12 operations for 32 bytes, which is probably fine, but
# only works for 2byte data.  Recursive algorithm might be too expensive.

# mm_shuffle(hi,lo)_epi16 also interesting.

def byte_transpose(arr, itemsize=None):
    if not itemsize:
        itemsize = arr.dtype.itemsize
    in_buf = arr.view(np.uint8)
    in_buf.shape = (in_buf.size,)
    nelem = in_buf.size // itemsize
    in_buf.shape = (nelem, itemsize)
    
    print
    # TODO, accellerate with SSE.
    out_buf = np.empty((itemsize, nelem), dtype=np.uint8)
    last = 0
    if itemsize == 1:
        out_buf.flat[:] = in_buf.flat[:]
        last = nelem
    elif itemsize == 2:
        # 16 elements at a time.
        for ii in range(0,nelem - 15,16):
            a0 = mm_load_si128(in_buf, ii * itemsize)
            b0 = mm_load_si128(in_buf, ii * itemsize + 16)
            print a0
            print b0
            print
            a1 = mm_unpacklo_epi8(a0, b0)
            b1 = mm_unpackhi_epi8(a0, b0)
            print a1
            print b1
            print
            a2 = mm_unpacklo_epi8(a1, b1)
            b2 = mm_unpackhi_epi8(a1, b1)
            print a2
            print b2
            print
            a1 = mm_unpacklo_epi8(a2, b2)
            b1 = mm_unpackhi_epi8(a2, b2)
            print a1
            print b1
            print
            a2 = mm_unpacklo_epi8(a1, b1)
            b2 = mm_unpackhi_epi8(a1, b1)
            print a2
            print b2
            print
            mm_store_si128(a2, out_buf, 0*nelem + ii)
            mm_store_si128(b2, out_buf, 1*nelem + ii)
        last = nelem - nelem % 16
    elif itemsize == 4:
        # 16 elements at a time.
        for ii in range(0,nelem - 15,16):
            a0 = mm_load_si128(in_buf, ii * itemsize)
            b0 = mm_load_si128(in_buf, ii * itemsize + 16)
            c0 = mm_load_si128(in_buf, ii * itemsize + 32)
            d0 = mm_load_si128(in_buf, ii * itemsize + 48)
            print a0
            print b0
            print c0
            print d0
            print
            a1 = mm_unpacklo_epi8(a0, b0)
            b1 = mm_unpackhi_epi8(a0, b0)
            c1 = mm_unpacklo_epi8(c0, d0)
            d1 = mm_unpackhi_epi8(c0, d0)
            print a1
            print b1
            print c1
            print d1
            print
            a2 = mm_unpacklo_epi8(a1, b1)
            b2 = mm_unpackhi_epi8(a1, b1)
            c2 = mm_unpacklo_epi8(c1, d1)
            d2 = mm_unpackhi_epi8(c1, d1)
            print a2
            print b2
            print c2
            print d2
            print
            a1 = mm_unpacklo_epi8(a2, b2)
            b1 = mm_unpackhi_epi8(a2, b2)
            c1 = mm_unpacklo_epi8(c2, d2)
            d1 = mm_unpackhi_epi8(c2, d2)
            print a1
            print b1
            print c1
            print d1
            print
            a2 = mm_unpacklo_epi64(a1, c1)
            b2 = mm_unpackhi_epi64(a1, c1)
            c2 = mm_unpacklo_epi64(b1, d1)
            d2 = mm_unpackhi_epi64(b1, d1)
            print a2
            print b2
            print c2
            print d2
            print
            mm_store_si128(a2, out_buf, 0*nelem + ii)
            mm_store_si128(b2, out_buf, 1*nelem + ii)
            mm_store_si128(c2, out_buf, 2*nelem + ii)
            mm_store_si128(d2, out_buf, 3*nelem + ii)
        last = nelem - nelem % 16
    elif itemsize == 8:
        # 16 elements at a time.
        for ii in range(0,nelem - 15,16):
            a0 = mm_load_si128(in_buf, ii * itemsize)
            b0 = mm_load_si128(in_buf, ii * itemsize + 16)
            c0 = mm_load_si128(in_buf, ii * itemsize + 32)
            d0 = mm_load_si128(in_buf, ii * itemsize + 48)
            e0 = mm_load_si128(in_buf, ii * itemsize + 64)
            f0 = mm_load_si128(in_buf, ii * itemsize + 80)
            g0 = mm_load_si128(in_buf, ii * itemsize + 96)
            h0 = mm_load_si128(in_buf, ii * itemsize + 112)
            print a0
            print b0
            print c0
            print d0
            print e0
            print f0
            print g0
            print h0
            print
            a1 = mm_unpacklo_epi8(a0, b0)
            b1 = mm_unpackhi_epi8(a0, b0)
            c1 = mm_unpacklo_epi8(c0, d0)
            d1 = mm_unpackhi_epi8(c0, d0)
            e1 = mm_unpacklo_epi8(e0, f0)
            f1 = mm_unpackhi_epi8(e0, f0)
            g1 = mm_unpacklo_epi8(g0, h0)
            h1 = mm_unpackhi_epi8(g0, h0)
            print a1
            print b1
            print c1
            print d1
            print e1
            print f1
            print g1
            print h1
            print
            a2 = mm_unpacklo_epi8(a1, b1)
            b2 = mm_unpackhi_epi8(a1, b1)
            c2 = mm_unpacklo_epi8(c1, d1)
            d2 = mm_unpackhi_epi8(c1, d1)
            e2 = mm_unpacklo_epi8(e1, f1)
            f2 = mm_unpackhi_epi8(e1, f1)
            g2 = mm_unpacklo_epi8(g1, h1)
            h2 = mm_unpackhi_epi8(g1, h1)
            print a2
            print b2
            print c2
            print d2
            print e2
            print f2
            print g2
            print h2
            print
            a1 = mm_unpacklo_epi32(a2, c2)
            b1 = mm_unpackhi_epi32(a2, c2)
            c1 = mm_unpacklo_epi32(b2, d2)
            d1 = mm_unpackhi_epi32(b2, d2)
            e1 = mm_unpacklo_epi32(e2, g2)
            f1 = mm_unpackhi_epi32(e2, g2)
            g1 = mm_unpacklo_epi32(f2, h2)
            h1 = mm_unpackhi_epi32(f2, h2)
            print a1
            print b1
            print c1
            print d1
            print e1
            print f1
            print g1
            print h1
            print
            a2 = mm_unpacklo_epi64(a1, e1)
            b2 = mm_unpackhi_epi64(a1, e1)
            c2 = mm_unpacklo_epi64(b1, f1)
            d2 = mm_unpackhi_epi64(b1, f1)
            e2 = mm_unpacklo_epi64(c1, g1)
            f2 = mm_unpackhi_epi64(c1, g1)
            g2 = mm_unpacklo_epi64(d1, h1)
            h2 = mm_unpackhi_epi64(d1, h1)
            print a2
            print b2
            print c2
            print d2
            print e2
            print f2
            print g2
            print h2
            print

            mm_store_si128(a2, out_buf, 0*nelem + ii)
            mm_store_si128(b2, out_buf, 1*nelem + ii)
            mm_store_si128(c2, out_buf, 2*nelem + ii)
            mm_store_si128(d2, out_buf, 3*nelem + ii)
            mm_store_si128(e2, out_buf, 4*nelem + ii)
            mm_store_si128(f2, out_buf, 5*nelem + ii)
            mm_store_si128(g2, out_buf, 6*nelem + ii)
            mm_store_si128(h2, out_buf, 7*nelem + ii)
        last = nelem - nelem % 16

    for ii in range(last, nelem):
        for jj in range(itemsize):
            out_buf[jj,ii] = in_buf[ii,jj]
    return out_buf



def bit_transpose(arr):
    nelem = arr.size
    itemsize = arr.dtype.itemsize
    in_buf = arr.view(np.uint8)
    in_buf.shape = (in_buf.size,)
    nbyte = nelem * itemsize
    if nbyte % 128:
        raise ValueError("Input array length must be multiple of 16 bytes.")

    # Byte transpose.
    buf0 = byte_transpose(in_buf, itemsize)

    # Transpose the bits and unpack.
    buf1 = np.empty((itemsize, 8, nelem // 8), dtype=np.uint8)

    # If we unrolled the mm loop we might be able to use the *register* keyword
    # when declaring buf_8x128_ui16.  This isn't strictly legal but works with
    # most compilers as long as you only index the array with literals.
    # Really, the compiler should perform this optimization anyway.

    buf_8x128_ui16 = np.empty((8, 8,), dtype=np.uint16)
    for ii in range(itemsize):
        for jj in range(0, nelem, 128):
            for kk in range(8):
                # XXX Replace with a single mm_load.
                this_m128 = mm_load_si128(buf0[ii,:], jj + 16*kk)

                # This fixes the fact that the little endianess of uint16 swaps
                # the two bytes.
                # XXX Can be replaced by a single call to mm_permute_epi
                m128_tmp = np.empty_like(this_m128)
                m128_tmp[:8] = this_m128[8:]
                m128_tmp[8:] = this_m128[:8]
                this_m128 = m128_tmp

                # If we unrolled this loop, we could eliminate an extra slli
                # but the compiler might do both of these anyway.
                for mm in range(8):
                    byte_trans = mm_movemask_epi8(this_m128)
                    # I think the order of the following two is optimal, since
                    # movemask depends on slli, and the store depends on
                    # movemask.
                    this_m128 = mm_slli_epi16(this_m128, 1)
                    # XXX Replace with mm_insert_epi16?  Yes, so we don't have
                    # to cast to uint16.  Also, this avoids all our endianess
                    # issues of the uint16, and we can delete the mm_permute.
                    buf_8x128_ui16[mm,kk] = byte_trans
            # Store the registers in the output.
            for mm in range(8):
                tmp_m128 = buf_8x128_ui16[mm,:].view(np.uint8)
                mm_store_si128(tmp_m128, buf1[ii,mm,:], jj // 8)
    buf1.shape = (buf1.size,)
    return buf1
