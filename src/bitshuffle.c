
#include "bitshuffle.h"


/* ---- Code that should compile on any machine. ---- */

/* Memory copy with bshuf call signature. For testing and profiling. */
int bshuf_copy(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;

    memcpy(B, A, size * elem_size);
    return 0;
}


/* Transpose bytes withing elements, starting partway through input. */
int bshuf_trans_byte_elem_remainder(void* in, void* out, const size_t size,
         const size_t elem_size, const size_t start) {

    char* A = (char*) in;
    char* B = (char*) out;
    // ii loop separated into 2 loops so the compiler can unroll the inner one.
    for (size_t ii = start; ii + 7 < size; ii += 8) {
        for (size_t jj = 0; jj < elem_size; jj++) {
            for (size_t kk = 0; kk < 8; kk++) {
                B[jj * size + ii + kk] = A[ii * elem_size + kk * elem_size + jj];
            }
        }
    }
    return 0;
}


/* Transpose bytes within elements, scalar algorithm. */
int bshuf_trans_byte_elem_scal(void* in, void* out, const size_t size,
         const size_t elem_size) {

    return bshuf_trans_byte_elem_remainder(in, out, size, elem_size, 0);
}

#define TRANS_BIT_8X8(x, t) {                                               \
        t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AALL;                          \
        x = x ^ t ^ (t << 7);                                               \
        t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCLL;                         \
        x = x ^ t ^ (t << 14);                                              \
        t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0LL;                         \
        x = x ^ t ^ (t << 28);                                              \
    }

/* Transpose bits within bytes. Does not use x86 specific instructions.
 * Code from the Hacker's Delight. */
int bshuf_trans_bit_byte(void* in, void* out, const size_t size,
         const size_t elem_size) {

    uint64_t* A = in;
    uint8_t* B = out;

    uint64_t x, t;

    size_t nbyte = elem_size * size;
    size_t nbyte_bitrow = nbyte / 8;

    for (size_t ii = 0; ii < nbyte_bitrow; ii ++) {
        x = A[ii];
        TRANS_BIT_8X8(x, t);
        for (int kk = 0; kk < 8; kk ++) {
            B[kk * nbyte_bitrow + ii] = x;
            x = x >> 8;
        }
    }
    return 0;
}


/* Transpose bits within bytes. Does not use x86 specific instructions.
 * Code from the Hacker's Delight. */
int bshuf_trans_bit_byte_unrolled(void* in, void* out, const size_t size,
         const size_t elem_size) {

    uint64_t* A = in;
    uint8_t* B = out;

    uint64_t x0, x1, x2, x3, t0, t1, t2, t3;

    size_t nbyte = elem_size * size;
    size_t nbyte_bitrow = nbyte / 8;

    for (size_t ii = 0; ii + 3 < nbyte_bitrow; ii += 4) {
        x0 = A[ii + 0];
        x1 = A[ii + 1];
        x2 = A[ii + 2];
        x3 = A[ii + 3];

        TRANS_BIT_8X8(x0, t0);
        TRANS_BIT_8X8(x1, t1);
        TRANS_BIT_8X8(x2, t2);
        TRANS_BIT_8X8(x3, t3);

        // Get back 4% if I unroll this.
        for (int kk = 0; kk < 8; kk ++) {
            B[kk * nbyte_bitrow + ii + 0] = x0;
            x0 = x0 >> 8;
            B[kk * nbyte_bitrow + ii + 1] = x1;
            x1 = x1 >> 8;
            B[kk * nbyte_bitrow + ii + 2] = x2;
            x2 = x2 >> 8;
            B[kk * nbyte_bitrow + ii + 3] = x3;
            x3 = x3 >> 8;
        }
    }
    return 0;
}


int bshuf_trans_bit_elem_scal(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;

    union {uint64_t x; char b[8];} mx0;
    uint64_t t0;
    size_t nbyte_bitrow = size / 8;

    for (size_t ii = 0; ii < nbyte_bitrow; ii ++) {
        for (size_t jj = 0; jj < elem_size; jj ++) {
            for (size_t kk = 0; kk < 8; kk ++) {
                mx0.b[kk] = A[(ii + 0) * 8 * elem_size + kk * elem_size + jj];
            }
            TRANS_BIT_8X8(mx0.x, t0);
            for (size_t kk = 0; kk < 8; kk ++) {
                B[jj * size + kk * nbyte_bitrow + ii + 0] = mx0.b[kk];
            }
        }
    }
    return 0;
}


int bshuf_untrans_bit_elem_scal(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;

    union {uint64_t x; char b[8];} mx0;
    uint64_t t0;
    size_t nbyte_bitrow = size / 8;

    for (size_t ii = 0; ii < nbyte_bitrow; ii ++) {
        for (size_t jj = 0; jj < elem_size; jj ++) {
            for (size_t kk = 0; kk < 8; kk ++) {
                mx0.b[kk] = A[jj * size + kk * nbyte_bitrow + ii + 0];
            }
            TRANS_BIT_8X8(mx0.x, t0);
            for (size_t kk = 0; kk < 8; kk ++) {
                B[(ii + 0) * 8 * elem_size + kk * elem_size + jj] = mx0.b[kk];
            }
        }
    }
    return 0;
}


/* Transpose of an array, optimized for small elements. */
#define TRANS_ELEM_TYPE(in, out, lda, ldb, type_t) {                        \
        type_t* A = (type_t*) in;                                           \
        type_t* B = (type_t*) out;                                          \
        for(size_t ii = 0; ii < lda; ii += 8) {                             \
            for(size_t jj = 0; jj < ldb; jj++) {                            \
                for(size_t kk = 0; kk < 8; kk++) {                          \
                    B[jj*lda + ii + kk] = A[ii*ldb + kk * ldb + jj];        \
                }                                                           \
            }                                                               \
        }                                                                   \
    }


/* General transpose of an array, (un)optimized for large element sizes. */
int bshuf_trans_elem(void* in, void* out, const size_t lda, const size_t ldb,
        const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;
    for(size_t ii = 0; ii < lda; ii++) {
        for(size_t jj = 0; jj < ldb; jj++) {
            memcpy(&B[(jj*lda + ii) * elem_size],
                   &A[(ii*ldb + jj) * elem_size], elem_size);
        }
    }
    return 0;
}


/* Transpose rows of shiffled bits (size / 8 bytes) within groups of 8. */
int bshuf_trans_bitrow_eight(void* in, void* out, const size_t size,
         const size_t elem_size) {

    size_t nbyte_bitrow = size / 8;
    return bshuf_trans_elem(in, out, 8, elem_size, nbyte_bitrow);
}


/* For data organized into a row for each bit (8 * elem_size rows), transpose
 * the bytes. */
int bshuf_trans_byte_bitrow(void* in, void* out, const size_t size,
         const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;

    size_t nbyte_row = size / 8;

    for (int ii = 0; ii < nbyte_row; ii++) {
        for (int jj = 0; jj < 8*elem_size; jj++) {
            B[ii * 8 * elem_size + jj] = A[jj * nbyte_row + ii];
        }
    }
    return 0;
}




/* ---- Code that requires SSE2. x86 architectures post Pentium 4. ---- */

#ifdef USESSE2

/* Transpose bytes within elements using SSE for 16 bit elements. */
int bshuf_trans_byte_elem_SSE_16(void* in, void* out, const size_t size) {

    char* A = (char*) in;
    char* B = (char*) out;
    __m128i a0, b0, a1, b1;

    for (size_t ii=0; ii + 15 < size; ii += 16) {
        a0 = _mm_loadu_si128((__m128i *) &A[2*ii + 0*16]);
        b0 = _mm_loadu_si128((__m128i *) &A[2*ii + 1*16]);

        a1 = _mm_unpacklo_epi8(a0, b0);
        b1 = _mm_unpackhi_epi8(a0, b0);

        a0 = _mm_unpacklo_epi8(a1, b1);
        b0 = _mm_unpackhi_epi8(a1, b1);

        a1 = _mm_unpacklo_epi8(a0, b0);
        b1 = _mm_unpackhi_epi8(a0, b0);

        a0 = _mm_unpacklo_epi8(a1, b1);
        b0 = _mm_unpackhi_epi8(a1, b1);

        _mm_storeu_si128((__m128i *) &B[0*size + ii], a0);
        _mm_storeu_si128((__m128i *) &B[1*size + ii], b0);
    }
    return bshuf_trans_byte_elem_remainder(in, out, size, 2,
            size - size % 16);
}


/* Transpose bytes within elements using SSE for 32 bit elements. */
int bshuf_trans_byte_elem_SSE_32(void* in, void* out, const size_t size) {

    char* A = (char*) in;
    char* B = (char*) out;
    __m128i a0, b0, c0, d0, a1, b1, c1, d1;

    for (size_t ii=0; ii + 15 < size; ii += 16) {
        a0 = _mm_loadu_si128((__m128i *) &A[4*ii + 0*16]);
        b0 = _mm_loadu_si128((__m128i *) &A[4*ii + 1*16]);
        c0 = _mm_loadu_si128((__m128i *) &A[4*ii + 2*16]);
        d0 = _mm_loadu_si128((__m128i *) &A[4*ii + 3*16]);

        a1 = _mm_unpacklo_epi8(a0, b0);
        b1 = _mm_unpackhi_epi8(a0, b0);
        c1 = _mm_unpacklo_epi8(c0, d0);
        d1 = _mm_unpackhi_epi8(c0, d0);

        a0 = _mm_unpacklo_epi8(a1, b1);
        b0 = _mm_unpackhi_epi8(a1, b1);
        c0 = _mm_unpacklo_epi8(c1, d1);
        d0 = _mm_unpackhi_epi8(c1, d1);

        a1 = _mm_unpacklo_epi8(a0, b0);
        b1 = _mm_unpackhi_epi8(a0, b0);
        c1 = _mm_unpacklo_epi8(c0, d0);
        d1 = _mm_unpackhi_epi8(c0, d0);

        a0 = _mm_unpacklo_epi64(a1, c1);
        b0 = _mm_unpackhi_epi64(a1, c1);
        c0 = _mm_unpacklo_epi64(b1, d1);
        d0 = _mm_unpackhi_epi64(b1, d1);

        _mm_storeu_si128((__m128i *) &B[0*size + ii], a0);
        _mm_storeu_si128((__m128i *) &B[1*size + ii], b0);
        _mm_storeu_si128((__m128i *) &B[2*size + ii], c0);
        _mm_storeu_si128((__m128i *) &B[3*size + ii], d0);
    }
    return bshuf_trans_byte_elem_remainder(in, out, size, 4,
            size - size % 16);
}


/* Transpose bytes within elements using SSE for 64 bit elements. */
int bshuf_trans_byte_elem_SSE_64(void* in, void* out, const size_t size) {

    char* A = (char*) in;
    char* B = (char*) out;
    __m128i a0, b0, c0, d0, e0, f0, g0, h0;
    __m128i a1, b1, c1, d1, e1, f1, g1, h1;

    for (size_t ii=0; ii + 15 < size; ii += 16) {
        a0 = _mm_loadu_si128((__m128i *) &A[8*ii + 0*16]);
        b0 = _mm_loadu_si128((__m128i *) &A[8*ii + 1*16]);
        c0 = _mm_loadu_si128((__m128i *) &A[8*ii + 2*16]);
        d0 = _mm_loadu_si128((__m128i *) &A[8*ii + 3*16]);
        e0 = _mm_loadu_si128((__m128i *) &A[8*ii + 4*16]);
        f0 = _mm_loadu_si128((__m128i *) &A[8*ii + 5*16]);
        g0 = _mm_loadu_si128((__m128i *) &A[8*ii + 6*16]);
        h0 = _mm_loadu_si128((__m128i *) &A[8*ii + 7*16]);

        a1 = _mm_unpacklo_epi8(a0, b0);
        b1 = _mm_unpackhi_epi8(a0, b0);
        c1 = _mm_unpacklo_epi8(c0, d0);
        d1 = _mm_unpackhi_epi8(c0, d0);
        e1 = _mm_unpacklo_epi8(e0, f0);
        f1 = _mm_unpackhi_epi8(e0, f0);
        g1 = _mm_unpacklo_epi8(g0, h0);
        h1 = _mm_unpackhi_epi8(g0, h0);

        a0 = _mm_unpacklo_epi8(a1, b1);
        b0 = _mm_unpackhi_epi8(a1, b1);
        c0 = _mm_unpacklo_epi8(c1, d1);
        d0 = _mm_unpackhi_epi8(c1, d1);
        e0 = _mm_unpacklo_epi8(e1, f1);
        f0 = _mm_unpackhi_epi8(e1, f1);
        g0 = _mm_unpacklo_epi8(g1, h1);
        h0 = _mm_unpackhi_epi8(g1, h1);

        a1 = _mm_unpacklo_epi32(a0, c0);
        b1 = _mm_unpackhi_epi32(a0, c0);
        c1 = _mm_unpacklo_epi32(b0, d0);
        d1 = _mm_unpackhi_epi32(b0, d0);
        e1 = _mm_unpacklo_epi32(e0, g0);
        f1 = _mm_unpackhi_epi32(e0, g0);
        g1 = _mm_unpacklo_epi32(f0, h0);
        h1 = _mm_unpackhi_epi32(f0, h0);

        a0 = _mm_unpacklo_epi64(a1, e1);
        b0 = _mm_unpackhi_epi64(a1, e1);
        c0 = _mm_unpacklo_epi64(b1, f1);
        d0 = _mm_unpackhi_epi64(b1, f1);
        e0 = _mm_unpacklo_epi64(c1, g1);
        f0 = _mm_unpackhi_epi64(c1, g1);
        g0 = _mm_unpacklo_epi64(d1, h1);
        h0 = _mm_unpackhi_epi64(d1, h1);

        _mm_storeu_si128((__m128i *) &B[0*size + ii], a0);
        _mm_storeu_si128((__m128i *) &B[1*size + ii], b0);
        _mm_storeu_si128((__m128i *) &B[2*size + ii], c0);
        _mm_storeu_si128((__m128i *) &B[3*size + ii], d0);
        _mm_storeu_si128((__m128i *) &B[4*size + ii], e0);
        _mm_storeu_si128((__m128i *) &B[5*size + ii], f0);
        _mm_storeu_si128((__m128i *) &B[6*size + ii], g0);
        _mm_storeu_si128((__m128i *) &B[7*size + ii], h0);
    }
    return bshuf_trans_byte_elem_remainder(in, out, size, 8,
            size - size % 16);
}


/* Transpose bytes within elements using best SSE algorithm available. */
int bshuf_trans_byte_elem_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {

    int err;

    // Trivial cases: power of 2 bytes.
    switch (elem_size) {
        case 1:
            err = bshuf_copy(in, out, size, elem_size);
            return err;
        case 2:
            err = bshuf_trans_byte_elem_SSE_16(in, out, size);
            return err;
        case 4:
            err = bshuf_trans_byte_elem_SSE_32(in, out, size);
            return err;
        case 8:
            err = bshuf_trans_byte_elem_SSE_64(in, out, size);
            return err;
    }

    // Worst case: odd number of bytes. Turns out that this is faster for
    // (odd * 2) byte elements as well (hense % 4).
    if (elem_size % 4) {
        err = bshuf_trans_byte_elem_scal(in, out, size, elem_size);
        return err;
    }

    // Multiple of power of 2: transpose hierarchically.
    {
        size_t nchunk_elem;
        void* tmp_buf = malloc(size * elem_size);

        if ((elem_size % 8) == 0) {
            nchunk_elem = elem_size / 8;
            TRANS_ELEM_TYPE(in, out, size, nchunk_elem, int64_t);
            err = bshuf_trans_byte_elem_SSE_64(out, tmp_buf, size * nchunk_elem);
            bshuf_trans_elem(tmp_buf, out, 8, nchunk_elem, size);
        } else if ((elem_size % 4) == 0) {
            nchunk_elem = elem_size / 4;
            TRANS_ELEM_TYPE(in, out, size, nchunk_elem, int32_t);
            err = bshuf_trans_byte_elem_SSE_32(out, tmp_buf, size * nchunk_elem);
            bshuf_trans_elem(tmp_buf, out, 4, nchunk_elem, size);
        } else {
            // Not used since scalar algorithm is faster.
            nchunk_elem = elem_size / 2;
            TRANS_ELEM_TYPE(in, out, size, nchunk_elem, int16_t);
            err = bshuf_trans_byte_elem_SSE_16(out, tmp_buf, size * nchunk_elem);
            bshuf_trans_elem(tmp_buf, out, 2, nchunk_elem, size);
        }

        free(tmp_buf);
        return err;
    }
}


/* Transpose bits within bytes using SSE. */
int bshuf_trans_bit_byte_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;
    uint16_t* Bui;

    size_t nbyte = elem_size * size;

    __m128i xmm;
    int bt;

    for (size_t ii = 0; ii + 15 < nbyte; ii += 16) {
        xmm = _mm_loadu_si128((__m128i *) &A[ii]);
        for (size_t kk = 0; kk < 8; kk++) {
            bt = _mm_movemask_epi8(xmm);
            xmm = _mm_slli_epi16(xmm, 1);
            Bui = (uint16_t*) &B[((7 - kk) * nbyte + ii) / 8];
            *Bui = bt;
        }
    }
    return 0;
}


/* For data organized into a row for each bit (8 * elem_size rows), transpose
 * the bytes. */
int bshuf_trans_byte_bitrow_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;

    size_t nrows = 8 * elem_size;
    size_t nbyte_row = size / 8;

    __m128i a0, b0, c0, d0, e0, f0, g0, h0;
    __m128i a1, b1, c1, d1, e1, f1, g1, h1;
    __m128 *as, *bs, *cs, *ds, *es, *fs, *gs, *hs;

    for (size_t ii = 0; ii + 7 < nrows; ii += 8) {
        for (size_t jj = 0; jj + 15 < nbyte_row; jj += 16) {
            a0 = _mm_loadu_si128((__m128i *) &A[(ii + 0)*nbyte_row + jj]);
            b0 = _mm_loadu_si128((__m128i *) &A[(ii + 1)*nbyte_row + jj]);
            c0 = _mm_loadu_si128((__m128i *) &A[(ii + 2)*nbyte_row + jj]);
            d0 = _mm_loadu_si128((__m128i *) &A[(ii + 3)*nbyte_row + jj]);
            e0 = _mm_loadu_si128((__m128i *) &A[(ii + 4)*nbyte_row + jj]);
            f0 = _mm_loadu_si128((__m128i *) &A[(ii + 5)*nbyte_row + jj]);
            g0 = _mm_loadu_si128((__m128i *) &A[(ii + 6)*nbyte_row + jj]);
            h0 = _mm_loadu_si128((__m128i *) &A[(ii + 7)*nbyte_row + jj]);


            a1 = _mm_unpacklo_epi8(a0, b0);
            b1 = _mm_unpacklo_epi8(c0, d0);
            c1 = _mm_unpacklo_epi8(e0, f0);
            d1 = _mm_unpacklo_epi8(g0, h0);
            e1 = _mm_unpackhi_epi8(a0, b0);
            f1 = _mm_unpackhi_epi8(c0, d0);
            g1 = _mm_unpackhi_epi8(e0, f0);
            h1 = _mm_unpackhi_epi8(g0, h0);


            a0 = _mm_unpacklo_epi16(a1, b1);
            b0 = _mm_unpacklo_epi16(c1, d1);
            c0 = _mm_unpackhi_epi16(a1, b1);
            d0 = _mm_unpackhi_epi16(c1, d1);

            e0 = _mm_unpacklo_epi16(e1, f1);
            f0 = _mm_unpacklo_epi16(g1, h1);
            g0 = _mm_unpackhi_epi16(e1, f1);
            h0 = _mm_unpackhi_epi16(g1, h1);


            a1 = _mm_unpacklo_epi32(a0, b0);
            b1 = _mm_unpackhi_epi32(a0, b0);

            c1 = _mm_unpacklo_epi32(c0, d0);
            d1 = _mm_unpackhi_epi32(c0, d0);

            e1 = _mm_unpacklo_epi32(e0, f0);
            f1 = _mm_unpackhi_epi32(e0, f0);

            g1 = _mm_unpacklo_epi32(g0, h0);
            h1 = _mm_unpackhi_epi32(g0, h0);

            /*
            _mm_storeu_si128((__m128i *) &B[(jj + 0) * nrows + ii], a1);
            _mm_storeu_si128((__m128i *) &B[(jj + 2) * nrows + ii], b1);
            _mm_storeu_si128((__m128i *) &B[(jj + 4) * nrows + ii], c1);
            _mm_storeu_si128((__m128i *) &B[(jj + 6) * nrows + ii], d1);
            _mm_storeu_si128((__m128i *) &B[(jj + 8) * nrows + ii], e1);
            _mm_storeu_si128((__m128i *) &B[(jj + 10) * nrows + ii], f1);
            _mm_storeu_si128((__m128i *) &B[(jj + 12) * nrows + ii], g1);
            _mm_storeu_si128((__m128i *) &B[(jj + 14) * nrows + ii], h1);
            */

            /*
            _mm_storel_epi64((__m128i *) &B[(jj + 0) * nrows + ii], a1);
            _mm_storel_epi64((__m128i *) &B[(jj + 2) * nrows + ii], b1);
            _mm_storel_epi64((__m128i *) &B[(jj + 4) * nrows + ii], c1);
            _mm_storel_epi64((__m128i *) &B[(jj + 6) * nrows + ii], d1);
            _mm_storel_epi64((__m128i *) &B[(jj + 8) * nrows + ii], e1);
            _mm_storel_epi64((__m128i *) &B[(jj + 10) * nrows + ii], f1);
            _mm_storel_epi64((__m128i *) &B[(jj + 12) * nrows + ii], g1);
            _mm_storel_epi64((__m128i *) &B[(jj + 14) * nrows + ii], h1);
            */

            // We don't have a storeh instruction for integers, so interpret
            // as a float. Have a storel (_mm_storel_epi64).
            as = (__m128 *) &a1;
            bs = (__m128 *) &b1;
            cs = (__m128 *) &c1;
            ds = (__m128 *) &d1;
            es = (__m128 *) &e1;
            fs = (__m128 *) &f1;
            gs = (__m128 *) &g1;
            hs = (__m128 *) &h1;

            _mm_storel_pi((__m64 *) &B[(jj + 0) * nrows + ii], *as);
            _mm_storel_pi((__m64 *) &B[(jj + 2) * nrows + ii], *bs);
            _mm_storel_pi((__m64 *) &B[(jj + 4) * nrows + ii], *cs);
            _mm_storel_pi((__m64 *) &B[(jj + 6) * nrows + ii], *ds);
            _mm_storel_pi((__m64 *) &B[(jj + 8) * nrows + ii], *es);
            _mm_storel_pi((__m64 *) &B[(jj + 10) * nrows + ii], *fs);
            _mm_storel_pi((__m64 *) &B[(jj + 12) * nrows + ii], *gs);
            _mm_storel_pi((__m64 *) &B[(jj + 14) * nrows + ii], *hs);

            _mm_storeh_pi((__m64 *) &B[(jj + 1) * nrows + ii], *as);
            _mm_storeh_pi((__m64 *) &B[(jj + 3) * nrows + ii], *bs);
            _mm_storeh_pi((__m64 *) &B[(jj + 5) * nrows + ii], *cs);
            _mm_storeh_pi((__m64 *) &B[(jj + 7) * nrows + ii], *ds);
            _mm_storeh_pi((__m64 *) &B[(jj + 9) * nrows + ii], *es);
            _mm_storeh_pi((__m64 *) &B[(jj + 11) * nrows + ii], *fs);
            _mm_storeh_pi((__m64 *) &B[(jj + 13) * nrows + ii], *gs);
            _mm_storeh_pi((__m64 *) &B[(jj + 15) * nrows + ii], *hs);
        }
    }
    return 0;
}


/* Tranpose bits within elements. */
int bshuf_trans_bit_elem_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {

    int err;

    void* tmp_buf = malloc(size * elem_size);
    if (tmp_buf == NULL) return 1;

    // Should acctually check errors individually.
    err = bshuf_trans_byte_elem_SSE(in, out, size, elem_size);
    err += bshuf_trans_bit_byte_SSE(out, tmp_buf, size, elem_size);
    err += bshuf_trans_bitrow_eight(tmp_buf, out, size, elem_size);

    free(tmp_buf);

    return err;
}


/* Shuffle bits within the bytes of eight element blocks. */
int bshuf_shuffle_bit_eightelem_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {

    // With a bit of care, this could be written such that such that it is
    // in_buf = out_buf safe.
    char* A = (char*) in;
    char* B = (char*) out;
    uint16_t* Bui = (uint16_t*) out;

    size_t nbyte = elem_size * size;

    __m128i xmm;
    int bt;

    if (elem_size == 1) {
        for (size_t ii = 0; ii + 15 < nbyte; ii += 16) {
            xmm = _mm_loadu_si128((__m128i *) &A[ii]);
            for (size_t kk = 0; kk < 8; kk++) {
                bt = _mm_movemask_epi8(xmm);
                xmm = _mm_slli_epi16(xmm, 1);
                Bui[ii / 2 + 7 - kk] = bt;
            }
        }

        __m128i a0, b0, a1, b1;
        for (size_t ii = 0; ii + 31 < nbyte; ii += 32) {
            a0 = _mm_loadu_si128((__m128i *) &B[ii]);
            b0 = _mm_loadu_si128((__m128i *) &B[ii + 16]);

            a1 = _mm_unpacklo_epi8(a0, b0);
            b1 = _mm_unpackhi_epi8(a0, b0);

            a0 = _mm_unpacklo_epi8(a1, b1);
            b0 = _mm_unpackhi_epi8(a1, b1);

            a1 = _mm_unpacklo_epi8(a0, b0);
            b1 = _mm_unpackhi_epi8(a0, b0);

            a0 = _mm_unpacklo_epi8(a1, b1);
            b0 = _mm_unpackhi_epi8(a1, b1);

            a1 = _mm_unpacklo_epi64(a0, b0);
            b1 = _mm_unpackhi_epi64(a0, b0);

            _mm_storeu_si128((__m128i *) &B[ii], a1);
            _mm_storeu_si128((__m128i *) &B[ii + 16], b1);
        }
    } else {
        for (size_t ii = 0; ii + 8 * elem_size - 1 < nbyte; ii += 8 * elem_size) {
            for (size_t jj = 0; jj < 8 * elem_size; jj += 16) {
                xmm = _mm_loadu_si128((__m128i *) &A[ii + jj]);
                for (size_t kk = 0; kk < 8; kk++) {
                    bt = _mm_movemask_epi8(xmm);
                    xmm = _mm_slli_epi16(xmm, 1);
                    int ind = (ii + jj / 8 + (7 - kk) * elem_size);
                    Bui[ind / 2] = bt;
                }
            }
        }
    }
    return 0;
}


int bshuf_untrans_bit_elem_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {

    int err;
    void* tmp_buf = malloc(size * elem_size);
    if (tmp_buf == NULL) return 1;

    // Should acctually check errors individually.
    err = bshuf_trans_byte_bitrow_SSE(in, tmp_buf, size, elem_size);
    err +=  bshuf_shuffle_bit_eightelem_SSE(tmp_buf, out, size, elem_size);

    free(tmp_buf);

    return err;
}

#else // #ifdef USESSE2


int bshuf_untrans_bit_elem_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {
    return 11;
}


int bshuf_trans_byte_bitrow_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {
    return 11;
}


int bshuf_trans_byte_bitrow_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {
    return 11;
}


int bshuf_trans_bit_byte_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {
    return 11;
}


int bshuf_trans_byte_elem_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {
    return 11;
}


int bshuf_trans_byte_elem_SSE_64(void* in, void* out, const size_t size) {
    return 11;
}


int bshuf_trans_byte_elem_SSE_32(void* in, void* out, const size_t size) {
    return 11;
}


int bshuf_trans_byte_elem_SSE_16(void* in, void* out, const size_t size) {
    return 11;
}

#endif // #ifdef USESSE2



/* ---- Code that requires AVX2. Intel Haswell (2013) and later. ---- */

#ifdef USEAVX2

/* Transpose bits within bytes using AVX. */
int bshuf_trans_bit_byte_AVX_unrolled(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;
    uint32_t* Bui;

    size_t nbyte = elem_size * size;
    size_t kk;

    __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    int bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7;

    // Turns out that doublly unrolling this loop (unrolling 2 loops of 8) 
    // gives a speed up roughly 70% for some problem sizes.  The compiler will
    // not automatically doublly unroll a loop, but will optimize the
    // order of operations within one long section.
    for (size_t ii = 0; ii + 32 * 8 - 1 < nbyte; ii += 32 * 8) {
        ymm0 = _mm256_loadu_si256((__m256i *) &A[ii + 0*32]);
        ymm1 = _mm256_loadu_si256((__m256i *) &A[ii + 1*32]);
        ymm2 = _mm256_loadu_si256((__m256i *) &A[ii + 2*32]);
        ymm3 = _mm256_loadu_si256((__m256i *) &A[ii + 3*32]);
        ymm4 = _mm256_loadu_si256((__m256i *) &A[ii + 4*32]);
        ymm5 = _mm256_loadu_si256((__m256i *) &A[ii + 5*32]);
        ymm6 = _mm256_loadu_si256((__m256i *) &A[ii + 6*32]);
        ymm7 = _mm256_loadu_si256((__m256i *) &A[ii + 7*32]);

        kk = 0;
        Bui = (uint32_t*) &B[((7 - kk) * nbyte + ii) / 8];
        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);
        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);
        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);
        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);
        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;
        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        kk = 1;
        Bui = (uint32_t*) &B[((7 - kk) * nbyte + ii) / 8];
        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);
        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);
        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);
        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);
        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;
        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        kk = 2;
        Bui = (uint32_t*) &B[((7 - kk) * nbyte + ii) / 8];
        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);
        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);
        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);
        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);
        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;
        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        kk = 3;
        Bui = (uint32_t*) &B[((7 - kk) * nbyte + ii) / 8];
        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);
        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);
        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);
        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);
        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;
        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        kk = 4;
        Bui = (uint32_t*) &B[((7 - kk) * nbyte + ii) / 8];
        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);
        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);
        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);
        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);
        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;
        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        kk = 5;
        Bui = (uint32_t*) &B[((7 - kk) * nbyte + ii) / 8];
        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);
        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);
        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);
        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);
        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;
        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        kk = 6;
        Bui = (uint32_t*) &B[((7 - kk) * nbyte + ii) / 8];
        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);
        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);
        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);
        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);
        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;
        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        kk = 7;
        Bui = (uint32_t*) &B[((7 - kk) * nbyte + ii) / 8];
        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);
        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);
        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);
        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);
        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;
        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;


    }
    return 0;
}


/* Transpose bits within bytes using AVX. Less optimized version. */
int bshuf_trans_bit_byte_AVX(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;
    uint32_t* Bui;

    size_t nbyte = elem_size * size;

    __m256i ymm;
    int bt;

    for (size_t ii = 0; ii + 31 < nbyte; ii += 32) {
        ymm = _mm256_loadu_si256((__m256i *) &A[ii]);
        for (size_t kk = 0; kk < 8; kk++) {
            bt = _mm256_movemask_epi8(ymm);
            ymm = _mm256_slli_epi16(ymm, 1);
            Bui = (uint32_t*) &B[((7 - kk) * nbyte + ii) / 8];
            *Bui = bt;
        }
    }
    return 0;
}


/* Tranpose bits within elements. */
int bshuf_trans_bit_elem_AVX(void* in, void* out, const size_t size,
         const size_t elem_size) {

    int err;

    void* tmp_buf = malloc(size * elem_size);
    if (tmp_buf == NULL) return 1;

    // Should acctually check errors individually.
    err = bshuf_trans_byte_elem_SSE(in, out, size, elem_size);
    err += bshuf_trans_bit_byte_AVX(out, tmp_buf, size, elem_size);
    err += bshuf_trans_bitrow_eight(tmp_buf, out, size, elem_size);

    free(tmp_buf);

    return err;
}

#else // #ifdef USEAVX2

int bshuf_trans_bit_byte_AVX(void* in, void* out, const size_t size,
         const size_t elem_size) {
    return 12;
}


int bshuf_trans_bit_byte_AVX1(void* in, void* out, const size_t size,
         const size_t elem_size) {
    return 12;
}


int bshuf_trans_bit_elem_AVX(void* in, void* out, const size_t size,
         const size_t elem_size) {
    return 12;
}

#endif // #ifdef USEAVX2


/* ---- Public functions ---- */

int bshuf_trans_bit_elem(void* in, void* out, const size_t size, 
        const size_t elem_size) {

#ifdef USEAVX2
    return bshuf_trans_bit_elem_AVX(in, out, size, elem_size);
#elif defined(USESSE2)
    return bshuf_trans_bit_elem_SSE(in, out, size, elem_size);
#else
    return bshuf_trans_bit_elem_scal(in, out, size, elem_size);
#endif
}


int bshuf_untrans_bit_elem(void* in, void* out, const size_t size, 
        const size_t elem_size) {

#ifdef USEAVX2
    //return bshuf_untrans_bit_elem_AVX(in, out, size, elem_size);
    return bshuf_untrans_bit_elem_SSE(in, out, size, elem_size);
#elif defined(USESSE2)
    return bshuf_untrans_bit_elem_SSE(in, out, size, elem_size);
#else
    return bshuf_untrans_bit_elem_scal(in, out, size, elem_size);
#endif
}

#define BSHUF_NELEM_SHUFF 2048
#define BSHUF_BLOCKED_MULT 128  // Eventuallty 8.

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

int bshuf_bitshuffle(void* in, void* out, const size_t size, 
        const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;

    int err;
    size_t block_size;
    size_t leftover;

    for (size_t ii = 0; ii < size; ii += BSHUF_NELEM_SHUFF) {
        block_size = MIN(size - ii, BSHUF_NELEM_SHUFF);
        block_size = block_size - block_size % BSHUF_BLOCKED_MULT;
        if (block_size) {
            err = bshuf_trans_bit_elem(&A[ii * elem_size],
                    &B[ii * elem_size], block_size, elem_size);
            if (err) return err;
        }
    }
    leftover = size % BSHUF_BLOCKED_MULT;
    memcpy(&B[(size - leftover) * elem_size], &A[(size - leftover) * elem_size],
            leftover * elem_size);

    return 0;
}


int bshuf_bitunshuffle(void* in, void* out, const size_t size, 
        const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;

    int err;
    size_t block_size;
    size_t leftover;

    for (size_t ii = 0; ii < size; ii += BSHUF_NELEM_SHUFF) {
        block_size = MIN(size - ii, BSHUF_NELEM_SHUFF);
        block_size = block_size - block_size % BSHUF_BLOCKED_MULT;
        if (block_size) {
            err = bshuf_untrans_bit_elem(&A[ii * elem_size],
                    &B[ii * elem_size], block_size, elem_size);
            if (err) return err;
        }
    }
    leftover = size % BSHUF_BLOCKED_MULT;
    memcpy(&B[(size - leftover) * elem_size], &A[(size - leftover) * elem_size],
            leftover * elem_size);

    return 0;
}





#undef TRANS_BIT_8X8
#undef TRANS_ELEM_TYPE

#undef USESSE2
#undef USEAVX2
