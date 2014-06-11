#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <xmmintrin.h>
#include <immintrin.h>


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
    for (size_t ii = start; ii < size; ii++) {
        for (size_t jj = 0; jj < elem_size; jj++) {
            B[jj * size + ii] = A[ii * elem_size + jj];
        }
    }
    return 0;
}


/* Transpose bytes within elements, simplest algorithm. */
int bshuf_trans_byte_elem_simple(void* in, void* out, const size_t size,
         const size_t elem_size) {
    return bshuf_trans_byte_elem_remainder(in, out, size, elem_size, 0);
}


/* Transpose bytes within elements using SSE for 16 bit elements. */
int bshuf_trans_byte_elem_SSE_16(void* in, void* out, const size_t size) {

    char* A = (char*) in;
    char* B = (char*) out;
    __m128i a0, b0, a1, b1;

    for (size_t ii=0; ii < size - 15; ii += 16) {
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

    for (size_t ii=0; ii < size - 15; ii += 16) {
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

    for (size_t ii=0; ii < size - 15; ii += 16) {
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


/* Transpose bytes within elements using best algorithm available. */
int bshuf_trans_byte_elem(void* in, void* out, const size_t size,
         const size_t elem_size) {

    int err;
    switch (elem_size) {
        case 1:
            err = bshuf_copy(in, out, size, elem_size);
            break;
        case 2:
            err = bshuf_trans_byte_elem_SSE_16(in, out, size);
            break;
        case 4:
            err = bshuf_trans_byte_elem_SSE_32(in, out, size);
            break;
        case 8:
            err = bshuf_trans_byte_elem_SSE_64(in, out, size);
            break;
        default:
            err = bshuf_trans_byte_elem_simple(in, out, size, elem_size);
    }
    return err;
}


/* Transpose bits within bytes using SSE. */
int bshuf_trans_bit_byte_SSE(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;
    uint16_t* Bui;

    size_t nbytes = elem_size * size;

    __m128i xmm;
    int bt;

    for (size_t ii = 0; ii < nbytes - 15; ii += 16) {
        xmm = _mm_loadu_si128((__m128i *) &A[ii]);
        for (size_t kk = 0; kk < 8; kk++) {
            bt = _mm_movemask_epi8(xmm);
            xmm = _mm_slli_epi16(xmm, 1);
            Bui = (uint16_t*) &B[((7 - kk) * nbytes + ii) / 8];
            *Bui = bt;
        }
    }
    return 0;
}


/* Transpose bits within bytes using AVX. */
int bshuf_trans_bit_byte_AVX(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;
    uint32_t* Bui;

    size_t nbytes = elem_size * size;
    size_t kk;

    __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    int bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7;

    // Turns out that doublly unrolling this loop (unrolling 2 loops of 8) 
    // gives a speed up roughly 70% for some problem sizes.  The compiler will
    // not automatically doublly unroll a loop, but will optimize the
    // order of operations within one long section.
    for (size_t ii = 0; ii < nbytes - 32 * 8 + 1; ii += 32 * 8) {
        ymm0 = _mm256_loadu_si256((__m256i *) &A[ii + 0*32]);
        ymm1 = _mm256_loadu_si256((__m256i *) &A[ii + 1*32]);
        ymm2 = _mm256_loadu_si256((__m256i *) &A[ii + 2*32]);
        ymm3 = _mm256_loadu_si256((__m256i *) &A[ii + 3*32]);
        ymm4 = _mm256_loadu_si256((__m256i *) &A[ii + 4*32]);
        ymm5 = _mm256_loadu_si256((__m256i *) &A[ii + 5*32]);
        ymm6 = _mm256_loadu_si256((__m256i *) &A[ii + 6*32]);
        ymm7 = _mm256_loadu_si256((__m256i *) &A[ii + 7*32]);

        kk = 0;
        Bui = (uint32_t*) &B[((7 - kk) * nbytes + ii) / 8];
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
        Bui = (uint32_t*) &B[((7 - kk) * nbytes + ii) / 8];
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
        Bui = (uint32_t*) &B[((7 - kk) * nbytes + ii) / 8];
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
        Bui = (uint32_t*) &B[((7 - kk) * nbytes + ii) / 8];
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
        Bui = (uint32_t*) &B[((7 - kk) * nbytes + ii) / 8];
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
        Bui = (uint32_t*) &B[((7 - kk) * nbytes + ii) / 8];
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
        Bui = (uint32_t*) &B[((7 - kk) * nbytes + ii) / 8];
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
        Bui = (uint32_t*) &B[((7 - kk) * nbytes + ii) / 8];
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
int bshuf_trans_bit_byte_AVX1(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;
    uint32_t* Bui;

    size_t nbytes = elem_size * size;

    __m256i ymm;
    int bt;

    for (size_t ii = 0; ii < nbytes - 31; ii += 32) {
        ymm = _mm256_loadu_si256((__m256i *) &A[ii]);
        for (size_t kk = 0; kk < 8; kk++) {
            bt = _mm256_movemask_epi8(ymm);
            ymm = _mm256_slli_epi16(ymm, 1);
            Bui = (uint32_t*) &B[((7 - kk) * nbytes + ii) / 8];
            *Bui = bt;
        }
    }
    return 0;
}


/* Transpose rows of shiffled bits (size / 8 bytes) within groups of 8. */
int bshuf_trans_bitrow_eight(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;

    size_t nbytes_bitrow = size / 8;

    for (size_t ii = 0; ii < elem_size; ii++) {
        for (size_t jj = 0; jj < 8; jj++) {
            memcpy((void*) &B[(ii * 8 + jj) * nbytes_bitrow],
                   (void*) &A[(jj * elem_size + ii) * nbytes_bitrow],
                   nbytes_bitrow);
        }
    }
    return 0;
}


/* Tranpose bits within elements. */
int bshuf_trans_bit_elem(void* in, void* out, const size_t size,
         const size_t elem_size) {

    int err;

    void* tmp_buf = malloc(size * elem_size);
    if (tmp_buf == NULL) return 1;

    // Should acctually check errors individually.
    err = bshuf_trans_byte_elem(in, out, size, elem_size);
    err += bshuf_trans_bit_byte_AVX(out, tmp_buf, size, elem_size);
    err += bshuf_trans_bitrow_eight(tmp_buf, out, size, elem_size);

    free(tmp_buf);

    return err;
}


int bshuf_trans_byte_bitrow_sse(void* in, void* out, const size_t size,
         const size_t elem_size) {

    char* A = (char*) in;
    char* B = (char*) out;

    size_t nbyte = size * elem_size;
    size_t nbyte_row = size / 8;

    __m128i a0, b0, c0, d0, e0, f0, g0, h0;
    __m128i a1, b1, c1, d1, e1, f1, g1, h1;

    for (int jj = 0; jj < elem_size; jj++) {
        for (int ii = 0; ii < nbyte_row; ii += 16) {
            a0 = _mm_loadu_si128((__m128i *) &A[(jj * 8 + 0)*nbyte_row + ii]);
            b0 = _mm_loadu_si128((__m128i *) &A[(jj * 8 + 1)*nbyte_row + ii]);
            c0 = _mm_loadu_si128((__m128i *) &A[(jj * 8 + 2)*nbyte_row + ii]);
            d0 = _mm_loadu_si128((__m128i *) &A[(jj * 8 + 3)*nbyte_row + ii]);
            e0 = _mm_loadu_si128((__m128i *) &A[(jj * 8 + 4)*nbyte_row + ii]);
            f0 = _mm_loadu_si128((__m128i *) &A[(jj * 8 + 5)*nbyte_row + ii]);
            g0 = _mm_loadu_si128((__m128i *) &A[(jj * 8 + 6)*nbyte_row + ii]);
            h0 = _mm_loadu_si128((__m128i *) &A[(jj * 8 + 7)*nbyte_row + ii]);


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

            _mm_storeu_si128((__m128i *) &B[(jj * 8 + 0)*nbyte_row + ii], a1);
            _mm_storeu_si128((__m128i *) &B[(jj * 8 + 1)*nbyte_row + ii], b1);
            _mm_storeu_si128((__m128i *) &B[(jj * 8 + 2)*nbyte_row + ii], c1);
            _mm_storeu_si128((__m128i *) &B[(jj * 8 + 3)*nbyte_row + ii], d1);
            _mm_storeu_si128((__m128i *) &B[(jj * 8 + 4)*nbyte_row + ii], e1);
            _mm_storeu_si128((__m128i *) &B[(jj * 8 + 5)*nbyte_row + ii], f1);
            _mm_storeu_si128((__m128i *) &B[(jj * 8 + 6)*nbyte_row + ii], g1);
            _mm_storeu_si128((__m128i *) &B[(jj * 8 + 7)*nbyte_row + ii], h1);

            /*
            _mm_storeu_si128((__m128i *) &B[(ii * 8 + 0) * 16], a1);
            _mm_storeu_si128((__m128i *) &B[(ii * 8 + 1) * 16], b1);
            _mm_storeu_si128((__m128i *) &B[(ii * 8 + 2) * 16], c1);
            _mm_storeu_si128((__m128i *) &B[(ii * 8 + 3) * 16], d1);
            _mm_storeu_si128((__m128i *) &B[(ii * 8 + 4) * 16], e1);
            _mm_storeu_si128((__m128i *) &B[(ii * 8 + 5) * 16], f1);
            _mm_storeu_si128((__m128i *) &B[(ii * 8 + 6) * 16], g1);
            _mm_storeu_si128((__m128i *) &B[(ii * 8 + 7) * 16], h1);
            */
        }
    }
    return 0;
}


/* For data organized into a row for each bit (8 * elem_size rows), transpose
 * the bytes. */
int bshuf_trans_byte_bitrow(void* in, void* out, const size_t size,
         const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;

    size_t nbyte = size * elem_size;
    size_t nbyte_row = size / 8;

    for (int ii = 0; ii < nbyte_row; ii++) {
        for (int jj = 0; jj < 8*elem_size; jj++) {
            B[ii*8*elem_size + jj] = A[jj*nbyte_row + ii];
        }
    }
    return 0;
}


/* Shuffle bits within the bytes of eight element blocks. */
int bshuf_shuffle_bit_eightelem(void* in, void* out, const size_t size,
         const size_t elem_size) {

    // With a bit of care, this could be written such that such that it is
    // in_buf = out_buf safe.
    char* A = (char*) in;
    char* B = (char*) out;
    uint16_t* Bui = (uint16_t*) out;

    size_t nbytes = elem_size * size;

    __m128i xmm;
    int bt;

    if (elem_size == 1) {
        for (size_t ii = 0; ii < nbytes - 15; ii += 16) {
            xmm = _mm_loadu_si128((__m128i *) &A[ii]);
            for (size_t kk = 0; kk < 8; kk++) {
                bt = _mm_movemask_epi8(xmm);
                xmm = _mm_slli_epi16(xmm, 1);
                Bui[ii / 2 + 7 - kk] = bt;
            }
        }

        __m128i a0, b0, a1, b1;
        for (size_t ii = 0; ii < nbytes - 31; ii += 32) {
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
        for (size_t ii = 0; ii < nbytes - 8 * elem_size + 1; ii += 8 * elem_size) {
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


int bshuf_untrans_bit_elem(void* in, void* out, const size_t size,
         const size_t elem_size) {
    int err;
    void* tmp_buf = malloc(size * elem_size);
    if (tmp_buf == NULL) return 1;

    // Should acctually check errors individually.
    err = bshuf_trans_byte_bitrow(in, tmp_buf, size, elem_size);
    err +=  bshuf_shuffle_bit_eightelem(tmp_buf, out, size, elem_size);

    free(tmp_buf);

    return err;
}

