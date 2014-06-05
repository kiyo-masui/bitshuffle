#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <xmmintrin.h>
#include <immintrin.h>


// Memory copy for testing and profiling.
int shuff_just_copy(void* in, void* out, const size_t size,
         const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;
    memcpy(B, A, size * elem_size);
    return 0;
}


// Transpose bytes within elements.
int shuff_byte_T_elem_simple(void* in, void* out, const size_t size,
         const size_t elem_size) {
    return shuff_byte_T_elem_remainder(in, out, size, elem_size, 0);
}


// Simple transpose starting a arbitrary location in array.
int shuff_byte_T_elem_remainder(void* in, void* out, const size_t size,
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


int shuff_byte_T_elem_SSE_16(void* in, void* out, const size_t size) {

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
    return shuff_byte_T_elem_remainder(in, out, size, 2,
            size - size % 16);
}


int shuff_byte_T_elem_SSE_32(void* in, void* out, const size_t size) {

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
    return shuff_byte_T_elem_remainder(in, out, size, 4,
            size - size % 16);
}


// Transpose bytes within elements.
int shuff_byte_T_elem_fast(void* in, void* out, const size_t size,
         const size_t elem_size) {

    int err;
    switch (elem_size) {
        case 1:
            err = shuff_just_copy(in, out, size, elem_size);
            break;
        case 2:
            err = shuff_byte_T_elem_SSE_16(in, out, size);
            break;
        case 4:
            err = shuff_byte_T_elem_SSE_32(in, out, size);
            break;
        default:
            err = shuff_byte_T_elem_simple(in, out, size, elem_size);
    }
    return err;
}


// Transpose bits within bytes.
int shuff_bit_T_byte(void* in, void* out, const size_t size,
         const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;
    uint16_t* Bui;

    size_t nbytes = elem_size * size;

    __m128i xmm;
    int bt;

    for (size_t ii = 0; ii < nbytes; ii += 16) {
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

