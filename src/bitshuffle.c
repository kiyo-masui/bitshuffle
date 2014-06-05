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
    char* A = (char*) in;
    char* B = (char*) out;
    for (size_t ii = 0; ii < size; ii++) {
        for (size_t jj = 0; jj < elem_size; jj++) {
            B[jj * size + ii] = A[ii * elem_size + jj];
        }
    }
    return 0;
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

