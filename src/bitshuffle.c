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


// Transpose bytes within elements.
int shuff_byte_T_elem_simple(void* in, void* out, const size_t size,
         const size_t elem_size) {
    return shuff_byte_T_elem_remainder(in, out, size, elem_size, 0);
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


// Transpose bits within bytes.
int shuff_bit_T_byte_avx2(void* in, void* out, const size_t size,
         const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;
    uint32_t* Bui;

    size_t nbytes = elem_size * size;

    __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    int bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7;

    for (size_t ii = 0; ii < nbytes; ii += 32 * 8) {
        ymm0 = _mm256_loadu_si256((__m256i *) &A[ii + 0*32]);
        ymm1 = _mm256_loadu_si256((__m256i *) &A[ii + 1*32]);
        ymm2 = _mm256_loadu_si256((__m256i *) &A[ii + 2*32]);
        ymm3 = _mm256_loadu_si256((__m256i *) &A[ii + 3*32]);
        ymm4 = _mm256_loadu_si256((__m256i *) &A[ii + 4*32]);
        ymm5 = _mm256_loadu_si256((__m256i *) &A[ii + 5*32]);
        ymm6 = _mm256_loadu_si256((__m256i *) &A[ii + 6*32]);
        ymm7 = _mm256_loadu_si256((__m256i *) &A[ii + 7*32]);

        // Bit start.
        Bui = (uint32_t*) &B[((7 - 0) * nbytes + ii) / 8];

        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);

        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);

        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;

        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);

        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);

        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        // Bit start.
        Bui = (uint32_t*) &B[((7 - 1) * nbytes + ii) / 8];

        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);

        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);

        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;

        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);

        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);

        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        // Bit start.
        Bui = (uint32_t*) &B[((7 - 2) * nbytes + ii) / 8];

        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);

        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);

        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;

        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);

        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);

        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        // Bit start.
        Bui = (uint32_t*) &B[((7 - 3) * nbytes + ii) / 8];

        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);

        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);

        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;

        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);

        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);

        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        // Bit start.
        Bui = (uint32_t*) &B[((7 - 4) * nbytes + ii) / 8];

        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);

        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);

        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;

        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);

        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);

        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        // Bit start.
        Bui = (uint32_t*) &B[((7 - 5) * nbytes + ii) / 8];

        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);

        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);

        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;

        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);

        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);

        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        // Bit start.
        Bui = (uint32_t*) &B[((7 - 6) * nbytes + ii) / 8];

        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);

        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);

        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;

        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);

        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);

        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

        // Bit start.
        Bui = (uint32_t*) &B[((7 - 7) * nbytes + ii) / 8];

        bt0 = _mm256_movemask_epi8(ymm0);
        bt1 = _mm256_movemask_epi8(ymm1);
        bt2 = _mm256_movemask_epi8(ymm2);
        bt3 = _mm256_movemask_epi8(ymm3);

        ymm0 = _mm256_slli_epi16(ymm0, 1);
        ymm1 = _mm256_slli_epi16(ymm1, 1);
        ymm2 = _mm256_slli_epi16(ymm2, 1);
        ymm3 = _mm256_slli_epi16(ymm3, 1);

        Bui[0] = bt0;
        Bui[1] = bt1;
        Bui[2] = bt2;
        Bui[3] = bt3;

        bt4 = _mm256_movemask_epi8(ymm4);
        bt5 = _mm256_movemask_epi8(ymm5);
        bt6 = _mm256_movemask_epi8(ymm6);
        bt7 = _mm256_movemask_epi8(ymm7);

        ymm4 = _mm256_slli_epi16(ymm4, 1);
        ymm5 = _mm256_slli_epi16(ymm5, 1);
        ymm6 = _mm256_slli_epi16(ymm6, 1);
        ymm7 = _mm256_slli_epi16(ymm7, 1);

        Bui[4] = bt4;
        Bui[5] = bt5;
        Bui[6] = bt6;
        Bui[7] = bt7;

    }
    return 0;
}


// Transpose bits within bytes.
int shuff_bit_T_byte_avx(void* in, void* out, const size_t size,
         const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;
    uint32_t* Bui;

    size_t nbytes = elem_size * size;
    size_t kk;

    __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    int bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7;

    for (size_t ii = 0; ii < nbytes; ii += 32 * 8) {
        ymm0 = _mm256_loadu_si256((__m256i *) &A[ii + 0*32]);
        ymm1 = _mm256_loadu_si256((__m256i *) &A[ii + 1*32]);
        ymm2 = _mm256_loadu_si256((__m256i *) &A[ii + 2*32]);
        ymm3 = _mm256_loadu_si256((__m256i *) &A[ii + 3*32]);
        ymm4 = _mm256_loadu_si256((__m256i *) &A[ii + 4*32]);
        ymm5 = _mm256_loadu_si256((__m256i *) &A[ii + 5*32]);
        ymm6 = _mm256_loadu_si256((__m256i *) &A[ii + 6*32]);
        ymm7 = _mm256_loadu_si256((__m256i *) &A[ii + 7*32]);

        // Bit start.
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

        // Bit start.
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

        // Bit start.
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

        // Bit start.
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

        // Bit start.
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

        // Bit start.
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

        // Bit start.
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

        // Bit start.
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


// Transpose bits within bytes.
int shuff_bit_T_byte_avx1(void* in, void* out, const size_t size,
         const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;
    uint32_t* Bui;

    size_t nbytes = elem_size * size;

    __m256i ymm;
    int bt;

    for (size_t ii = 0; ii < nbytes; ii += 32) {
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


