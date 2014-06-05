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
int shuff_bit_T_byte0(void* in, void* out, const size_t size,
         const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;
    size_t nbytes = elem_size * size;

    __m128i this_xmm;
    int byte_trans;

    /*  Won't compile because the _mm_insert_epi16 needs a constant jj.
    __m128i buf_8x128[8];
    for (size_t ii = 0; ii < nbytes; ii += 128) {
        for (size_t jj = 0; jj < 8; jj++) {
            this_xmm = _mm_loadu_si128((__m128i *) &A[ii + 16 * jj]);
            for (size_t kk = 0; kk < 8; kk++) {
                byte_trans = _mm_movemask_epi8(this_xmm);
                this_xmm = _mm_slli_epi16(this_xmm, 1);
                buf_8x128[kk] = _mm_insert_epi16(buf_8x128[kk], byte_trans, jj);
            }
        }
        for (size_t kk = 0; kk < 8; kk++) {
            _mm_storeu_si128((__m128i *) &B[(kk * nbytes + ii) / 8], buf_8x128[kk]);
        }
    }
    */

    union { __m128i x[8]; uint16_t ui[64]; } buf_8x128;

    for (size_t ii = 0; ii < nbytes; ii += 128) {
        for (size_t jj = 0; jj < 8; jj++) {
            this_xmm = _mm_loadu_si128((__m128i *) &A[ii + 16 * jj]);
            for (size_t kk = 0; kk < 8; kk++) {
                byte_trans = _mm_movemask_epi8(this_xmm);
                this_xmm = _mm_slli_epi16(this_xmm, 1);
                buf_8x128.ui[(7 - kk) * 8 + jj] = byte_trans;
            }
        }
        for (size_t kk = 0; kk < 8; kk++) {
            _mm_storeu_si128((__m128i *) &B[(kk * nbytes + ii) / 8], buf_8x128.x[kk]);
        }
    }

    return 0;
}


// Transpose bits within bytes.
int shuff_bit_T_byte1(void* in, void* out, const size_t size,
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
int shuff_bit_T_byte2(void* in, void* out, const size_t size,
         const size_t elem_size) {
    char* A = (char*) in;
    char* B = (char*) out;
    size_t nbytes = elem_size * size;
    uint16_t* Bui;

    __m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
    int bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7;

    //union { __m128i x[8]; uint16_t ui[64]; } buf_8x128;

    for (size_t ii = 0; ii < nbytes; ii += 128) {
        //for (size_t jj = 0; jj < 8; jj++) {
            xmm0 = _mm_loadu_si128((__m128i *) &A[ii + 16 * 0]);
            xmm1 = _mm_loadu_si128((__m128i *) &A[ii + 16 * 1]);
            xmm2 = _mm_loadu_si128((__m128i *) &A[ii + 16 * 2]);
            xmm3 = _mm_loadu_si128((__m128i *) &A[ii + 16 * 3]);
            xmm4 = _mm_loadu_si128((__m128i *) &A[ii + 16 * 4]);
            xmm5 = _mm_loadu_si128((__m128i *) &A[ii + 16 * 5]);
            xmm6 = _mm_loadu_si128((__m128i *) &A[ii + 16 * 6]);
            xmm7 = _mm_loadu_si128((__m128i *) &A[ii + 16 * 7]);
            for (size_t kk = 0; kk < 8; kk++) {
                bt0 = _mm_movemask_epi8(xmm0);
                bt1 = _mm_movemask_epi8(xmm1);
                bt2 = _mm_movemask_epi8(xmm2);
                bt3 = _mm_movemask_epi8(xmm3);
                bt4 = _mm_movemask_epi8(xmm4);
                bt5 = _mm_movemask_epi8(xmm5);
                bt6 = _mm_movemask_epi8(xmm6);
                bt7 = _mm_movemask_epi8(xmm7);

                xmm0 = _mm_slli_epi16(xmm0, 1);
                xmm1 = _mm_slli_epi16(xmm1, 1);
                xmm2 = _mm_slli_epi16(xmm2, 1);
                xmm3 = _mm_slli_epi16(xmm3, 1);
                xmm4 = _mm_slli_epi16(xmm4, 1);
                xmm5 = _mm_slli_epi16(xmm5, 1);
                xmm6 = _mm_slli_epi16(xmm6, 1);
                xmm7 = _mm_slli_epi16(xmm7, 1);

                //buf_8x128.x[7 - kk] = _mm_insert_epi16(buf_8x128.x[7 - kk], bt0, 0);
                //buf_8x128.ui[(7 - kk) * 8 + 0] = bt0;

                //buf_8x128.x[7 - kk] = _mm_set_epi16(bt7, bt6, bt5, bt4,
                //                                    bt3, bt2, bt1, bt0);

                //_mm_storeu_si128((__m128i *) &B[((7 - kk) * nbytes + ii) / 8],
                //        _mm_set_epi16(bt7, bt6, bt5, bt4, bt3, bt2, bt1, bt0));

                
                Bui = (uint16_t*) &B[((7 - kk) * nbytes + ii) / 8];
                Bui[0] = bt0;
                Bui[1] = bt1;
                Bui[2] = bt2;
                Bui[3] = bt3;
                Bui[4] = bt4;
                Bui[5] = bt5;
                Bui[6] = bt6;
                Bui[7] = bt7;
                
            }
        //}
        /*
        for (size_t kk = 0; kk < 8; kk++) {
            _mm_storeu_si128((__m128i *) &B[(kk * nbytes + ii) / 8], buf_8x128.x[kk]);
        }
        */
    }

    return 0;
}
