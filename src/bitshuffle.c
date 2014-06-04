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
    for (int i=0; i < size; i++) {
        for (int j=0; j < elem_size; j++) {
            B[j * size + i] = A[i * elem_size + j];
        }
    }
    return 0;
}


