/* Error codes:
 *      11 : Missing SSE
 *      12 : Missing AVX
 *      80 : Input not a multiple of 8.
 */


#ifndef BITSHUFFLE_H
#define BITSHUFFLE_H


#if defined(__AVX2__) && defined (__SSE2__)
#define USEAVX2
#endif

#if defined(__SSE2__)
#define USESSE2
#endif


#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>

// Conditional includes for SSE2 and AVX2.
#ifdef USEAVX2
#include <immintrin.h>
#elif defined USESSE2
#include <emmintrin.h>
#endif


#define BSHUF_VERSION 0


int bshuf_using_SSE2(void);


int bshuf_using_AVX2(void);


int bshuf_bitshuffle(void* in, void* out, const size_t size, 
        const size_t elem_size);


int bshuf_bitunshuffle(void* in, void* out, const size_t size, 
        const size_t elem_size);


#endif  // BITSHUFFLE
