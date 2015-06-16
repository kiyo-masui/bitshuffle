/*
 * Bitshuffle - Filter for improving compression of typed binary data.
 *
 * This file is part of Bitshuffle
 * Author: Kiyoshi Masui <kiyo@physics.ubc.ca>
 * Website: http://www.github.com/kiyo-masui/bitshuffle
 * Created: 2014
 *
 * See LICENSE file for details about copyright and rights to use.
 *
 *
 * Header File
 *
 * Worker routines return an int64_t which is the number of bytes processed
 * if positive or an error code if negative.
 *
 * Error codes:
 *      -1    : Failed to allocate memory.
 *      -11   : Missing SSE.
 *      -12   : Missing AVX.
 *      -80   : Input size not a multiple of 8.
 *      -81   : block_size not multiple of 8.
 *      -91   : Decompression error, wrong number of bytes processed.
 *      -1YYY : Error internal to compression routine with error code -YYY.
 */


#ifndef BITSHUFFLE_H
#define BITSHUFFLE_H


#include <stdint.h>
#include <stdlib.h>


// These are usually set in the setup.py.
#ifndef BSHUF_VERSION_MAJOR
#define BSHUF_VERSION_MAJOR 0
#define BSHUF_VERSION_MINOR 2
#define BSHUF_VERSION_POINT 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --- bshuf_using_SSE2 ----
 *
 * Whether routines where compiled with the SSE2 instruction set.
 *
 * Returns
 * -------
 *  1 if using SSE2, 0 otherwise.
 *
 */
int bshuf_using_SSE2(void);


/* ---- bshuf_using_AVX2 ----
 *
 * Whether routines where compiled with the AVX2 instruction set.
 *
 * Returns
 * -------
 *  1 if using AVX2, 0 otherwise.
 *
 */
int bshuf_using_AVX2(void);


/* ---- bshuf_default_block_size ----
 *
 * The default block size as function of element size.
 *
 * This is the block size used by the blocked routines (any routine
 * taking a *block_size* argument) when the block_size is not provided
 * (zero is passed).
 *
 * The results of this routine are guaranteed to be stable such that
 * shuffled/compressed data can always be decompressed.
 *
 * Parameters
 * ----------
 *  elem_size : element size of data to be shuffled/compressed.
 *
 */
size_t bshuf_default_block_size(const size_t elem_size);


/* ---- bshuf_bitshuffle ----
 *
 * Bitshuffle the data.
 *
 * Transpose the bits within elements, in blocks of *block_size*
 * elements.
 *
 * Parameters
 * ----------
 *  in : input buffer, must be of size * elem_size bytes
 *  out : output buffer, must be of size * elem_size bytes
 *  size : number of elements in input
 *  elem_size : element size of typed data
 *  block_size : Do transpose in blocks of this many elements. Pass 0 to
 *  select automatically (recommended).
 *
 * Returns
 * -------
 *  number of bytes processed, negative error-code if failed.
 *
 */
int64_t bshuf_bitshuffle(void* in, void* out, const size_t size,
        const size_t elem_size, size_t block_size);


/* ---- bshuf_bitunshuffle ----
 *
 * Unshuffle bitshuffled data.
 *
 * Untranspose the bits within elements, in blocks of *block_size*
 * elements.
 *
 * To properly unshuffle bitshuffled data, *size*, *elem_size* and *block_size*
 * must match the parameters used to shuffle the data.
 *
 * Parameters
 * ----------
 *  in : input buffer, must be of size * elem_size bytes
 *  out : output buffer, must be of size * elem_size bytes
 *  size : number of elements in input
 *  elem_size : element size of typed data
 *  block_size : Do transpose in blocks of this many elements. Pass 0 to
 *  select automatically (recommended).
 *
 * Returns
 * -------
 *  number of bytes processed, negative error-code if failed.
 *
 */
int64_t bshuf_bitunshuffle(void* in, void* out, const size_t size,
        const size_t elem_size, size_t block_size);


/* ---- bshuf_compress_lz4_bound ----
 *
 * Bound on size of data compressed with *bshuf_compress_lz4*.
 *
 * Parameters
 * ----------
 *  size : number of elements in input
 *  elem_size : element size of typed data
 *  block_size : Process in blocks of this many elements. Pass 0 to
 *  select automatically (recommended).
 *
 * Returns
 * -------
 *  Bound on compressed data size.
 *
 */
size_t bshuf_compress_lz4_bound(const size_t size,
        const size_t elem_size, size_t block_size);


/* ---- bshuf_compress_lz4 ----
 *
 * Bitshuffled and compress the data using LZ4.
 *
 * Transpose within elements, in blocks of data of *block_size* elements then
 * compress the blocks using LZ4.  In the output buffer, each block is prefixed
 * by a 4 byte integer giving the compressed size of that block.
 *
 * Output buffer must be large enough to hold the compressed data.  This could
 * be in principle substantially larger than the input buffer.  Use the routine
 * *bshuf_compress_lz4_bound* to get an upper limit.
 *
 * Parameters
 * ----------
 *  in : input buffer, must be of size * elem_size bytes 
 *  out : output buffer, must be large enough to hold data.
 *  size : number of elements in input
 *  elem_size : element size of typed data
 *  block_size : Process in blocks of this many elements. Pass 0 to
 *  select automatically (recommended).
 *
 * Returns
 * -------
 *  number of bytes used in output buffer, negative error-code if failed.
 *
 */
int64_t bshuf_compress_lz4(void* in, void* out, const size_t size, const size_t
        elem_size, size_t block_size);


/* ---- bshuf_decompress_lz4 ----
 *
 * Undo compression and bitshuffling.
 *
 * Decompress data then un-bitshuffle it in blocks of *block_size* elements.
 *
 * To properly unshuffle bitshuffled data, *size*, *elem_size* and *block_size*
 * must patch the parameters used to compress the data.
 *
 * NOT TO BE USED WITH UNTRUSTED DATA: This routine uses the function
 * LZ4_decompress_fast from LZ4, which does not protect against maliciously
 * formed datasets. By modifying the compressed data, this function could be
 * coerced into leaving the boundaries of the input buffer.
 *
 * Parameters
 * ----------
 *  in : input buffer
 *  out : output buffer, must be of size * elem_size bytes
 *  size : number of elements in input
 *  elem_size : element size of typed data
 *  block_size : Process in blocks of this many elements. Pass 0 to
 *  select automatically (recommended).
 *
 * Returns
 * -------
 *  number of bytes consumed in *input* buffer, negative error-code if failed.
 *
 */
int64_t bshuf_decompress_lz4(void* in, void* out, const size_t size,
        const size_t elem_size, size_t block_size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif  // BITSHUFFLE_H
