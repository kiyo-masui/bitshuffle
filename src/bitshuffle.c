/*
 * Bitshuffle - Filter for improving compression of typed binary data.
 *
 * Author: Kiyoshi Masui <kiyo@physics.ubc.ca>
 * Website: http://www.github.com/kiyo-masui/bitshuffle
 * Created: 2014
 *
 * See LICENSE file for details about copyright and rights to use.
 *
 */

#include "bitshuffle.h"
#include "bitshuffle_core.h"
#include "bitshuffle_internals.h"
#include "lz4.h"

#ifdef ZSTD_SUPPORT
#include "zstd.h"
#endif

#include <stdio.h>
#include <string.h>


// Macros.
#define CHECK_ERR_FREE_LZ(count, buf) if (count < 0) {                      \
    free(buf); return count - 1000; }


/* Bitshuffle and compress a single block. */
int64_t bshuf_compress_lz4_block(ioc_chain *C_ptr, \
        const size_t size, const size_t elem_size, const int option) {

    int64_t nbytes, count;
    void *tmp_buf_bshuf;
    void *tmp_buf_lz4;
    size_t this_iter;
    const void *in;
    void *out;

    tmp_buf_bshuf = malloc(size * elem_size);
    if (tmp_buf_bshuf == NULL) return -1;

    int dst_capacity = LZ4_compressBound(size * elem_size);
    tmp_buf_lz4 = malloc(dst_capacity);
    if (tmp_buf_lz4 == NULL){
        free(tmp_buf_bshuf);
        return -1;
    }


    in = ioc_get_in(C_ptr, &this_iter);
    ioc_set_next_in(C_ptr, &this_iter, (void*) ((char*) in + size * elem_size));

    count = bshuf_trans_bit_elem(in, tmp_buf_bshuf, size, elem_size);
    if (count < 0) {
        free(tmp_buf_lz4);
        free(tmp_buf_bshuf);
        return count;
    }
    nbytes = LZ4_compress_default((const char*) tmp_buf_bshuf, (char*) tmp_buf_lz4, size * elem_size, dst_capacity);
    free(tmp_buf_bshuf);
    CHECK_ERR_FREE_LZ(nbytes, tmp_buf_lz4);

    out = ioc_get_out(C_ptr, &this_iter);
    ioc_set_next_out(C_ptr, &this_iter, (void *) ((char *) out + nbytes + 4));

    bshuf_write_uint32_BE(out, nbytes);
    memcpy((char *) out + 4, tmp_buf_lz4, nbytes);

    free(tmp_buf_lz4);

    return nbytes + 4;
}


/* Decompress and bitunshuffle a single block. */
int64_t bshuf_decompress_lz4_block(ioc_chain *C_ptr,
        const size_t size, const size_t elem_size, const int option) {

    int64_t nbytes, count;
    void *out, *tmp_buf;
    const void *in;
    size_t this_iter;
    int32_t nbytes_from_header;

    in = ioc_get_in(C_ptr, &this_iter);
    nbytes_from_header = bshuf_read_uint32_BE(in);
    ioc_set_next_in(C_ptr, &this_iter,
            (void*) ((char*) in + nbytes_from_header + 4));

    out = ioc_get_out(C_ptr, &this_iter);
    ioc_set_next_out(C_ptr, &this_iter,
            (void *) ((char *) out + size * elem_size));

    tmp_buf = malloc(size * elem_size);
    if (tmp_buf == NULL) return -1;

    nbytes = LZ4_decompress_safe((const char*) in + 4, (char *) tmp_buf, nbytes_from_header,
                                 size * elem_size);
    CHECK_ERR_FREE_LZ(nbytes, tmp_buf);
    if (nbytes != size * elem_size) {
        free(tmp_buf);
        return -91;
    }
    nbytes = nbytes_from_header;

    count = bshuf_untrans_bit_elem(tmp_buf, out, size, elem_size);
    CHECK_ERR_FREE(count, tmp_buf);
    nbytes += 4;

    free(tmp_buf);
    return nbytes;
}

#ifdef ZSTD_SUPPORT
/* Bitshuffle and compress a single block. */
int64_t bshuf_compress_zstd_block(ioc_chain *C_ptr, \
        const size_t size, const size_t elem_size, const int comp_lvl) {

    int64_t nbytes, count;
    void *tmp_buf_bshuf;
    void *tmp_buf_zstd;
    size_t this_iter;
    const void *in;
    void *out;

    tmp_buf_bshuf = malloc(size * elem_size);
    if (tmp_buf_bshuf == NULL) return -1;

    size_t tmp_buf_zstd_size = ZSTD_compressBound(size * elem_size);
    tmp_buf_zstd = malloc(tmp_buf_zstd_size);
    if (tmp_buf_zstd == NULL){
        free(tmp_buf_bshuf);
        return -1;
    }

    in = ioc_get_in(C_ptr, &this_iter);
    ioc_set_next_in(C_ptr, &this_iter, (void*) ((char*) in + size * elem_size));

    count = bshuf_trans_bit_elem(in, tmp_buf_bshuf, size, elem_size);
    if (count < 0) {
        free(tmp_buf_zstd);
        free(tmp_buf_bshuf);
        return count;
    }
    nbytes = ZSTD_compress(tmp_buf_zstd, tmp_buf_zstd_size, (const void*)tmp_buf_bshuf,  size * elem_size, comp_lvl);
    free(tmp_buf_bshuf);
    CHECK_ERR_FREE_LZ(nbytes, tmp_buf_zstd);

    out = ioc_get_out(C_ptr, &this_iter);
    ioc_set_next_out(C_ptr, &this_iter, (void *) ((char *) out + nbytes + 4));

    bshuf_write_uint32_BE(out, nbytes);
    memcpy((char *) out + 4, tmp_buf_zstd, nbytes);

    free(tmp_buf_zstd);

    return nbytes + 4;
}


/* Decompress and bitunshuffle a single block. */
int64_t bshuf_decompress_zstd_block(ioc_chain *C_ptr,
        const size_t size, const size_t elem_size, const int option) {

    int64_t nbytes, count;
    void *out, *tmp_buf;
    const void *in;
    size_t this_iter;
    int32_t nbytes_from_header;

    in = ioc_get_in(C_ptr, &this_iter);
    nbytes_from_header = bshuf_read_uint32_BE(in);
    ioc_set_next_in(C_ptr, &this_iter,
            (void*) ((char*) in + nbytes_from_header + 4));

    out = ioc_get_out(C_ptr, &this_iter);
    ioc_set_next_out(C_ptr, &this_iter,
            (void *) ((char *) out + size * elem_size));

    tmp_buf = malloc(size * elem_size);
    if (tmp_buf == NULL) return -1;

    nbytes = ZSTD_decompress(tmp_buf, size * elem_size, in + 4, nbytes_from_header);
    CHECK_ERR_FREE_LZ(nbytes, tmp_buf);
    if (nbytes != size * elem_size) {
        free(tmp_buf);
        return -91;
    }

    nbytes = nbytes_from_header;
    count = bshuf_untrans_bit_elem(tmp_buf, out, size, elem_size);
    CHECK_ERR_FREE(count, tmp_buf);
    nbytes += 4;

    free(tmp_buf);
    return nbytes;
}
#endif // ZSTD_SUPPORT


/* ---- Public functions ----
 *
 * See header file for description and usage.
 *
 */

size_t bshuf_compress_lz4_bound(const size_t size,
        const size_t elem_size, size_t block_size) {

    size_t bound, leftover;

    if (block_size == 0) {
        block_size = bshuf_default_block_size(elem_size);
    }
    if (block_size % BSHUF_BLOCKED_MULT) return -81;

    // Note that each block gets a 4 byte header.
    // Size of full blocks.
    bound = (LZ4_compressBound(block_size * elem_size) + 4) * (size / block_size);
    // Size of partial blocks, if any.
    leftover = ((size % block_size) / BSHUF_BLOCKED_MULT) * BSHUF_BLOCKED_MULT;
    if (leftover) bound += LZ4_compressBound(leftover * elem_size) + 4;
    // Size of uncompressed data not fitting into any blocks.
    bound += (size % BSHUF_BLOCKED_MULT) * elem_size;
    return bound;
}


int64_t bshuf_compress_lz4(const void* in, void* out, const size_t size,
        const size_t elem_size, size_t block_size) {
    return bshuf_blocked_wrap_fun(&bshuf_compress_lz4_block, in, out, size,
            elem_size, block_size, 0/*option*/);
}


int64_t bshuf_decompress_lz4(const void* in, void* out, const size_t size,
        const size_t elem_size, size_t block_size) {
    return bshuf_blocked_wrap_fun(&bshuf_decompress_lz4_block, in, out, size,
            elem_size, block_size, 0/*option*/);
}

#ifdef ZSTD_SUPPORT
size_t bshuf_compress_zstd_bound(const size_t size,
        const size_t elem_size, size_t block_size) {

    size_t bound, leftover;

    if (block_size == 0) {
        block_size = bshuf_default_block_size(elem_size);
    }
    if (block_size % BSHUF_BLOCKED_MULT) return -81;

    // Note that each block gets a 4 byte header.
    // Size of full blocks.
    bound = (ZSTD_compressBound(block_size * elem_size) + 4) * (size / block_size);
    // Size of partial blocks, if any.
    leftover = ((size % block_size) / BSHUF_BLOCKED_MULT) * BSHUF_BLOCKED_MULT;
    if (leftover) bound += ZSTD_compressBound(leftover * elem_size) + 4;
    // Size of uncompressed data not fitting into any blocks.
    bound += (size % BSHUF_BLOCKED_MULT) * elem_size;
    return bound;
}


int64_t bshuf_compress_zstd(const void* in, void* out, const size_t size,
        const size_t elem_size, size_t block_size, const int comp_lvl) {
    return bshuf_blocked_wrap_fun(&bshuf_compress_zstd_block, in, out, size,
            elem_size, block_size, comp_lvl);
}


int64_t bshuf_decompress_zstd(const void* in, void* out, const size_t size,
        const size_t elem_size, size_t block_size) {
    return bshuf_blocked_wrap_fun(&bshuf_decompress_zstd_block, in, out, size,
            elem_size, block_size, 0/*option*/);
}
#endif // ZSTD_SUPPORT
