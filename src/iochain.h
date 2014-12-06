/*
 * Parallelization construct for distributing a chain of dependant IO events.
 *
 * Similar in concept to a queue. Each task includes reading an input
 * and writing output, but the location of the input/output (the pointers)
 * depend on the previous item in the chain.
 *
 * This is designed for parallelizing blocked compression/decompression IO,
 * where the destination of a compressed block depends on the compressed size
 * of all previous blocks.
 *
 * Implemented with OpenMP locks.
 *
 *
 * Usage
 * -----
 *  - Call `ioc_init` in serial block.
 *  - Each thread should create a local variable *size_t this_iter* and 
 *    pass its address to all function calls. Its value will be set
 *    inside the functions and is used to identify the thread.
 *  - Each thread must call each of the `ioc_get*` and `ioc_set*` methods
 *    exactly once per iteration, starting with `ioc_get_in` and ending
 *    with `ioc_set_next_out`.
 *  - The order (`ioc_get_in`, `ioc_set_next_in`, *work*, `ioc_get_out`,
 *    `ioc_set_next_out`, *work*) is most efficient.
 *  - Have each thread call `ioc_end_pop`.
 *  - `ioc_get_in` is blocked until the previous entry's
 *    `ioc_set_next_in` is called.
 *  - `ioc_get_out` is blocked until the previous entry's
 *    `ioc_set_next_out` is called.
 *  - There are no blocks on the very first iteration.
 *  - Call `ioc_destroy` in serial block.
 *  - Safe for num_threads >= IOC_SIZE (but less efficient).
 *
 */


#ifndef IOCHAIN_H
#define IOCHAIN_H


#include <stdlib.h>
#include <omp.h>


#define IOC_SIZE 33


typedef struct ioc_ptr_and_lock {
    omp_lock_t lock;
    void *ptr;
} ptr_and_lock;


typedef struct ioc_chain {
    omp_lock_t next_lock;
    size_t next;
    ptr_and_lock in_pl[IOC_SIZE];
    ptr_and_lock out_pl[IOC_SIZE];
} ioc_chain;


void ioc_init(ioc_chain *C, void *in_ptr_0, void *out_ptr_0) {
    omp_init_lock(&C->next_lock);
    for (size_t ii = 0; ii < IOC_SIZE; ii ++) {
        omp_init_lock(&(C->in_pl[ii].lock));
        omp_init_lock(&(C->out_pl[ii].lock));
    }
    C->next = 0;
    C->in_pl[0].ptr = in_ptr_0;
    C->out_pl[0].ptr = out_ptr_0;
}


void ioc_destroy(ioc_chain *C) {
    omp_destroy_lock(&C->next_lock);
    for (size_t ii = 0; ii < IOC_SIZE; ii ++) {
        omp_destroy_lock(&(C->in_pl[ii].lock));
        omp_destroy_lock(&(C->out_pl[ii].lock));
    }
}


void * ioc_get_in(ioc_chain *C, size_t *this_iter) {
    omp_set_lock(&C->next_lock);
    #pragma omp flush
    *this_iter = C->next;
    C->next ++;
    omp_set_lock(&(C->in_pl[*this_iter % IOC_SIZE].lock));
    omp_set_lock(&(C->in_pl[(*this_iter + 1) % IOC_SIZE].lock));
    omp_set_lock(&(C->out_pl[(*this_iter + 1) % IOC_SIZE].lock));
    omp_unset_lock(&C->next_lock);
    return C->in_pl[*this_iter % IOC_SIZE].ptr;
}


void ioc_set_next_in(ioc_chain *C, size_t* this_iter, void* in_ptr) {
    C->in_pl[(*this_iter + 1) % IOC_SIZE].ptr = in_ptr;
    omp_unset_lock(&(C->in_pl[(*this_iter + 1) % IOC_SIZE].lock));
}


void * ioc_get_out(ioc_chain *C, size_t *this_iter) {
    omp_set_lock(&(C->out_pl[(*this_iter) % IOC_SIZE].lock));
    #pragma omp flush
    void *out_ptr = C->out_pl[*this_iter % IOC_SIZE].ptr;
    omp_unset_lock(&(C->out_pl[(*this_iter) % IOC_SIZE].lock));
    return out_ptr;
}


void ioc_set_next_out(ioc_chain *C, size_t *this_iter, void* out_ptr) {
    C->out_pl[(*this_iter + 1) % IOC_SIZE].ptr = out_ptr;
    omp_unset_lock(&(C->out_pl[(*this_iter + 1) % IOC_SIZE].lock));
    // *in_pl[this_iter]* lock released at the end of the iteration to avoid being
    // overtaken by previous threads and having *out_pl[this_iter]* corrupted.
    // Especially worried about thread 0, iteration 0.
    omp_unset_lock(&(C->in_pl[(*this_iter) % IOC_SIZE].lock));
}


#endif  // IOCHAIN_H

