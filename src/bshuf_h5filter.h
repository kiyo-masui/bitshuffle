

#ifndef BSHUF_H5FILTER_H
#define BSHUF_H5FILTER_H

#define H5Z_class_t_vers 2
#include "hdf5.h"

#include "bitshuffle.h"


#define BSHUF_H5FILTER 32008


#define BSHUF_H5FILTER_VERSION 0


#define BSHUF_H5_COMPRESS_LZ4 2


H5Z_class_t bshuf_H5Filter[1];


int bshuf_register_h5filter(void);


#endif // BSHUF_H5FILTER_H
