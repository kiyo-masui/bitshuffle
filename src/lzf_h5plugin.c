/*
 * Dynamically loaded filter plugin for HDF5 LZF filter.
 *
 * This file is part of Bitshuffle
 * Author: Kiyoshi Masui <kiyo@physics.ubc.ca>
 * Website: https://www.github.com/kiyo-masui/bitshuffle
 * Created: 2014
 *
 * See LICENSE file for details about copyright and rights to use.
 *
 */


#define H5Z_class_t_vers 2
#include "lzf_filter.h"
#include "H5PLextern.h"

#include <stdint.h>


size_t lzf_filter(unsigned flags, size_t cd_nelmts,
                  const unsigned cd_values[], size_t nbytes,
                  size_t *buf_size, void **buf);


herr_t lzf_set_local(hid_t dcpl, hid_t type, hid_t space);


H5Z_class_t lzf_H5Filter[1] = {{
    H5Z_CLASS_T_VERS,
    (H5Z_filter_t)(H5PY_FILTER_LZF),
    1, 1,
    "lzf",
    NULL,
    (H5Z_set_local_func_t)(lzf_set_local),
    (H5Z_func_t)(lzf_filter)
}};
