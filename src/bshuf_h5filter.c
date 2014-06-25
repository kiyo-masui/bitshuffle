#include "bshuf_h5filter.h"


// Only called on compresion, not on reverse.
herr_t bshuf_h5_set_local(hid_t dcpl, hid_t type, hid_t space){

    herr_t r;

    unsigned int elem_size;

    unsigned int flags;
    size_t nelements = 8;
    size_t nelem_max = 11;
    unsigned values[] = {0,0,0,0,0,0,0,0,0,0,0};
    unsigned tmp_values[] = {0,0,0,0,0,0,0,0};

    r = H5Pget_filter_by_id2(dcpl, BSHUF_H5FILTER, &flags, &nelements,
            tmp_values, 0, NULL, NULL);
    if(r<0) return -1;

    // First 3 slots reserved. Move any passed options to higher addresses.
    for (size_t ii=0; ii < nelements && ii + 3 < nelem_max; ii++) {
        values[ii + 3] = tmp_values[ii];
    }

    nelements = 3 + nelements;

    values[0] = BSHUF_H5FILTER_VERSION;
    values[1] = BSHUF_VERSION;

    elem_size = H5Tget_size(type);
    if(elem_size == 0) return -1;

    values[2] = elem_size;

    r = H5Pmodify_filter(dcpl, BSHUF_H5FILTER, flags, nelements, values);
    if(r<0) return -1;

    return 1;
}


size_t bshuf_h5_filter(unsigned int flags, size_t cd_nelmts,
           const unsigned int cd_values[], size_t nbytes,
           size_t *buf_size, void **buf) {

    size_t size, elem_size;
    int err;


    if (cd_nelmts < 3) return 0;
    elem_size = cd_values[2];
    if (nbytes % elem_size) return 0;
    size = nbytes / elem_size;

    void* out_buf;
    out_buf = malloc(size * elem_size);
    if (out_buf == NULL) return 0;

    if (flags & H5Z_FLAG_REVERSE) {
        // Bit unshuffle.
        err = bshuf_bitunshuffle(*buf, out_buf, size, elem_size);
    } else {
        // Bit unshuffle.
        err = bshuf_bitshuffle(*buf, out_buf, size, elem_size);
    }

    if (err) {
        free(out_buf);
        return 0;
    }

    free(*buf);
    *buf = out_buf;
    *buf_size = nbytes;

    return nbytes;
}


H5Z_class_t bshuf_H5Filter[1] = {{
    H5Z_CLASS_T_VERS,
    (H5Z_filter_t)(BSHUF_H5FILTER),
    1, 1,
    "bitshuffle; see https://github.com/kiyo-masui/bitshuffle",
    NULL,
    (H5Z_set_local_func_t)(bshuf_h5_set_local),
    (H5Z_func_t)(bshuf_h5_filter)
}};


int bshuf_register_h5filter(void){

    int retval;

    retval = H5Zregister(bshuf_H5Filter);
    if(retval<0){
        H5Epush1(__FILE__, "bshuf_register_h5filter", __LINE__, H5E_PLINE,
                 H5E_CANTREGISTER, "Can't register bitshuffle filter");
    }
    return retval;
}

