#include "bshuf_h5filter.h"

H5PL_type_t H5PLget_plugin_type(void) {return H5PL_TYPE_FILTER;}
const void* H5PLget_plugin_info(void) {return bshuf_H5Filter;}

