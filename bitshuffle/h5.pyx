import numpy
import h5py
from h5py import h5d, h5s, h5t, filters

cimport cython


cdef extern from "bshuf_h5filter.h":
    int bshuf_register_h5filter()
    int BSHUF_H5FILTER
    int BSHUF_H5_COMPRESS_LZ4

cdef int LZF_FILTER = 32000

H5_COMPRESS_LZ4 = BSHUF_H5_COMPRESS_LZ4


def register_h5_filter():
    ret = bshuf_register_h5filter()
    if ret < 0:
        raise RuntimeError("Failed to register bitshuffle HDF5 filter.", ret)


register_h5_filter()


def create_dataset(parent, name, shape, dtype, chunks=None, maxshape=None,
                   fillvalue=None, track_times=None,
                   filter_pipeline=(), filter_flags=None, filter_opts=None):
    """Create a dataset with an arbitrary filter pipeline.

    Return a new low-level dataset identifier.

    """

    if hasattr(filter_pipeline, "__getitem__"):
        filter_pipeline = list(filter_pipeline)
    else:
        filter_pipeline = [filter_pipeline]
        filter_flags = [filter_flags]
        filter_opts = [filter_opts]
    nfilters = len(filter_pipeline)
    if filter_flags is None:
        filter_flags = [None] * nfilters
    if filter_opts is None:
        filter_opts = [None] * nfilters
    if not len(filter_flags) == nfilters or not len(filter_opts) == nfilters:
        msg = "Supplied incompatible number of filters, flags, and options."
        raise ValueError(msg)

    shape = tuple(shape)

    tmp_shape = maxshape if maxshape is not None else shape
    # Validate chunk shape
    chunks_larger = (-numpy.array([ i>=j 
                     for i,j in zip(tmp_shape,chunks) if i is not None])).any()
    if isinstance(chunks, tuple) and chunks_larger:
        errmsg = ("Chunk shape must not be greater than data shape in any "
                  "dimension. {} is not compatible with {}".format(chunks, shape))
        raise ValueError(errmsg)

    if isinstance(dtype, h5py.Datatype):
        # Named types are used as-is
        tid = dtype.id
        dtype = tid.dtype  # Following code needs this
    else:
        # Validate dtype
        dtype = numpy.dtype(dtype)
        tid = h5t.py_create(dtype, logical=1)

    dcpl = filters.generate_dcpl(shape, dtype, chunks, None, None, None, 
                                 None, maxshape, None)

    if fillvalue is not None:
        fillvalue = numpy.array(fillvalue)
        dcpl.set_fill_value(fillvalue)

    if track_times in (True, False):
        dcpl.set_obj_track_times(track_times)
    elif track_times is not None:
        raise TypeError("track_times must be either True or False")

    for ii in range(nfilters):
        this_filter = filter_pipeline[ii]
        this_flags = filter_flags[ii]
        this_opts = filter_opts[ii]
        if this_flags is None:
            this_flags = 0
        if this_opts is None:
            this_opts = ()
        dcpl.set_filter(this_filter, this_flags, this_opts)

    if maxshape is not None:
        maxshape = tuple(m if m is not None else h5s.UNLIMITED
                         for m in maxshape)
    sid = h5s.create_simple(shape, maxshape)

    dset_id = h5d.create(parent.id, name, tid, sid, dcpl=dcpl)

    return dset_id


def create_bitshuffle_lzf_dataset(parent, name, shape, dtype, chunks=None,
                                  maxshape=None, fillvalue=None,
                                  track_times=None):
    filter_pipeline = (BSHUF_H5FILTER, LZF_FILTER)
    dset_id = create_dataset(parent, name, shape, dtype, chunks=chunks,
                             filter_pipeline=filter_pipeline, maxshape=maxshape,
                             fillvalue=fillvalue, track_times=track_times)
    return dset_id

