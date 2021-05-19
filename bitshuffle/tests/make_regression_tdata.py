"""
Script to create data used for regression testing.

"""

import numpy as np
from numpy import random
import h5py

import bitshuffle
from bitshuffle import h5
from h5py import h5z

BLOCK_SIZE = 64   # Smallish such that datasets have many blocks but are small.
COMP_LVL = 10 # ZSTD compression level
FILTER_PIPELINE = [h5.H5FILTER,]
FILTER_OPTS = [[(BLOCK_SIZE, h5.H5_COMPRESS_LZ4)], [(BLOCK_SIZE, h5.H5_COMPRESS_ZSTD, COMP_LVL)]]

OUT_FILE = ["bitshuffle/tests/data/lz4_regression_%s.h5" % bitshuffle.__version__,
            "bitshuffle/tests/data/zstd_regression_%s.h5" % bitshuffle.__version__]

DTYPES = ['a1', 'a2', 'a3', 'a4', 'a6', 'a8', 'a10']

for i, fil_op in enumerate(FILTER_OPTS):
    f = h5py.File(OUT_FILE[i], 'w')
    g_comp = f.create_group("compressed")
    g_orig = f.create_group("origional")
    for dtype in DTYPES:
        for rep in ['a', 'b', 'c']:
            dset_name = "%s_%s" % (dtype, rep)
            dtype = np.dtype(dtype)
            n_elem = 3 * BLOCK_SIZE + random.randint(0, BLOCK_SIZE)
            shape = (n_elem,)
            chunks = shape
            data = random.randint(0, 255, n_elem * dtype.itemsize)
            data = data.astype(np.uint8).view(dtype)
    
            g_orig.create_dataset(dset_name, data=data)
    
            h5.create_dataset(g_comp, bytes(dset_name, 'utf-8'), shape, dtype, chunks=chunks,
                    filter_pipeline=FILTER_PIPELINE, filter_flags=(h5z.FLAG_MANDATORY,), 
                    filter_opts=fil_op)
            g_comp[dset_name][:] = data
    
    f.close()
