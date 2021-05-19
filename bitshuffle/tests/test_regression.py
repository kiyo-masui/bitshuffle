"""
Test that data encoded with earlier versions can still be decoded correctly.

"""

from __future__ import absolute_import, division, print_function

import unittest
from os import path

import numpy as np
import h5py

import bitshuffle
from bitshuffle import h5
from packaging import version


TEST_DATA_DIR = path.dirname(bitshuffle.__file__) + "/tests/data"

OUT_FILE_TEMPLATE = TEST_DATA_DIR + "/regression_%s.h5"

VERSIONS = ["0.1.3", "0.3.6",]


class TestAll(unittest.TestCase):

    def test_regression(self):
        for rev in VERSIONS:
            file_name = OUT_FILE_TEMPLATE % (rev)
            f = h5py.File(file_name)
            g_orig = f["origional"]
            g_comp = f["compressed"]

            for dset_name in g_comp.keys():
                self.assertTrue(np.all(g_comp[dset_name][:]
                                       == g_orig[dset_name][:]))
            
            # Only run ZSTD comparison on versions >= 0.3.6
            if version.parse(rev) >= version.parse("0.3.6"):
                g_comp_zstd = f["compressed_zstd"]
                for dset_name in g_comp_zstd.keys():
                    self.assertTrue(np.all(g_comp_zstd[dset_name][:]
                                           == g_orig[dset_name][:]))

if __name__ == "__main__":
    unittest.main()
