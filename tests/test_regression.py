"""
Test that data encoded with earlier versions can still be decoded correctly.

"""

from __future__ import absolute_import, division, print_function

import pathlib
import unittest

import numpy as np
import h5py


TEST_DATA_DIR = pathlib.Path(__file__).parent / "data"

OUT_FILE_TEMPLATE = "regression_%s.h5"

VERSIONS = [
    "0.1.3",
]


class TestAll(unittest.TestCase):
    def test_regression(self):
        for version in VERSIONS:
            file_name = TEST_DATA_DIR / (OUT_FILE_TEMPLATE % version)
            f = h5py.File(file_name, "r")
            g_orig = f["origional"]
            g_comp = f["compressed"]

            for dset_name in g_comp.keys():
                self.assertTrue(np.all(g_comp[dset_name][:] == g_orig[dset_name][:]))


if __name__ == "__main__":
    unittest.main()
