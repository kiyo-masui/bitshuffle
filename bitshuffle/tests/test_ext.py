import unittest
import time
import timeit

import numpy as np
from numpy import random

from bitshuffle import ext


# If we are doing timeings and by what factor in increase workload.
TIME = 256


class TestProfile(unittest.TestCase):

    def setUp(self):
        n = 1096
        if TIME:
            n *= TIME
        self.data = random.rand(n) * 2**40
        self.fun = lambda x: x**2
        self.case = "None"

    def test_copy(self):
        self.case = "memcpy"
        self.fun = ext.just_copy

    def test_byte_T_32(self):
        self.case = "btye T 32"
        self.data = self.data.astype(np.float32)
        self.fun = ext.byte_T_elem_simple

    def test_byte_T_64(self):
        self.case = "btye T 64"
        self.data = self.data.astype(np.float64)
        self.fun = ext.byte_T_elem_simple

    def test_byte_T_16(self):
        self.case = "btye T 16"
        self.data = self.data.astype(np.int16)
        self.fun = ext.byte_T_elem_simple

    def tearDown(self):
        delta_ts = []
        for ii in range(5):
            t0 = time.time()
            self.fun(self.data)
            delta_ts.append(time.time() - t0)
        delta_t = min(delta_ts)
        size = self.data.size * self.data.dtype.itemsize
        speed = round(size / delta_t / 1024**2)   # MB/s
        if TIME:
            print "%s: %5.3f s, %5.0f Mb/s" % (self.case, delta_t, speed)



if __name__ == "__main__":
    unittest.main()
