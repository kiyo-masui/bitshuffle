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

    def test_bit_T_32(self):
        self.case = "bit T 32"
        self.data = self.data.astype(np.float32)
        self.fun = ext.bit_T_byte

    def tearDown(self):
        delta_ts = []
        for ii in range(10):
            t0 = time.time()
            self.fun(self.data)
            delta_ts.append(time.time() - t0)
        delta_t = min(delta_ts)
        size = self.data.size * self.data.dtype.itemsize
        speed = round(size / delta_t / 1024**2)   # MB/s
        if TIME:
            print "%s: %5.3f s, %5.0f Mb/s" % (self.case, delta_t, speed)


class TestRandNumbers(unittest.TestCase):

    def setUp(self):
        n = 4096 * 8    # bytes
        data = random.randint(-2**31, 2**31 - 1, n // 4).astype(np.int32)
        self.data = data.view(np.uint8)

    def test_byte_elem_simple_int32(self):
        data = self.data.view(np.int32)
        out = ext.byte_T_elem_simple(data)
        self.assertTrue(np.all(byte_T_elem(data) == out))

    def test_byte_elem_simple_float64(self):
        data = self.data.view(np.float64)
        out = ext.byte_T_elem_simple(data)
        self.assertTrue(np.all(byte_T_elem(data) == out))

    def test_bit_byte_int32(self):
        data = self.data.view(np.int32)
        out = ext.bit_T_byte(data)
        self.assertTrue(np.all(bit_T_byte(data) == out))


class TestDevCases(unittest.TestCase):

    def test_bit_byte(self):
        d = np.arange(128, dtype=np.uint8)
        #d = np.zeros(128, dtype=np.uint8)
        #d[0] = 1
        #d = random.randint(0, 127, 128).astype(np.uint8)
        #print np.reshape(ext.bit_T_byte(d), (8, 16))
        #print np.reshape(bit_T_byte(d), (8, 16))
        self.assertTrue(np.all(ext.bit_T_byte(d) == bit_T_byte(d)))



# Python implementations for testing.

def byte_T_elem(arr):
    itemsize = arr.dtype.itemsize
    in_buf = arr.flat[:].view(np.uint8)
    nelem = in_buf.size // itemsize
    in_buf.shape = (nelem, itemsize)

    out_buf = np.empty((itemsize, nelem), dtype=np.uint8)
    for ii in range(nelem):
        for jj in range(itemsize):
            out_buf[jj,ii] = in_buf[ii,jj]
    return out_buf.flat[:]


def bit_T_byte(arr):
    n = arr.size
    dtype = arr.dtype
    itemsize = dtype.itemsize
    bits = np.unpackbits(arr.view(np.uint8))
    bits.shape = (n * itemsize, 8)
    # We have to reverse the order of the bits both for unpackin and packing,
    # since we want to call the least significant bit the first bit.
    bits = bits[:,::-1]
    bits_shuff = (bits.T).copy()
    bits_shuff.shape = (n * itemsize, 8)
    bits_shuff = bits_shuff[:,::-1]
    arr_bt = np.packbits(bits_shuff.flat[:])
    return arr_bt





if __name__ == "__main__":
    unittest.main()
