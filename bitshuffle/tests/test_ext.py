import unittest
import time
import timeit

import numpy as np
from numpy import random

from bitshuffle import ext


# If we are doing timeings and by what factor in increase workload.
TIME = 0


TEST_DTYPES = [np.uint8, np.uint16, np.int32, np.uint64, np.float32,
               np.float64, np.complex128]
TEST_DTYPES += ['a3', 'a5', 'a7', 'a9', 'a11', 'a12', 'a24', 'a48']

class TestProfile(unittest.TestCase):

    def setUp(self):
        n = 1024  # bytes.
        if TIME:
            n *= TIME
        self.data = random.randint(0, 8, n).astype(np.uint8)  # Random bits.
        self.fun = ext.copy
        self.check = None
        self.check_data = None
        self.case = "None"

    def tearDown(self):
        """Performs all tests and timings."""
        if TIME:
            reps = 10
        else:
            reps = 1
        delta_ts = []
        try:
            for ii in range(reps):
                t0 = time.time()
                out = self.fun(self.data)
                delta_ts.append(time.time() - t0)
        except RuntimeError as err:
            if (err.args[1] == 11) and not ext.using_SSE2():
                return
            if (err.args[1] == 12) and not ext.using_AVX2():
                return
            else:
                raise
        delta_t = min(delta_ts)
        size = self.data.size * self.data.dtype.itemsize
        speed = (ext.REPEAT * size / delta_t / 1024**3)   # GB/s
        if TIME:
            print "%-20s: %5.2f s/GB,   %5.2f GB/s" % (self.case, 1./speed, speed)
        if not self.check is None:
            ans = self.check(self.data).view(np.uint8)
            self.assertTrue(np.all(ans == out.view(np.uint8)))
        if not self.check_data is None:
            ans = self.check_data.view(np.uint8)
            self.assertTrue(np.all(ans == out.view(np.uint8)))

    def test_0_copy(self):
        self.case = "copy"
        self.fun = ext.copy
        self.check = lambda x: x

    def test_1_trans_byte_elem_scal(self):
        self.case = "byte T elem scal 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_byte_elem_scal
        self.check = trans_byte_elem

    def test_2a_trans_byte_elem_16(self):
        self.case = "byte T elem SSE 16"
        self.data = self.data.view(np.int16)
        self.fun = ext.trans_byte_elem_SSE
        self.check = trans_byte_elem

    def test_2b_trans_byte_elem_32(self):
        self.case = "byte T elem SSE 32"
        self.data = self.data.view(np.float32)
        self.fun = ext.trans_byte_elem_SSE
        self.check = trans_byte_elem

    def test_2c_trans_byte_elem_64(self):
        self.case = "byte T elem SSE 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_byte_elem_SSE
        self.check = trans_byte_elem

    def test_2d_trans_byte_elem_128(self):
        self.case = "byte T elem SSE 128"
        self.data = self.data.view(np.complex128)
        self.fun = ext.trans_byte_elem_SSE
        self.check = trans_byte_elem

    def test_2f_trans_byte_elem_96(self):
        self.case = "byte T elem SSE 96"
        n = self.data.size // 128 * 96
        dt = np.dtype([('a', 'i4'), ('b', 'i4'), ('c', 'i4')])
        self.data = self.data[:n].view(dt)
        self.fun = ext.trans_byte_elem_SSE
        self.check = trans_byte_elem

    def test_2g_trans_byte_elem_80(self):
        self.case = "byte T elem SSE 80"
        n = self.data.size // 128 * 80
        dt = np.dtype([('a', 'i2'), ('b', 'i2'), ('c', 'i2'), 
                       ('d', 'i2'), ('e', 'i2')])
        self.data = self.data[:n].view(dt)
        self.fun = ext.trans_byte_elem_SSE
        self.check = trans_byte_elem

    def test_3a_trans_bit_byte(self):
        self.case = "bit T byte 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_bit_byte_scal
        self.check = trans_bit_byte

    def test_3b_trans_bit_byte1(self):
        self.case = "bit T byte un 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_bit_byte_scal_unrolled
        self.check = trans_bit_byte

    def test_3d_trans_bit_byte_SSE(self):
        self.case = "bit T byte SSE 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_bit_byte_SSE
        self.check = trans_bit_byte

    def test_3f_trans_bit_byte_AVX(self):
        self.case = "bit T byte AVX 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_bit_byte_AVX
        self.check = trans_bit_byte

    def test_3g_trans_bit_byte_AVX_32(self):
        self.case = "bit T byte AVX 32"
        self.data = self.data.view(np.float32)
        self.fun = ext.trans_bit_byte_AVX
        self.check = trans_bit_byte

    def test_3f_trans_bit_byte_AVX1(self):
        self.case = "bit T byte AVX un 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_bit_byte_AVX_unrolled
        self.check = trans_bit_byte

    def test_4a_trans_bit_elem_AVX(self):
        self.case = "bit T elem AVX 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_bit_elem_AVX
        self.check = trans_bit_elem

    def test_4b_trans_bit_elem_AVX_128(self):
        self.case = "bit T elem AVX 128"
        self.data = self.data.view(np.complex128)
        self.fun = ext.trans_bit_elem_AVX
        self.check = trans_bit_elem

    def test_4c_trans_bit_elem_AVX_32(self):
        self.case = "bit T elem AVX 32"
        self.data = self.data.view(np.float32)
        self.fun = ext.trans_bit_elem_AVX
        self.check = trans_bit_elem

    def test_4d_trans_bit_elem_AVX_16(self):
        self.case = "bit T elem AVX 16"
        self.data = self.data.view(np.int16)
        self.fun = ext.trans_bit_elem_AVX
        self.check = trans_bit_elem

    def test_4e_trans_bit_elem_64(self):
        self.case = "bit T elem scal 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_bit_elem_scal
        self.check = trans_bit_elem

    def test_5a_untrans_bit_elem_16(self):
        self.case = "bit U elem SSE 16"
        pre_trans = self.data.view(np.int16)
        self.data = trans_bit_elem(pre_trans)
        self.fun = ext.untrans_bit_elem_SSE
        self.check_data = pre_trans

    def test_5b_untrans_bit_elem_128(self):
        self.case = "bit U elem SSE 128"
        pre_trans = self.data.view(np.complex128)
        self.data = trans_bit_elem(pre_trans)
        self.fun = ext.untrans_bit_elem_SSE
        self.check_data = pre_trans

    def test_5c_untrans_bit_elem_32(self):
        self.case = "bit U elem SSE 32"
        pre_trans = self.data.view(np.float32)
        self.data = trans_bit_elem(pre_trans)
        self.fun = ext.untrans_bit_elem_SSE
        self.check_data = pre_trans

    def test_5d_untrans_bit_elem_64(self):
        self.case = "bit U elem SSE 64"
        pre_trans = self.data.view(np.float64)
        self.data = trans_bit_elem(pre_trans)
        self.fun = ext.untrans_bit_elem_SSE
        self.check_data = pre_trans

    def test_5e_untrans_bit_elem_64(self):
        self.case = "bit U elem scal 64"
        pre_trans = self.data.view(np.float64)
        self.data = trans_bit_elem(pre_trans)
        self.fun = ext.untrans_bit_elem_scal
        self.check_data = pre_trans

    def test_6a_trans_byte_bitrow_64(self):
        self.case = "byte T row 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_byte_bitrow

    def test_6b_trans_byte_bitrow_SSE_64(self):
        self.case = "byte T row SSE 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_byte_bitrow_SSE
        self.check = ext.trans_byte_bitrow

    def test_7a_shuffle_bit_eight_SSE_64(self):
        self.case = "bit S eight SSE 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.shuffle_bit_eightelem_SSE

    def test_8a_trans_bit_elem_64(self):
        self.case = "bit T elem 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_bit_elem
        self.check = trans_bit_elem

    def test_8b_untrans_bit_elem_64(self):
        self.case = "bit U elem 64"
        pre_trans = self.data.view(np.float64)
        self.data = trans_bit_elem(pre_trans)
        self.fun = ext.untrans_bit_elem
        self.check_data = pre_trans

    def test_9a_bitshuffle_64(self):
        self.case = "bitshuffle 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.bitshuffle

    def test_9b_bitunshuffle_64(self):
        self.case = "bitunshuffle 64"
        pre_trans = self.data.view(np.float64)
        self.data = ext.bitshuffle(pre_trans)
        self.fun = ext.bitunshuffle
        self.check_data = pre_trans


class TestDevCases(unittest.TestCase):

    def deactivated_test_trans_bit_byte(self):
        d = np.arange(16, dtype=np.uint16)
        t = ext.trans_bit_byte_scal(d)
        #print t
        t1 = trans_bit_byte(d)
        #print t1
        self.assertTrue(np.all(t == t1))

    def deactivated_test_trans_byte_bitrow_SSE(self):
        d = np.arange(256, dtype = np.uint8)
        t = ext.trans_byte_bitrow(d)
        #print np.reshape(t, (32, 8))
        t1 = ext.trans_byte_bitrow_SSE(d)
        #print np.reshape(t1, (32, 8))
        self.assertTrue(np.all(t == t1))

    def deactivated_test_trans_byte_elem_SSE(self):
        d = np.empty(16, dtype=([('a', 'u4'), ('b', 'u4'), ('c', 'u4')]))
        d['a'] = np.arange(16) * 1
        d['b'] = np.arange(16) * 2
        d['c'] = np.arange(16) * 3
        #print d.dtype.itemsize
        #print np.reshape(d.view(np.uint8), (16, 12))
        t1 = ext.trans_byte_elem_SSE(d)
        #print np.reshape(t1.view(np.uint8), (12, 16))
        t0 = trans_byte_elem(d)
        #print np.reshape(t0.view(np.uint8), (12, 16))
        self.assertTrue(np.all(t0.view(np.uint8) == t1.view(np.uint8)))

    def deactivated_test_bitshuffle(self):
        d = np.arange(128, dtype=np.uint16)
        t1 = ext.bitshuffle(d)
        #print t1
        t2 = ext.bitunshuffle(t1)
        #print t2
        self.assertTrue(np.all(t2.view(np.uint8) == d.view(np.uint8)))


class TestOddLengths(unittest.TestCase):

    def setUp(self):
        self.reps = 10
        self.nmax = 128 * 8
        #self.nmax = 4 * 8    # XXX
        self.fun = ext.copy
        self.check = lambda x: x

    def test_trans_bit_elem_SSE(self):
        self.fun = ext.trans_bit_elem_SSE
        self.check = trans_bit_elem

    def test_untrans_bit_elem_SSE(self):
        self.fun = lambda x: ext.untrans_bit_elem_SSE(ext.trans_bit_elem(x))
        self.check = lambda x: x

    def test_trans_bit_elem_AVX(self):
        self.fun = ext.trans_bit_elem_AVX
        self.check = trans_bit_elem

    def test_untrans_bit_elem_AVX(self):
        self.fun = lambda x: ext.untrans_bit_elem_SSE(ext.trans_bit_elem(x))
        self.check = lambda x: x

    def test_trans_bit_elem_scal(self):
        self.fun = ext.trans_bit_elem_scal
        self.check = trans_bit_elem

    def test_untrans_bit_elem_scal(self):
        self.fun = lambda x: ext.untrans_bit_elem_scal(ext.trans_bit_elem(x))
        self.check = lambda x: x

    def test_trans_byte_elem_SSE(self):
        self.fun = ext.trans_byte_elem_SSE
        self.check = trans_byte_elem

    def tearDown(self):
        try:
            for dtype in TEST_DTYPES:
                itemsize = np.dtype(dtype).itemsize
                nbyte_max = self.nmax * itemsize
                dbuf = random.randint(0, 255, nbyte_max).astype(np.uint8)
                dbuf = dbuf.view(dtype)
                for ii in range(self.reps):
                    n = random.randint(0, self.nmax / 8, 1) * 8
                    data = dbuf[:n]
                    out = self.fun(data).view(np.uint8)
                    ans = self.check(data).view(np.uint8)
                    self.assertTrue(np.all(out == ans))
        except RuntimeError as err:
            if (err.args[1] == 11) and not ext.using_SSE2():
                return
            if (err.args[1] == 12) and not ext.using_AVX2():
                return
            else:
                raise


class TestBitShuffleCircle(unittest.TestCase):
    """Ensure that final filter is circularly consistant for any data type and
    any length buffer."""

    def test_circle(self):
        nmax = 10000
        reps = 100
        for dtype in TEST_DTYPES:
            itemsize = np.dtype(dtype).itemsize
            nbyte_max = nmax * itemsize
            dbuf = random.randint(0, 255, nbyte_max).astype(np.uint8)
            dbuf = dbuf.view(dtype)
            for ii in range(reps):
                n = random.randint(0, nmax, 1)
                data = dbuf[:n]
                shuff = ext.bitshuffle(data)
                out = ext.bitunshuffle(shuff)
                self.assertTrue(out.dtype is data.dtype)
                self.assertTrue(np.all(data.view(np.uint8)
                                       == out.view(np.uint8)))


# Python implementations for checking results.

def trans_byte_elem(arr):
    dtype = arr.dtype
    itemsize = dtype.itemsize
    in_buf = arr.flat[:].view(np.uint8)
    nelem = in_buf.size // itemsize
    in_buf.shape = (nelem, itemsize)

    out_buf = np.empty((itemsize, nelem), dtype=np.uint8)
    for ii in range(nelem):
        for jj in range(itemsize):
            out_buf[jj,ii] = in_buf[ii,jj]
    return out_buf.flat[:].view(dtype)


def trans_bit_byte(arr):
    n = arr.size
    dtype = arr.dtype
    itemsize = dtype.itemsize
    bits = np.unpackbits(arr.view(np.uint8))
    bits.shape = (n * itemsize, 8)
    # We have to reverse the order of the bits both for unpacking and packing,
    # since we want to call the least significant bit the first bit.
    bits = bits[:,::-1]
    bits_shuff = (bits.T).copy()
    bits_shuff.shape = (n * itemsize, 8)
    bits_shuff = bits_shuff[:,::-1]
    arr_bt = np.packbits(bits_shuff.flat[:])
    return arr_bt.view(dtype)


def trans_bit_elem(arr):
    n = arr.size
    dtype = arr.dtype
    itemsize = dtype.itemsize
    bits = np.unpackbits(arr.view(np.uint8))
    bits.shape = (n * itemsize, 8)
    # We have to reverse the order of the bits both for unpacking and packing,
    # since we want to call the least significant bit the first bit.
    bits = bits[:,::-1].copy()
    bits.shape = (n, itemsize * 8)
    bits_shuff = (bits.T).copy()
    bits_shuff.shape = (n * itemsize, 8)
    bits_shuff = bits_shuff[:,::-1]
    arr_bt = np.packbits(bits_shuff.flat[:])
    return arr_bt.view(dtype)



if __name__ == "__main__":
    unittest.main()
