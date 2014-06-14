import unittest
import time
import timeit

import numpy as np
from numpy import random

from bitshuffle import ext


# If we are doing timeings and by what factor in increase workload.
TIME = 32


class TestProfileRandData(unittest.TestCase):

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
        for ii in range(reps):
            t0 = time.time()
            out = self.fun(self.data)
            delta_ts.append(time.time() - t0)
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
        self.fun = ext.trans_bit_byte
        self.check = trans_bit_byte

    def test_3b_trans_bit_byte1(self):
        self.case = "bit T byte un 64"
        self.data = self.data.view(np.float64)
        self.fun = ext.trans_bit_byte_unrolled
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

    def test_trans_bit_byte(self):
        d = np.arange(16, dtype=np.uint16)
        t = ext.trans_bit_byte(d)
        #print t
        t1 = trans_bit_byte(d)
        #print t1
        self.assertTrue(np.all(t == t1))

    def test_trans_byte_bitrow_SSE(self):
        d = np.arange(256, dtype = np.uint8)
        t = ext.trans_byte_bitrow(d)
        #print np.reshape(t, (32, 8))
        t1 = ext.trans_byte_bitrow_SSE(d)
        #print np.reshape(t1, (32, 8))
        self.assertTrue(np.all(t == t1))

    def test_trans_byte_elem_SSE(self):
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

    def test_bitshuffle(self):
        d = np.arange(128, dtype=np.uint16)
        t1 = ext.bitshuffle(d)
        #print t1
        t2 = ext.bitunshuffle(t1)
        #print t2
        self.assertTrue(np.all(t2.view(np.uint8) == d.view(np.uint8)))


    def atest_bit_byte(self):
        d = np.arange(128, dtype=np.uint8)
        #d = np.zeros(128, dtype=np.uint8)
        #d[0] = 1
        #d = random.randint(0, 127, 128).astype(np.uint8)
        #print np.reshape(ext.bit_T_byte(d), (8, 16))
        #print np.reshape(bit_T_byte(d), (8, 16))
        self.assertTrue(np.all(ext.bit_T_byte(d) == bit_T_byte(d)))

    def atest_sse_byte_trans(self):
        d = np.arange(16, dtype = np.uint16)
        t = ext.byte_T_elem_fast(d)
        t1 = byte_T_elem(d)
        #print np.reshape(t.view(np.uint8), (2, 16))
        #print np.reshape(t1.view(np.uint8), (2, 16))
        self.assertTrue(np.all(t == t1))

    def atest_bit_rows_T_byte_rows(self):
        DTYPE = np.uint32
        n = 128
        d = np.arange(n, dtype=DTYPE)
        s = d.dtype.itemsize
        d = d + 2**15 * d
        t1 = byte_T_elem(d)
        #print np.reshape(d.view(np.uint8), (16, 4))
        #print np.reshape(t1.view(np.uint8), (4, 16))
        tt1 = bit_T_byte(t1)
        #print np.reshape(tt1.view(np.uint8), (32, 2))
        ttt1 = ext.bit_rows_T_byte_rows(tt1.view(DTYPE))
        #print np.reshape(ttt1.view(np.uint8), (32, 2))
        ttt2 = bit_T_elem(d)
        ttt3 = ext.bit_T_elem(d)
        #print np.reshape(ttt2.view(np.uint8), (s * 8, n // 8))
        #print np.reshape(ttt3.view(np.uint8), (s * 8, n // 8))
        self.assertTrue(np.all(ttt2 == ttt1))
        self.assertTrue(np.all(ttt3 == ttt1))

    def atest_byte_T_8xN_8(self):
        d = np.arange(128, dtype=np.uint8)
        t1 = bit_T_elem(d)
        #print np.reshape(t1, (8, 16))
        t2 = ext.byte_T_8xN(t1)
        #print np.reshape(t2, (16, 8))
        t3 = ext.bit_uT_byte(t2)
        #print np.reshape(t3, (8, 16))
        self.assertTrue(np.all(t3 == d))

    def atest_byte_T_8xN_32(self):
        d = np.arange(128, dtype=np.uint32)
        t1 = bit_T_elem(d)
        #print np.reshape(t1, (32, 16))
        t2 = ext.byte_T_8xN(t1.view(np.uint32))
        #print np.reshape(t2, (16, 32))
        t3 = ext.bit_uT_byte(t2.view(np.uint32)).view(np.uint32)
        #print t3
        self.assertTrue(np.all(t3 == d))

    def atest_byte_T_8xN_64(self):
        d = np.arange(128, dtype=np.int64)
        t1 = bit_T_elem(d)
        print np.reshape(t1, (64, 16))
        t2 = ext.byte_T_8xN(t1.view(np.int64))
        print np.reshape(t2, (16, 64))
        t3 = ext.bit_uT_byte(t2.view(np.int64)).view(np.int64)
        print t3
        self.assertTrue(np.all(t3 == d))

    def atest_byte_T_8xN_16(self):
        d = np.arange(128, dtype=np.uint16)
        t1 = bit_T_elem(d)
        #print np.reshape(t1, (16, 16))
        t2 = ext.byte_T_8xN(t1.view(np.uint16))
        #print np.reshape(t2, (16, 16))
        t3 = ext.bit_uT_byte(t2.view(np.uint16)).view(np.uint16)
        #print t3
        self.assertTrue(np.all(t3 == d))


class TestOddLengths(unittest.TestCase):

    def setUp(self):
        n = 1103    # prime
        data = random.randint(-2**31, 2**31 - 1, n)
        self.data = data

    def atest_byte_elem_int16(self):
        data = self.data.astype(np.uint16)
        out = ext.byte_T_elem_fast(data)
        self.assertTrue(np.all(byte_T_elem(data) == out))

    def atest_byte_elem_int32(self):
        data = self.data.astype(np.uint32)
        out = ext.byte_T_elem_fast(data)
        self.assertTrue(np.all(byte_T_elem(data) == out))

# Python implementations for testing.

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
