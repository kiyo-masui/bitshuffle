import unittest

import numpy as np
from numpy import random

from bitshuffle import mockup

def rand_m128():
    return random.randint(0, 256, 16).astype(np.uint8)


def zeros_m128():
    return np.zeros(16, dtype=np.uint8)


class TestSSEMock(unittest.TestCase):

    def setUp(self):
        self.m128_0 = rand_m128()
        self.m128_1 = rand_m128()
        self.m128_2 = rand_m128()
        self.m128_3 = rand_m128()

    def test_movemask(self):
        mask = mockup.mm_movemask_epi8(self.m128_0)
        right_ans = 0
        for ii in range(16):
            right_ans += 2**ii * (self.m128_0[16 - 1 - ii] // 2**7)
        self.assertEqual(mask, right_ans)

    def test_slli_epi16(self):
        a = self.m128_0
        a[:] = 0
        a[-1] = 2
        a[2] = 64  # shifted out
        a[0] = 5
        b = mockup.mm_slli_epi16(a, 3)
        self.assertEqual(b[0], a[0] * 2**3)
        self.assertEqual(b[-1], a[-1] * 2**3)
        self.assertEqual(b[1], 0)
        self.assertEqual(b[2], 0)

    def test_unpack(self):
        a = self.m128_0
        a[:] = np.arange(16)
        b = self.m128_1
        b[:] = np.arange(16, 32)

        c = mockup.mm_unpacklo_epi8(a, b)
        res = np.array([0, 16, 1, 17, 2, 18, 3, 19,
                        4, 20, 5, 21, 6, 22, 7, 23])
        self.assertTrue(np.all(c == res))
        c = mockup.mm_unpackhi_epi8(a, b)
        res = res + 8
        self.assertTrue(np.all(c == res))

        c = mockup.mm_unpacklo_epi16(a, b)
        res = np.array([0, 1, 16, 17, 2, 3, 18, 19,
                        4, 5, 20, 21, 6, 7, 22, 23])
        self.assertTrue(np.all(c == res))
        c = mockup.mm_unpackhi_epi16(a, b)
        res = res + 8
        self.assertTrue(np.all(c == res))

        c = mockup.mm_unpacklo_epi32(a, b)
        res = np.array([0, 1, 2, 3, 16, 17, 18, 19,
                        4, 5, 6, 7, 20, 21, 22, 23])
        self.assertTrue(np.all(c == res))
        c = mockup.mm_unpackhi_epi32(a, b)
        res = res + 8
        self.assertTrue(np.all(c == res))

        c = mockup.mm_unpacklo_epi64(a, b)
        res = np.array([0, 1, 2, 3, 4, 5, 6, 7,
                        16, 17, 18, 19, 20, 21, 22, 23])
        self.assertTrue(np.all(c == res))
        c = mockup.mm_unpackhi_epi64(a, b)
        res = res + 8
        self.assertTrue(np.all(c == res))

    def test_shuffle(self):
        a = self.m128_0
        a[:] = np.arange(16)
        b = mockup.mm_shufflehi_epi16(a, 0xe4)
        self.assertTrue(np.all(a == b))
        b = mockup.mm_shufflelo_epi16(a, 0xe4)
        self.assertTrue(np.all(a == b))


class TestBitTranspose16(unittest.TestCase):

    def test_ones(self):
        m128 = zeros_m128() + 1
        out = mockup.bt16x8(m128)
        self.assertEqual(out[-1], 255)
        self.assertEqual(out[-2], 255)
        self.assertTrue(np.all(out[:-2] == 0))

    def test_some_ones(self):
        m128 = zeros_m128()
        m128[:4] = 1
        out = mockup.bt16x8(m128)
        self.assertEqual(out[-1], 0)
        self.assertEqual(out[-2], 2**7 + 2**6 + 2**5 + 2**4)
        self.assertTrue(np.all(out[:-2] == 0))


class TestByteTranspose(unittest.TestCase):

    def test_8(self):
        a = np.arange(128).astype(np.uint8)
        b = mockup.byte_transpose(a)
        self.assertTrue(np.all(a == b))

    def test_16(self):
        a = np.zeros(16, dtype=np.int16)
        a[:9] = np.arange(1, 10)
        a = a * 10 + (a * 10 + 1) * 2**8
        b = mockup.byte_transpose(a)
        self.assertTrue(np.all(self.byte_transpose(a) == b))

    def test_32(self):
        a = np.zeros(16, dtype=np.int32)
        a[:9] = np.arange(1, 10)
        a = (a * 10 + (a * 10 + 1) * 2**8 + (a * 10 + 2) * 2**16
             + (a * 10 + 3) * 2**24)
        b = mockup.byte_transpose(a)
        self.assertTrue(np.all(self.byte_transpose(a) == b))

    def test_64(self):
        a = np.zeros(16, dtype=np.int64)
        a[:9] = np.arange(1, 10)
        a = (a * 10 + (a * 10 + 1) * 2**8 + (a * 10 + 2) * 2**16
             + (a * 10 + 3) * 2**24 + (a * 10 + 4) * 2**32
             + (a * 10 + 5) * 2**40 + (a * 10 + 6) * 2**48
             + (a * 10 + 7) * 2**56)
        b = mockup.byte_transpose(a)
        self.assertTrue(np.all(self.byte_transpose(a) == b))

    def byte_transpose(self, arr):
        itemsize = arr.dtype.itemsize
        in_buf = arr.flat[:].view(np.uint8)
        nelem = in_buf.size // itemsize
        in_buf.shape = (nelem, itemsize)

        out_buf = np.empty((itemsize, nelem), dtype=np.uint8)
        for ii in range(nelem):
            for jj in range(itemsize):
                out_buf[jj,ii] = in_buf[ii,jj]
        return out_buf



class TestBitTranspose(unittest.TestCase):


    def test_arange(self):
        arr = np.arange(128 * 7, dtype=np.uint16)
        bt_arr = mockup.bit_transpose(arr)
        self.assertTrue(np.all(bt_arr == bit_transpose(arr)))

    def test_random(self):
        arr = random.randint(-12345, 98877890, 128 * 13).astype(dtype=np.uint64)
        bt_arr = mockup.bit_transpose(arr)
        self.assertTrue(np.all(bt_arr == bit_transpose(arr)))


def bit_transpose(arr):
    """Simple implementation of bit-transpose for result checking."""
    n = arr.size
    dtype = arr.dtype
    itemsize = dtype.itemsize
    bits = np.unpackbits(arr.view(np.uint8))
    bits.shape = (n, itemsize * 8)
    bits_shuff = (bits.T).copy()
    arr_bt = np.packbits(bits_shuff.flat[:])
    return arr_bt

class TestAVXMock(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
