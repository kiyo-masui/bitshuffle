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


class TestBitTranspose(unittest.TestCase):


    def test_arange(self):
        arr = np.arange(32, dtype=np.uint16)
        bt_arr = mockup.bit_transpose(arr)
        print bt_arr
        print bit_transpose(arr)
        self.assertTrue(np.all(bt_arr == bit_transpose(arr)))


def bit_transpose(arr):
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
