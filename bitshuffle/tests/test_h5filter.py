import unittest
import os
import glob

import numpy as np
import h5py
from h5py import h5f, h5d, h5z, h5t, h5s, filters

os.environ["HDF5_PLUGIN_PATH"] = "/Users/kiyo/working/bitshuffle/plugins"
#os.environ["HDF5_PLUGIN_PATH"] = ("/Users/kiyo/Downloads/BZIP2-plugin/plugins/lib"
#                                  ":/Users/kiyo/working/bitshuffle/bitshuffle"
#                                  ":/Users/kiyo/working/bitshuffle/src"
#                                  )


class TestBZIP2(unittest.TestCase):

    def atest_dump_shell(self):
        os.system('h5dump '
                  '/Users/kiyo/Downloads/BZIP2-plugin/example/h5ex_d_bzip2.h5')

    def test_create_dataset(self):
        shape = (32 * 1024,)
        chunks = (4 * 1024,)
        dtype = np.int64
        data = np.arange(shape[0])
        f = h5py.File("tmp_test_bzip.h5")
        tid = h5t.py_create(dtype, logical=1)
        sid = h5s.create_simple(shape, shape)
        dcpl = filters.generate_dcpl(shape, dtype, chunks, None, None,
                  None, None, None, None)
        #dcpl.set_filter(h5z.FILTER_SHUFFLE)
        dcpl.set_filter(32008, h5z.FLAG_MANDATORY)
        #dcpl.set_filter(32008)
        dcpl.set_filter(307)

        dset_id = h5d.create(f.id, "stuff", tid, sid, dcpl=dcpl)
        dset_id.write(h5s.ALL, h5s.ALL, data)

        f.close()

        os.system("h5dump -H -p tmp_test_bzip.h5")
        #os.system("h5dump tmp_test_bzip.h5")

        f = h5py.File("tmp_test_bzip.h5", 'r')
        d = f['stuff'][:]
        self.assertTrue(np.all(d == data))


    def tearDown(self):
        files = glob.glob("tmp_test_*")
        for f in files:
            os.remove(f)




if __name__ == "__main__":
    unittest.main()
