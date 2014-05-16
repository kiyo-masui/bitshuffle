"""Initial script for playing with the bitshuffle concept for compression."""

import numpy as np
import h5py
from scipy import random
import matplotlib.pyplot as plt

n = 1024 * 64
sigma = 4
amp =  4
offset = 0
#amp = 0
#offset = 1

DTYPE = np.int32
#DTYPE = np.float32
SOURCE = 'GEN'

TRANS = 't'

f_andata = h5py.File('/Users/kiyo/data/some_andata/data.h5', 'r')
real_data = f_andata['vis'][:]
prods = f_andata['index_map/prod'][:]
f_andata.close()
freq = 8    # Doesn't really matter.
real_data *= 13.64  # Random factor.

cross_data = real_data[:,[1, 2, 4],:].copy().astype(np.complex64)
# Caculate the normalization.
norm = np.empty((real_data.shape[0], 3, real_data.shape[2]), np.float32)
norm[:,0,:] = np.sqrt(real_data[:,0,:] * real_data[:,3,:]).real
norm[:,1,:] = np.sqrt(real_data[:,0,:] * real_data[:,5,:]).real
norm[:,2,:] = np.sqrt(real_data[:,3,:] * real_data[:,5,:]).real

if TRANS == 'r':
    # Store as cross correlation coef.
    cross_data /= norm
    cross_data *= 2**15
elif TRANS == 't':
    err = norm / 2**15
    err = 2**np.floor(np.log(err) / np.log(2))
    cross_data.real = err * np.round(cross_data.real / err)
    cross_data.imag = err * np.round(cross_data.imag / err)

if SOURCE == 'GEN':
    #tmp_data = random.randint(offset, offset + width, 2*n)
    tmp_data = random.randn(2 * n) * sigma
    tmp_data += offset
    tmp_data += amp * np.sin(np.arange(2 * n) * 10 / n) / 2
    tmp_data = tmp_data[:n] + 1j * tmp_data[n:]
elif SOURCE == 'NS':
    tmp_data = cross_data[freq, 0, :n]
elif SOURCE == 'EW':
    tmp_data = cross_data[freq, 2, :n]
data = np.empty(n, dtype=[('r', DTYPE), ('i', DTYPE)])
data['r'] = tmp_data.real
data['i'] = tmp_data.imag

plt.plot(data['r'])
plt.plot(data['i'])

#plt.figure()
#plt.plot(norm[0,0,:n])
#plt.show()


print data['r']

f = h5py.File('tmp_shuffle_test.h5', 'w')

# Standard filters.
f.create_dataset('uncompressed', data=data, chunks=(n,))
f.create_dataset('gzip', data=data, chunks=(n,), compression='gzip',
        compression_opts=1)
f.create_dataset('lzf', data=data, chunks=(n,), compression='lzf')
f.create_dataset('gzip_shuff', data=data, chunks=(n,), compression='gzip',
        compression_opts=1, shuffle=True)
f.create_dataset('lzf_shuff', data=data, chunks=(n,), compression='lzf',
        shuffle=True)

#data = random.randint(-123, 123, n).astype(np.int32) * 2**11
#print data

# Bit shuffle non-filter.
itemsize = data.dtype.itemsize
bitshuff_data = data.copy().view(np.uint8)
print np.reshape(bitshuff_data, (n, itemsize))[:,::-1]
bitshuff_data = np.unpackbits(bitshuff_data)
bitshuff_data.shape = (n, itemsize * 8)
bitshuff_data = (bitshuff_data.T).copy()
bitshuff_data = np.packbits(bitshuff_data.flat[:])
print np.reshape(np.reshape(bitshuff_data, (itemsize, n))[::-1], (itemsize * 8, n//8))
data = bitshuff_data.view(np.int32)
f.create_dataset('gzip_bit_shuff', data=data, chunks=(n,), compression='gzip',
        compression_opts=1)
f.create_dataset('lzf_bit_shuff', data=data, chunks=(n,), compression='lzf',)

f.close()

# Print results: ``$ h5dump -H -p tmp_shuffle_test.h5 | grep 'SIZE\|DATASET'``.
