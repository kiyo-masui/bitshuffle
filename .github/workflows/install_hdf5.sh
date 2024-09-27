HDF5_DISTRIBUTION=$1
# Extract the patch number
IFS='-' read -ra ADDR <<< "$HDF5_DISTRIBUTION"

HDF5_VERSION=${ADDR[0]}

# Download and install HDF5 $HDF5_DISTRIBUTION from source for building wheels
curl https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION%.*}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_DISTRIBUTION.tar.gz -O -s
tar -xzf hdf5-$HDF5_DISTRIBUTION.tar.gz
cd hdf5-$HDF5_DISTRIBUTION
./configure --prefix=/usr/local
make -j 2
make install
cd ..
