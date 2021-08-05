HDF5_VERSION=$1

# Download and install HDF5 $HDF5_VERSION from source for building wheels
curl https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_VERSION%.*}/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz --output hdf5-$HDF5_VERSION.tar.gz --silent
tar -xvf hdf5-$HDF5_VERSION.tar.gz
cd hdf5-$HDF5_VERSION
chmod +x autogen.sh
./autogen.sh
CFLAGS=-std=c99 ./configure --prefix=/usr/local
make -j 6
make install
cd ..
