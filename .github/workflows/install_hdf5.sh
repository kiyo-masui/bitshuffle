# Download and install HDF5 1.10.7 from source for building wheels
curl https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.7/src/hdf5-1.10.7.tar.gz --output hdf5-1.10.7.tar.gz --silent
tar -xvf hdf5-1.10.7.tar.gz
cd hdf5-1.10.7
chmod +x autogen.sh
./autogen.sh
./configure --prefix=/usr/local
make -j 6
make install
cd ..
