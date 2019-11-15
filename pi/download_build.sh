#!/bin/bash
# Modified downloads and builds code for the raspberry pi.

HOME=$(pwd)

# Download tvm and build the runtime
git clone --recursive https://github.com/apache/incubator-tvm.git tvm
git checkout d4ca627a5a5df88f477bd6cc89ee2e3e06931c29
cd tvm
mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make runtime

# Download pytorch
cd $HOME
git clone --recursive  https://github.com/pytorch/pytorch.git
cd pytorch
echo $(pwd)
git checkout 7a3c38ab595ea78f16935df788c4982a0ec56966
git submodule init
# Ignore this unneeded submodule that break the build
git submodule deinit third_party/nervanagpu
git submodule deinit third_party/benchmark
git submodule deinit third_party/ideep
git submodule update
# Add test scripts + code for measuring time, also update cmake.config to use -O3
cp ../ulp* caffe2/core
cp ../CMakeLists.txt caffe2/
cp ../build_raspbian.sh scripts/

./scripts/build_raspbian.sh