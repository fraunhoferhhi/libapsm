#!/bin/env bash

# show bin files in a better way
git config --local diff.hex.textconv "hexdump -v -C"
git config --local diff.hex.binary true

# clone all submodules
git submodule update --init --recursive
#git submodule update --remote

#export CMAKE_GENERATOR="Unix Makefiles"
#export CMAKE_BUILD_PARALLEL_LEVEL=3
#export CMAKE_GENERATOR="Ninja"
export CMAKE_BUILD_TYPE=Release
#export CMAKE_EXPORT_COMPILE_COMMANDS=ON

git clean -xdf build
cmake -S cpp -B build
cmake --build build --target all
