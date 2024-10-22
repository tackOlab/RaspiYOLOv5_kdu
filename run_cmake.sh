#!/bin/bash

cmake -B./build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=/usr/bin/clang++-16 -DCMAKE_LINKER=/usr/bin/ld.lld-16
