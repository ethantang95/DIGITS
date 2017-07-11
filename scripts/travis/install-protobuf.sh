#!/bin/bash
# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.

set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ "$#" -ne 1 ];
then
    echo "Usage: $0 INSTALL_DIR"
    exit 1
fi

INSTALL_DIR=$(readlink -f "$1")
"${LOCAL_DIR}/bust-cache.sh" "$INSTALL_DIR"
if [ -d "$INSTALL_DIR" ] && [ -e "$INSTALL_DIR/install/bin/th" ]; then
    echo "Using cached build at $INSTALL_DIR ..."
    exit 0
fi
rm -rf "$INSTALL_DIR"

set -x

git clone https://github.com/google/protobuf.git -b '3.3.x' $INSTALL_DIR
cd $INSTALL_DIR
./autogen.sh
./configure --prefix=/usr
make -j$(nproc)
make install
ldconfig
cd $INSTALL_DIR/python
python setup.py install --cpp_implementation