#!/bin/bash
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ ! -z "$DEB_BUILD" ]; then
    echo "Skipping for deb build"
    exit 0
fi

set -x

pip install protobuf -U

pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0rc0-cp27-none-linux_x86_64.whl --upgrade
