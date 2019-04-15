#!/bin/bash

set -e

######## PARAMETERS #########################################
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dst_dir="${script_dir}/tensorflow"
tmp_dir="/tmp/build_tf"
tmp_pkg_dir="/tmp/tensorflow_pkg"

tf_commit_tag="v1.13.1"

cc_opt_flags="${CC_OPT_FLAGS:--march=native}"

export PYTHON_BIN_PATH="/usr/bin/python3"
export PYTHON_LIB_PATH="/usr/lib/python3/dist-packages"
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=10.1
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export CUDNN_INSTALL_PATH="/usr/local/cuda"
export TF_CUDNN_VERSION=7
export TF_NEED_TENSORRT=0
export TF_CUDA_COMPUTE_CAPABILITIES="6.1"
export TF_CUDA_CLANG=0
export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"
export TF_ENABLE_XLA=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_MPI=0
export TF_SET_ANDROID_WORKSPACE=0

#############################################################

rm -rfd ${tmp_dir}
rm -rfd ${tmp_pkg_dir}
mkdir -p ${tmp_dir}

rm -rf ${dst_dir}/*
mkdir -p ${dst_dir}

echo "Cloning tensorflow to ${tmp_dir}"
git clone https://github.com/tensorflow/tensorflow "${tmp_dir}"

pushd "${tmp_dir}"

echo "Checking out ${tf_commit_tag}"
git checkout "${tf_commit_tag}"

echo "Configuring tensorflow"
./configure

echo "Building tensorflow package"
bazel build -c opt --config=opt --copt="${cc_opt_flags}" //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${tmp_pkg_dir}

echo "Tensorflow built-ish"
echo "Unpacking tensorflow package..."
unzip -q ${tmp_pkg_dir}/tensorflow-*.whl -d ${tmp_dir}

echo "Copying tensor flow headers to ${dst_dir}"
cp -r ${tmp_dir}/tensorflow-*.data/purelib/tensorflow/include/* "${dst_dir}"

echo "Building tensorflow libraries"

bazel build -c opt --config=opt --copt="${cc_opt_flags}" \
    //tensorflow:libtensorflow_cc.so \
    //tensorflow:libtensorflow_framework.so

echo "Copying tensorflow libraries to ${dst_dir}"
cp bazel-bin/tensorflow/libtensorflow_*.so "${dst_dir}"
