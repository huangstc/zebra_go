#!/bin/bash

# Because Sabaki overwrites this environment variable. The program needs this
# CUDA library path.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >&2

cd "`dirname $0`/.."
echo "current directory: '$PWD'" >&2

mkdir -p log
export GLOG_log_dir=$PWD/log
echo "log to '$GLOG_log_dir'"

MODEL="testdata/20190422.pb"
#MODEL=“testdata/20190530.pb”
INPUT_LAYER="go_input"
OUTPUT_LAYER_PREFIX="go_output/"

exec bazel-bin/engine/zebra_go  \
  --logtostderr=false --log_dir=$GLOG_log_dir \
  --simple_engine=false --simple_scorer=false  \
  --model=$MODEL \
  --input_layer_name=$INPUT_LAYER \
  --output_layer_prefix=$OUTPUT_LAYER_PREFIX
