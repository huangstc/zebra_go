#!/bin/bash

# Generate data sets from these SGF files.
SGF_TRAINING="/home/tc/SGF/training/*/*.sgf"
SGF_TEST="/home/tc/SGF/test/*/*.sgf"

# Training data and test data will be written to these locations.
TRAINING_RIO="/home/tc/SGF/rio/training/data"
TEST_RIO="/home/tc/SGF/rio/test/data"

function gen_datasets() {
  bazel build -c opt model:gen_dataset

  echo "Generating the test data set from $SGF_TEST:"
  bazel-bin/model/gen_dataset  --examples_per_file=100000 \
    --sgf_files="$SGF_TEST" \
    --output="$TEST_RIO"

  echo "Generating the training data set from $SGF_TRAINING:"
  bazel-bin/model/gen_dataset  --examples_per_file=100000 \
    --sgf_files="$SGF_TRAINING" \
    --output="$TRAINING_RIO"
}

function eval() {
  echo "Not implemented."
}

function print_help_message() {
  echo "Usage: train.sh gen_dataset"
}

if [[ "$1" = "gen_dataset" ]]; then
  gen_datasets
elif [[ "$1" = "eval" ]]; then
  eval
else
  echo "Unknown command."
  print_help_message
fi
