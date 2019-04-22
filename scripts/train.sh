#!/bin/bash

# Generate data sets from these SGF files.
SGF_TRAINING="/home/tc/SGF/training/*/*.sgf"
SGF_TEST="/home/tc/SGF/test/*/*.sgf"

# Training data and test data will be written to these locations.
TRAINING_RIO="/home/tc/SGF/rio/training/data"
TEST_RIO="/home/tc/SGF/rio/test/data"

NUM_EPOCHS=10
RESUME_FROM=""
TRAINED_MODEL="/home/tc/SGF/models/`date "+%Y%m%d"`.h5"

# Replace the extension with .pb
CONVERTED_MODEL="${TRAINED_MODEL%.h5}.pb"

function gen_datasets() {
  bazel build -c opt model:gen_dataset

  echo "Generating the test data set from $SGF_TEST:"
  bazel-bin/model/gen_dataset   \
    --examples_per_file=100000  \
    --sgf_files="$SGF_TEST"     \
    --output="$TEST_RIO"

  echo "Generating the training data set from $SGF_TRAINING:"
  bazel-bin/model/gen_dataset    \
    --examples_per_file=100000   \
    --sgf_files="$SGF_TRAINING"  \
    --output="$TRAINING_RIO"
}

function train() {
  python ./model/train_dual_net.py           \
    --train_records="${TRAINING_RIO}-*.rio"  \
    --test_records="${TEST_RIO}-*.rio"       \
    --num_epochs="${NUM_EPOCHS}"             \
    --load_model="${RESUME_FROM}"            \
    --trained_model="${TRAINED_MODEL}"
}

function eval() {
  bazel build -c opt model:eval

  echo "Converting the trained model from .h5 to .pb"
  python model/keras_to_tensorflow.py      \
    --input_model="${TRAINED_MODEL}"       \
    --output_model="${CONVERTED_MODEL}"    \
    --output_nodes_prefix="go_output/"

  echo "Run eval..."
  bazel-bin/model/eval           \
  --sgf_files="${SGF_TEST}"      \
  --model="${CONVERTED_MODEL}"   \
  --input_layer_name="go_input"  \
  --output_layer_prefix="go_output/"
}

function print_help_message() {
  echo "Usage: train.sh [gen_data|train|eval]"
}

if [[ "$1" = "gen_data" ]]; then
  gen_datasets
elif [[ "$1" = "train" ]]; then
  train
elif [[ "$1" = "eval" ]]; then
  eval
else
  echo "Unknown command."
  print_help_message
fi
