# zebra_go

Yet another computer Go program.

# Guide for Players and Developers
## Prerequisites

The following packages and software are required to build and run this project:
* [Bazel](https://github.com/bazelbuild/bazel)
* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive)
* [Sabaki](https://sabaki.yichuanshen.de/)

Choose the CUDA version that is compatible with your TensorFlow version. For example, as of 2019/04/14, CUDA 10.1 is not officially supported by TensorFlow 1.13.1.

## Building
First, run ```model/configure_tf.sh``` to build TensorFlow from source codes. Some parameters at the beginning of the script may need tweaking to correctly build TensorFlow on your system. TensorFlow version that will be used by all C++ codes is also configured here.

```bash
$ git clone https://github.com/huangstc/zebra_go.git
$ cd zebra_go
$ chmod +x ./model/configure_tf.sh
$ ./model/configure_tf.sh
```

Then build the project:
```bash
$ bazel build engine:all
$ bazel build model:all
```

## Generate Training/Test Data
The model is trained on human games. It is recommended to train it with professional or strong amateur players' games. [Sensei's library](https://senseis.xmp.net/?GoDatabases) lists some databases of Go game records.

Assume you have collected some records in SGF format and put them in ```/tmp/training_sgf``` and ```/tmp/test_sgf``` respectively. Edit ```scripts/train.sh``` to set inputs and outputs:

```
# Generate data sets from these SGF files.
SGF_TRAINING="/tmp/training_sgf"
SGF_TEST="/tmp/test_sgf"

# Training data and test data will be written to these locations.
TRAINING_RIO="/tmp/training_rio/data"
TEST_RIO="/tmp/test_rio/data"
```

Then, run this command to build training data and test data from SGF files.
```bash
$ chmod +x ./scripts/train.sh
$ ./scripts/train.sh gen_data
```

## Train and Evaluation

First of all, it is important to make sure that the version of TensorFlow used for training exactly matches the one linked to C++ binaries, otherwise the trained model may not be recognized by C++ binaries. The version for C++ is configured in `model/configure_tf.sh`.

Then, set the following parameters in `scripts/train.sh`:

```bash
# Data sets output from previous step.
TRAINING_RIO="/tmp/training_rio/data"
TEST_RIO="/tmp/test_rio/data"

NUM_EPOCHS=10
RESUME_FROM=""  # If you need to resume from a trained model.
TRAINED_MODEL="/tmp/models/`date "+%Y%m%d"`.h5"
```

Run the command to start training:
```bash
./scripts/train.sh train
```

After the training is done, run evaluation:
```bash
./scripts/train.sh eval
```

## Run

You can play againt the program with [Sabaki](https://sabaki.yichuanshen.de/), using the trained model in `testdata/<yyyymmdd>.pb`. First, build the binary:

```
bazel build -c opt engine:zebra_go
```

And make sure `scripts/start.sh` is using the latest model:
```bash
MODEL="testdata/20190422.pb"
INPUT_LAYER="go_input"
OUTPUT_LAYER_PREFIX="go_output/"
```

Then add ZebraGo as an engine of Sakaki, with its path pointing to `/your/git/dir/zebra_go/scripts/start.sh`.

# Acknowledgments

The project uses codes or ideas from the following projects:
* `model/configure_tf.sh` is copied from [MiniGo](https://github.com/tensorflow/minigo/blob/master/cc/configure_tensorflow.sh), with minor modifications.
* `model/keras_to_tensorflow.py` is copied from [Keras to TensorFlow](https://github.com/amir-abdi/keras_to_tensorflow) authored by Amir H. Abdi.
* Tencent's [PhoenixGo](https://github.com/Tencent/PhoenixGo) for connecting with Sabaki through Go Text Protocol.
