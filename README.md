# zebra_go

Yet another computer Go program.

## Prerequisites

The following packages and software are required to build and run this project:
* [Bazel](https://github.com/bazelbuild/bazel)
* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive).

Choose the CUDA version that is compatible with your TensorFlow version. For example, as of 2019/04/14, CUDA 10.1 is not officially supported by TensorFlow 1.13.1.

## Building
First, run ```model/configure_tf.sh``` to build TensorFlow from source codes. Some parameters at the beginning of the script may need tweaking to correctly build TensorFlow on your system. TensorFlow version that will be used by all C++ codes is also configured here.

```bash
$ git clone https://github.com/huangstc/zebra_go.git
$ cd zebra_go
$ chmod u+x ./model/configure_tf.sh
$ ./model/configure_tf.sh
```

Then build the project:
```bash
$ bazel build engine:all
$ bazel build model:all
```

## Generate Training/Test Data
The model is trained by human games. It is recommended to train it with professional or strong amateur players games. [Sensei's library](https://senseis.xmp.net/?GoDatabases) lists some databases of game records.

Assume you have collected some records in SGF format and put them in ```/tmp/training_sgf``` and ```/tmp/test_sgf``` respectively. Edit ```scripts/train.sh``` to set inputs and outputs:

```
# Generate data sets from these SGF files.
SGF_TRAINING="/tmp/training_sgf"
SGF_TEST="`/tmp/test_sgf"

# Training data and test data will be written to these locations.
TRAINING_RIO="/tmp/training_rio/data"
TEST_RIO="/tmp/test_rio/data"
```

Then, run this command to build training data and test data from SGF files.
```bash
$ chmod +x ./scripts/train.sh
$ ./scripts/train.sh gen_datasets
```
