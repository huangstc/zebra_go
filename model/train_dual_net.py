import glob
import os
import numpy as np
import tensorflow as tf

from absl import app, flags
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import load_model

flags.DEFINE_integer('num_epochs', 10, 'Number of training epochs.')
flags.DEFINE_string('load_model', None, 'Resume training from a checkpoint.')
flags.DEFINE_string('train_records', None, 'Path to the training dataset.')
flags.DEFINE_string('test_records', None, 'Path to the test dataset.')
flags.DEFINE_string('trained_model', None, 'Path to the saved model, ending with .h5')

FLAGS = flags.FLAGS

BOARD_SIZE = 19
NUM_CHANNELS = 7
BATCH_SIZE = 64
SHUFFLE_BUFFER = 12800

FEATURES = {"next": tf.FixedLenFeature([], tf.int64),
            "outcome": tf.FixedLenFeature([], tf.float32),
            "note": tf.VarLenFeature(tf.string),
            "orig": tf.VarLenFeature(tf.float32),
            "b1": tf.VarLenFeature(tf.float32),
            "b2": tf.VarLenFeature(tf.float32),
            "b3": tf.VarLenFeature(tf.float32),
            "w1": tf.VarLenFeature(tf.float32),
            "w2": tf.VarLenFeature(tf.float32),
            "w3": tf.VarLenFeature(tf.float32)}


def _parse_full_example(example):
    return tf.parse_single_example(example, FEATURES)


def _reshape_plane(t):
    return tf.reshape(tf.sparse.to_dense(t), [BOARD_SIZE, BOARD_SIZE])


def _reshape_planes(feature_dict, names):
    return [_reshape_plane(feature_dict[name]) for name in names]


def ParseForPolicyValue(example):
    parsed = _parse_full_example(example)
    planes = _reshape_planes(parsed, ["orig", "b1", "b2", "b3", "w1", "w2", "w3"])
    return tf.stack(planes, axis=2), (parsed["next"], parsed["outcome"])


def BuildCommonLayers(input):
    x = Conv2D(filters=64, kernel_size=(7,7), activation="relu", padding="same",
               data_format='channels_last')(input)
    x = Conv2D(filters=64, kernel_size=(7,7), activation="relu", padding="same",
               data_format='channels_last')(x)
    x = Conv2D(filters=32, kernel_size=(5,5), activation="relu", padding="same",
               data_format='channels_last')(x)
    x = Conv2D(filters=32, kernel_size=(5,5), activation="relu", padding="same",
               data_format='channels_last')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=32, kernel_size=(5,5), activation="relu", padding="same",
               data_format='channels_last')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    return x


def BuildDualNetModel():
    input = Input(shape=(BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), name="go_input")
    common = BuildCommonLayers(input)

    # policy network
    x = Dense(BOARD_SIZE*BOARD_SIZE)(common)
    policy_output = Activation("softmax", name="policy_output")(x)

    # value network
    x = Dense(1)(common)
    value_output = Activation("sigmoid", name="value_output")(x)

    model = keras.models.Model(inputs=input, outputs=[policy_output, value_output])
    return model


def GetModel():
    model = load_model(FLAGS.load_model) if FLAGS.load_model else BuildDualNetModel()
    return model


def CompileModel(model):
    losses = {"policy_output": "sparse_categorical_crossentropy",
              "value_output": "binary_crossentropy"}
    loss_weights = {"policy_output": 1.0, "value_output": 1.0}
    model.compile(optimizer="sgd", loss=losses, loss_weights=loss_weights,
                  metrics=["accuracy"])
    print(model.summary())
    return model


def CreateDataset(files, batch_size):
    # Also set in cc/gen_dataset.cc
    dataset = tf.data.TFRecordDataset(files, "ZLIB")
    dataset = dataset.map(ParseForPolicyValue)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset


def main(argv):
    del argv  # Unused
    flags.mark_flag_as_required('train_records')
    flags.mark_flag_as_required('test_records')
    flags.mark_flag_as_required('trained_model')
    print("Training with %s" % FLAGS.train_records)
    print("Model will be saved to %s" % FLAGS.trained_model)

    train_dataset = CreateDataset(glob.glob(FLAGS.train_records), BATCH_SIZE)
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER)

    model = GetModel()
    model = CompileModel(model)
    checkpoint_path = "checkpoints/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                          save_weights_only=True, verbose=1)

    model.fit(train_dataset, epochs=FLAGS.num_epochs, steps_per_epoch=100000,
              callbacks = [checkpoint_callback])
    model.save(FLAGS.trained_model)
    print("Model is saved to %s" % FLAGS.trained_model)

    print("Testing with %s" % FLAGS.test_records)
    test_dataset = CreateDataset(glob.glob(FLAGS.test_records), BATCH_SIZE)
    eval_result = model.evaluate(test_dataset, steps=1000)
    print("weighted loss: %f" % eval_result[0])
    print("policy loss: %f" % eval_result[1])
    print("value loss: %f" % eval_result[2])
    print("policy accuracy: %f" % eval_result[3])
    print("value accuracy: %f" % eval_result[4])


if __name__ == '__main__':
    app.run(main)
