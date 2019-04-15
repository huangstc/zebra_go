import numpy as np
import tensorflow as tf

from visualization import SvgGoBoard

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
    return tf.reshape(tf.sparse.to_dense(t), [19, 19])


def _reshape_planes(feature_dict, names):
    return [_reshape_plane(feature_dict[name]) for name in names]


def ParseForPolicy(example):
    parsed = _parse_full_example(example)
    planes = _reshape_planes(parsed, ["orig", "b1", "b2", "b3", "w1", "w2", "w3"])
    return tf.stack(planes, axis=2), parsed["next"]  # , parsed["outcome"]


def LabelToCoord(label):
    x = label % 19
    y = int(label / 19)
    return (x, y)


def DrawGoBoard(plane):
    shape = plane.shape
    board = SvgGoBoard(shape[0], shape[1], '00')
    for x in range(shape[0]):
        for y in range(shape[1]):
            v = plane[y, x]
            if v > 0.5:
                board.AddStone(x, y, 1)
            elif v < -0.5:
                board.AddStone(x, y, -1)
    return board