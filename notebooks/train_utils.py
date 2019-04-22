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
    """Parse an example for training the policy network.
    """
    parsed = _parse_full_example(example)
    planes = _reshape_planes(parsed, ["orig", "b1", "b2", "b3", "w1", "w2", "w3"])
    return tf.stack(planes, axis=2), parsed["next"]


def ParseForPolicyValue(example):
    parsed = _parse_full_example(example)
    planes = _reshape_planes(parsed, ["orig", "b1", "b2", "b3", "w1", "w2", "w3"])
    return tf.stack(planes, axis=2), (parsed["next"], parsed["outcome"])


def CreateDataset(files, batch_size):
    # Also set in cc/gen_dataset.cc
    dataset = tf.data.TFRecordDataset(files, "ZLIB")
    dataset = dataset.map(ParseForPolicy)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset


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


def TestExample(model, input_tensor):
    feature_set = _parse_full_example(input_tensor)
    print(feature_set["note"])
    
    planes, label = ParseForPolicy(input_tensor)
    output = model(tf.reshape(planes, shape=(1,19,19,7)))
    output_arr = output.numpy()  # shape=(1, 361)
    top_10 = np.flip(output_arr[0,:].argsort()[-10:])
    
    bb = DrawGoBoard(planes.numpy()[:,:,0])
    golden_x, golden_y = LabelToCoord(label)
    print("Golden: (%d,%d)" % (golden_x, golden_y))

    idx = 0
    for p in top_10:
        x, y = LabelToCoord(p)
        bb.AddSquare(x, y, idx)
        print("#" + str(idx) + ": (%d,%d) with score=%f" % (x, y, output_arr[0, p]))
        idx += 1
    return bb