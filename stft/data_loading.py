import os

import tensorflow as tf

from stft.config import PARAMS


def _parse_function(proto):
    f = {
        "composer_id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "song_id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "x": tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)
    }
    parsed_features = tf.io.parse_single_example(proto, f)
    comp_id = parsed_features["composer_id"]
    # song_id = parsed_features["song_id"]
    x = tf.reshape(tf.cast(parsed_features["x"], tf.float32), PARAMS['x.shape'][1:])
    return x, comp_id  # , song_id


def create_tfrecords_dataset(input_path, batch_size, shuffle_buffer=None):
    dataset = tf.data.TFRecordDataset(input_path).map(_parse_function, num_parallel_calls=16)
    if shuffle_buffer is None:
        dataset = dataset.repeat().batch(batch_size).prefetch(2)
    else:
        dataset = dataset.shuffle(shuffle_buffer).repeat().batch(batch_size).prefetch(2)
    return dataset
