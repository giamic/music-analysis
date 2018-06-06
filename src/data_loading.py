"""
Functions to pass the data to the
"""

import os

import tensorflow as tf


def _parse_function(proto):
    f = {
        "song_id": tf.FixedLenSequenceFeature([], tf.int64, default_value=1, allow_missing=True),
        "time": tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        "x": tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)
    }
    parsed_features = tf.parse_single_example(proto, f)
    song_id = parsed_features["song_id"]
    time = parsed_features["time"]
    x = parsed_features["x"]
    return x, song_id, time


def create_tfrecords_iterator(input_path, batch_size, shuffle_buffer):
    """

    :param input_path:
    :param batch_size:
    :param shuffle_buffer:
    :return:
    """
    if os.path.isdir(input_path):
        data_file = [os.path.join(input_path, fp) for fp in os.listdir(input_path)]
    elif os.path.isfile(input_path):
        data_file = input_path
    else:
        raise ValueError("please specify a valid path, folder or file")
    dataset = tf.data.TFRecordDataset(data_file)
    if shuffle_buffer is None:
        dataset = dataset.map(_parse_function, num_parallel_calls=16).repeat().batch(batch_size)
    else:
        dataset = dataset.map(_parse_function, num_parallel_calls=16).shuffle(shuffle_buffer).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator()
