"""
Functions to pass the data to the
"""

import os

import tensorflow as tf


def _parse_function(proto):
    f = {
        "composer_id": tf.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
        "recording_id": tf.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
        "song_id": tf.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
        "time": tf.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
        "x": tf.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)
    }
    parsed_features = tf.parse_single_example(proto, f)
    comp_id = parsed_features["composer_id"]
    reco_id = parsed_features["recording_id"]
    song_id = parsed_features["song_id"]
    time = parsed_features["time"]
    x = tf.cast(parsed_features["x"], tf.float32)
    return x, comp_id, reco_id, song_id, time


def create_tfrecords_iterator(input_path, batch_size, shuffle_buffer):
    """
    Create an iterator over the TFRecords file with chroma features.

    :param input_path: can accept both a file and a folder
    :param batch_size:
    :param shuffle_buffer: if None, don't shuffle
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

