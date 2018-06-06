"""
Functions to pass the data to the
"""

import os

import tensorflow as tf

DEFAULTS = [['']] + [[0.]] * 1537  # 1537 = 1 (time) + 12*128 (chroma features)
DEFAULTS[0] = ['']


def parse_csv(line):
    columns = tf.decode_csv(line, record_defaults=DEFAULTS)  # take a line at a time
    song_id = columns[0]
    time = columns[1]
    x = tf.stack(columns[2:])
    return x, song_id, time


def train_input_fn(input_path, batch_size=128, shuffle_buffer=100_000):
    """Generate an iterator to produce the training input."""
    if os.path.isdir(input_path):
        data_file = [os.path.join(input_path, fp) for fp in os.listdir(input_path)]
    elif os.path.isfile(input_path):
        data_file = input_path
    else:
        raise ValueError("please specify a valid path, folder or file")

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    # We call repeat after shuffling, rather than before, to prevent separate epochs from blending together.
    if shuffle_buffer is None:
        dataset = dataset.map(parse_csv).repeat().batch(batch_size)
    else:
        dataset = dataset.map(parse_csv).shuffle(shuffle_buffer).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator()


def test_input_fn(input_path, batch_size, shuffle_buffer):
    """Generate an iterator to produce the test input."""
    if os.path.isdir(input_path):
        data_file = [input_path + fp for fp in os.listdir(input_path)]
    elif os.path.isfile(input_path):
        data_file = input_path
    else:
        raise ValueError("please specify a valid path, folder or file")

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)
    if shuffle_buffer is None:
        dataset = dataset.map(parse_csv).repeat().batch(batch_size)
    else:
        dataset = dataset.map(parse_csv).shuffle(shuffle_buffer).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator()
