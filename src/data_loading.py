"""
Functions to pass the data to the
"""
import tensorflow as tf
import os

data_folder = '../data/dataset_audiolabs_crossera/'
DEFAULTS = [['']] + [[0.]] * 1537  # 1537 = 1 (time) + 12*128 (chroma features)
DEFAULTS[0] = ['']


# COLUMNS = ['songID', 'time', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']


def parse_csv(line):
    columns = tf.decode_csv(line, record_defaults=DEFAULTS)  # take a line at a time
    labels = columns[0]
    time = columns[1]
    x = tf.stack(columns[2:])
    return x, time, labels


def train_input_fn(input_path, batch_size=128, shuffle_buffer=10_000):
    """Generate an iterator to produce the training input."""
    if os.path.isdir(input_path):
        data_file = [input_path + fp for fp in os.listdir(input_path)]
    elif os.path.isfile(input_path):
        data_file = input_path
    else:
        raise ValueError("please specify a valid path, folder or file")

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    # We call repeat after shuffling, rather than before, to prevent separate epochs from blending together.
    dataset = dataset.map(parse_csv).shuffle(shuffle_buffer).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator()


def test_input_fn(data_file, batch_size=100, shuffle_buffer=100):
    """Generate an iterator to produce the test input."""
    assert tf.gfile.Exists(data_file), (
            '%s not found.' % data_file)

    # Extract lines from input files using the Dataset API.
    # the test.csv file contains 5 songs: the three selected for training + 2 more
    dataset = tf.data.TextLineDataset(data_file)
    # It contains also just 50 examples
    dataset = dataset.map(parse_csv, num_parallel_calls=3).shuffle(shuffle_buffer).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator()
