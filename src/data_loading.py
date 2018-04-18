"""
Functions to pass the data to the
"""
import tensorflow as tf
import os

data_folder = '../data/dataset_audiolabs_crossera/'
DEFAULTS = [['']] + [[0.]] * 1537  # 1537 = 1 (time) + 12*128 (chroma features)
DEFAULTS[0] = ['']

# COLUMNS = ['songID', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']


def parse_csv(line):
    columns = tf.decode_csv(line, record_defaults=DEFAULTS)  # take a line at a time
    # columns = tf.reshape(columns, [-1, 85, 12])
    # features = dict(zip(COLUMNS, columns))  # create a dictionary out of the features

    features = {'songID': columns[0], 'time': columns[1], 'x': tf.stack(columns[2:])}  # create a dictionary out of the features
    labels = features.pop('songID')  # define the label
    return features, labels


def train_input_fn(folder="../data/dataset_audiolabs_crossera/by_song/", batch_size=128):
    """Generate an input function for the Estimator."""
    # data_file = [folder + fp for fp in os.listdir(folder)][0:3]  # take only three songs so far
    # to simplify, take only three songs (predefined)
    file_paths = ["chroma-nnls_CrossEra-0861.csv", "chroma-nnls_CrossEra-1444.csv", "chroma-nnls_CrossEra-1516.csv"]
    data_file = [folder + fp for fp in file_paths]  # take only three songs so far

    # Extract lines from input files using the Dataset API.
    # input_string = tf.train.string_input_producer(data_file)
    dataset = tf.data.TextLineDataset(data_file)

    dataset = dataset.map(parse_csv)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.shuffle(10_000_000).repeat().batch(batch_size)

    # A longer example of what we do in the return
    # iterator = dataset.make_one_shot_iterator()
    # features, labels = iterator.get_next()
    # return features, labels
    return dataset.make_one_shot_iterator().get_next()  # this will create a tuple (features, labels)


def test_input_fn(data_file="../data/dataset_audiolabs_crossera/test.csv"):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found.' % data_file)

    # Extract lines from input files using the Dataset API.
    # the test.csv file contains 5 songs: the three selected for training + 2 more
    dataset = tf.data.TextLineDataset(data_file)
    # It contains also just 50 examples
    dataset = dataset.map(parse_csv, num_parallel_calls=3).shuffle(50).repeat().batch(50)

    return dataset.make_one_shot_iterator().get_next()  # this will create a tuple (features, labels)
