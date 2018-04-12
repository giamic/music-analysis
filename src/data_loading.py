"""
Functions to pass the data to the
"""
import tensorflow as tf

data_folder = '../data/dataset_audiolabs_crossera/'
DEFAULTS = [[0.] for i in range(1009)]
DEFAULTS[0] = [0]

# COLUMNS = ['songID', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']


def parse_csv(line):
    print('Parsing')
    columns = tf.decode_csv(line, record_defaults=DEFAULTS)  # take a line at a time
    # columns = tf.reshape(columns, [-1, 85, 12])
    # features = dict(zip(COLUMNS, columns))  # create a dictionary out of the features

    features = {'songID': columns[0], 'x': tf.stack(columns[1:])}  # create a dictionary out of the features
    labels = features.pop('songID')  # define the label
    return features, labels


def train_input_fn(data_file="../data/dataset_audiolabs_crossera/chroma_expanded.csv", batch_size=128):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found.' % data_file)

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    dataset = dataset.map(parse_csv)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.shuffle(1_000_000).repeat().batch(batch_size)

    # A longer example of what we do in the return
    # iterator = dataset.make_one_shot_iterator()
    # features, labels = iterator.get_next()
    # return features, labels
    return dataset.make_one_shot_iterator().get_next()  # this will create a tuple (features, labels)


def test_input_fn(data_file="../data/dataset_audiolabs_crossera/chroma_expanded.csv"):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found.' % data_file)

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    dataset = dataset.map(parse_csv, num_parallel_calls=3).shuffle(1_000_000)

    return dataset.make_one_shot_iterator().get_next()  # this will create a tuple (features, labels)
