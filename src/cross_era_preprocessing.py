"""
All the functions that we need in order to read the database in cross_era and obtain features.
The database can be found at the URL https://www.audiolabs-erlangen.de/resources/MIR/cross-era
We modify it slightly to change the first column into a label column, where only the CrossEra ID is kept, and to
remove the time column.
The idea today is to have a huge training file containing all chroma features and on the left column the label.
Then we could fish N consecutive lines from this file and keep them iff the label is always the same (not to mix songs)
We repeat this B times and we create our batch of training examples.
"""
import numpy as np
import tensorflow as tf

data_folder = '../data/dataset_audiolabs_crossera/'
DEFAULTS = [[0.] for i in range(13)]
DEFAULTS[0] = [0]

COLUMNS = ['songID', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']


def parse_csv(line):
    print('Parsing')
    columns = tf.decode_csv(line, record_defaults=DEFAULTS)  # take a line at a time
    features = dict(zip(COLUMNS, columns))  # create a dictionary out of the features
    labels = features.pop('songID')  # define the label
    return features, labels


def train_input_fn(data_file="../data/dataset_audiolabs_crossera/chroma.csv", batch_size=128):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found.' % data_file)

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    dataset = dataset.map(parse_csv)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.shuffle(1000000).repeat().batch(batch_size)

    # A longer example of what we do in the return
    # iterator = dataset.make_one_shot_iterator()
    # features, labels = iterator.get_next()
    # return features, labels
    return dataset.make_one_shot_iterator().get_next()  # this will create a tuple (features, labels)


def test_input_fn(data_file="../data/dataset_audiolabs_crossera/chroma.csv"):
    """Generate an input function for the Estimator."""
    assert tf.gfile.Exists(data_file), (
        '%s not found.' % data_file)

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    dataset = dataset.map(parse_csv, num_parallel_calls=5)

    return dataset.make_one_shot_iterator().get_next()  # this will create a tuple (features, labels)


# # filename_queue = tf.train.string_input_producer([data_folder + 'chroma_0849.csv', data_folder + 'chroma_1422.csv'])
# filename_queue = tf.train.string_input_producer([data_folder + 'chroma.csv'])
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)
#
# # Default values, in case of empty columns. Also specifies the type of the decoded result.
# record_defaults = [[0.] for i in range(13)]
# record_defaults[0] = [0]
# features = tf.decode_csv(value, record_defaults=record_defaults)
# label = features.pop(0)
#
# with tf.Session() as sess:
#     # Start populating the filename queue.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for _ in range(10):
#         # Retrieve a single line:
#         example = sess.run([features, label])
#         print(example)
#
#     coord.request_stop()
#     coord.join(threads)
#
