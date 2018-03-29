"""
All the functions that we need in order to read the database in cross_era and obtain features.
The idea today is to have a huge training file containing all chroma features and on the left column the ID.
Then we could fish N consecutive lines from this file and keep them iff the ID is always the same (not to mix songs)
We repeat this B times and we create our batch of training examples.
"""
import numpy as np
import tensorflow as tf

data_folder = '../data/dataset_audiolabs_crossera/'

# filename_queue = tf.train.string_input_producer([data_folder + 'chroma_0849.csv', data_folder + 'chroma_1422.csv'])
filename_queue = tf.train.string_input_producer([data_folder + 'chroma_1422.csv'])
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the decoded result.
record_defaults = [[0.] for i in range(12)]
features = tf.decode_csv(value, record_defaults=record_defaults)


with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1200):
        # Retrieve a single instance:
        example = sess.run([features, key])
        print(example)

    coord.request_stop()
    coord.join(threads)

