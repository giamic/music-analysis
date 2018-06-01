"""
Functions to pass the data to the
"""
from itertools import islice
from random import choice

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import csv

DEFAULTS = [['']] + [[0.]] * 1537  # 1537 = 1 (time) + 12*128 (chroma features)
DEFAULTS[0] = ['']


# COLUMNS = ['songID', 'time', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']


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
    # the test.csv file contains 5 songs: the three selected for training + 2 more
    dataset = tf.data.TextLineDataset(data_file)
    # It contains also just 50 examples
    dataset = dataset.map(parse_csv).shuffle(shuffle_buffer).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator()


def g(file_paths, composers, lengths, steps):
    while True:
        idx_cmp = choice(np.arange(len(file_paths)))
        c = composers[idx_cmp]
        fp = file_paths[idx_cmp]
        l = lengths[idx_cmp]
        idx_sng = choice(np.arange(len(l) - 1))
        ni, nf = l[idx_sng], l[idx_sng + 1]
        n = choice(np.arange(ni, max(nf - steps, ni)))
        with open(fp, 'r') as f:
            lines = list(islice(f, n, n + steps))
            # datareader = csv.reader(f, delimiter=',')
            # x = []
            # for _ in range(n):
            #     next(datareader)
            # for i in range(steps):
            #     x.append(next(datareader)[2:])
        # yield c, n, np.array(x)
        yield lines


if __name__ == '__main__':
    data_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data',
                               'dataset_audiolabs_crosscomposer', 'chroma_features')
    sl = pd.read_csv(os.path.join(data_folder, 'song_lengths.csv'), header=None)
    sl = sl.values[:, 1:]
    paths = sorted([os.path.join(data_folder, x) for x in os.listdir(data_folder)])
    composers = [
        'Bach JS',
        'Beethoven',
        'Brahms',
        'Dvorak',
        'Handel',
        'Haydn',
        'Mendelssohn',
        'Mozart',
        'Rameau',
        'Schubert',
        'Shostakovich',
    ]
    steps = 10
    z = g(paths, composers, sl, steps)
    print(next(z))
