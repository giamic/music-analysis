import logging
import os
from random import shuffle

import numpy as np
import pandas as pd


def create_random_dataset(data_folder, path_output, steps, n_excerpts, n_songs=None):
    """
    Create a file containing a dataset with random picks from the different songs.

    :param data_folder: where the data is stored per song
    :param path_output: where to store the test data
    :param steps: the number of time steps per row, convenience value to set the correct number of columns
    :param n_excerpts: the number of rows we take from each song
    :param n_songs: the number of songs we consider; if None, take all
    :return:
    """
    logger = logging.getLogger(__name__)
    if n_songs is None:
        file_paths = [data_folder + fp for fp in os.listdir(data_folder)]  # take all the songs
    else:  # take only n_songs so far
        file_paths = np.random.choice([data_folder + fp for fp in os.listdir(data_folder)], n_songs, replace=False)
    original_labels = ['songID', 'time', 'A_t', 'A#_t', 'B_t', 'C_t', 'C#_t', 'D_t', 'D#_t', 'E_t', 'F_t', 'F#_t',
                       'G_t', 'G#_t']
    N = len(file_paths)
    labels = original_labels[0:2]
    chromas = original_labels[2:]
    labels = labels + [c + str(s) for s in range(steps) for c in chromas]
    df_random = pd.DataFrame(columns=labels)
    for n, fp in enumerate(file_paths):
        logger.info("Working on {}, file {} out of {}".format(fp, n + 1, N))
        df = pd.read_csv(fp, header=None, names=labels)
        logger.info("File read, now concatenating")
        df_random = pd.concat([df_random, df.sample(n_excerpts)], ignore_index=True)
    df_random = df_random.sample(frac=1)  # shuffle the df_random
    df_random.to_csv(path_output, header=False, index=False)
    return


def create_test_dataset(data_folder, path_output, composers, id2cmp, steps, n_excerpts=10, n_songs=10):
    """
    Create a file containing a dataset with random picks from the different songs.

    :param data_folder: where the data is stored per song
    :param path_output: where to store the test data
    :param steps: the number of time steps per row, convenience value to set the correct number of columns
    :param n_excerpts: the number of rows we take from each song
    :param n_songs: the number of songs we consider; if None, take all
    :return:
    """
    composers = dict(zip(composers, np.ones(len(composers)) * n_songs))
    logger = logging.getLogger(__name__)
    file_paths = [data_folder + fp for fp in os.listdir(data_folder)]  # take all the songs
    shuffle(file_paths)
    original_labels = ['songID', 'time', 'A_t', 'A#_t', 'B_t', 'C_t', 'C#_t', 'D_t', 'D#_t', 'E_t', 'F_t', 'F#_t',
                       'G_t', 'G#_t']
    N = len(file_paths)
    labels = original_labels[0:2]
    chromas = original_labels[2:]
    labels = labels + [c + str(s) for s in range(steps) for c in chromas]
    df_random = pd.DataFrame(columns=labels)
    while sum(composers.values()) > 0:
        for n, fp in enumerate(file_paths):
            # logger.info("Working on {}, file {} out of {}".format(fp, n + 1, N))
            with open(fp) as f:
                id = f.readline()[:13]
                c = id2cmp[id]
            if c in composers and composers[c] > 0:
                df = pd.read_csv(fp, header=None, names=labels)
                df_random = pd.concat([df_random, df.sample(n_excerpts)], ignore_index=True)
                composers[c] -= 1
                logger.info("This piece was by {}. Still {} by him to find. Still {} in total.".format(c, composers[c], sum(composers.values())))
                if composers[c] == 0:
                    logger.info("Finished to analyse pieces by {}".format(c))
        if sum(composers.values()) > 0:
            logger.info("Doing a second round trip.")
            logger.info("{}".format(composers))
    df_random = df_random.sample(frac=1)  # shuffle the df_random
    df_random.to_csv(path_output, header=False, index=False)
    return


def create_annotations(data_folder, output_folder, ids, times):
    annotations = pd.read_csv(os.path.join(data_folder, 'cross-era_annotations.csv'))
    names = ["index", "CrossEra-ID", "ClipTime"]
    df = pd.DataFrame(dict(zip(names, [np.arange(len(ids)), ids, times])))
    res = df.merge(annotations, on="CrossEra-ID")
    res = res.sort_values("index")
    res = res[['Composer', 'CrossEra-ID', 'ClipTime', 'CompLifetime', 'SongYear']]
    res.to_csv(os.path.join(output_folder, 'metadata.tab'), sep='\t', header=True)
    return res


def find_id2cmp(input_path):
    logger = logging.getLogger(__name__)
    logger.info("Constructing the ID to composer look-up table...")
    data = pd.read_csv(input_path)
    logger.info("...done!")
    return data.iloc[:, 2].values, data.iloc[:, 6].values


def store_song_lengths(data_folder, output_file):
    paths = [os.path.join(data_folder, x) for x in os.listdir(data_folder)]
    with open(output_file, 'a') as f:
        for n, fp in enumerate(sorted(paths)):
            if os.path.split(fp)[-1].startswith("chroma-nnls"):
                df = pd.read_csv(fp, header=None)
                name = df.iloc[:, 0]
                starts = list(map(str, np.where(~name.isna())[0]))
                # line = os.path.split(fp)[-1] + ',' + ','.join(starts) + ',' + str(len(df)) + '\n'
                line = str(n) + ',' + ','.join(starts) + ',' + str(len(df)) + '\n'
                f.write(line)
    return


if __name__ == '__main__':
    # general_folder = "../data/dataset_audiolabs_crossera/"
    # by_song_folder = "/media/gianluca/data/PycharmProjects/music-analysis/data/dataset_audiolabs_crossera/by_song/"
    # T = 128  # how many successive steps we want to put in a single row
    #
    # composers = [
    #     'Bach; Johann Sebastian',
    #     'Beethoven; Ludwig van',
    #     'Brahms; Johannes',
    #     'Chopin; Frederic',
    #     'Debussy; Claude',
    #     'Grieg; Edvard',
    #     'Mahler; Gustav',
    #     'Hindemith; Paul',
    #     'Mozart; Wolfgang Amadeus',
    #     'Prokofiew; Sergej',
    #     'Shostakovich; Dmitri',
    # ]
    #
    # logging.basicConfig(level=logging.INFO)
    # preprocess(general_folder, by_song_folder, T)
    # create_random_dataset(by_song_folder, general_folder + 'train2.csv', T, 20)
    # create_random_dataset(by_song_folder, general_folder + 'test2.csv', T, 5, 20)

    # ids, cmp = find_id2cmp(general_folder + 'cross-era_annotations.csv')
    # id2cmp = dict(zip(ids, cmp))
    # create_test_dataset(by_song_folder, general_folder + 'test_manual.csv', composers, id2cmp, T)

    # df = pd.read_csv('/home/gianluca/PycharmProjects/music-analysis/models/model_large_dataset_3/test/50/temp.txt',
    #                  sep='\t', header=None)
    # ids = df.iloc[:, 1]
    # times = df.iloc[:, 2]
    # res = create_annotations(general_folder, ids, times)
    # pass

    general_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 'dataset_audiolabs_crosscomposer')
    output_file = os.path.join(general_folder, "song_lengths.csv")
    store_song_lengths(general_folder, output_file)
