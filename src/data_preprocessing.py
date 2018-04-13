"""
All the functions that we need in order to adapt the database in cross_era to our needs.
The database can be found at the URL https://www.audiolabs-erlangen.de/resources/MIR/cross-era
We modify it in the following way:
  1. We change the first column into a label column where only the CrossEra ID is kept
  2. We remove the time column
  3. We make each row of the dataset contain information about N samplings of chroma features. This
     effectively multiplies the size of the dataset by N.
  4. We remove incomplete lines or lines that mix songs (the last N-1 lines from each song)
"""
import os
import re
import logging
from warnings import warn

import numpy as np
import pandas as pd

logging.getLogger('__main__')
logging.basicConfig(level="INFO")


def create_datapoints_for_dnn(df, T):
    """
    Here we take the data frame with chroma features at time t and create all features at times t+1, t+2, ..., t+N-1.

    :param df: initial data frame of chroma features
    :param T: number of time steps to keep
    :return: expanded data frame of chroma features
    """
    res = df.copy()
    original_labels = df.columns.values
    n_steps = df.shape[0]  # the number of time steps in this song
    nans = pd.Series(np.full(n_steps, np.NaN)).values  # a column of nans of the correct length
    for n in range(1, T):
        new_labels = [ol + '+' + str(n) for ol in original_labels[2:]]
        for nl, ol in zip(new_labels, original_labels[2:]):
            # df.assign would use the name "nl" instead of what nl contains, so we build and unpack a dictionary
            res = res.assign(**{nl: nans})  # create a new column
            # res.iloc[:-n, res.columns.get_loc(nl)] = df.iloc[:, df.columns.get_loc(ol)].shift(-n)
            res.iloc[:-n, res.columns.get_loc(nl)] = df.iloc[n:, df.columns.get_loc(ol)].values  # this is maybe faster
    return res[: - (T - 1)]  # drop the last N-1 rows because time t+N-1 is not defined for them


def take_id(song_name):
    """
    Retrieve the ID from the name of the song in the CrossEra dataset

    :param song_name:
    :return:
    """
    if not isinstance(song_name, str):
        warn("The given song_name ({}) is not a valid string. Returning a nan.".format(song_name))
        return np.NaN
    match = re.search(r'CrossEra-\d+', song_name)
    if match is not None:
        return match.group(0)
    else:
        warn("CrossEra ID not found in song name \"{}\". Returning a nan.".format(song_name))
    return np.NaN


def preprocess(input_file_paths, output_folder, T, overwrite=False):
    """
    Transform the data from CrossEra database into data that is useful to us.

    :param input_file_paths: a list of the path to all input files
    :param output_folder: where to store the csv we generate
    :param T: the number of time steps to keep
    :return:
    """
    original_labels = ['songID', 'time', 'A_t', 'A#_t', 'B_t', 'C_t', 'C#_t', 'D_t', 'D#_t', 'E_t', 'F_t', 'F#_t',
                       'G_t', 'G#_t']
    for f in input_file_paths:

        logging.info("Working on file {}".format(f))
        data = pd.read_csv(f, header=None, names=original_labels)
        data['songID'] = data['songID'].apply(take_id)  # take just the ID of the song
        data['songID'] = data['songID'].fillna(method='ffill')  # repeat the ID for all rows
        for s in set(data['songID']):
            path_output = output_folder + 'by_song/chroma-nnls_' + s + '.csv'
            if not overwrite and os.path.isfile(path_output):
                logging.info("Output file {} already exists. Skipping songID {}".format(path_output, s))
                continue
            logging.info("Working on songID {}".format(s))
            df = data.loc[data['songID'] == s]  # select one song at a time not to use too much memory
            df = create_datapoints_for_dnn(df, T)  # add the desired columns
            df.to_csv(path_output, header=False, index=False)  # write the df in a file
    return


def create_file_paths(folder):
    """
    Given the folder, create the file paths to link to the CrossEra database
    :param folder:
    :return:
    """
    debut = "chroma-nnls"
    instrument = ["piano", "orchestra"]
    style = ["baroque", "classical", "romantic", "modern", "addon"]
    file_names = ["_".join([debut, i, s]) for i in instrument for s in style]
    # file_names = ["test0"]

    return [folder + fn + ".csv" for fn in file_names]


def create_test_file(folder, T):
    folder_by_song = folder+'by_song/'
    file_paths = [folder_by_song + fp for fp in os.listdir(folder_by_song)][0:5]  # take only five songs so far
    original_labels = ['songID', 'time', 'A_t', 'A#_t', 'B_t', 'C_t', 'C#_t', 'D_t', 'D#_t', 'E_t', 'F_t', 'F#_t', 'G_t', 'G#_t']
    labels = original_labels[0:2]
    chromas = original_labels[2:]
    labels = labels + [c + str(t) for t in range(T) for c in chromas]
    df_test = pd.DataFrame(columns=labels)
    for fp in file_paths:
        df = pd.read_csv(fp, header=None, names=labels)
        idx = np.random.randint(0, df.shape[0], 10)
        df_test = pd.concat([df_test, df.iloc[idx]], ignore_index=True)
    path_output = folder + 'test.csv'
    df_test.to_csv(path_output, header=False, index=False)
    return


if __name__ == '__main__':
    folder = "../data/dataset_audiolabs_crossera/"
    T = 128  # how many successive steps we want to put in a single row

    # input_fp = create_file_paths(folder)
    # preprocess(input_fp, folder, T)
    create_test_file(folder, T)