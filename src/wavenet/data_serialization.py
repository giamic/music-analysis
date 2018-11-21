import os
import random

import librosa
import numpy as np
import tensorflow as tf

from wavenet.config import TRAIN_PATH, VALIDATION_PATH, EXTERNAL_DATA_FOLDER, SR, MAX_FILENAME_LENGTH


def transform_into_tfrecord(data_path, output_path):
    if os.path.isfile(output_path):
        raise PermissionError("The output file already exists, exiting to avoid data loss.")
    file_names = os.listdir(data_path)
    random.shuffle(file_names)
    n, N = 0, len(file_names)
    T = 2 ** 20  # max number of samples to take, circa equivalent to 23.777 seconds with 44100 Hz.
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for fn in file_names:
            if n % 100 == 0:
                print("Recording {} of {}".format(n, N))
            x, sample_rate = librosa.load(os.path.join(data_path, fn), sr=SR)
            if len(x.shape) == 2:
                x = np.average(x, axis=1)  # magically transform the file to mono if needed
            if len(x) < T:  # 2 ** 20 = 1_048_576 = 44_100 * 23.777...
                print("Skipping a short recording! file: {}, shape: {}".format(fn, x.shape))
                continue
            x = x[:T]
            assert len(x) == 1048576
            temp = fn.split('_')
            composer_id, song_id = int(temp[0]), int(temp[2])
            example = tf.train.Example()
            example.features.feature["composer_id"].int64_list.value.append(composer_id)
            example.features.feature["song_id"].int64_list.value.append(song_id)
            example.features.feature["x"].float_list.value.extend(x)
            writer.write(example.SerializeToString())
            n += 1
    return


def train_validation_split(data_folder):
    os.makedirs(os.path.join(data_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_folder, 'validation'), exist_ok=True)

    artists = sorted([x for x in os.listdir(data_folder) if x[0] == '0'])
    if len(artists) == 0:
        print("No data available for splitting, I'm leaving.")
        return

    for a in artists:
        print("Moving data for {}".format(a))
        recordings = os.listdir(os.path.join(data_folder, a))
        if len(recordings) == 0:
            print("No data available for splitting for {}, I'm leaving.".format(a))
            continue

        for r in recordings:
            if np.random.random() > 0.9:
                os.rename(os.path.join(data_folder, a, r),
                          os.path.join(data_folder, 'validation', (a + '_' + r)[:MAX_FILENAME_LENGTH]))
            else:
                os.rename(os.path.join(data_folder, a, r),
                          os.path.join(data_folder, 'train', (a + '_' + r)[:MAX_FILENAME_LENGTH]))
    return


if __name__ == '__main__':
    train_validation_split(os.path.join(EXTERNAL_DATA_FOLDER, 'recordings'))
    transform_into_tfrecord(os.path.join(EXTERNAL_DATA_FOLDER, 'recordings', 'train'), TRAIN_PATH)
    transform_into_tfrecord(os.path.join(EXTERNAL_DATA_FOLDER, 'recordings', 'validation'), VALIDATION_PATH)
