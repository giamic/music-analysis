import os
import random

import librosa
import numpy as np
import tensorflow as tf

from wavenet_mfcc.config import TRAIN_PATH, VALIDATION_PATH, EXTERNAL_DATA_FOLDER, SR, MAX_FILENAME_LENGTH, SR


def _analyse_single_file(audio_file):
    samples, sample_rate = librosa.load(audio_file, sr=SR)

    if len(samples.shape) == 2:
        samples = np.average(samples, axis=1)  # magically transform the file to mono
    # frequency resolution = SR * window_main_lobe_size / FRAME_SIZE = 44100 Hz * 4 / 2048 = 86.13 Hz
    # time resolution on single frame = FRAME_SIZE / SR = 2048 / 44100 Hz = 0.0464 s = 46.4 ms
    # time distance between frames = HOP_SIZE / SR = 512 / 44100 Hz = 11.6 ms
    mfcc = librosa.feature.mfcc(samples, sample_rate)
    return np.transpose(mfcc)


def transform_into_tfrecord(data_path, output_path):
    if os.path.isfile(output_path):
        raise PermissionError("The output file already exists, exiting to avoid data loss.")
    file_names = os.listdir(data_path)
    random.shuffle(file_names)
    n, N = 0, len(file_names)
    T = 2048  # desired data length
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for fn in file_names:
            if n % 100 == 0:
                print("Recording {} of {}".format(n, N))
            # x, sample_rate = librosa.load(os.path.join(data_path, fn), sr=SR)
            x = _analyse_single_file(os.path.join(data_path, fn))
            if len(x) < T:  # 2 ** 11 = 2048
                print("Skipping a short recording! file: {}, shape: {}".format(fn, x.shape))
                continue
            x = x[:T].flatten()
            assert len(x) == 2048 * 20
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
    # train_validation_split(os.path.join(EXTERNAL_DATA_FOLDER, 'recordings'))
    transform_into_tfrecord(os.path.join(EXTERNAL_DATA_FOLDER, 'recordings', 'train'), TRAIN_PATH)
    transform_into_tfrecord(os.path.join(EXTERNAL_DATA_FOLDER, 'recordings', 'validation'), VALIDATION_PATH)


