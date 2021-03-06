import os
import random

import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray, rgba2rgb
from skimage.io import imread

from stft.config import TRAIN_PATH, VALIDATION_PATH, EXTERNAL_DATA_FOLDER


def transform_into_tfrecord(data_path, output_path):
    if os.path.isfile(output_path):
        raise PermissionError("The output file already exists, exiting to avoid data loss.")
    file_names = os.listdir(data_path)
    random.shuffle(file_names)
    n, N = 0, len(file_names)
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for fn in file_names:
            if n % 100 == 0:
                print("Image {} of {}".format(n, N))
            x = imread(os.path.join(data_path, fn))
            if x.shape != (233, 1323, 4):
                print("Skipping a weird image! file: {}, shape: {}".format(fn, x.shape))
                continue
            x = rgb2gray(rgba2rgb(x))
            x = (x * 256).astype(np.uint8).flatten()
            assert len(x) == 308259
            temp = fn.split('_')
            composer_id, song_id = int(temp[1]), int(temp[2])
            example = tf.train.Example()
            example.features.feature["composer_id"].int64_list.value.append(composer_id)
            example.features.feature["song_id"].int64_list.value.append(song_id)
            example.features.feature["x"].int64_list.value.extend(x)
            writer.write(example.SerializeToString())
            n += 1
    return


def train_validation_split(data_folder):
    images = os.listdir(data_folder)
    images = [i for i in images if i[-4:] == '.png']  # remove directories and other files and keep only the images

    if len(images) == 0:
        print("No data available for splitting, I'm leaving.")
        return

    for i in images:
        if np.random.random() > 0.9:
            os.rename(os.path.join(data_folder, i), os.path.join(data_folder, 'validation', i))
        else:
            os.rename(os.path.join(data_folder, i), os.path.join(data_folder, 'train', i))

    return


if __name__ == '__main__':
    # train_validation_split(EXTERNAL_DATA_FOLDER)
    transform_into_tfrecord(os.path.join(EXTERNAL_DATA_FOLDER, 'images', 'train'), TRAIN_PATH)
    transform_into_tfrecord(os.path.join(EXTERNAL_DATA_FOLDER, 'images', 'validation'), VALIDATION_PATH)
