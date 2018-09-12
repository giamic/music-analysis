import os

import numpy as np
import tensorflow as tf
from skimage.color import rgba2rgb, rgb2gray
from skimage.io import imread


def train_validation_split(data_folder):
    images = os.listdir(data_folder)
    images = [i for i in images if i[-4:] == '.png']  # remove directories and other files and keep only the images
    if len(images) == 0:
        print("No data available for splitting, I'm leaving.")
        return
    for i in images:
        if np.random.random() > 0.8:
            os.rename(os.path.join(data_folder, i), os.path.join(data_folder, 'validation', i))
        else:
            os.rename(os.path.join(data_folder, i), os.path.join(data_folder, 'train', i))
    return


def transform_into_tfrecord(data_path, output_path):
    file_names = os.listdir(data_path)
    file_paths = [os.path.join(data_path, fn) for fn in file_names]
    # image = tf.image.decode_png(image_file, channels=1, dtype=tf.uint8, name=None)
    n, N = 0, len(file_names)
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for fn, fp in zip(file_names, file_paths):
            if n % 10 == 0:
                print("Image {} of {}".format(n, N))
            x = imread(fp)
            x = rgb2gray(rgba2rgb(x))
            x = (x*256).astype(np.uint8).flatten()
            temp = fn.split('_')
            composer_id, song_id, time = int(temp[1]), int(temp[2]), int(temp[3][:-4])
            example = tf.train.Example()
            example.features.feature["composer_id"].int64_list.value.append(composer_id)
            example.features.feature["song_id"].int64_list.value.append(song_id)
            example.features.feature["time"].int64_list.value.append(time)
            example.features.feature["x"].int64_list.value.extend(x)
            writer.write(example.SerializeToString())
            n += 1
    return


if __name__ == '__main__':
    data_folder = os.path.join(os.path.curdir, '..', '..', 'data', 'images')
    train_validation_split(data_folder)
    transform_into_tfrecord(os.path.join(data_folder, 'train'), os.path.join(data_folder, 'train.tfrecords'))
    transform_into_tfrecord(os.path.join(data_folder, 'validation'), os.path.join(data_folder, 'validation.tfrecords'))
