import os

import numpy as np
import tensorflow as tf
from skimage.color import rgba2rgb, rgb2gray
from skimage.io import imread


def train_validation_split(data_folder, n_composers=6):
    images = os.listdir(data_folder)
    images = [i for i in images if i[-4:] == '.png']  # remove directories and other files and keep only the images

    if len(images) == 0:
        print("No data available for splitting, I'm leaving.")
        return

    for n in range(n_composers):
        im = [i for i in images if i.split('_')[1] == str(n)]
        recs = set([i.split('_')[2] for i in im])
        for r in recs:
            im_r = [i for i in im if i.split('_')[2] == r]
            songs = set([i.split('_')[3] for i in im_r])
            for s in songs:
                im_rs = [i for i in im_r if i.split('_')[3] == s]
                if np.random.random() > 0.9:
                    for i in im_rs:
                        os.rename(os.path.join(data_folder, i), os.path.join(data_folder, 'validation', i))
                else:
                    for i in im_rs:
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
            x = (x * 256).astype(np.uint8).flatten()
            temp = fn.split('_')
            composer_id, recording_id, song_id, time = int(temp[1]), int(temp[2]), int(temp[3]), int(temp[4][:-4])
            example = tf.train.Example()
            example.features.feature["composer_id"].int64_list.value.append(composer_id)
            example.features.feature["recording_id"].int64_list.value.append(recording_id)
            example.features.feature["song_id"].int64_list.value.append(song_id)
            example.features.feature["time"].int64_list.value.append(time)
            example.features.feature["x"].int64_list.value.extend(x)
            writer.write(example.SerializeToString())
            n += 1
    return


def pairwise_distances_array(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = np.dot(embeddings, embeddings.T)

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = dot_product.diagonal()

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm[np.newaxis, :] - 2.0 * dot_product + square_norm[:, np.newaxis]

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = np.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = (distances == 0.0).astype(np.float32)
        distances = distances + mask * 1e-16

        distances = np.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


if __name__ == '__main__':
    data_folder = os.path.join(os.path.curdir, '..', '..', 'data', 'images')
    train_validation_split(data_folder)
    transform_into_tfrecord(os.path.join(data_folder, 'train'), os.path.join(data_folder, 'train.tfrecords'))
    transform_into_tfrecord(os.path.join(data_folder, 'validation'), os.path.join(data_folder, 'validation.tfrecords'))
