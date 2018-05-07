"""
IDEA FOR THE ALGORITHM:
  - put labels on all the chroma features depending on the ID of the song
  - read the chroma features, all numerical columns
  - create a convolution neural network
  - apply triplet loss (thanks to the amazing guy that posted it on github)
  - train
"""

import numpy as np
import tensorflow as tf

from data_loading import train_input_fn, test_input_fn
from triplet_loss import batch_hard_triplet_loss, pairwise_distances

train_folder = "/media/gianluca/data/PycharmProjects/music-analysis/data/dataset_audiolabs_crossera/by_song/"
train_file = "../data/dataset_audiolabs_crossera/train.csv"
test_file = "../data/dataset_audiolabs_crossera/test.csv"

trn_itr = train_input_fn(train_file)
tst_itr = test_input_fn(test_file)

handle = tf.placeholder(tf.string, shape=[])
x, time, y_ = tf.data.Iterator.from_string_handle(handle, trn_itr.output_types, trn_itr.output_shapes).get_next()

input_layer = tf.reshape(x, [-1, 128, 12])

# Convolutional Layer #1 and Pooling Layer #1
conv1 = tf.layers.conv1d(
    inputs=input_layer,
    filters=32,
    kernel_size=4,
    padding="same",
    activation=tf.nn.relu)
pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)  # size 64
norm1 = tf.layers.batch_normalization(inputs=pool1)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv1d(
    inputs=norm1,
    filters=64,
    kernel_size=4,
    padding="same",
    activation=tf.nn.relu)
pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)  # size 32
norm2 = tf.layers.batch_normalization(inputs=pool2)

# Convolutional Layer #3 and Pooling Layer #3
conv3 = tf.layers.conv1d(
    inputs=norm2,
    filters=128,
    kernel_size=4,
    padding="same",
    activation=tf.nn.relu)
pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)  # size 16
norm3 = tf.layers.batch_normalization(inputs=pool3)

# Dense Layer
norm3_flat = tf.reshape(norm3, [-1, 16 * 128])
embeddings = tf.layers.dense(inputs=norm3_flat, units=1024)

with tf.name_scope("training") as scope:
    loss = batch_hard_triplet_loss(labels=y_, embeddings=embeddings, margin=0.2)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # needed for batch normalizations
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=tf.train.create_global_step())
    tf.summary.scalar('loss', loss)

with tf.name_scope('summaries') as scope:
    distance_matrix = pairwise_distances(embeddings)
    tf.summary.tensor_summary("distance_matrix", distance_matrix)

with tf.Session() as sess:
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name.replace(":", "_"), var)
        if "kernel" in var.name and "conv1d" in var.name:
            tf.summary.image(var.name.replace(":", "_") + "_image", tf.expand_dims(var, -1))
    merged = tf.summary.merge_all()  # compute all the summaries (why name merge?)
    folder = '../models/model1/'
    train_writer = tf.summary.FileWriter(folder + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(folder + 'test')

    tf.global_variables_initializer().run()
    trn_handle = sess.run(trn_itr.string_handle())
    tst_handle = sess.run(tst_itr.string_handle())

    N = 10_001
    for n in range(N):
        if n == N - 1:
            print("step {} of {}, global_step set to {}".format(n, N - 1, sess.run(tf.train.get_global_step())))
            summary, dm, labels, times = sess.run([merged, distance_matrix, y_, time], feed_dict={handle: tst_handle})
            evo_id = np.array([tf.compat.as_text(l) + "_t=" + str(t) for l, t in zip(labels, times)])
            np.savetxt(folder + 'test/dm.txt', dm)
            np.savetxt(folder + 'test/labels.txt', evo_id, fmt="%s")
            test_writer.add_summary(summary, global_step=sess.run(tf.train.get_global_step()))
        else:
            if n % 50 == 0:
                print("step {} of {}, global_step set to {}".format(n, N - 1, sess.run(tf.train.get_global_step())))
            summary, _ = sess.run([merged, train_step], feed_dict={handle: trn_handle})
            train_writer.add_summary(summary, global_step=sess.run(tf.train.get_global_step()))
