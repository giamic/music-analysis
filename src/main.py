"""
IDEA FOR THE ALGORITHM:
  - put labels on all the chroma features depending on the ID of the song
  - read the chroma features, all numerical columns
  - create a convolution neural network
  - apply triplet loss (thanks to the amazing guy that posted it on github)
  - train
"""

import logging
import os

import numpy as np
import tensorflow as tf

from data_loading import train_input_fn, test_input_fn, find_id2cmp
from tree import reconstruct_tree
from models import three_layers_conv
from triplet_loss import pairwise_distances, batch_all_triplet_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 'dataset_audiolabs_crossera')
train_file = os.path.join(data_folder, 'train2.csv')
test_file = os.path.join(data_folder, 'test2.csv')
annotations_file = os.path.join(data_folder, 'cross-era_annotations.csv')
model_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models', 'model_large_dataset_2')

""" Data input """
trn_itr = train_input_fn(train_file, batch_size=128, shuffle_buffer=40_000)  # total of 40_000 lines in train2.csv
tst_itr = test_input_fn(test_file, batch_size=2_000, shuffle_buffer=2_000)  # total of 2_000 lines in test2.csv

handle = tf.placeholder(tf.string, shape=[])
x, song_id, time = tf.data.Iterator.from_string_handle(handle, trn_itr.output_types, trn_itr.output_shapes).get_next()

keys, values = find_id2cmp(annotations_file)
id2cmp = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), "")
y_ = id2cmp.lookup(song_id)

input_layer = tf.reshape(x, [-1, 128, 12])

""" Calculations """
embeddings = three_layers_conv(input_layer)

with tf.name_scope("training") as scope:
    loss, positive_triplets = batch_all_triplet_loss(labels=y_, embeddings=embeddings, margin=10)
    # loss = batch_hard_triplet_loss(labels=y_, embeddings=embeddings, margin=100)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # needed for batch normalizations
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss, global_step=tf.train.create_global_step())
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('positive_triplets', positive_triplets)

""" Creation of the distance matrix for the tree reconstruction """
with tf.name_scope('summaries') as scope:
    distance_matrix = pairwise_distances(embeddings)
    tf.summary.tensor_summary("distance_matrix", distance_matrix)

""" Session run """
with tf.Session() as sess:
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name.replace(":", "_"), var)
        if "kernel" in var.name and "conv1d" in var.name:
            tf.summary.image(var.name.replace(":", "_") + "_image", tf.expand_dims(var, -1))
    merged = tf.summary.merge_all()  # compute all the summaries (why name merge?)
    train_writer = tf.summary.FileWriter(os.path.join(model_folder, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(model_folder, 'test'))

    tf.global_variables_initializer().run()
    tf.tables_initializer().run()
    saver = tf.train.Saver()

    try:
        logger.info("Trying to find a previous model checkpoint.")
        saver.restore(sess, os.path.join(model_folder, "model.ckpt"))
        logger.info("Previous model restored")
    except tf.errors.NotFoundError:
        logger.warning("No model checkpoint found. Initializing a new model.")

    trn_handle = sess.run(trn_itr.string_handle())
    tst_handle = sess.run(tst_itr.string_handle())
    # test = sess.run([y_, song_id, time], feed_dict={handle: tst_handle})
    # print(test)

    N = 100_001
    count = 0
    for n in range(N):
        print("step {} of {}, global_step set to {}".format(n, N - 1, sess.run(tf.train.get_global_step())))

        if n == N - 1 or (n > 0 and n % 2_000 == 0):  # log the results on the test set and reconstruct the tree
            summary, dm, labels, ids, times, l, pt = sess.run(
                [merged, distance_matrix, y_, song_id, time, loss, positive_triplets], feed_dict={handle: tst_handle})
            evo_id = np.array(
                [tf.compat.as_text(l) + tf.compat.as_text(i) + "_t=" + str(t) for l, i, t in zip(labels, ids, times)])
            tree_folder = os.path.join(model_folder, 'test', str(count))
            os.mkdir(tree_folder)
            np.savetxt(os.path.join(tree_folder, 'dm.txt'), dm)
            np.savetxt(os.path.join(tree_folder, 'labels.txt'), evo_id, fmt="%s")
            reconstruct_tree(tree_folder)
            test_writer.add_summary(summary, global_step=sess.run(tf.train.get_global_step()))
            saver.save(sess, os.path.join(model_folder, "model.ckpt"))
            count += 1
        else:  # train the network
            summary, _ = sess.run([merged, train_step], feed_dict={handle: trn_handle})
            train_writer.add_summary(summary, global_step=sess.run(tf.train.get_global_step()))
