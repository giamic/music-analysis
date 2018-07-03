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
from datetime import datetime

import pandas as pd
import tensorflow as tf
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler.model_analyzer import Profiler

from data_loading import create_tfrecords_iterator
from models import match_3cl_pool_sigm, match_5cl_pool_sigm
from runs import test_run, profiled_run, logged_run, training_run
from triplet_loss import pairwise_distances, batch_all_triplet_loss
from utils import count_params, encode_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = match_5cl_pool_sigm
data_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 'dataset_audiolabs_crosscomposer')
train_path = os.path.join(data_folder, 'train', 'chroma_features', 'train.tfrecords')
test_path = os.path.join(data_folder, 'test', 'chroma_features', 'test_long.tfrecords')
annotations_path = os.path.join(data_folder, 'cross-composer_annotations.csv')
model_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models',
                            model.__name__ + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

try:
    os.mkdir(model_folder)
except FileExistsError:
    pass

""" Config """
params = {
    'bs_test': 1_100,  # batch_size
    'sb_test': None,  # shuffle_buffer, total of 1_100 lines in test.csv, don't shuffle if None
    'bs_train': 128,
    'sb_train': 45_354,  # total of 45_354 lines in cross_composer/train/chroma_features/by_song folder
    'x.shape': [-1, 128, 12],
    'loss_margin': 10,
    'lr': 0.001,  # learning rate
    'f1': 16,  # number of filters in the 1st layer
    'f2': 24,
    'f3': 32,
    'k1': 4,  # kernel size of filters in the 1st layer (length of the filter vector)
    'k2': 4,
    'k3': 4,
    'n_embeddings': 32,  # number of elements in the final embeddings vector
    'n_composers': 11,  # number of composers in the classification task
    'steps': 30_001,  # number of training steps, one epoch is 354 steps, avoid over-fitting
    'test_step': 1_000,
    'log_step': 50,
    'profile_step': -1,
}
params['steps_per_epoch'] = params['sb_train'] / params['bs_train']

with open(os.path.join(model_folder, 'params.txt'), 'w') as file:
    for (k, v) in params.items():
        file.write("{}: {}\n".format(k, v))
annotations = pd.read_csv(annotations_path)

""" Data input """
with tf.name_scope("data_input") as scope:
    trn_itr = create_tfrecords_iterator(train_path, batch_size=params['bs_train'], shuffle_buffer=params['sb_train'])
    tst_itr = create_tfrecords_iterator(test_path, batch_size=params['bs_test'], shuffle_buffer=params['sb_test'])

    handle = tf.placeholder(tf.string, shape=[])
    x, song_id, time = tf.data.Iterator.from_string_handle(handle, trn_itr.output_types,
                                                           trn_itr.output_shapes).get_next()

    input_layer = tf.reshape(x, params['x.shape'])
    y_ = encode_labels(song_id, one_hot=False)

""" Calculations """
embeddings = model(input_layer, params)

with tf.name_scope("training") as scope:
    loss, positive_triplets = batch_all_triplet_loss(labels=y_, embeddings=embeddings, margin=params['loss_margin'])
    # loss = batch_hard_triplet_loss(labels=y_, embeddings=embeddings, margin=100)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # needed for batch normalizations
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(params['lr']).minimize(loss, global_step=tf.train.create_global_step())
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
    profiler = Profiler(sess.graph)

    try:
        logger.info("Trying to find a previous model checkpoint.")
        saver.restore(sess, os.path.join(model_folder, "model.ckpt"))
        logger.info("Previous model restored")
    except tf.errors.NotFoundError:
        logger.warning("No model checkpoint found. Initializing a new model.")

    trn_handle = sess.run(trn_itr.string_handle())
    tst_handle = sess.run(tst_itr.string_handle())
    # test = sess.run([x, song_id, time, y_], feed_dict={handle: trn_handle})
    # print(test)

    opts = (option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.trainable_variables_parameter())
            .with_file_output(os.path.join(model_folder, 'profile_model.txt')).build())
    profiler.profile_name_scope(options=opts)
    targets = [embeddings, y_, distance_matrix, song_id, time, loss, positive_triplets]

    steps = params['steps']
    for n in range(steps):
        global_step = sess.run(tf.train.get_global_step())

        # if n == 0:  # log the results on the test set and reconstruct the tree
        if (n > 0 and n % params['test_step'] == 0) or n == steps - 1:
            print("step {} of {}, global_step set to {}. Test time!".format(n, steps - 1, global_step))
            test_run(sess, targets, merged, handle, tst_handle, global_step, model_folder, annotations, test_writer, saver, clustering=True, tree=True)
        elif params['profile_step'] > 0 and n > 0 and n % params['profile_step'] == 0:  # train and log
            print("step {} of {}, global_step set to {}. Profiling!".format(n, steps - 1, global_step))
            profiled_run(sess, train_step, merged, handle, trn_handle, global_step, model_folder, profiler, n, train_writer, mode='raw_data')
        elif n % params['log_step'] == 0:  # train and log
            print("step {} of {}, global_step set to {}".format(n, steps - 1, global_step))
            logged_run(sess, train_step, merged, handle, trn_handle, global_step, train_writer)
        else:  # just train
            training_run(sess, train_step, handle, trn_handle)
    # Profiler advice
    ALL_ADVICE = {'ExpensiveOperationChecker': {},
                  'AcceleratorUtilizationChecker': {},
                  'JobChecker': {},  # Only available internally.
                  'OperationChecker': {}}
    profiler.advise(ALL_ADVICE)
