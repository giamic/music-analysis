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

import tensorflow as tf
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler.model_analyzer import Profiler

from data_loading import create_tfrecords_iterator
from models import classify, classify_2cl_relu, classify_2cl_sigm
from utils import encode_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = classify_2cl_sigm
data_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 'dataset_audiolabs_crosscomposer')
train_path = os.path.join(data_folder, 'train', 'chroma_features', 'train.tfrecords')
test_path = os.path.join(data_folder, 'test', 'chroma_features', 'test.tfrecords')
annotations_file = os.path.join(data_folder, 'cross-composer_annotations.csv')
model_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models',
                            model.__name__ + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

try:
    os.mkdir(model_folder)
except FileExistsError:
    pass


def test_run(sess, writer=None, saver=None):
    """
    Run the model on the test database, then write the summaries to disk and save the model.
    One can cluster the embeddings and use the distance matrix to reconstruct a tree.

    :param sess:
    :param writer:
    :param saver: if a tf.train.Saver() is provided, save the model
    :param clustering:
    :param tree:
    :return:
    """
    print("step {} of {}, global_step set to {}. Test time!".format(n, steps - 1, global_step))
    summary, acc = sess.run([merged, accuracy], feed_dict={handle: tst_handle})
    if writer is not None:
        writer.add_summary(summary, global_step=global_step)
    if saver is not None:
        saver.save(sess, os.path.join(model_folder, "model.ckpt"))
    return


def profiled_run(sess, writer=None, log_step=False, mode='raw_data'):
    """

    :param sess:
    :param writer:
    :param log_step:
    :param mode: either 'timeline' or 'raw_data'
    :return:
    """
    assert mode == 'timeline' or mode == 'raw_data'

    if log_step:
        print("step {} of {}, global_step set to {}".format(n, steps - 1, global_step))
    run_meta = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={handle: trn_handle},
                          options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_meta)
    if writer is not None:
        writer.add_summary(summary, global_step=global_step)
    profiler.add_step(n, run_meta)

    if mode == 'raw_data':
        opts = (option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.time_and_memory())
                .with_step(-1)
                .with_file_output(os.path.join(model_folder, 'profile_time.txt')).build())
        profiler.profile_operations(options=opts)

    if mode == 'timeline':
        opts = (option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.time_and_memory())
                .with_step(-1)
                .with_timeline_output(os.path.join(model_folder, 'profile_graph.txt')).build())
        profiler.profile_graph(options=opts)
    return


def logged_run(sess, writer):
    summary, _ = sess.run([merged, train_step], feed_dict={handle: trn_handle})
    writer.add_summary(summary, global_step=global_step)
    return


def training_run(sess):
    _ = sess.run(train_step, feed_dict={handle: trn_handle})
    return


""" Config """
params = {
    'bs_test': 1_100,  # batch_size
    'sb_test': None,  # shuffle_buffer, don't shuffle if None, total of 1_100 lines in test.csv
    'n_test': 1_100,  # number of test examples
    'bs_train': 128,
    'sb_train': 1_275,  # total of 45_354 lines in cross_composer/train/chroma_features/by_song folder
    'n_train': 45_354,  # number of training examples
    'x.shape': [-1, 128, 12],
    # 'x.shape': [-1, 1, 1],
    'lr': 0.001,  # learning rate
    'f1': 16,  # number of filters in the 1st layer
    'f2': 24,
    'f3': 32,
    'k1': 8,  # kernel size of filters in the 1st layer (length of the filter vector)
    'k2': 8,
    'k3': 8,
    'n_embeddings': 64,  # number of elements in the final embeddings vector
    'n_composers': 11,  # number of composers in the classification task
    'steps': 100_001,  # number of training steps, one epoch is 354 steps, avoid over-fitting
    'test_step': 350,
    'log_step': 19,
    'profile_step': 95,
    'epsilon': 1e-8  # for numerical stability
}
params['steps_per_epoch'] = params['n_train'] / params['bs_train']  # do it outside because I need to call the dict
with open(os.path.join(model_folder, 'params.txt'), 'w') as file:
    for (k, v) in params.items():
        file.write("{}: {}\n".format(k, v))

""" Data input """
trn_itr = create_tfrecords_iterator(train_path, batch_size=params['bs_train'], shuffle_buffer=params['sb_train'])
tst_itr = create_tfrecords_iterator(test_path, batch_size=params['bs_test'], shuffle_buffer=params['sb_test'])

handle = tf.placeholder(tf.string, shape=[])
x, song_id, time = tf.data.Iterator.from_string_handle(handle, trn_itr.output_types, trn_itr.output_shapes).get_next()

input_layer = tf.reshape(x, params['x.shape'])
y_ = encode_labels(song_id, one_hot=True)

# """ Calculations """
predictions = model(input_layer, params)

with tf.name_scope("training") as scope:
    loss = tf.losses.softmax_cross_entropy(y_, predictions)
    # loss = tf.reduce_mean(-tf.multiply(y_, tf.log(predictions + params['epsilon'])))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # needed for batch normalizations
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(params['lr']).minimize(loss, global_step=tf.train.create_global_step())
    tf.summary.scalar('loss', loss)

""" Creation of the distance matrix for the tree reconstruction """
with tf.name_scope('summaries') as scope:
    class_predicted = tf.argmax(predictions, axis=1)
    class_true = tf.argmax(y_, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(class_predicted, class_true), tf.float32))
    tf.summary.scalar('accuracy', accuracy)


""" Session run """
with tf.Session() as sess:
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.name.replace(":", "_"), var)
    #     if "kernel" in var.name and "conv1d" in var.name:
    #         tf.summary.image(var.name.replace(":", "_") + "_image", tf.expand_dims(var, -1))
    merged = tf.summary.merge_all()  # compute all the summaries (why name merge?)
    train_writer = tf.summary.FileWriter(os.path.join(model_folder, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(model_folder, 'test'))

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
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
    # test = sess.run([x, y_, embeddings, loss, class_predicted, class_true, accuracy], feed_dict={handle: trn_handle})

    opts = (option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.trainable_variables_parameter())
            .with_file_output(os.path.join(model_folder, 'profile_model.txt')).build())
    profiler.profile_name_scope(options=opts)

    steps = params['steps']
    for n in range(steps):
        global_step = sess.run(tf.train.get_global_step())

        # if n == 0:  # log the results on the test set and reconstruct the tree
        if (n > 0 and n % params['test_step'] == 0) or n == steps - 1:
            test_run(sess, test_writer, saver)
        elif n > 0 and n % params['profile_step'] == 0:  # train and log
            profiled_run(sess, train_writer)
        elif n % params['log_step'] == 0:  # train and log
            logged_run(sess, train_writer)
        else:  # just train
            training_run(sess)
    # Profiler advice
    ALL_ADVICE = {'ExpensiveOperationChecker': {},
                  'AcceleratorUtilizationChecker': {},
                  'JobChecker': {},  # Only available internally.
                  'OperationChecker': {}}
    profiler.advise(ALL_ADVICE)
