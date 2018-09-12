"""
IDEA FOR THE ALGORITHM:
  - CNN on the magnitude of the STFT
  - apply triplet loss (thanks to the amazing guy that posted it on github)
  - train
"""

import logging
import os
from datetime import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler.model_analyzer import Profiler

from models import match_5cl_pool_sigm, classify_4c2_bn_pool_sigmoid, classify_3c2_rnn_bn_pool_sigmoid
from stft.classify_runs import logged_run, training_run, test_run, profiled_run
from stft.data_loading import create_tfrecords_iterator
from stft.tree import tree_analysis
from triplet_loss import pairwise_distances

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = classify_3c2_rnn_bn_pool_sigmoid
data_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data', 'images')
train_path = os.path.join(data_folder, 'train.tfrecords')
test_path = os.path.join(data_folder, 'validation.tfrecords')
# annotations_path = os.path.join(data_folder, 'cross-composer_annotations.csv')
model_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'models',
                            model.__name__ + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

try:
    os.mkdir(model_folder)
except FileExistsError:
    pass

""" Config """
params = {
    'bs_test': 16,  # batch_size
    'sb_test': 218,  # shuffle_buffer, total of 218 STFTs, don't shuffle if None
    'step_validation': 27,  # it means we consider 8*27 = 216 images per validation instead of 218
    'bs_train': 8,
    'sb_train': 904,  # total of 904 STFTs in training set
    'x.shape': [-1, 233, 1_764, 1],
    'loss_margin': 10,
    'lr': 0.001,  # learning rate
    'f1': 8,  # number of filters in the 1st layer
    'f2': 8,
    'f3': 16,
    'f4': 16,
    'k1_f': 8,  # kernel size of filters in the 1st layer (length of the filter vector)
    'k1_t': 16,  # kernel size of filters in the 1st layer (length of the filter vector)
    'k2_f': 8,  # kernel size of filters in the 1st layer (length of the filter vector)
    'k2_t': 16,  # kernel size of filters in the 1st layer (length of the filter vector)
    'k3_f': 8,
    'k3_t': 16,
    'k4_f': 8,
    'k4_t': 16,
    'n_embeddings': 32,  # number of elements in the final embeddings vector
    'n_composers': 6,  # number of composers in the classification task
    'steps': 16_001,  # number of training steps, one epoch is 113 steps, avoid over-fitting
    'test_step': 1,
    'log_step': 15,
    'profile_step': -1,
}
params['steps_per_epoch'] = params['sb_train'] / params['bs_train']

with open(os.path.join(model_folder, 'params.txt'), 'w') as file:
    for (k, v) in params.items():
        file.write("{}: {}\n".format(k, v))
# annotations = pd.read_csv(annotations_path)

""" Data input """
with tf.name_scope("data_input") as scope:
    trn_itr = create_tfrecords_iterator(train_path, batch_size=params['bs_train'], shuffle_buffer=params['sb_train'])
    tst_itr = create_tfrecords_iterator(test_path, batch_size=params['bs_test'], shuffle_buffer=params['sb_test'])

    handle = tf.placeholder(tf.string, shape=[])
    x, comp_id, song_id, time = tf.data.Iterator.from_string_handle(
        handle, trn_itr.output_types, trn_itr.output_shapes).get_next()

    temp = tf.reshape(x, params['x.shape'])  # shape [-1, 233, 1_764, 1]
    input_layer = tf.layers.max_pooling2d(inputs=temp, pool_size=(1, 4), strides=(1, 4),
                                          padding='same')  # shape (-1, 233, 441, 1)
    # Are we sure we need this pooling at the beginning?

    y_ = tf.one_hot(comp_id, 6)
    y_ = tf.squeeze(y_, 1)  # squeeze because tf.graph doesn't know that there is only one songID per data point

""" Calculations """
logits, embeddings = model(input_layer, params)
distance_matrix = pairwise_distances(embeddings)

with tf.name_scope("training") as scope:
    loss = tf.losses.softmax_cross_entropy(y_, logits)
    # loss = tf.reduce_mean(-tf.multiply(y_, tf.log(predictions + params['epsilon'])))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # needed for batch normalizations
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(params['lr']).minimize(loss, global_step=tf.train.create_global_step())
tf.summary.scalar('loss', loss)

""" Creation of the distance matrix for the tree reconstruction """
with tf.name_scope('summaries') as scope:
    class_predicted = tf.argmax(logits, axis=1)
    class_true = tf.argmax(y_, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(class_predicted, class_true), tf.float32))
tf.summary.scalar('accuracy', accuracy)

""" Session run """
with tf.Session() as sess:
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name.replace(":", "_"), var)
        if "kernel" in var.name and "conv1d" in var.name:
            tf.summary.image(var.name.replace(":", "_") + "_image", tf.expand_dims(var, -1))
    merged = tf.summary.merge_all()  # compute all the summaries (why name merge?)
    train_writer = tf.summary.FileWriter(os.path.join(model_folder, 'train'), sess.graph)
    classify_writer = tf.summary.FileWriter(os.path.join(model_folder, 'classify'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(model_folder, 'test'))

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    tf.tables_initializer().run()
    saver = tf.train.Saver()
    profiler = Profiler(sess.graph)

    # try:
    #     logger.info("Trying to find a previous model checkpoint.")
    #     saver.restore(sess, os.path.join(model_folder, "model.ckpt"))
    #     logger.info("Previous model restored")
    # except tf.errors.NotFoundError:
    #     logger.warning("No model checkpoint found. Initializing a new model.")

    trn_handle = sess.run(trn_itr.string_handle())
    tst_handle = sess.run(tst_itr.string_handle())
    test = sess.run([x, y_, logits], feed_dict={handle: trn_handle})
    print(test[2])

    opts = (option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.trainable_variables_parameter())
            .with_file_output(os.path.join(model_folder, 'profile_model.txt')).build())
    profiler.profile_name_scope(options=opts)

    value_lv = None
    lv = tf.Summary()
    lv.value.add(tag='loss', simple_value=value_lv)
    value_av = None
    av = tf.Summary()
    av.value.add(tag='accuracy', simple_value=value_av)

    steps = params['steps']
    for n in range(steps):
        global_step = sess.run(tf.train.get_global_step())

        # if n == 0:  # log the results on the test set and reconstruct the tree
        if (n % params['test_step'] == 0) or n == steps - 1:
            print("step {} of {}, global_step set to {}. Test time!".format(n, steps - 1, global_step))
            acc_validation, lss_validation = 0., 0.
            for i in range(27):
                summary, dm, c_ids, s_ids, times, acc, lss = sess.run([merged, distance_matrix, comp_id, song_id, time, accuracy, loss], feed_dict={handle: tst_handle})
                acc_validation += acc
                lss_validation += lss
                output_folder = os.path.join(model_folder, 'test', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
                tree_analysis(dm, c_ids, s_ids, times, output_folder)
            acc_validation /= 27.
            lss_validation /= 27.
            print(acc_validation, lss_validation)
            av.value[0].simple_value = acc_validation
            lv.value[0].simple_value = lss_validation
            test_writer.add_summary(av, global_step=global_step)
            test_writer.add_summary(lv, global_step=global_step)

        elif n > 0 and params['profile_step'] > 0 and n % params['profile_step'] == 0:  # train and log
            print("step {} of {}, global_step set to {}. Profiling!".format(n, steps - 1, global_step))
            profiled_run(sess, train_step, merged, handle, trn_handle, global_step, model_folder, profiler, n,
                         train_writer)
        elif n % params['log_step'] == 0:  # train and log
            print("step {} of {}, global_step set to {}".format(n, steps - 1, global_step))
            logged_run(sess, [train_step], merged, handle, trn_handle, global_step, train_writer)
        else:  # just train
            training_run(sess, [train_step], handle, trn_handle)
    # Profiler advice
    ALL_ADVICE = {'ExpensiveOperationChecker': {},
                  'AcceleratorUtilizationChecker': {},
                  'JobChecker': {},  # Only available internally.
                  'OperationChecker': {}}
    profiler.advise(ALL_ADVICE)
