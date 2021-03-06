"""
IDEA FOR THE ALGORITHM:
  - CNN on the magnitude of the STFT
  - apply triplet loss (thanks to the amazing guy that posted it on github)
  - train
"""

import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.profiler import option_builder
from tensorflow.python.profiler.model_analyzer import Profiler

from models import extract_3c2_rnn_bn_pool_sigmoid
from stft.classify_runs import logged_run, training_run, profiled_run
from stft.config import TRAIN_PATH, VALIDATION_PATH, MODELS_FOLDER, PARAMS
from stft.data_loading import create_tfrecords_iterator
from stft.tree import tree_analysis
from stft.utils import pairwise_distances_array
from triplet_loss import pairwise_distances, batch_hard_triplet_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = extract_3c2_rnn_bn_pool_sigmoid
model_folder = os.path.join(MODELS_FOLDER, model.__name__ + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

try:
    os.mkdir(model_folder)
except FileExistsError:
    pass

""" Config """
with open(os.path.join(model_folder, 'params.txt'), 'w') as file:
    for (k, v) in PARAMS.items():
        file.write("{}: {}\n".format(k, v))

""" Data input """
with tf.name_scope("data_input") as scope:
    trn_itr = create_tfrecords_iterator(TRAIN_PATH, batch_size=PARAMS['bs_train'], shuffle_buffer=PARAMS['sb_train'])
    tst_itr = create_tfrecords_iterator(VALIDATION_PATH, batch_size=PARAMS['bs_test'], shuffle_buffer=PARAMS['sb_test'])

    handle = tf.placeholder(tf.string, shape=[])
    x, comp_id, song_id = tf.data.Iterator.from_string_handle(
        handle, trn_itr.output_types, trn_itr.output_shapes).get_next()

    input_layer = tf.reshape(x, PARAMS['x.shape'])  # shape [-1, 233, 1_323, 1]
    y_ = tf.one_hot(comp_id, PARAMS['n_composers'])
    y_ = tf.squeeze(y_, 1)  # squeeze because tf.graph doesn't know that there is only one comp_id per data point

""" Calculations """
_, embeddings = model(input_layer, PARAMS)
distance_matrix = pairwise_distances(embeddings)

with tf.name_scope("training") as scope:
    loss = batch_hard_triplet_loss(tf.squeeze(comp_id, 1), embeddings, PARAMS['triplet_loss_margin'])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # needed for batch normalizations
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(PARAMS['lr']).minimize(loss, global_step=tf.train.create_global_step())
tf.summary.scalar('loss', loss)


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
    test = sess.run([x, y_, embeddings], feed_dict={handle: trn_handle})
    print(test[2])

    opts = (option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.trainable_variables_parameter())
            .with_file_output(os.path.join(model_folder, 'profile_model.txt')).build())
    profiler.profile_name_scope(options=opts)

    value_lv = None
    lv = tf.Summary()
    lv.value.add(tag='loss', simple_value=value_lv)

    # steps = 1
    for n in range(PARAMS['steps']):
        global_step = sess.run(tf.train.get_global_step())

        # if n == 0:  # log the results on the test set and reconstruct the tree
        if (n % PARAMS['test_step'] == 0) or n == PARAMS['steps'] - 1:
            print("step {} of {}, global_step set to {}. Test time!".format(n, PARAMS['steps'] - 1, global_step))
            lss_validation = 0.
            ems, c_ids, s_ids = [], [], []
            for i in range(PARAMS['steps_validation']):
                summary, em, cs, ss, lss = sess.run(
                    [merged, embeddings, comp_id, song_id, loss],
                    feed_dict={handle: tst_handle})
                lss_validation += lss
                ems.extend(em), c_ids.extend(cs), s_ids.extend(ss)

            lss_validation /= PARAMS['steps_validation']
            print(lss_validation)

            lv.value[0].simple_value = lss_validation
            test_writer.add_summary(lv, global_step=global_step)

            ems = np.array(ems)
            dm = pairwise_distances_array(ems)
            output_folder = os.path.join(model_folder, 'test', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
            try:
                os.mkdir(output_folder)
            except FileExistsError:
                pass
            logger.warning("Doing the tree analysis")
            tree_analysis(dm, c_ids, s_ids, output_folder, run_analysis=False)
            saver.save(sess, os.path.join(model_folder, "model.ckpt"))

        elif n > 0 and PARAMS['profile_step'] > 0 and n % PARAMS['profile_step'] == 0:  # train and log
            print("step {} of {}, global_step set to {}. Profiling!".format(n, PARAMS['steps'] - 1, global_step))
            profiled_run(sess, train_step, merged, handle, trn_handle, global_step, model_folder, profiler, n,
                         train_writer)
        elif n % PARAMS['log_step'] == 0:  # train and log
            print("step {} of {}, global_step set to {}".format(n, PARAMS['steps'] - 1, global_step))
            logged_run(sess, [train_step], merged, handle, trn_handle, global_step, train_writer)
        else:  # just train
            training_run(sess, [train_step], handle, trn_handle)
    # Profiler advice
    ALL_ADVICE = {'ExpensiveOperationChecker': {},
                  'AcceleratorUtilizationChecker': {},
                  'JobChecker': {},  # Only available internally.
                  'OperationChecker': {}}
    profiler.advise(ALL_ADVICE)
