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

import numpy as np
import tensorflow as tf

from data_loading import train_input_fn, test_input_fn
from models import classify
from utils import find_id2cmp, count_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = classify
data_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 'dataset_audiolabs_crosscomposer')
train_path = os.path.join(data_folder, 'train', 'chroma_features', 'train.csv')
# train_path = os.path.join(data_folder, 'train', 'chroma_features', 'by_song')
test_path = os.path.join(data_folder, 'test', 'chroma_features', 'test.csv')
annotations_file = os.path.join(data_folder, 'cross-composer_annotations.csv')
model_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'models',
                            model.__name__ + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

try:
    os.mkdir(model_folder)
except FileExistsError:
    pass

""" Config """
params = {
    'bs_test': 1_100,  # batch_size
    'sb_test': None,  # shuffle_buffer, don't shuffle if None, total of 1_100 lines in test.csv
    'n_test': 1_100,  # number of test examples
    'bs_train': 128,
    'sb_train': 1_275,  # total of 45_354 lines in cross_composer/train/chroma_features/by_song folder
    'n_train': 45_354,  # number of training examples
    'x.shape': [-1, 128, 12],
    'lr': 0.001,  # learning rate
    'f1': 16,  # number of filters in the 1st layer
    'f2': 32,
    'f3': 64,
    'k1': 8,  # kernel size of filters in the 1st layer (length of the filter vector)
    'k2': 8,
    'k3': 8,
    'n_embeddings': 128,  # number of elements in the final embeddings vector
    'n_composers': 11,  # number of composers in the classification task
    'steps': 300_001,  # number of training steps, one epoch is 354 steps, avoid over-fitting
    'epsilon': 1e-8  # for numerical stability
}
params['steps_per_epoch'] = params['n_train'] / params['bs_train']

composers = [
    'Bach; Johann Sebastian',
    'Beethoven; Ludwig van',
    'Brahms; Johannes',
    'Dvorak; Antonin',
    'Handel; George Frideric',
    'Haydn; Franz Joseph',
    'Mendelssohn; Felix',
    'Mozart; Wolfgang Amadeus',
    'Rameau; Jean-Philippe',
    'Schubert; Franz',
    'Shostakovich; Dmitry'
]

encoded = np.identity(len(composers))

with open(os.path.join(model_folder, 'params.txt'), 'w') as file:
    for (k, v) in params.items():
        file.write("{}: {}\n".format(k, v))

""" Data input """
trn_itr = train_input_fn(train_path, batch_size=params['bs_train'], shuffle_buffer=params['sb_train'])
tst_itr = test_input_fn(test_path, batch_size=params['bs_test'], shuffle_buffer=params['sb_test'])

handle = tf.placeholder(tf.string, shape=[])
x, song_id, time = tf.data.Iterator.from_string_handle(handle, trn_itr.output_types, trn_itr.output_shapes).get_next()

keys, values = find_id2cmp(annotations_file)

id2cmp = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values), "")
cmp2oh = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(composers, np.arange(len(composers))),
                                     -1)
cmp = id2cmp.lookup(song_id)
oh = cmp2oh.lookup(cmp)
y_ = tf.one_hot(oh, len(composers))

input_layer = tf.reshape(x, params['x.shape'])

""" Calculations """
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
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name.replace(":", "_"), var)
        if "kernel" in var.name and "conv1d" in var.name:
            tf.summary.image(var.name.replace(":", "_") + "_image", tf.expand_dims(var, -1))
    merged = tf.summary.merge_all()  # compute all the summaries (why name merge?)
    train_writer = tf.summary.FileWriter(os.path.join(model_folder, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(model_folder, 'test'))

    tf.global_variables_initializer().run()
    tf.tables_initializer().run()
    tf.local_variables_initializer().run()
    saver = tf.train.Saver()

    count_params(tf.trainable_variables(), os.path.join(model_folder, 'params.txt'))

    try:
        logger.info("Trying to find a previous model checkpoint.")
        saver.restore(sess, os.path.join(model_folder, "model.ckpt"))
        logger.info("Previous model restored")
    except tf.errors.NotFoundError:
        logger.warning("No model checkpoint found. Initializing a new model.")

    trn_handle = sess.run(trn_itr.string_handle())
    tst_handle = sess.run(tst_itr.string_handle())
    # test = sess.run([x, y_, embeddings, loss, class_predicted, class_true, accuracy], feed_dict={handle: trn_handle})
    # test = sess.run([x], feed_dict={handle: trn_handle})
    # print(test)

    steps = params['steps']
    for n in range(steps):
        global_step = sess.run(tf.train.get_global_step())
        print("step {} of {}, global_step set to {}".format(n, steps - 1, global_step))
        # if n == 0:  # log the results on the test set and reconstruct the tree
        if n == steps - 1 or (n > 0 and n % 200 == 0):  # log the results on the test set and reconstruct the tree
            summary, acc = sess.run([merged, accuracy], feed_dict={handle: tst_handle})
            # summary, acc, cp = sess.run([merged, accuracy, class_predicted], feed_dict={handle: tst_handle})
            # print(acc, [p for p in cp])
            test_writer.add_summary(summary, global_step=global_step)
            saver.save(sess, os.path.join(model_folder, "model.ckpt"))
        elif global_step % 10 == 0:  # train and log
            summary, _ = sess.run([merged, train_step], feed_dict={handle: trn_handle})
            train_writer.add_summary(summary, global_step=global_step)
        else:  # just train
            _ = sess.run([train_step], feed_dict={handle: trn_handle})
