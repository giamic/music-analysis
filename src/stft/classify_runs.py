import os
from datetime import datetime

import tensorflow as tf
from tensorflow.python.profiler import option_builder

from tree import tree_analysis
from chroma.utils import clustering_classification


def test_run(sess, targets, merged, handle, h, global_step, model_folder, writer=None, saver=None):
    """
    Run the model on the test database, then write the summaries to disk and save the model.
    One can cluster the embeddings and use the distance matrix to reconstruct a tree.

    :param sess:
    :param targets:
    :param merged:
    :param handle:
    :param h:
    :param global_step:
    :param model_folder:
    :param writer:
    :param saver: if a tf.train.Saver() is provided, save the model
    :return:
    """

    summary, acc = sess.run([merged] + targets, feed_dict={handle: h})
    if writer is not None:
        writer.add_summary(summary, global_step=global_step)
    if saver is not None:
        saver.save(sess, os.path.join(model_folder, "model.ckpt"))
    return


def profiled_run(sess, train_step, merged, handle, h, global_step, model_folder, profiler, n, writer, mode='raw_data'):
    """

    :param sess:
    :param writer:
    :param log_step:
    :param mode: either 'timeline' or 'raw_data'
    :return:
    """
    assert mode == 'timeline' or mode == 'raw_data'

    run_meta = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={handle: h},
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


def logged_run(sess, targets, merged, handle, h, global_step, writer):
    summary, _ = sess.run([merged] + targets, feed_dict={handle: h})
    writer.add_summary(summary, global_step=global_step)
    return


def training_run(sess, targets, handle, h):
    _ = sess.run(targets, feed_dict={handle: h})
    return
