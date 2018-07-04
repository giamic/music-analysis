import os
from datetime import datetime

import tensorflow as tf
from tensorflow.python.profiler import option_builder

from tree import tree_analysis
from utils import clustering_classification


def test_run(sess, targets, merged_summary, handle, h, global_step, model_folder, annotations_file, writer=None,
             saver=None,
             clustering=True, tree=True):
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
    summary, y, labels, dm, ids, times = sess.run([merged_summary] + targets, feed_dict={handle: h})
    if writer is not None:
        writer.add_summary(summary, global_step=global_step)
    if saver is not None:
        saver.save(sess, os.path.join(model_folder, "model.ckpt"))
    if clustering or tree:
        output_folder = os.path.join(model_folder, 'test', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.mkdir(output_folder)
        if clustering:
            clustering_classification(labels, y, output_folder)
        if tree:
            tree_analysis(dm, ids, times, annotations_file, output_folder)
        return output_folder
    return


def profiled_test_run(sess, targets, merged_summary, handle, h, global_step, model_folder, annotations, profiler, n,
                      writer=None, saver=None, clustering=True, tree=True, mode='raw_data'):
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
    assert mode == 'timeline' or mode == 'raw_data'

    run_meta = tf.RunMetadata()
    summary, y, labels, dm, ids, times = sess.run([merged_summary] + targets, feed_dict={handle: h},
                                                  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                                  run_metadata=run_meta)
    if writer is not None:
        writer.add_summary(summary, global_step=global_step)
    if saver is not None:
        saver.save(sess, os.path.join(model_folder, "model.ckpt"))

    profiler.add_step(n, run_meta)

    if mode == 'raw_data':
        opts = (option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.time_and_memory())
                .with_step(-1)  # average of all steps
                .with_file_output(os.path.join(model_folder, 'profile_time.txt')).build())
        profiler.profile_operations(options=opts)

    if mode == 'timeline':
        opts = (option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.time_and_memory())
                .with_step(-1)  # average of all steps
                .with_timeline_output(os.path.join(model_folder, 'profile_graph.txt')).build())
        profiler.profile_graph(options=opts)

    if clustering or tree:
        output_folder = os.path.join(model_folder, 'test', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.mkdir(output_folder)
        if clustering:
            clustering_classification(labels, y, output_folder)
        if tree:
            tree_analysis(dm, ids, times, annotations, output_folder)
        return output_folder

    return


def profiled_run(sess, train_step, merged_summary, handle, h, global_step, model_folder, profiler, n, writer,
                 mode='raw_data'):
    """

    :param sess:
    :param writer:
    :param log_step:
    :param mode: either 'timeline' or 'raw_data'
    :return:
    """
    assert mode == 'timeline' or mode == 'raw_data'

    run_meta = tf.RunMetadata()
    summary, _ = sess.run([merged_summary, train_step], feed_dict={handle: h},
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


def logged_run(sess, train_step, merged_summary, handle, h, global_step, writer):
    summary, _ = sess.run([merged_summary, train_step], feed_dict={handle: h})
    writer.add_summary(summary, global_step=global_step)
    return


def training_run(sess, train_step, handle, h):
    _ = sess.run(train_step, feed_dict={handle: h})
    return


def classifier_run(sess, cl_train, merged_summary, class_predicted, class_true, handle, th, vh, global_step,
                   output_folder, writer=None):
    for i in range(1_000):
        _ = sess.run([cl_train], feed_dict={handle: th})
    summary, cp, ct = sess.run([merged_summary, class_predicted, class_true], feed_dict={handle: vh})
    if writer is not None:
        writer.add_summary(summary, global_step=global_step)
    with open(os.path.join(output_folder, 'classification.txt'), 'a') as f:
        f.write('pred\ttrue\n')
        for p, t in zip(cp, ct):
            f.write('{}\t{}\n'.format(p, t))
    return
