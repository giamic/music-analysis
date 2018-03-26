import tensorflow as tf


def cnn_model_fn(features, labels, mode):
    """
    Model function for CNN.

    :param features:
    :param labels:
    :param mode:
    :return:
    """
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 128, 12])

    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv1d(
        inputs=input_layer,
        filters=32,
        kernel_size=4,
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=64,
        kernel_size=4,
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)

    # Convolutional Layer #1 and Pooling Layer #3
    conv3 = tf.layers.conv1d(
        inputs=pool2,
        filters=128,
        kernel_size=4,
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)

    # Dense Layer
    pool3_flat = tf.reshape(pool3, [-1, 16*128])
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # TODO: study what happens from now on.

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
