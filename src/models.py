import tensorflow as tf


def three_layers_conv(input_layer, params):
    """

    :param input_layer:
    :param params: a dictionary with number of filters (f1, f2, f3), kernel sizes (k1, k2, k3), and embedding size (n)
    :return:
    """
    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv1d(
        inputs=input_layer,
        filters=params['f1'],
        kernel_size=params['k1'],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)  # size 64
    norm1 = tf.layers.batch_normalization(inputs=pool1)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
        inputs=norm1,
        filters=params['f2'],
        kernel_size=params['k2'],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)  # size 32
    norm2 = tf.layers.batch_normalization(inputs=pool2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv1d(
        inputs=norm2,
        filters=params['f3'],
        kernel_size=params['k3'],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)  # size 16
    norm3 = tf.layers.batch_normalization(inputs=pool3)

    # Dense Layer
    norm3_flat = tf.reshape(norm3, [-1, 16 * 128])
    embeddings = tf.layers.dense(inputs=norm3_flat, units=params['n'])
    return embeddings
