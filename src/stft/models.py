import tensorflow as tf
from tensorflow.contrib import rnn


def classify_3c2_rnn_bn_pool_sigmoid(input_layer, params):
    """

    :param input_layer:
    :param params: a dictionary with number of filters (fi), kernel sizes (ki_f, ki_t), and embedding size (n)
    :return:
    """
    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,  # Dimension of input_layer = (-1, 233, 1323, 1)
        filters=params['f1'],
        kernel_size=(params['k1_f'], params['k1_t']),
        padding="same",
        activation=tf.nn.sigmoid)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 4), strides=(2, 4), padding='same')  # shape (-1, 117, 331, 8)
    norm1 = tf.layers.batch_normalization(inputs=pool1)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=norm1,
        filters=params['f2'],
        kernel_size=(params['k2_f'], params['k2_t']),
        padding="same",
        activation=tf.nn.sigmoid)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 4), strides=(2, 4), padding='same')  # shape (-1, 59, 83, 8)
    norm2 = tf.layers.batch_normalization(inputs=pool2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=norm2,
        filters=params['f3'],
        kernel_size=(params['k3_f'], params['k3_t']),
        padding="same",
        activation=tf.nn.sigmoid)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 4), strides=(2, 4), padding='same')  # shape (-1, 30, 21, 16)
    norm3 = tf.layers.batch_normalization(inputs=pool3)

    # TODO: make this reshape automatic, without having to hard-code the dimensions.
    time_steps = 21
    features = 30 * params['f3']  # frequencies * channels
    temp = tf.reshape(tf.transpose(norm3, [0, 2, 1, 3]), [-1, time_steps, features])
    # embeddings = tf.reshape(temp, [-1, 14*480])
    rnn_input = tf.unstack(temp, time_steps, axis=1)
    lstm_cell = rnn.BasicLSTMCell(params['n_embeddings'], forget_bias=1.0)
    embeddings, state = rnn.static_rnn(lstm_cell, rnn_input, dtype=tf.float32)
    logits = tf.layers.dense(inputs=embeddings[-1], units=params['n_composers'])
    return logits, embeddings[-1]


def extract_3c2_rnn_bn_pool_sigmoid(input_layer, params):
    """

    :param input_layer:
    :param params: a dictionary with number of filters (fi), kernel sizes (ki_f, ki_t), and embedding size (n)
    :return:
    """
    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,  # Dimension of input_layer = (-1, 233, 1323, 1)
        filters=params['f1'],
        kernel_size=(params['k1_f'], params['k1_t']),
        padding="same",
        activation=tf.nn.sigmoid)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 4), strides=(2, 4), padding='same')  # shape (-1, 117, 331, 8)
    norm1 = tf.layers.batch_normalization(inputs=pool1)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=norm1,
        filters=params['f2'],
        kernel_size=(params['k2_f'], params['k2_t']),
        padding="same",
        activation=tf.nn.sigmoid)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 4), strides=(2, 4), padding='same')  # shape (-1, 59, 83, 8)
    norm2 = tf.layers.batch_normalization(inputs=pool2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=norm2,
        filters=params['f3'],
        kernel_size=(params['k3_f'], params['k3_t']),
        padding="same",
        activation=tf.nn.sigmoid)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 4), strides=(2, 4), padding='same')  # shape (-1, 30, 21, 16)
    norm3 = tf.layers.batch_normalization(inputs=pool3)

    # TODO: make this reshape automatic, without having to hard-code the dimensions.
    time_steps = 21
    features = 30 * params['f3']  # frequencies * channels
    temp = tf.reshape(tf.transpose(norm3, [0, 2, 1, 3]), [-1, time_steps, features])
    # embeddings = tf.reshape(temp, [-1, 14*480])
    rnn_input = tf.unstack(temp, time_steps, axis=1)
    lstm_cell = rnn.BasicLSTMCell(params['n_embeddings'], forget_bias=1.0)
    embeddings, state = rnn.static_rnn(lstm_cell, rnn_input, dtype=tf.float32)
    return [], embeddings[-1]