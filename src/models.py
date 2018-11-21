import tensorflow as tf


def classify_4c2_bn_pool_sigmoid(input_layer, params):
    """

    :param input_layer:
    :param params: a dictionary with number of filters (fi), kernel sizes (ki_f, ki_t), and embedding size (n)
    :return:
    """
    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,  # Dimension of input_layer = (-1, 233, 441, 1)
        filters=params['f1'],
        kernel_size=(params['k1_f'], params['k1_t']),
        padding="same",
        activation=tf.nn.sigmoid)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 4), strides=(2, 4), padding='same')  # shape (-1, 117, 111, 8)
    norm1 = tf.layers.batch_normalization(inputs=pool1)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=norm1,
        filters=params['f2'],
        kernel_size=(params['k2_f'], params['k2_t']),
        padding="same",
        activation=tf.nn.sigmoid)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 4), strides=(2, 4), padding='same')  # shape (-1, 59, 28, 8)
    norm2 = tf.layers.batch_normalization(inputs=pool2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv2d(
        inputs=norm2,
        filters=params['f3'],
        kernel_size=(params['k3_f'], params['k3_t']),
        padding="same",
        activation=tf.nn.sigmoid)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 4), strides=(2, 4), padding='same')  # shape (-1, 30, 7, 16)
    norm3 = tf.layers.batch_normalization(inputs=pool3)

    # Convolutional Layer #4 and Pooling Layer #4
    conv4 = tf.layers.conv2d(
        inputs=norm3,
        filters=params['f4'],
        kernel_size=(params['k4_f'], params['k4_t']),
        padding="same",
        activation=tf.nn.sigmoid)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(2, 4), strides=(2, 4), padding='same')  # shape (-1, 15, 2, 16)
    norm4 = tf.layers.batch_normalization(inputs=pool4)

    # Dense Layer
    flat = tf.reshape(norm4, [-1, 15 * 2 * params['f4']])
    embeddings = tf.layers.dense(inputs=flat, units=params['n_embeddings'], activation=tf.nn.sigmoid)
    logits = tf.layers.dense(inputs=embeddings, units=params['n_composers'])
    return logits


def match_3cl_bn_pool_relu(input_layer, params):
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
    norm3_flat = tf.reshape(norm3, [-1, 16 * params['f3']])
    embeddings = tf.layers.dense(inputs=norm3_flat, units=params['n_embeddings'])
    return embeddings


def match_3cl_pool_sigm(input_layer, params):
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
        activation=tf.nn.sigmoid)
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)  # size 64

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=params['f2'],
        kernel_size=params['k2'],
        padding="same",
        activation=tf.nn.sigmoid)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)  # size 32

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv1d(
        inputs=pool2,
        filters=params['f3'],
        kernel_size=params['k3'],
        padding="same",
        activation=tf.nn.sigmoid)
    pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)  # size 16

    # Dense Layer
    flat = tf.reshape(pool3, [-1, 16 * params['f3']])
    embeddings = tf.layers.dense(inputs=flat, units=params['n_embeddings'])
    return embeddings


def match_5cl_pool_sigm(input_layer, params):
    """

    :param input_layer:
    :param params: a dictionary with number of filters (f1, f2, f3), kernel sizes (k1, k2, k3), and embedding size (n)
    :return:
    """
    # Convolutional Layer #1 and Pooling Layer #1
    conv1a = tf.layers.conv1d(
        inputs=input_layer,
        filters=params['f1'],
        kernel_size=params['k1'],
        padding="same",
        activation=tf.nn.sigmoid)
    conv1b = tf.layers.conv1d(
        inputs=conv1a,
        filters=params['f1'],
        kernel_size=params['k1'],
        padding="same",
        activation=tf.nn.sigmoid)
    pool1 = tf.layers.max_pooling1d(inputs=conv1b, pool_size=2, strides=2)  # size 64

    # Convolutional Layer #2 and Pooling Layer #2
    conv2a = tf.layers.conv1d(
        inputs=pool1,
        filters=params['f2'],
        kernel_size=params['k2'],
        padding="same",
        activation=tf.nn.sigmoid)
    conv2b = tf.layers.conv1d(
        inputs=conv2a,
        filters=params['f2'],
        kernel_size=params['k2'],
        padding="same",
        activation=tf.nn.sigmoid)
    pool2 = tf.layers.max_pooling1d(inputs=conv2b, pool_size=2, strides=2)  # size 32

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv1d(
        inputs=pool2,
        filters=params['f3'],
        kernel_size=params['k3'],
        padding="same",
        activation=tf.nn.sigmoid)
    pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)  # size 16

    # Dense Layer
    flat = tf.reshape(pool3, [-1, 16 * params['f3']])
    embeddings = tf.layers.dense(inputs=flat, units=params['n_embeddings'])
    return embeddings


def classify(input_layer, params):
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
        activation=tf.nn.sigmoid)
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)  # size 64

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=params['f2'],
        kernel_size=params['k2'],
        padding="same",
        activation=tf.nn.sigmoid)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)  # size 32

    # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv1d(
        inputs=pool2,
        filters=params['f3'],
        kernel_size=params['k3'],
        padding="same",
        activation=tf.nn.sigmoid)
    pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2)  # size 16

    # Dense Layer
    flat = tf.reshape(pool3, [-1, 16 * params['f3']])
    embeddings = tf.layers.dense(inputs=flat, units=params['n_embeddings'], activation=tf.nn.sigmoid)
    predictions = tf.layers.dense(inputs=embeddings, units=params['n_composers'], activation=tf.nn.softmax)
    return predictions


def classify_2cl_sigm(input_layer, params):
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
        activation=tf.nn.sigmoid)
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)  # size 64

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=params['f2'],
        kernel_size=params['k2'],
        padding="same",
        activation=tf.nn.sigmoid)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)  # size 32

    # Dense Layer
    flat = tf.reshape(pool2, [-1, 32 * params['f2']])
    embeddings = tf.layers.dense(inputs=flat, units=params['n_embeddings'], activation=tf.nn.sigmoid)
    predictions = tf.layers.dense(inputs=embeddings, units=params['n_composers'], activation=tf.nn.softmax)
    return predictions


def classify_2cl_relu(input_layer, params):
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

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=params['f2'],
        kernel_size=params['k2'],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)  # size 32

    # Dense Layer
    flat = tf.reshape(pool2, [-1, 32 * params['f2']])
    embeddings = tf.layers.dense(inputs=flat, units=params['n_embeddings'], activation=tf.nn.sigmoid)
    predictions = tf.layers.dense(inputs=embeddings, units=params['n_composers'], activation=tf.nn.softmax)
    return predictions


def classify_logistic(embeddings, params):
    predictions = tf.layers.dense(inputs=embeddings, units=params['n_composers'], activation=tf.nn.softmax,
                                  name='classify')
    return predictions
