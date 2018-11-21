import tensorflow as tf
from tensorflow.contrib import rnn


def wavenet_s2_11l_relu_classify(input_layer, params):
    """
    Do a wavenet-like implementation with filter size 2, 20 layers, and ReLU activation
    :param input_layer:
    :param params: a dictionary with n_composers
    :return:
    """
    conv0 = input_layer
    conv1 = tf.layers.conv1d(inputs=conv0, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=32)
    conv2 = tf.layers.conv1d(inputs=conv1, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=32)
    conv3 = tf.layers.conv1d(inputs=conv2, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=32)
    conv4 = tf.layers.conv1d(inputs=conv3, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=64)
    conv5 = tf.layers.conv1d(inputs=conv4, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=64)
    conv6 = tf.layers.conv1d(inputs=conv5, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=64)
    conv7 = tf.layers.conv1d(inputs=conv6, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=128)
    conv8 = tf.layers.conv1d(inputs=conv7, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=128)
    conv9 = tf.layers.conv1d(inputs=conv8, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=256)
    conv10 = tf.layers.conv1d(inputs=conv9, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=256)
    conv11 = tf.layers.conv1d(inputs=conv10, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=512)
    embeddings = tf.reshape(conv11, [-1, 512])
    logits = tf.layers.dense(inputs=embeddings, units=params['n_composers'])
    return logits, embeddings


def wavenet_rnn_s2_7l_relu_classify(input_layer, params):
    """
    Do a wavenet-like implementation with filter size 2, 20 layers, and ReLU activation
    :param input_layer:
    :param params: a dictionary with n_composers
    :return:
    """

    def _conv_block(x, f):
        conv = tf.layers.conv1d(inputs=x, kernel_size=2, padding="valid", strides=2, filters=f,
                                activation=tf.nn.leaky_relu,
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))
        # return conv
        return tf.layers.batch_normalization(inputs=conv)

    # out0 = input_layer
    out0 = tf.layers.batch_normalization(inputs=input_layer)
    out1 = _conv_block(out0, 32)
    out2 = _conv_block(out1, 32)
    out3 = _conv_block(out2, 32)
    out4 = _conv_block(out3, 64)
    out5 = _conv_block(out4, 64)
    out6 = _conv_block(out5, 64)
    out7 = _conv_block(out6, 128)
    # conv1 = tf.layers.conv1d(inputs=conv0, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
    #                          filters=32)
    # conv2 = tf.layers.conv1d(inputs=conv1, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
    #                          filters=32)
    # conv3 = tf.layers.conv1d(inputs=conv2, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
    #                          filters=32)
    # conv4 = tf.layers.conv1d(inputs=conv3, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
    #                          filters=64)
    # conv5 = tf.layers.conv1d(inputs=conv4, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
    #                          filters=64)
    # conv6 = tf.layers.conv1d(inputs=conv5, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
    #                          filters=64)
    # conv7 = tf.layers.conv1d(inputs=conv6, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
    #                          filters=128)
    time_steps = 16
    rnn_input = tf.unstack(out7, time_steps, axis=1)
    lstm_cell = rnn.BasicLSTMCell(params['n_embeddings'], forget_bias=1.0)
    embeddings, state = rnn.static_rnn(lstm_cell, rnn_input, dtype=tf.float32)
    logits = tf.layers.dense(inputs=embeddings[-1], units=params['n_composers'])
    return logits, embeddings[-1], out0, out1, out2, out3, out4, out5, out6, out7


def classify_3c1_rnn_bn_pool_sigmoid(input_layer, params):
    """

    :param input_layer:
    :param params: a dictionary with number of filters (fi), kernel sizes (ki_f, ki_t), and embedding size (n)
    :return:
    """
    def _conv_block(x, f, k, p):
        conv = tf.layers.conv1d(
            inputs=x,
            filters=f,
            kernel_size=k,
            padding="same",
            activation=tf.nn.sigmoid)
        pool = tf.layers.max_pooling1d(
            inputs=conv,
            pool_size=p, strides=p,
            padding='same')
        norm = tf.layers.batch_normalization(inputs=pool)
        return norm

    out0 = input_layer
    out1 = _conv_block(out0, 32, 16, 4)
    out2 = _conv_block(out1, 64, 16, 4)
    out3 = _conv_block(out2, 128, 16, 4)
    # TODO: remove the following line, inserted just for compatibility with the other model during debug
    out4, out5, out6, out7 = [], [], [], []

    rnn_input = tf.unstack(out3, axis=1)
    lstm_cell = rnn.BasicLSTMCell(params['n_embeddings'], forget_bias=1.0)
    embeddings, state = rnn.static_rnn(lstm_cell, rnn_input, dtype=tf.float32)
    logits = tf.layers.dense(inputs=embeddings[-1], units=params['n_composers'])
    return logits, embeddings[-1], out0, out1, out2, out3, out4, out5, out6, out7
