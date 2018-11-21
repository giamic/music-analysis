import tensorflow as tf


def wavenet_s2_20l_relu_classify(input_layer, params):
    """
    Do a wavenet-like implementation with filter size 2, 20 layers, and ReLU activation
    :param input_layer:
    :param params: a dictionary with n_composers
    :return:
    """
    conv0 = input_layer
    conv1 = tf.layers.conv1d(inputs=conv0, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=2)
    conv2 = tf.layers.conv1d(inputs=conv1, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=2)
    conv3 = tf.layers.conv1d(inputs=conv2, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=2)
    conv4 = tf.layers.conv1d(inputs=conv3, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=4)
    conv5 = tf.layers.conv1d(inputs=conv4, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=4)
    conv6 = tf.layers.conv1d(inputs=conv5, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=4)
    conv7 = tf.layers.conv1d(inputs=conv6, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=8)
    conv8 = tf.layers.conv1d(inputs=conv7, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=8)
    conv9 = tf.layers.conv1d(inputs=conv8, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                             filters=8)
    conv10 = tf.layers.conv1d(inputs=conv9, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=16)
    conv11 = tf.layers.conv1d(inputs=conv10, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=16)
    conv12 = tf.layers.conv1d(inputs=conv11, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=16)
    conv13 = tf.layers.conv1d(inputs=conv12, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=32)
    conv14 = tf.layers.conv1d(inputs=conv13, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=32)
    conv15 = tf.layers.conv1d(inputs=conv14, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=32)
    conv16 = tf.layers.conv1d(inputs=conv15, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=64)
    conv17 = tf.layers.conv1d(inputs=conv16, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=64)
    conv18 = tf.layers.conv1d(inputs=conv17, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=128)
    conv19 = tf.layers.conv1d(inputs=conv18, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=128)
    conv20 = tf.layers.conv1d(inputs=conv19, kernel_size=2, padding="valid", strides=2, activation=tf.nn.relu,
                              filters=256)
    embeddings = tf.reshape(conv20, [-1, 256])
    logits = tf.layers.dense(inputs=embeddings, units=params['n_composers'])
    return logits, embeddings


def wavenet_s4_10l_relu_classify(input_layer, params):
    """
    Do a wavenet-like implementation with filter size 2, 20 layers, and ReLU activation
    :param input_layer:
    :param params: a dictionary with n_composers
    :return:
    """
    conv0 = input_layer
    conv1 = tf.layers.conv1d(inputs=conv0, kernel_size=4, padding="valid", strides=4, activation=tf.nn.relu,
                             filters=8)
    conv2 = tf.layers.conv1d(inputs=conv1, kernel_size=4, padding="valid", strides=4, activation=tf.nn.relu,
                             filters=8)
    conv3 = tf.layers.conv1d(inputs=conv2, kernel_size=4, padding="valid", strides=4, activation=tf.nn.relu,
                             filters=16)
    conv4 = tf.layers.conv1d(inputs=conv3, kernel_size=4, padding="valid", strides=4, activation=tf.nn.relu,
                             filters=16)
    conv5 = tf.layers.conv1d(inputs=conv4, kernel_size=4, padding="valid", strides=4, activation=tf.nn.relu,
                             filters=32)
    conv6 = tf.layers.conv1d(inputs=conv5, kernel_size=4, padding="valid", strides=4, activation=tf.nn.relu,
                             filters=32)
    conv7 = tf.layers.conv1d(inputs=conv6, kernel_size=4, padding="valid", strides=4, activation=tf.nn.relu,
                             filters=64)
    conv8 = tf.layers.conv1d(inputs=conv7, kernel_size=4, padding="valid", strides=4, activation=tf.nn.relu,
                             filters=64)
    conv9 = tf.layers.conv1d(inputs=conv8, kernel_size=4, padding="valid", strides=4, activation=tf.nn.relu,
                             filters=128)
    conv10 = tf.layers.conv1d(inputs=conv9, kernel_size=4, padding="valid", strides=4, activation=tf.nn.relu,
                              filters=256)
    embeddings = tf.reshape(conv10, [-1, 256])
    logits = tf.layers.dense(inputs=embeddings, units=params['n_composers'])
    return logits, embeddings
