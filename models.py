import time

import tensorflow as tf
from tensorflow import name_scope
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Concatenate, Bidirectional, Dense, MaxPool2D, GRU

from stft.config import PARAMS


class TimeOut(Callback):
    def __init__(self, t0, timeout):
        super().__init__()
        self.t0 = t0
        self.timeout = timeout  # time in minutes

    def on_train_batch_end(self, batch, logs=None):
        if time.time() - self.t0 > self.timeout * 60:  # 58 minutes
            print(f"\nReached {(time.time() - self.t0) / 60:.3f} minutes of training, stopping")
            self.model.stop_training = True


def DenseNetLayer2D(x, l, k, f, d=0.3, n=1):
    """
    Implementation of a DenseNetLayer
    :param x: input
    :param l: number of elementary blocks in the layer
    :param k: features generated at every block
    :param f: tuple containing the size of the filters
    :param d: dropout rate
    :param n: unique identifier of the DenseNetLayer
    :return:
    """
    with name_scope(f"denseNet_{n}"):
        for _ in range(l):
            # x = Dropout(d)(x)
            y = Conv2D(filters=4 * k, kernel_size=(1, 1), padding='same', data_format='channels_last',
                       activation='relu')(x)
            y = BatchNormalization()(y)
            z = Conv2D(filters=k, kernel_size=f, padding='same', data_format='channels_last', activation='relu')(y)
            z = BatchNormalization()(z)
            x = Concatenate()([x, z])
    return x


def classify_3dn2_gru_bn(l, k):
    """

    :param l: the number of layers in each dense network
    :param k: the number of additional features in each layer of the dense network
    :return:
    """
    x = Input(shape=(PARAMS['x.shape']), name="piano_roll_input")  # shape (-1, 233, 1323, 1)
    z = DenseNetLayer2D(x, l, k, (PARAMS['k1_f'], PARAMS['k1_t']), d=0.3, n=1)
    z = MaxPool2D(pool_size=(2, 4), strides=(2, 4), padding='same')(z)  # shape (-1, 117, 331, 1+l*k)
    z = DenseNetLayer2D(z, l, k, (PARAMS['k1_f'], PARAMS['k1_t']), d=0.3, n=2)
    z = MaxPool2D(pool_size=(2, 4), strides=(2, 4), padding='same')(z)  # shape (-1, 59, 83, 1+2l*k)
    z = DenseNetLayer2D(z, l, k, (PARAMS['k1_f'], PARAMS['k1_t']), d=0.3, n=3)
    z = MaxPool2D(pool_size=(2, 4), strides=(2, 4), padding='same')(z)  # shape (-1, 30, 21, 1+3l*k)
    z = tf.reshape(tf.transpose(z, [0, 2, 1, 3]),
                   [-1, z.shape[2], z.shape[1] * z.shape[3]])  # shape (-1, 21, 30*(1+3l*k))
    z = Bidirectional(GRU(48, return_sequences=False, dropout=0.3))(z)
    y = Dense(PARAMS['n_composers'], activation='sigmoid')(z)
    model = Model(inputs=x, outputs=y, name='DenseNet_GRU')
    return model


def classify_2dn2_gru_bn(l, k):
    """

    :param l: the number of layers in each dense network
    :param k: the number of additional features in each layer of the dense network
    :return:
    """
    x = Input(shape=(PARAMS['x.shape']), name="piano_roll_input")  # shape (-1, 233, 1323, 1)
    z = MaxPool2D(pool_size=(2, 4), strides=(2, 4), padding='same')(x)  # shape (-1, 117, 331, 1)
    z = DenseNetLayer2D(z, l, k, (PARAMS['k1_f'], PARAMS['k1_t']), d=0.3, n=2)
    z = MaxPool2D(pool_size=(2, 4), strides=(2, 4), padding='same')(z)  # shape (-1, 59, 83, 1+l*k)
    z = DenseNetLayer2D(z, l, k, (PARAMS['k1_f'], PARAMS['k1_t']), d=0.3, n=3)
    z = MaxPool2D(pool_size=(2, 4), strides=(2, 4), padding='same')(z)  # shape (-1, 30, 21, 1+2l*k)
    z = tf.reshape(tf.transpose(z, [0, 2, 1, 3]),
                   [-1, z.shape[2], z.shape[1] * z.shape[3]])  # shape (-1, 21, 30*(1+3l*k))
    z = Bidirectional(GRU(48, return_sequences=False, dropout=0.3))(z)
    y = Dense(PARAMS['n_composers'], activation='sigmoid')(z)
    model = tf.keras.Model(inputs=x, outputs=y, name='DenseNet_GRU')
    return model
