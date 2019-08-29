import tensorflow as tf
from tensorflow.python import name_scope
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Concatenate, BatchNormalization, Dropout, Conv2D, MaxPooling2D, \
    Bidirectional, GRU, Dense

from stft.config import PARAMS


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
            x = BatchNormalization()(x)
            x = Dropout(d)(x)
            y = Conv2D(filters=4 * k, kernel_size=(1, 1), padding='same', data_format='channels_last',
                       activation='relu')(x)
            y = BatchNormalization()(y)
            z = Conv2D(filters=k, kernel_size=f, padding='same', data_format='channels_last', activation='relu')(y)
            x = Concatenate()([x, z])
    return x


def classify_3c2_rnn_bn_pool_sigmoid():
    """

    :param input_layer:
    :param PARAMS: a dictionary with number of filters (fi), kernel sizes (ki_f, ki_t), and embedding size (n)
    :return:
    """
    x = Input(shape=(PARAMS['x.shape'][1:]), name="piano_roll_input")
    z = DenseNetLayer2D(x, 4, 4, (PARAMS['k1_f'], PARAMS['k1_t']), d=0.3, n=1)
    z = MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding='same')(z)  # shape (-1, 117, 331, 17)
    z = DenseNetLayer2D(z, 4, 4, (PARAMS['k1_f'], PARAMS['k1_t']), d=0.3, n=2)
    z = MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding='same')(z)  # shape (-1, 59, 83, 33)
    z = DenseNetLayer2D(z, 4, 4, (PARAMS['k1_f'], PARAMS['k1_t']), d=0.3, n=3)
    z = MaxPooling2D(pool_size=(2, 4), strides=(2, 4), padding='same')(z)  # shape (-1, 30, 21, 49)
    z = tf.reshape(tf.transpose(z, [0, 2, 1, 3]), [-1, z.shape[2], z.shape[1] * z.shape[3]])
    z = Bidirectional(GRU(48, return_sequences=False, dropout=0.3))(z)
    y = Dense(PARAMS['n_composers'], activation='sigmoid')(z)
    model = Model(inputs=x, outputs=y, name='DenseNet_GRU')
    return model
