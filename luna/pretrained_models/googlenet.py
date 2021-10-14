"""
A Keras Implementation of the GoogleNet (InceptionV1) from
https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14
"""
# pylint: skip-file
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, ZeroPadding2D, Dropout, Flatten
from tensorflow.keras.layers import Concatenate, Activation
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.utils.conv_utils import convert_kernel


class PoolHelper(Layer):
    """Helper class for the googlenet creation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs[:, :, 1:, 1:]

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LRN(Layer):
    """Helper class for the googlenet creation."""

    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2  # half the local region
        input_sqr = K.square(x)  # square the input
        if K.backend() == 'theano':
            # make an empty tensor with zero pads along channel dimension
            zeros = T.alloc(0., b, ch + 2*half_n, r, c)
            # set the center to be the squared input
            input_sqr = T.set_subtensor(
                zeros[:, half_n:half_n+ch, :, :], input_sqr)
        else:
            input_sqr = tf.pad(
                input_sqr, [[0, 0], [half_n, half_n], [0, 0], [0, 0]])
        scale = self.k  # offset for the scale
        norm_alpha = self.alpha / self.n  # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_googlenet(user_weight_path=None):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    input = Input(shape=(3, 224, 224))

    input_pad = ZeroPadding2D(padding=(3, 3))(input)
    conv1_7x7_s2 = Conv2D(64, (7, 7), strides=(2, 2), padding='valid',
                          activation='relu', name='conv1/7x7_s2',
                          kernel_regularizer=l2(0.0002))(input_pad)
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
    pool1_helper = PoolHelper()(conv1_zero_pad)
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid',
                                name='pool1/3x3_s2')(pool1_helper)
    pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)

    conv2_3x3_reduce = Conv2D(64, (1, 1), padding='same', activation='relu',
                              name='conv2/3x3_reduce',
                              kernel_regularizer=l2(0.0002))(pool1_norm1)
    conv2_3x3 = Conv2D(192, (3, 3), padding='same', activation='relu',
                       name='conv2/3x3',
                       kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
    conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
    pool2_helper = PoolHelper()(conv2_zero_pad)
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid',
                                name='pool2/3x3_s2')(pool2_helper)

    inception_3a_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu',
                              name='inception_3a/1x1',
                              kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3_reduce = Conv2D(96, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_3a/3x3_reduce',
                                     kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_3x3_pad = ZeroPadding2D(
        padding=(1, 1))(inception_3a_3x3_reduce)
    inception_3a_3x3 = Conv2D(128, (3, 3), padding='valid', activation='relu',
                              name='inception_3a/3x3',
                              kernel_regularizer=l2(0.0002))(inception_3a_3x3_pad)
    inception_3a_5x5_reduce = Conv2D(16, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_3a/5x5_reduce',
                                     kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
    inception_3a_5x5_pad = ZeroPadding2D(
        padding=(2, 2))(inception_3a_5x5_reduce)
    inception_3a_5x5 = Conv2D(32, (5, 5), padding='valid',
                              activation='relu', name='inception_3a/5x5',
                              kernel_regularizer=l2(0.0002))(inception_3a_5x5_pad)
    inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same',
                                     name='inception_3a/pool')(pool2_3x3_s2)
    inception_3a_pool_proj = Conv2D(32, (1, 1), padding='same', activation='relu',
                                    name='inception_3a/pool_proj',
                                    kernel_regularizer=l2(0.0002))(inception_3a_pool)
    inception_3a_output = Concatenate(axis=1, name='inception_3a/output')(
        [inception_3a_1x1, inception_3a_3x3,
         inception_3a_5x5, inception_3a_pool_proj])

    inception_3b_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                              name='inception_3b/1x1',
                              kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3_reduce = Conv2D(128, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_3b/3x3_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_3x3_pad = ZeroPadding2D(
        padding=(1, 1))(inception_3b_3x3_reduce)
    inception_3b_3x3 = Conv2D(192, (3, 3), padding='valid', activation='relu',
                              name='inception_3b/3x3',
                              kernel_regularizer=l2(0.0002))(inception_3b_3x3_pad)
    inception_3b_5x5_reduce = Conv2D(32, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_3b/5x5_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_3a_output)
    inception_3b_5x5_pad = ZeroPadding2D(
        padding=(2, 2))(inception_3b_5x5_reduce)
    inception_3b_5x5 = Conv2D(96, (5, 5), padding='valid', activation='relu',
                              name='inception_3b/5x5',
                              kernel_regularizer=l2(0.0002))(inception_3b_5x5_pad)
    inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same', name='inception_3b/pool')(inception_3a_output)
    inception_3b_pool_proj = Conv2D(64, (1, 1), padding='same',
                                    activation='relu',
                                    name='inception_3b/pool_proj',
                                    kernel_regularizer=l2(0.0002))(inception_3b_pool)
    inception_3b_output = Concatenate(axis=1, name='inception_3b/output')(
        [inception_3b_1x1, inception_3b_3x3,
         inception_3b_5x5, inception_3b_pool_proj])

    inception_3b_output_zero_pad = ZeroPadding2D(
        padding=(1, 1))(inception_3b_output)
    pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                padding='valid',
                                name='pool3/3x3_s2')(pool3_helper)

    inception_4a_1x1 = Conv2D(192, (1, 1), padding='same', activation='relu',
                              name='inception_4a/1x1',
                              kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3_reduce = Conv2D(96, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_4a/3x3_reduce',
                                     kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_3x3_pad = ZeroPadding2D(
        padding=(1, 1))(inception_4a_3x3_reduce)
    inception_4a_3x3 = Conv2D(208, (3, 3), padding='valid',
                              activation='relu', name='inception_4a/3x3',
                              kernel_regularizer=l2(0.0002))(inception_4a_3x3_pad)
    inception_4a_5x5_reduce = Conv2D(16, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_4a/5x5_reduce',
                                     kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
    inception_4a_5x5_pad = ZeroPadding2D(
        padding=(2, 2))(inception_4a_5x5_reduce)
    inception_4a_5x5 = Conv2D(48, (5, 5), padding='valid', activation='relu',
                              name='inception_4a/5x5',
                              kernel_regularizer=l2(0.0002))(inception_4a_5x5_pad)
    inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same',
                                     name='inception_4a/pool')(pool3_3x3_s2)
    inception_4a_pool_proj = Conv2D(64, (1, 1), padding='same',
                                    activation='relu',
                                    name='inception_4a/pool_proj',
                                    kernel_regularizer=l2(0.0002))(
                                        inception_4a_pool)
    inception_4a_output = Concatenate(axis=1, name='inception_4a/output')(
        [inception_4a_1x1, inception_4a_3x3,
         inception_4a_5x5, inception_4a_pool_proj])

    loss1_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3),
                                      name='loss1/ave_pool')(inception_4a_output)
    loss1_conv = Conv2D(128, (1, 1), padding='same', activation='relu',
                        name='loss1/conv', kernel_regularizer=l2(0.0002))(loss1_ave_pool)
    loss1_flat = Flatten()(loss1_conv)
    loss1_fc = Dense(1024, activation='relu', name='loss1/fc',
                     kernel_regularizer=l2(0.0002))(loss1_flat)
    loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
    loss1_classifier = Dense(1000, name='loss1/classifier',
                             kernel_regularizer=l2(0.0002))(loss1_drop_fc)
    loss1_classifier_act = Activation('softmax')(loss1_classifier)

    inception_4b_1x1 = Conv2D(160, (1, 1), padding='same', activation='relu',
                              name='inception_4b/1x1',
                              kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3_reduce = Conv2D(112, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_4b/3x3_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_3x3_pad = ZeroPadding2D(
        padding=(1, 1))(inception_4b_3x3_reduce)
    inception_4b_3x3 = Conv2D(224, (3, 3), padding='valid', activation='relu',
                              name='inception_4b/3x3',
                              kernel_regularizer=l2(0.0002))(inception_4b_3x3_pad)
    inception_4b_5x5_reduce = Conv2D(24, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_4b/5x5_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_4a_output)
    inception_4b_5x5_pad = ZeroPadding2D(
        padding=(2, 2))(inception_4b_5x5_reduce)
    inception_4b_5x5 = Conv2D(64, (5, 5), padding='valid', activation='relu',
                              name='inception_4b/5x5',
                              kernel_regularizer=l2(0.0002))(inception_4b_5x5_pad)
    inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same',
                                     name='inception_4b/pool')(inception_4a_output)
    inception_4b_pool_proj = Conv2D(64, (1, 1), padding='same',
                                    activation='relu',
                                    name='inception_4b/pool_proj',
                                    kernel_regularizer=l2(0.0002))(inception_4b_pool)
    inception_4b_output = Concatenate(axis=1, name='inception_4b/output')(
        [inception_4b_1x1, inception_4b_3x3,
         inception_4b_5x5, inception_4b_pool_proj])

    inception_4c_1x1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                              name='inception_4c/1x1',
                              kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3_reduce = Conv2D(128, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_4c/3x3_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_3x3_pad = ZeroPadding2D(
        padding=(1, 1))(inception_4c_3x3_reduce)
    inception_4c_3x3 = Conv2D(256, (3, 3), padding='valid', activation='relu',
                              name='inception_4c/3x3',
                              kernel_regularizer=l2(0.0002))(inception_4c_3x3_pad)
    inception_4c_5x5_reduce = Conv2D(24, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_4c/5x5_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_4b_output)
    inception_4c_5x5_pad = ZeroPadding2D(
        padding=(2, 2))(inception_4c_5x5_reduce)
    inception_4c_5x5 = Conv2D(64, (5, 5), padding='valid', activation='relu',
                              name='inception_4c/5x5',
                              kernel_regularizer=l2(0.0002))(inception_4c_5x5_pad)
    inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same', name='inception_4c/pool')(inception_4b_output)
    inception_4c_pool_proj = Conv2D(64, (1, 1), padding='same',
                                    activation='relu',
                                    name='inception_4c/pool_proj',
                                    kernel_regularizer=l2(0.0002))(inception_4c_pool)
    inception_4c_output = Concatenate(axis=1, name='inception_4c/output')(
        [inception_4c_1x1, inception_4c_3x3,
         inception_4c_5x5, inception_4c_pool_proj])

    inception_4d_1x1 = Conv2D(112, (1, 1), padding='same', activation='relu',
                              name='inception_4d/1x1',
                              kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3_reduce = Conv2D(144, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_4d/3x3_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_3x3_pad = ZeroPadding2D(
        padding=(1, 1))(inception_4d_3x3_reduce)
    inception_4d_3x3 = Conv2D(288, (3, 3), padding='valid', activation='relu',
                              name='inception_4d/3x3',
                              kernel_regularizer=l2(0.0002))(inception_4d_3x3_pad)
    inception_4d_5x5_reduce = Conv2D(32, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_4d/5x5_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_4c_output)
    inception_4d_5x5_pad = ZeroPadding2D(
        padding=(2, 2))(inception_4d_5x5_reduce)
    inception_4d_5x5 = Conv2D(64, (5, 5), padding='valid', activation='relu',
                              name='inception_4d/5x5',
                              kernel_regularizer=l2(0.0002))(inception_4d_5x5_pad)
    inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same',
                                     name='inception_4d/pool')(inception_4c_output)
    inception_4d_pool_proj = Conv2D(64, (1, 1), padding='same',
                                    activation='relu',
                                    name='inception_4d/pool_proj',
                                    kernel_regularizer=l2(0.0002))(inception_4d_pool)
    inception_4d_output = Concatenate(axis=1, name='inception_4d/output')(
        [inception_4d_1x1, inception_4d_3x3,
         inception_4d_5x5, inception_4d_pool_proj])

    loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3),
                                      name='loss2/ave_pool')(inception_4d_output)
    loss2_conv = Conv2D(128, (1, 1), padding='same', activation='relu',
                        name='loss2/conv', kernel_regularizer=l2(0.0002))(loss2_ave_pool)
    loss2_flat = Flatten()(loss2_conv)
    loss2_fc = Dense(1024, activation='relu', name='loss2/fc',
                     kernel_regularizer=l2(0.0002))(loss2_flat)
    loss2_drop_fc = Dropout(rate=0.7)(loss2_fc)
    loss2_classifier = Dense(1000, name='loss2/classifier',
                             kernel_regularizer=l2(0.0002))(loss2_drop_fc)
    loss2_classifier_act = Activation('softmax')(loss2_classifier)

    inception_4e_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu',
                              name='inception_4e/1x1',
                              kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_3x3_reduce = Conv2D(160, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_4e/3x3_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_3x3_pad = ZeroPadding2D(
        padding=(1, 1))(inception_4e_3x3_reduce)
    inception_4e_3x3 = Conv2D(320, (3, 3), padding='valid', activation='relu',
                              name='inception_4e/3x3',
                              kernel_regularizer=l2(0.0002))(inception_4e_3x3_pad)
    inception_4e_5x5_reduce = Conv2D(32, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_4e/5x5_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_4d_output)
    inception_4e_5x5_pad = ZeroPadding2D(
        padding=(2, 2))(inception_4e_5x5_reduce)
    inception_4e_5x5 = Conv2D(128, (5, 5), padding='valid', activation='relu',
                              name='inception_4e/5x5',
                              kernel_regularizer=l2(0.0002))(inception_4e_5x5_pad)
    inception_4e_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same', name='inception_4e/pool')(inception_4d_output)
    inception_4e_pool_proj = Conv2D(128, (1, 1), padding='same',
                                    activation='relu',
                                    name='inception_4e/pool_proj',
                                    kernel_regularizer=l2(0.0002))(inception_4e_pool)
    inception_4e_output = Concatenate(axis=1, name='inception_4e/output')(
        [inception_4e_1x1, inception_4e_3x3,
         inception_4e_5x5, inception_4e_pool_proj])

    inception_4e_output_zero_pad = ZeroPadding2D(
        padding=(1, 1))(inception_4e_output)
    pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid',
                                name='pool4/3x3_s2')(pool4_helper)

    inception_5a_1x1 = Conv2D(256, (1, 1), padding='same', activation='relu',
                              name='inception_5a/1x1',
                              kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_3x3_reduce = Conv2D(160, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_5a/3x3_reduce',
                                     kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_3x3_pad = ZeroPadding2D(
        padding=(1, 1))(inception_5a_3x3_reduce)
    inception_5a_3x3 = Conv2D(320, (3, 3), padding='valid', activation='relu',
                              name='inception_5a/3x3',
                              kernel_regularizer=l2(0.0002))(inception_5a_3x3_pad)
    inception_5a_5x5_reduce = Conv2D(32, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_5a/5x5_reduce',
                                     kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
    inception_5a_5x5_pad = ZeroPadding2D(
        padding=(2, 2))(inception_5a_5x5_reduce)
    inception_5a_5x5 = Conv2D(128, (5, 5), padding='valid', activation='relu',
                              name='inception_5a/5x5',
                              kernel_regularizer=l2(0.0002))(inception_5a_5x5_pad)
    inception_5a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same',
                                     name='inception_5a/pool')(pool4_3x3_s2)
    inception_5a_pool_proj = Conv2D(128, (1, 1), padding='same',
                                    activation='relu',
                                    name='inception_5a/pool_proj',
                                    kernel_regularizer=l2(0.0002))(inception_5a_pool)
    inception_5a_output = Concatenate(axis=1, name='inception_5a/output')(
        [inception_5a_1x1, inception_5a_3x3,
         inception_5a_5x5, inception_5a_pool_proj])

    inception_5b_1x1 = Conv2D(384, (1, 1), padding='same', activation='relu',
                              name='inception_5b/1x1',
                              kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_3x3_reduce = Conv2D(192, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_5b/3x3_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_3x3_pad = ZeroPadding2D(
        padding=(1, 1))(inception_5b_3x3_reduce)
    inception_5b_3x3 = Conv2D(384, (3, 3), padding='valid', activation='relu',
                              name='inception_5b/3x3',
                              kernel_regularizer=l2(0.0002))(inception_5b_3x3_pad)
    inception_5b_5x5_reduce = Conv2D(48, (1, 1), padding='same',
                                     activation='relu',
                                     name='inception_5b/5x5_reduce',
                                     kernel_regularizer=l2(0.0002))(inception_5a_output)
    inception_5b_5x5_pad = ZeroPadding2D(
        padding=(2, 2))(inception_5b_5x5_reduce)
    inception_5b_5x5 = Conv2D(128, (5, 5), padding='valid', activation='relu',
                              name='inception_5b/5x5',
                              kernel_regularizer=l2(0.0002))(inception_5b_5x5_pad)
    inception_5b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same',
                                     name='inception_5b/pool')(inception_5a_output)
    inception_5b_pool_proj = Conv2D(128, (1, 1), padding='same',
                                    activation='relu',
                                    name='inception_5b/pool_proj',
                                    kernel_regularizer=l2(0.0002))(inception_5b_pool)
    inception_5b_output = Concatenate(axis=1, name='inception_5b/output')(
        [inception_5b_1x1, inception_5b_3x3,
         inception_5b_5x5, inception_5b_pool_proj])

    pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                    name='pool5/7x7_s2')(inception_5b_output)
    loss3_flat = Flatten()(pool5_7x7_s1)
    pool5_drop_7x7_s1 = Dropout(rate=0.4)(loss3_flat)
    loss3_classifier = Dense(1000, name='loss3/classifier',
                             kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
    loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

    googlenet = Model(inputs=input, outputs=[loss1_classifier_act,
                                             loss2_classifier_act,
                                             loss3_classifier_act])

    if user_weight_path:
        print("Model loaded")
        googlenet.load_weights(user_weight_path)
    else:
        print("No model directory given. Downloading Model...")
        weights_path = get_file(
            'googlenet_weights.h5',
            'https://github.com/pinae/GoogLeNet-Keras-Test/raw/master/googlenet_weights.h5')
        googlenet.load_weights(weights_path)

    if keras.backend.backend() == 'tensorflow':
        # convert the convolutional kernels for tensorflow
        for layer in googlenet.layers:
            if layer.__class__.__name__ == 'Conv2D':
                original_w = tf.compat.v1.keras.backend.get_value(layer.kernel)
                converted_w = convert_kernel(original_w)
                tf.compat.v1.assign(layer.kernel, converted_w)

    return googlenet
