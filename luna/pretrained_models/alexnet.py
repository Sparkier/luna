"""
A Keras Implementation for Alexnet from
https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
"""
from  pathlib import Path
from keras import backend as K
from keras.engine import Layer
from keras.models import Model
from keras.layers import (Activation, Dense, Dropout, Flatten, Input,
                          concatenate, Lambda)
from keras.layers.convolutional import (MaxPooling2D,
                                        ZeroPadding2D)
from keras.utils.data_utils import get_file
from tensorflow.keras.layers import Conv2D




def cross_channel_normalization(alpha=1e-4, k=2, beta=0.75, num=5, **kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """

    def channel_norm(filt):
        _, channel, _, _ = filt.shape
        half = num // 2
        square = K.square(filt)
        extra_channels = K.spatial_2d_padding(
            K.permute_dimensions(square, (0, 2, 3, 1)), (0, half))
        extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
        scale = k
        for i in range(num):
            scale += alpha * extra_channels[:, i:i + channel, :, :]
        scale = scale ** beta
        return filt / scale

    return Lambda(channel_norm, output_shape=lambda input_shape: input_shape, **kwargs)


def splittensor(axis=1, ratio_split=1, id_split=0, **kwargs):
    """Function for splitting tensor

    Args:
        axis (int, optional): chosen axis. Defaults to 1.
        ratio_split (int, optional): Ratio of split. Defaults to 1.
        id_split (int, optional): Split index. Defaults to 0.
    """
    def split_tensor(tensor):
        div = tensor.shape[axis] // ratio_split

        if axis == 0:
            output = tensor[id_split * div:(id_split + 1) * div, :, :, :]
        elif axis == 1:
            output = tensor[:, id_split * div:(id_split + 1) * div, :, :]
        elif axis == 2:
            output = tensor[:, :, id_split * div:(id_split + 1) * div, :]
        elif axis == 3:
            output = tensor[:, :, :, id_split * div:(id_split + 1) * div]
        else:
            raise ValueError('This axis is not possible')

        return output

    def reshape_tensor(input_shape):
        output_shape = list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(split_tensor, output_shape=reshape_tensor, **kwargs)


class Softmax4D(Layer):
    """Class for applying softmax on the desired Layer

    Args:
        Layer ([type]): The Layer to apply the Softmax to.
    """

    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        exp = K.exp(inputs - K.max(inputs, axis=self.axis, keepdims=True))
        sum_of_exp = K.sum(exp, axis=self.axis, keepdims=True)
        return exp / sum_of_exp


def alex_net(user_weight_path = None, heatmap=False):
    """Generates the Network
    """
    if heatmap:
        inputs = Input(shape=(None, None, 3))
    else:
        inputs = Input(shape=(227, 227, 3))

    conv_1 = Conv2D(96, 11,strides=(4, 4), activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = cross_channel_normalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = concatenate([
        Conv2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_2)
        ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = cross_channel_normalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = concatenate([
        Conv2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_4)
        ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = concatenate([
        Conv2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_5)
        ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

    if heatmap:
        dense_1 = Conv2D(
            4096, 6, 6, activation='relu', name='dense_1')(dense_1)
        dense_2 = Conv2D(
            4096, 1, 1, activation='relu', name='dense_2')(dense_1)
        dense_3 = Conv2D(1000, 1, 1, name='dense_3')(dense_2)
        prediction = Softmax4D(axis=1, name='softmax')(dense_3)
    else:
        dense_1 = Flatten(name='flatten')(dense_1)
        dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(1000, name='dense_3')(dense_3)
        prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(input=inputs, output=prediction)

    weights_path = Path(str(Path.home()) + r"~\.keras\datasets\alexnet_weights.h5")

    if weights_path.is_file():
        model.load_weights(weights_path)
    elif user_weight_path:
        model.load_weights(user_weight_path)
    else:
        weights_path = get_file(
            'alexnet_weights.h5',
            'http://files.heuritech.com/weights/alexnet_weights.h5')
        model.load_weights(weights_path)
    return model
