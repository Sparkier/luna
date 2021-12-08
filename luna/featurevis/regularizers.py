"""
The regularizers for feature vis process
"""
import tensorflow as tf
import numpy as np


def blur_channel(activation, channel_num=3):
    """function for blurring the activation

    Args:
        activation (list): filter activation
        channel_num (int, optional): channel number. Defaults to 3.

    Returns:
        list: blurred activation
    """
    activation_size = activation.shape[-1]
    blur_matrix = np.zeros(
        [channel_num, channel_num, activation_size, activation_size])
    for channel in range(activation_size):
        blur_matrix_channel = blur_matrix[:, :, channel, channel]
        blur_matrix_channel[:, :] = 0.5
        blur_matrix_channel[1:-1, 1:-1] = 1.0

    def conv_k(activation):
        return tf.nn.conv2d(activation, blur_matrix, [1, 1, 1, 1], "SAME")
    return conv_k(activation) / conv_k(tf.ones_like(activation))


def l1_regularization(activation, l1_value=-0.05):
    """apply l1 regularization to filter activation

    Args:
        activation (list): filter activation
        l1 (float, optional): regularizer value. Defaults to -0.05.

    Returns:
        list: regularized activation
    """
    return l1_value * tf.reduce_sum(tf.abs(activation) - 0.5)


def total_variation(activation, tv_value=-0.25):
    """apply total variation regularization to filter activation

    Args:
        activation (list): filter activation
        tv_value (float, optional): regularizer value. Defaults to -0.25.

    Returns:
        list: regularized activation
    """
    return tv_value * tf.image.total_variation(activation)


def blur_activation(activation, blur=-1.0):
    """apply blur regularization to filter activation

    Args:
        activation (list): filter activation
        blur (float, optional): regularizer value. Defaults to -1.0.

    Returns:
        list: regularized activation
    """
    activation_blurred = tf.stop_gradient(blur_channel(activation))
    return blur * 0.5 * (tf.reduce_sum((activation - activation_blurred) ** 2))
