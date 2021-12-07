"""
The regularizers for feature vis process
"""
import tensorflow as tf
import numpy as np


def apply_blur(activation, channel_num=3):
    """function for blurring the activation

    Args:
        activation (list): filter activation
        channel_num (int, optional): channel number. Defaults to 3.

    Returns:
        list: blurred activation
    """
    activation_size = activation.shape[-1]
    blur_matrix = np.zeros([channel_num, channel_num, activation_size, activation_size])
    for channel in range(activation_size):
        blur_matrix_channel = blur_matrix[:, :, channel, channel]
        blur_matrix_channel[:, :] = 0.5
        blur_matrix_channel[1:-1, 1:-1] = 1.0

    conv_k = lambda t: tf.nn.conv2d(t, blur_matrix, [1, 1, 1, 1], "SAME")
    return conv_k(activation) / conv_k(tf.ones_like(activation))


def l1_regularizer(activation, l1_value=-0.05):
    """apply l1 regularization to filter activation

    Args:
        activation (list): filter activation
        l1 (float, optional): regularizer value. Defaults to -0.05.

    Returns:
        list: regularized activation
    """
    return l1_value * tf.reduce_sum(tf.abs(activation) - 0.5)


def tv_regularizer(activation, tv_value=-0.25):
    """apply total variation regularization to filter activation

    Args:
        activation (list): filter activation
        tv_value (float, optional): regularizer value. Defaults to -0.25.

    Returns:
        list: regularized activation
    """
    return tv_value * tf.image.total_variation(activation)


def blur_regularizer(activation, blur=-1.0):
    """apply blur regularization to filter activation

    Args:
        activation (list): filter activation
        blur (float, optional): regularizer value. Defaults to -1.0.

    Returns:
        list: regularized activation
    """
    activation_blurred = tf.stop_gradient(apply_blur(activation))
    return blur * 0.5 * (tf.reduce_sum((activation - activation_blurred) ** 2))


def perform_regularization(activation, activation_score, regularization_func):
    """Perfrom a sequence of regularizations

    Args:
        activation (list): filter activation.
        activation_score (array): value of the activation of the layer/channel.
        regularization_func (function): a function defining the regularizations
                                        to be perfromed.


    Returns:
        list: regularized activation_score.
    """

    activation_score = regularization_func(activation, activation_score)
    return activation_score
