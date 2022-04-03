"""
Adopted from [2017] [https://github.com/tensorflow/lucid]
"""

import tensorflow as tf


def redirected_relu_grad(image, grad):
    """Compute ReLu gradient.

    Args:
        image (array): the image to be modified by the feature vis process.
        grad (tf.Tensor): the gradient tensor.

    Returns:
        Redirected gradients smaller than 0.
    """

    # Compute ReLu gradient (indices of image input smaller than zero)
    relu_grad = tf.where(image < 0., tf.zeros_like(grad), grad)

    # Compute redirected gradient: where do we need to zero out incoming gradient
    # to prevent input going lower if its already negative
    neg_pushing_lower = tf.logical_and(image < 0., grad > 0.)
    redirected_grad = tf.where(neg_pushing_lower, tf.zeros_like(grad), grad)

    # Ensure we have at least a rank 2 tensor, as we expect a batch dimension
    assert_op = tf.Assert(tf.greater(
        tf.rank(relu_grad), 1), [tf.rank(relu_grad)])
    with tf.control_dependencies([assert_op]):
        # only use redirected gradient where nothing got through original gradient
        batch = tf.shape(relu_grad)[0]
        reshaped_relu_grad = tf.reshape(relu_grad, [batch, -1])
        relu_grad_mag = tf.norm(reshaped_relu_grad, axis=1)
    result_grad = tf.where(relu_grad_mag > 0., relu_grad, redirected_grad)

    global_step_t = tf.compat.v1.train.get_or_create_global_step()
    return_relu_grad = tf.greater(global_step_t, tf.constant(16, tf.int64))

    return tf.where(return_relu_grad, relu_grad, result_grad)


def redirected_relu6_grad(image, grad):
    """Compute ReLu6 gradients

    Args:
        image (array): the image to be modified by the feature vis process.
        grad (tf.Tensor): the gradient tensor.

    Returns:
        Redirected gradients bigger than 6.
    """

    # Compute ReLu gradient
    relu6_cond = tf.logical_or(image < 0., image > 6.)
    relu_grad = tf.where(relu6_cond, tf.zeros_like(grad), grad)

    # Compute redirected gradient: where do we need to zero out incoming gradient
    # to prevent input going lower if its already negative, or going higher if
    # already bigger than 6?
    neg_pushing_lower = tf.logical_and(image < 0., grad > 0.)
    pos_pushing_higher = tf.logical_and(image > 6., grad < 0.)
    dir_filter = tf.logical_or(neg_pushing_lower, pos_pushing_higher)
    redirected_grad = tf.where(dir_filter, tf.zeros_like(grad), grad)

    # Ensure we have at least a rank 2 tensor, as we expect a batch dimension
    assert_op = tf.Assert(tf.greater(
        tf.rank(relu_grad), 1), [tf.rank(relu_grad)])
    with tf.control_dependencies([assert_op]):
        # only use redirected gradient where nothing got through original gradient
        batch = tf.shape(relu_grad)[0]
        reshaped_relu_grad = tf.reshape(relu_grad, [batch, -1])
        relu_grad_mag = tf.norm(reshaped_relu_grad, axis=1)
    result_grad = tf.where(relu_grad_mag > 0., relu_grad, redirected_grad)

    global_step_t = tf.compat.v1.train.get_or_create_global_step()
    return_relu_grad = tf.greater(global_step_t, tf.constant(16, tf.int64))

    return tf.where(return_relu_grad, relu_grad, result_grad)
