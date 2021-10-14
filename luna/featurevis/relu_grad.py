"""
Adopted from [2017] [https://github.com/tensorflow/lucid]
"""

from contextlib import contextmanager
import uuid
import tensorflow as tf


def redirected_relu_grad(operation, grad):
    """Compute ReLu gradient.

    Args:
        operation (str): the type of operation to convert, needs to be Relu.
        grad (tf.Tensor): the gradient tensor.

    Returns:
        Redirected gradients smaller than 0.
    """
    assert operation.type == "Relu"
    relu_input = operation.inputs[0]

    # Compute ReLu gradient
    relu_grad = tf.where(relu_input < 0., tf.zeros_like(grad), grad)

    # Compute redirected gradient: where do we need to zero out incoming gradient
    # to prevent input going lower if its already negative
    neg_pushing_lower = tf.logical_and(relu_input < 0., grad > 0.)
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

#pylint: disable=R0914


def redirected_relu6_grad(operation, grad):
    """Compute ReLu6 gradients

    Args:
        operation (str): the type of operation to convert, needs to be Relu6.
        grad (tf.Tensor): the gradient tensor.

    Returns:
        Redirected gradients bigger than 6.
    """
    assert operation.type == "Relu6"
    relu_input = operation.inputs[0]

    # Compute ReLu gradient
    relu6_cond = tf.logical_or(relu_input < 0., relu_input > 6.)
    relu_grad = tf.where(relu6_cond, tf.zeros_like(grad), grad)

    # Compute redirected gradient: where do we need to zero out incoming gradient
    # to prevent input going lower if its already negative, or going higher if
    # already bigger than 6?
    neg_pushing_lower = tf.logical_and(relu_input < 0., grad > 0.)
    pos_pushing_higher = tf.logical_and(relu_input > 6., grad < 0.)
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


def register_to_random_name(grad_f):
    """Register a gradient function to a random string.
    In order to use a custom gradient in TensorFlow, it must be registered to a string.
    This is both a hassle, and -- because only one function can every be registered to a
    string -- annoying to iterate on in an interactive environemnt.
    This function registers a function to a unique random string of the form:
    {FUNCTION_NAME}_{RANDOM_SALT}
    and then returns the random string. This is a helper in creating more convenient
    gradient overrides.

    Args:
        grad_f: gradient function to register. Should map (op, grad) -> grad(s)

    Returns:
        String that gradient function was registered to.
    """

    grad_f_name = grad_f.__name__ + "_" + str(uuid.uuid4())
    tf.RegisterGradient(grad_f_name)(grad_f)
    return grad_f_name


@contextmanager
def gradient_override_map(override_dict):
    """Convenience wrapper for graph.gradient_override_map().
    This functions provides two conveniences over normal tensorflow gradient
    overrides: it auomatically uses the default graph instead of you needing to
    find the graph, and it automatically
    Example:
        def _foo_grad_alt(op, grad): ...
        with gradient_override({"Foo": _foo_grad_alt}):

    Args:
        override_dict: A dictionary describing how to override the gradient.
        keys: strings correponding to the op type that should have their gradient
            overriden.
        values: functions or strings registered to gradient functions
    """
    override_dict_by_name = {}
    for (op_name, grad_f) in override_dict.items():
        if isinstance(grad_f, str):
            override_dict_by_name[op_name] = grad_f
        else:
            override_dict_by_name[op_name] = register_to_random_name(grad_f)
    with tf.compat.v1.get_default_graph().gradient_override_map(override_dict_by_name):
        yield
