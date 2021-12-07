"""
The main file for the feature vis process
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras

from luna.featurevis import relu_grad as rg
from luna.featurevis import images as imgs
from luna.featurevis import transformations as trans

# pylint: disable=too-few-public-methods


class OptimizationParameters():
    """object for generalizing optimization parameters.

    Args:
        iterations (number): how many iterations to optimize for.
        learning_rate (number): update amount after each iteration.
    """

    def __init__(self, iterations, learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate

#pylint: disable=too-many-locals

def visualize_filter(
    image,
    model,
    layer,
    filter_index,
    opt_param,
    trans_func = None,
):
    """Create a feature visualization for a filter in a layer of the model.

    Args:
        image (array): the image to be modified by the feature vis process.
        model (object): the model to be used for the feature visualization.
        layer (string): the name of the layer to be used in the visualization.
        filter_index (number): the index of the filter to be visualized.
        opt_param (class): the optimizer class to be applied.
        trans_func (function): a function defining the transformations to be perfromed.


    Returns:
        tuple: activation and result image for the process.
    """
    image = tf.Variable(image)
    feature_extractor = get_feature_extractor(model, layer)
    print("Starting Feature Vis Process")
    for iteration in range(opt_param.iterations):
        pctg = int(iteration / opt_param.iterations * 100)

        if trans_func:
            if not callable(trans_func):
                raise ValueError("The transformations need to be a function.")
            image = trans.perform_trans(image, trans_func)
        else:
            image = trans.standard_transforms(image)

        activation, image = gradient_ascent_step(
            image, feature_extractor, filter_index, opt_param.learning_rate
        )

        print('>>', pctg, '%', end="\r", flush=True)
    print('>> 100 %')
    if image.shape[1] < 299 or image.shape[2] < 299:
        image = tf.image.resize(image, [299, 299])
    # Decode the resulting input image
    image = imgs.deprocess_image(image[0].numpy())

    return activation, image


def compute_activation(input_image, model, filter_index):
    """Computes the loss for the feature visualization process.

    Args:
        input_image (array): the image that is used to compute the loss.
        model (object): the model on which to compute the loss.
        filter_index (number): for which filter to compute the loss.
        Defaults to False.

    Returns:
        number: the activation for the specified setting
    """
    with rg.gradient_override_map(
            {'Relu': rg.redirected_relu_grad, 'Relu6': rg.redirected_relu6_grad}):
        activation = model(input_image)
    if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
        filter_activation = activation[:, filter_index, :, :]
    else:
        filter_activation = activation[:, :, :, filter_index]
    return tf.reduce_mean(filter_activation)


def gradient_ascent_step(img, model, filter_index, learning_rate):
    """Performing one step of gradient ascend.

    Args:
        img (array): the image to be changed by the gradiend ascend.
        model (object): the model with which to perform the image change.
        filter_index (number): which filter to optimize for.
        learning_rate (number): how much to change the image per iteration.

    Returns:
        tuple: the activation and the modified image
    """
    with tf.GradientTape() as tape:
        tape.watch(img)
        activation = compute_activation(img, model, filter_index)
    # Compute gradients.
    grads = tape.gradient(activation, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img = img + learning_rate * grads
    return activation, img


def get_feature_extractor(model, layer_name):
    """Builds a model that that returns the activation of the specified layer.

    Args:
        model (object): the model used as a basis for the feature extractor.
        layer (string): the layer at which to cap the original model.
    """
    layer = model.get_layer(name=layer_name)
    return keras.Model(inputs=model.inputs, outputs=layer.output)
