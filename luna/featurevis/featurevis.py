"""
The main file for the feature vis process
"""
from __future__ import absolute_import, division, print_function

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras

from luna.featurevis import images as imgs
from luna.featurevis import relu_grad as rg
from luna.featurevis import transformations as trans
from luna.featurevis import regularizers as regs


@dataclass
class OptimizationParameters():
    """object for generalizing optimization parameters."""
    iterations: int
    learning_rate: int


def visualize_filter(
    image,
    model,
    layer,
    filter_index,
    optimization_parameters,
    transformation=None,
    regularizers=None,
):
    """Create a feature visualization for a filter in a layer of the model.

    Args:
        image (array): the image to be modified by the feature vis process.
        model (object): the model to be used for the feature visualization.
        layer (string): the name of the layer to be used in the visualization.
        filter_index (number): the index of the filter to be visualized.
        optimization_parameters (OptimizationParameters): the optimizer class to be applied.
        transformations (function): a function defining the transformations to be perfromed.
        regularizers (dict): customized regularizers to be applied. Defaults to None.


    Returns:
        tuple: activation and result image for the process.
    """
    image = tf.Variable(image)
    feature_extractor = get_feature_extractor(model, layer)
    print("Starting Feature Vis Process")
    for iteration in range(optimization_parameters.iterations):
        pctg = int(iteration / optimization_parameters.iterations * 100)

        if transformation:
            if not callable(transformation):
                raise ValueError("The transformations need to be a function.")
            image = transformation(image)
        else:
            image = trans.standard_transformation(image)

        activation, image = gradient_ascent_step(
            image, feature_extractor, filter_index, regularizers, optimization_parameters.learning_rate
        )

        print('>>', pctg, '%', end="\r", flush=True)
    print('>> 100 %')
    if image.shape[1] < 299 or image.shape[2] < 299:
        image = tf.image.resize(image, [299, 299])
    # Decode the resulting input image
    image = imgs.deprocess_image(image[0].numpy())

    return activation, image


def compute_activation(input_image, model, filter_index, regularizers):
    """Computes the loss for the feature visualization process.

    Args:
        input_image (array): the image that is used to compute the loss.
        model (object): the model on which to compute the loss.
        filter_index (int): for which filter to compute the loss.
        Defaults to False.
        regularizers (dict): Dictionary of regularizers and their values.

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
    activation_score = tf.reduce_mean(filter_activation)
    if regularizers:
        obj = regs.perform_regularization(activation, regularizers)
        return obj
    return activation_score


def gradient_ascent_step(img, model, filter_index, regularizers, learning_rate):
    """Performing one step of gradient ascend.

    Args:
        img (array): the image to be changed by the gradiend ascend.
        model (object): the model with which to perform the image change.
        filter_index (number): which filter to optimize for.
        regularizers (dict): Dictionary of regularizers and their values.
        learning_rate (number): how much to change the image per iteration.

    Returns:
        tuple: the activation and the modified image
    """
    with tf.GradientTape() as tape:
        tape.watch(img)
        activation = compute_activation(img, model, filter_index, regularizers)
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
