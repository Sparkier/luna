"""
The main file for the feature vis process
"""
from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf
from tensorflow import keras

from matplotlib.pyplot import figure, imshow, axis

from luna.featurevis import images as imgs
from luna.featurevis import objectives


@dataclass
class OptimizationParameters():
    """object for generalizing optimization parameters."""
    iterations: int
    learning_rate: Optional[int]
    optimizer: Optional[object]


def visualize(
    image,
    objective,
    optimization_parameters,
    transformation=None,
    threshold=None,
    minimize=False
):
    """Create a feature visualization for a filter in a layer or a whole layer of the model.

    Args:
        image (array): the image to be modified by the feature vis process.
        objective (object): computes the loss to optimize visualization for (filter or layer).
        optimization_parameters (OptimizationParameters): the optimizer class to be applied.
        filter_index (number): the index of the filter to be visualized. Whole layer if None.
        transformation (function): a function defining the transformations to be performed.
        threshold (list): Intermediate steps for visualization. Defaults to None.
        minimize (bool): whether or not to apply minimize as opposed to calling apply_gradient()
                         for adam optimizer.

    Returns:
        tuple: activation and result image for the process.
    """
    _threshold_figures = figure(figsize=(15, 10), dpi=200)

    def transform_image(img):
        if transformation:
            if not callable(transformation):
                raise ValueError("transformation needs to be a function.")
            transformed = transformation(img)

            if transformed.shape[1] != img.shape[1] or transformed.shape[2] != img.shape[2]:
                transformed = tf.image.resize(
                    transformed, [img.shape[1], img.shape[2]])
            return transformed
        return img

    # Can only compute gradients for variables
    image = tf.Variable(image)
    print("Starting Feature Vis Process")
    for iteration in range(optimization_parameters.iterations):
        activation, image = optimize(
            image, objective, transform_image, optimization_parameters, minimize)

        print('>>', int(iteration / optimization_parameters.iterations * 100), '%',
              end="\r", flush=True)

        # Routine for creating a threshold image for Jupyter Notebooks
        if isinstance(threshold, list) and (iteration in threshold):
            threshold_image = _threshold_figures.add_subplot(
                1, len(threshold), threshold.index(iteration) + 1
            )
            threshold_image.title.set_text(f"Step {iteration}")
            threshold_view(image)

    print('>> 100 %')

    # Decode the resulting input image when gradient ascent is used.
    if (minimize is False) and (optimization_parameters.optimizer is None):
        image = imgs.deprocess_image(image[0].numpy())
    else:
        image = image[0].numpy()

    return activation, image


def visualize_filter(
    image,
    model,
    layer,
    filter_index,
    optimization_parameters,
    transformation=None,
    regularization=None,
    threshold=None,
    minimize=False
):
    """Create a feature visualization for a filter in a layer of the model.

    Args:
        image (array): the image to be modified by the feature vis process.
        model (object): the model to be used for the feature visualization.
        layer (string): the name of the layer to be used in the visualization.
        filter_index (number): the index of the filter to be visualized.
        optimization_parameters (OptimizationParameters): the optimizer class to be applied.
        transformation (function): a function defining the transformations to be performed.
        regularization (function): customized regularizers to be applied. Defaults to None.
        threshold (list): Intermediate steps for visualization. Defaults to None.
        minimize (bool): whether or not to apply minimize as opposed to calling apply_gradient()
                         for adam optimizer.

    Returns:
        tuple: activation and result image for the process.
    """
    objective = objectives.FilterObjective(
        model, layer, filter_index, regularization)
    return visualize(image, objective, optimization_parameters, transformation, threshold, minimize)


def visualize_layer(
    image,
    model,
    layer,
    optimization_parameters,
    transformation=None,
    regularization=None,
    threshold=None,
    minimize=False
):
    """Create deep dream visualization of a layer in the model.

    Args:
        image (array): the image to be modified by the feature vis process.
        model (object): the model to be used for the feature visualization.
        layer (string): the name of the layer to be used in the visualization.
        optimization_parameters (OptimizationParameters): the optimizer class to be applied.
        transformation (function): a function defining the transformations to be performed.
        regularization (function): customized regularizers to be applied. Defaults to None.
        threshold (list): Intermediate steps for visualization. Defaults to None.
        minimize (bool): whether or not to apply minimize as opposed to calling apply_gradient()
                         for adam optimizer.

    Returns:
        tuple: activation and result image for the process.
    """
    objective = objectives.LayerObjective(model, layer, regularization)
    return visualize(image, objective, optimization_parameters, transformation, threshold, minimize)


def optimize(img, objective, transform_image, optimization_parameters, minimize):
    """Performs one step of gradient ascent.

      Args:
          img (array): the image to be changed by the gradiend ascend.
          objective (object): computes the loss to optimize visualization for (filter or layer).
          transform_image (function): a function defining the transformations to be performed.
          optimization_parameters (OptimizationParameters): optimizer (only Adam is supported)
          minimize (bool): whether or not to apply minimize as opposed to calling apply_gradient()
                           for adam optimizer.
          filter_index (number): which filter to optimize for. Whole layer if None.

      Returns:
          tuple: the activation and the modified image
    """

    def compute_loss():
        transformed_image = transform_image(img)
        return objective.loss(transformed_image)

    if not minimize:
        with tf.GradientTape() as tape:
            tape.watch(img)
            # Record operations to compute gradients with respect to img
            loss = compute_loss()

        # Compute gradients with respect to img using backpropagation.
        grads = tape.gradient(loss, img)

        # Normalize gradients.
        if optimization_parameters.optimizer is None:
            grads = tf.math.l2_normalize(grads)
            # fallback to standard learning for apply gradient ascent
            learning_rate = optimization_parameters.learning_rate or 0.7
            img = img + learning_rate * grads
        else:
            # Relu gradient overrides do not seem to work,
            # or needs to be performed more fine grained.
            # grads_relu_0 = rg.redirected_relu_grad(img, grads)
            # grads_modified = rg.redirected_relu6_grad(img, grads_relu_0)
            # optimization_parameters.optimizer.apply_gradients(
            #     zip([grads_modified], [img]))
            optimization_parameters.optimizer.apply_gradients(
                zip([grads], [img]))
        activation = -loss
    else:
        activation = optimization_parameters.optimizer.minimize(compute_loss, [img])

    return activation, img


def threshold_view(image):
    """Intermediate visualizer.

    Args:
        image (list): Image.
    """
    # Process image
    image = imgs.deprocess_image(image[0].numpy())
    image = keras.preprocessing.image.img_to_array(image)

    if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
        image = tf.transpose(image, [0, 2, 1])

    image = keras.preprocessing.image.array_to_img(
        image, data_format="channels_last")
    imshow(image)
    axis("off")
