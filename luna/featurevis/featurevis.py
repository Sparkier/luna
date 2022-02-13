"""
The main file for the feature vis process
"""
from __future__ import absolute_import, division, print_function

from dataclasses import dataclass
from typing import final

import tensorflow as tf

from tensorflow import keras

from matplotlib.pyplot import figure, imshow, axis
import matplotlib.pyplot as plt

from luna.featurevis import relu_grad as rg
from luna.featurevis import images as imgs
from luna.featurevis import transformations as trans
import json

all_score = {}
@dataclass
class OptimizationParameters():
    """object for generalizing optimization parameters."""
    iterations: int
    learning_rate: int
    optimizer: object = None #tf.keras.optimizer


def visualize_filter(
    image_opt,
    model,
    layer,
    filter_index,
    optimization_parameters,
    transformation=None,
    regularization=None,
    threshold=None,
):
    """Create a feature visualization for a filter in a layer of the model.

    Args:
        image (array): the image to be modified by the feature vis process.
        model (object): the model to be used for the feature visualization.
        layer (string): the name of the layer to be used in the visualization.
        filter_index (number): the index of the filter to be visualized.
        optimization_parameters (OptimizationParameters): the optimizer class to be applied.
        transformations (function): a function defining the transformations to be perfromed.
        regularization (function): customized regularizers to be applied. Defaults to None.
        threshold (list): Intermediate steps for visualization. Defaults to None.

    Returns:
        tuple: activation and result image for the process.
    """

    image_opt = tf.Variable(image_opt)
    optimizer = tf.keras.optimizers.Adam(epsilon=1e-08, learning_rate=0.05)
    #optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.05)
    print(f"we are in feature vis function, the input image is {image_opt}")
    feature_extractor = get_feature_extractor(model, layer)
    #print(f"the activation layer is {feature_extractor}")
    #_threshold_figures = figure(figsize=(15, 10), dpi=200)
    print("Starting Feature Vis Process")
    #lr = 0.05
    #decay = lr / optimization_parameters.iterations

    # def lr_time_based_decay(iteration, lr):
    #     return lr * 1 / (1 + decay * iteration)


    #@tf.function
    def activation_score():
        with rg.gradient_override_map({'Relu': rg.redirected_relu_grad, 'Relu6': rg.redirected_relu6_grad}):
            activation = feature_extractor(image_opt)
        #print(f"activation value with gradient override is {activation} and the shape is {activation.shape}")
        if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
            filter_activation = activation[:, filter_index, :, :]
        else:
            filter_activation = activation[:, :, :, filter_index]

        #print(f"filter activation with filer number {filter_index} is {filter_activation}")
        activation_score = tf.math.reduce_mean(filter_activation)
        print(f"activation score from luna is {activation_score}")
        #print(f"activation score is {activation_score}")
        #score= compute_activation(image_opt, feature_extractor, filter_index)

        #all_score.update({f"step {iteration}": activation_score.numpy()})

        return -1 * activation_score


    for iteration in tf.range(optimization_parameters.iterations):
        pctg = int(iteration / optimization_parameters.iterations * 100)
        #print(model.get_layer(layer).get_weights()[0][:,:,:,filter_index])
        # if transformation:
        #     if not callable(transformation):
        #         raise ValueError("The transformations need to be a function.")
        #     image_opt = transformation(image_opt)
        # else:
        #     #image = trans.standard_transformation(image)
        #     print("no transformation is applied.")
            #image = image
        # activation, image_opt = gradient_ascent_step(optimization_parameters.iterations, iteration,
        #     image_opt, feature_extractor, filter_index, regularization,
        #     optimization_parameters
        # )
        #lr = lr_time_based_decay(iteration.numpy(), lr)
        #optimizer = tf.keras.optimizers.Adam(epsilon=1e-08, learning_rate=lr)
        optimizer.minimize(activation_score, [image_opt])
        #print(f"after step {iteration} image is {image_opt}")
        # if iteration == optimization_parameters.iterations -1:
        #     print(f"image in the last step is {image_opt}")
        print('>>', pctg, '%', end="\r", flush=True)

    final_score = activation_score()
    all_score.update({"loss": final_score.numpy()})
    file = open(f'test_score/all_score_layer_{layer}_channel_{filter_index}.json', 'w')
    file= json.dump(str(all_score), file)
    print('>> 100 %')
    # if image.shape[1] < 299 or image.shape[2] < 299:
    #     image = tf.image.resize(image, [299, 299])
    # Decode the resulting input image
    # print(image_opt)
    # print(image_opt.shape)
    # print(type(image_opt))
    print(image_opt)
    image_opt = imgs.deprocess_image(image_opt[0].numpy())
    # print(image_opt.shape)
    # print(type(image_opt))
    print(image_opt)
    #file = open(f'all_score_layer_{layer}_channel_{filter_index}.json', 'w')
    #file= json.dump(str(all_score), file)
    return image_opt


def compute_activation(input_image, activation_layer, filter_index):
    """Computes the loss for the feature visualization process.

    Args:
        input_image (array): the image that is used to compute the loss.
        model (object): the model on which to compute the loss.
        filter_index (int): for which filter to compute the loss.
        Defaults to False.
        regularization (function): a function defining the regularizations to be perfromed.

    Returns:
        number: the activation for the specified setting
    """
    activation = activation_layer(input_image)
    print(f"activation value with gradient override is {activation} and the shape is {activation.shape}")
    if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
        filter_activation = activation[:, filter_index, :, :]
    else:
        filter_activation = activation[:, :, :, filter_index]

    print(f"filter activation with filer number {filter_index} is {filter_activation}")
    activation_score = tf.reduce_mean(filter_activation)
    print(f"activation score is {activation_score}")
    return (-1)*activation_score

# def visualize_filter(
#     image_opt,
#     model,
#     layer,
#     filter_index,
#     optimization_parameters,
#     transformation=None,
#     regularization=None,
#     threshold=None,
# ):
#     """Create a feature visualization for a filter in a layer of the model.

#     Args:
#         image (array): the image to be modified by the feature vis process.
#         model (object): the model to be used for the feature visualization.
#         layer (string): the name of the layer to be used in the visualization.
#         filter_index (number): the index of the filter to be visualized.
#         optimization_parameters (OptimizationParameters): the optimizer class to be applied.
#         transformations (function): a function defining the transformations to be perfromed.
#         regularization (function): customized regularizers to be applied. Defaults to None.
#         threshold (list): Intermediate steps for visualization. Defaults to None.

#     Returns:
#         tuple: activation and result image for the process.
#     """
#     image_opt = tf.Variable(image_opt)
#     print(f"we are in feature vis function, the input image is {image_opt}")
#     feature_extractor = get_feature_extractor(model, layer)
#     print(f"the activation layer is {feature_extractor}")
#     _threshold_figures = figure(figsize=(15, 10), dpi=200)
#     print("Starting Feature Vis Process")
#     for iteration in tf.range(optimization_parameters.iterations):
#         pctg = int(iteration / optimization_parameters.iterations * 100)

#         if transformation:
#             if not callable(transformation):
#                 raise ValueError("The transformations need to be a function.")
#             image_opt = transformation(image_opt)
#         else:
#             #image = trans.standard_transformation(image)
#             print("no transformation is applied.")
#             #image = image
#         activation, image_opt = gradient_ascent_step(optimization_parameters.iterations, iteration,
#             image_opt, feature_extractor, filter_index, regularization,
#             optimization_parameters
#         )

#         print(f"after step {iteration} image is {image_opt}")
#         print('>>', pctg, '%', end="\r", flush=True)

#         # Routine for creating a threshold image for Jupyter Notebooks
#         if isinstance(threshold, list) and (iteration in threshold):
#             threshold_image = _threshold_figures.add_subplot(
#                 1, len(threshold), threshold.index(iteration) + 1
#             )
#             threshold_image.title.set_text(f"Step {iteration}")
#             threshold_view(image_opt)
#             #tf.get_variable_scope().reuse_variables()

#     print('>> 100 %')
#     # if image.shape[1] < 299 or image.shape[2] < 299:
#     #     image = tf.image.resize(image, [299, 299])
#     # Decode the resulting input image
#     image_opt = imgs.deprocess_image(image_opt[0].numpy())

#     return activation, image_opt


# def compute_activation(total_iteration,iteration, input_image, activation_layer, filter_index, regularization):
#     """Computes the loss for the feature visualization process.

#     Args:
#         input_image (array): the image that is used to compute the loss.
#         model (object): the model on which to compute the loss.
#         filter_index (int): for which filter to compute the loss.
#         Defaults to False.
#         regularization (function): a function defining the regularizations to be perfromed.

#     Returns:
#         number: the activation for the specified setting
#     """
#     # with rg.gradient_override_map(
#     #     {'Relu': rg.redirected_relu_grad, 'Relu6': rg.redirected_relu6_grad}):
#     # #print(f"input image is {input_image}")

#     activation = activation_layer(input_image)
#     print(f"activation value with gradient override is {activation} and the shape is {activation.shape}")
#     if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
#         filter_activation = activation[:, filter_index, :, :]
#     else:
#         filter_activation = activation[:, :, :, filter_index]
#     print(f"filter activation with filer number {filter_index} is {filter_activation}")
#     activation_score = tf.reduce_mean(filter_activation)

#     all_score.update({f"step {iteration}": activation_score.numpy()})

#     print(all_score)
#     if iteration == total_iteration-1:
#         file = open(f'all_score_layer_channel_{filter_index}.json', 'w')
#         file= json.dump(str(all_score), file)

#     print(f"loss or in other word activation score is {activation_score}")
#     print(f"regularization is {regularization}")
#     if regularization:
#         if not callable(regularization):
#             raise ValueError("The regularizations need to be a function.")
#         activation_score = regularization(activation, activation_score)
#     return activation_score  #(-1)*activation_score
# #@tf.function
def gradient_ascent_step(total_iteration, iteration,img, activation_layer, filter_index, regularization, optimization_parameters):
    """Performing one step of gradient ascend.

      Args:
          img (array): the image to be changed by the gradiend ascend.
          model (object): the model with which to perform the image change.
          filter_index (number): which filter to optimize for.
          regularization (function): a function defining the regularizations to be perfromed.
          learning_rate (number): how much to change the image per iteration.
          optimization_parameters (OptimizationParameters): optimizer (only Adam is supported)

      Returns:
          tuple: the activation and the modified image
    """
    img = tf.Variable(img)
    #print(f"for whatever reason image is variable again and the value is {img}")

    # with tf.GradientTape() as tape:
    #    tape.watch(img)
    #    print(f"image in tape watch is {img}")
    #    activation = compute_activation(img, model, filter_index, regularization)
    #optimizer = tf.keras.optimizers.Adam(epsilon=1e-08)
    def compute_loss():
        activation = compute_activation(total_iteration,iteration,img, activation_layer, filter_index, regularization)
        print(f" after compute loss the activation is {activation}")
        activation = (-1)*activation
        return activation
    # Compute gradients.
    # grads = tape.gradient(activation, img)
    #print(f"Again activation value with gradient override is {activation} and the shape is {activation.shape}")
    #print(f"gradients are {grads}" + f" the grad shape is {grads.shape}")
    #Normalize gradients.
    if optimization_parameters.optimizer is None:
        with tf.GradientTape() as tape:
            tape.watch(img)
            print(f"image in tape watch is {img}")
            activation = compute_activation(total_iteration,iteration,img, activation_layer, filter_index, regularization)
        grads = tape.gradient(activation, img)
        grads = tf.math.l2_normalize(grads)
        img = img + optimization_parameters.learning_rate * grads
        print(f"image after applying gradient ascent is {img}")
    else:
        #optimizer.minimize(compute_loss, img)
        #activation = 1
        activation = optimization_parameters.optimizer.minimize(compute_loss, variable_name=img)
        #optimization_parameters.optimizer.apply_gradients(zip([grads], [img]))

        print(f"image and activation after applying adam is {img} and {activation}")
    return activation, img


def get_feature_extractor(model, layer_name):
    """Builds a model that that returns the activation of the specified layer.

    Args:
        model (object): the model used as a basis for the feature extractor.
        layer (string): the layer at which to cap the original model.
    """
    layer = model.get_layer(name=layer_name)
    return keras.Model(inputs=model.inputs, outputs=layer.output)


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
    plt.savefig("test.png")
