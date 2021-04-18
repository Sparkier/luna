"""
The main file for the feature vis process
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras

from luna.featurevis import relu_grad as rg
from luna.featurevis import images as imgs
from luna.featurevis import transformations as trans

#pylint: disable=too-many-locals
#pylint: disable=too-many-arguments
def visualize_filter(image, model, layer, filter_index, iterations,
                     learning_rate, noise, blur, scale, pad_crop, rotation, flip, color_aug):
    """Create a feature visualization for a filter in a layer of the model.

    Args:
        image (array): the image to be modified by the feature vis process
        model (object): the model to be used for the feature visualization
        layer (string): the name of the layer to be used in the visualization
        filter_index (number): the index of the filter to be visualized
        iterations (number): hoe many iterations to optimize for
        learning_rate (number): update amount after each iteration
        noise (number): how much noise to add to the image
        blur (number): how much blur to add to the image
        scale (number): how much to scale the image

    Returns:
        tuple: loss and result image for the process
    """
    image = tf.Variable(image)
    feature_extractor = get_feature_extractor(model, layer)

    # Temporary method for random choice of
    print('Starting Feature Vis Process')
    for iteration in range(iterations):
        pctg = int(iteration / iterations * 100)
        image = trans.crop_or_pad(image, pad_crop)
        image = trans.add_noise(image, noise)
        image = trans.rescale_image(image, scale)
        image = trans.blur_image(image, blur)
        image = trans.random_flip(image, flip)
        image = trans.vert_rotation(image, rotation)
        image = trans.color_augmentation(image, color_aug)
        loss, image = gradient_ascent_step(
            image, feature_extractor, filter_index, learning_rate)

        print('>>', pctg, '%', end="\r", flush=True)
    print('>> 100 %')
    # Decode the resulting input image
    image = imgs.deprocess_image(image[0].numpy())

    return loss, image


def compute_loss(input_image, model, filter_index):
    """Computes the loss for the feature visualization process.

    Args:
        input_image (array): the image that is used to compute the loss
        model (object): the model on which to compute the loss
        filter_index (number): for which filter to compute the loss
        channels_first (bool, optional): Whether the image is channels first.
        Defaults to False.

    Returns:
        number: the loss for the specified setting
    """
    with rg.gradient_override_map(
        {'Relu': rg.redirected_relu_grad,'Relu6': rg.redirected_relu6_grad}):
        activation = model(input_image)
    if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
        filter_activation = activation[:, filter_index, :, :]
    else:
        filter_activation = activation[:, :, :, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function(experimental_relax_shapes=True)
def gradient_ascent_step(img, model, filter_index, learning_rate):
    """Performing one step of gradient ascend.

    Args:
        img (array): the image to be changed by the gradiend ascend
        model (object): the model with which to perform the image change
        filter_index (number): which filter to optimize for
        learning_rate (number): how much to change the image per iteration
        Defaults to False.

    Returns:
        tuple: the loss and the modified image
    """
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, model, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def get_feature_extractor(model, layer_name):
    """Builds a model that that returns the activation of the specified layer.

    Args:
        model (object): the model used as a basis for the feature extractor
        layer (string): the layer at which to cap the original model
    """
    layer = model.get_layer(name=layer_name)
    return keras.Model(inputs=model.inputs, outputs=layer.output)
