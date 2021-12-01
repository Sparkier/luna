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


class OptimizationParameters:
    """object for generalizing optimization parameters.

    Args:
        iterations (number): how many iterations to optimize for.
        learning_rate (number): update amount after each iteration.
        optimizer (class): optimizer (only Adam is supported)
    """

    def __init__(self, iterations, learning_rate, optimizer) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        if isinstance(optimizer, tf.keras.optimizers.Adam):
            self.optimizer = optimizer
            self.optimizer.learning_rate.assign(self.learning_rate)
        elif optimizer is None:
            self.optimizer = None
        else:
            raise ValueError(
                "The given optimizer is not supported. Needs to be one of None"
                + "(for traditional gradient ascent) or tf.keras.optimizers.Adam"
            )


class AuxiliaryTransformationParameters:
    """Object for generalizing auxiliary augmentation parameters.

    Args:
        add_blur (bool): whether or not gaussian blur is applied.
        scale (bool): whether or not scaling is applied.
        pad_crop (bool): whether or not random pad or crop are applied.
        add_flip(bool): whether or not flip is applied.
        add_rotation(bool): whether or not vert_rotation is applied.
        add_noise (bool): whether or not noise is applied.
        color_aug(bool): whether or not color augmentation is applied.
    """

    def __init__(
        self,
        add_blur=False,
        scale=False,
        pad_crop=False,
        add_flip=False,
        add_rotation=False,
        add_noise=False,
        color_aug=False,
    ) -> None:
        self.add_blur = add_blur
        self.scale = scale
        self.pad_crop = pad_crop
        self.add_flip = add_flip
        self.add_rotation = add_rotation
        self.add_noise = add_noise
        self.color_aug = color_aug


class TransformationParameters:
    """Object for generalizing augmentation parameters.

    Args:
        pad_size (float): the amount of padding to be applied.
        pad_mode (str): type of padding to be applied.
        add_jitter (float): the amount of jitter to be applied.
        bilinear (list): the amount of bilinear scaling to be applied.
        rotation (list): the amount of rotation to be applied.
    """

    def __init__(
        self,
        pad_size=None,
        pad_mode="REFLECT",
        add_jitter=None,
        bilinear=None,
        rotation=None,
    ) -> None:
        self.pad_size = pad_size
        self.pad_mode = pad_mode
        self.add_jitter = add_jitter
        self.rescale_val = bilinear
        self.angles = rotation


# pylint: disable=too-many-locals


def visualize_filter(
    image,
    model,
    layer,
    filter_index,
    opt_param,
    trans_param=None,
    aux_trans_param=None,
    custom_trans=None,
):
    """Create a feature visualization for a filter in a layer of the model.

    Args:
        image (array): the image to be modified by the feature vis process.
        model (object): the model to be used for the feature visualization.
        layer (string): the name of the layer to be used in the visualization.
        filter_index (number): the index of the filter to be visualized.


    Returns:
        tuple: activation and result image for the process.
    """
    image = tf.Variable(image)
    feature_extractor = get_feature_extractor(model, layer)

    # Temporary method for random choice of
    print("Starting Feature Vis Process")
    for iteration in range(opt_param.iterations):
        pctg = int(iteration / opt_param.iterations * 100)
        if not aux_trans_param and not trans_param and not custom_trans:
            image = trans.standard_transforms(image)
        elif custom_trans:
            image = trans.perform_custom_trans(image, custom_trans)
        else:
            if aux_trans_param:
                if isinstance(aux_trans_param, AuxiliaryTransformationParameters):
                    image = trans.perform_aux_trans(image, aux_trans_param)
                else:
                    raise TypeError(
                        "Wrong class was given. Expected a"
                        + "AuxiliaryTransformationParameters class"
                    )
                # TransParam
            if trans_param:
                if isinstance(aux_trans_param, TransformationParameters):
                    image = trans.perform_trans(image, trans_param)
                else:
                    raise TypeError(
                        "Wrong class was given. Expected a"
                        + "TransformationParameters class"
                    )
        activation, image = gradient_ascent_step(
            image, feature_extractor, filter_index, optimizer=opt_param
        )

        print(">>", pctg, "%", end="\r", flush=True)
    print(">> 100 %")
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
        {"Relu": rg.redirected_relu_grad, "Relu6": rg.redirected_relu6_grad}
    ):
        activation = model(input_image)
    if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
        filter_activation = activation[:, filter_index, :, :]
    else:
        filter_activation = activation[:, :, :, filter_index]
    return tf.reduce_mean(filter_activation)


# @tf.function()
def gradient_ascent_step(img, model, filter_index, optimizer=None):
    """Performing one step of gradient ascend.

    Args:
        img (array): the image to be changed by the gradiend ascend.
        model (object): the model with which to perform the image change.
        filter_index (number): which filter to optimize for.
        learning_rate (number): how much to change the image per iteration.

    Returns:
        tuple: the activation and the modified image
    """
    img = tf.Variable(img)

    with tf.GradientTape() as tape:
        tape.watch(img)
        activation = compute_activation(img, model, filter_index)
    # Compute gradients.
    grads = tape.gradient(activation, img)
    # Normalize gradients.
    if optimizer.optimizer is None:
        grads = tf.math.l2_normalize(grads)
        img = img + optimizer.learning_rate * grads
    else:
        optimizer.optimizer.apply_gradients(zip([grads], [img]))
    return activation, img


def get_feature_extractor(model, layer_name):
    """Builds a model that that returns the activation of the specified layer.

    Args:
        model (object): the model used as a basis for the feature extractor.
        layer (string): the layer at which to cap the original model.
    """
    layer = model.get_layer(name=layer_name)
    return keras.Model(inputs=model.inputs, outputs=layer.output)
