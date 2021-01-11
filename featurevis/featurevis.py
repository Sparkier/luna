"""
The main file for the feature vis process
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

from luna.featurevis import images as imgs
from luna.featurevis import global_constants


def add_noise(img, noise, pctg):
    """
    if noise is true will add random noise values to the current image

    :param img: the current state of the feature vis image
    :param noise: true, if noise should be added
    :param pctg: the amount of noise in percentage
    :return: the altered image
    """
    if noise:
        if global_constants.model_info["name"] == global_constants.INCEPTIONV1["name"]:
            img_noise = tf.random.uniform(
                (1, 3, global_constants.IMG_WIDTH, global_constants.IMG_HEIGHT),
                dtype=tf.dtypes.float32)
        else:
            img_noise = tf.random.uniform(
                (1, global_constants.IMG_WIDTH, global_constants.IMG_HEIGHT, 3),
                dtype=tf.dtypes.float32)
        img_noise = (img_noise - 0.5) * 0.25 * ((100 - pctg) / 100)
        img = img + img_noise
        img = tf.clip_by_value(img, -1, 1)
    return img


def blur_image(img, blur, pctg):
    """
    if blur is true will gaussian blur the current image

    :param img: the current state of the feature vis image
    :param blur: true, if blur should be added
    :param pctg: the amount of blur in percentage
    :return: the altered image
    """
    if blur:
        img = gaussian_blur(img, sigma=0.001 + ((100-pctg) / 100) * 1)
        img = tf.clip_by_value(img, -1, 1)
    return img


def rescale_image(img, scale, pctg):
    """
    Will rescale the current state of the image

    :param img: the current state of the feature vis image
    :param scale: true, if image should be randomly scaled
    :param pctg: the amount of scaling in percentage
    :return: the altered image
    """
    if scale:
        scale_factor = tf.random.normal([1], 1, pctg)
        img *= scale_factor[0]  # not working
    return img


# https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319
def gaussian_blur(img, kernel_size=3, sigma=5):
    """
    helper function for blurring the image, will soon be replaced by
    tfa.image.gaussian_filter2d

    :param img: the current state of the image
    :param kernel_size: size of the convolution kernel used for blurring
    :param sigma: gaussian blurring constant
    :return: the altered image
    """
    def gauss_kernel(channels, kernel_size, sigma):
        """
        Calculates the gaussian convolution kernel for the blurring process

        :param channels: amount of feature channels
        :param kernel_size: size of the kernel
        :param sigma: gaussian blurring constant
        :return: the kernel for the given values
        """
        axis = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xvals, yvals = tf.meshgrid(axis, axis)
        kernel = tf.exp(-(xvals ** 2 + yvals ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')


def visualize_filter(filter_index, noise, blur, scale):
    """
    The outer function that sets up the remaining objects and calls the
    feature vis methods

    :param filter_index: the index of the filter (channel)
      that should be visualized
    :param noise: true, if noise should be used in the process
    :param blur: true, if the image should be blurred to reduce high frequencies
    :param scale: true, if rescaling should be applied each step

    :return: The loss matrix of the training process and
      the visualisation of the filter
    """
    # We run gradient ascent for 20 steps
    currpctg = 0
    global_constants.MAIN_IMG = imgs.initialize_image(
        global_constants.IMG_WIDTH, global_constants.IMG_HEIGHT)
    img = global_constants.MAIN_IMG

    print('Starting Feature Vis Process')
    for iteration in range(global_constants.ITERATIONS):
        pctg = int(iteration / global_constants.ITERATIONS * 100)
        img = add_noise(img, noise, pctg)
        img = blur_image(img, blur, pctg)
        img = rescale_image(img, scale, pctg)
        loss, img = gradient_ascent_step(
            img, filter_index, global_constants.LEARNING_RATE)
        if pctg >= currpctg:
            print('>>', currpctg, '%')
            currpctg += 10
    print('>> 100 %')
    # print(img)
    # Decode the resulting input image
    img = imgs.deprocess_image(img[0].numpy())
    return loss, img


def compute_loss(input_image, filter_index):
    """
    Calculates the Loss when activating the filter with the given image

    :param input_image: The current state of the created feature image
    :param filter_index: The index of the filter that should be visualised
    :return: The mean activation of the filter within he given layer
    """
    activation = global_constants.feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    if global_constants.model_info["name"] == global_constants.INCEPTIONV1["name"]:
        filter_activation = activation[:, filter_index, 2:-2, 2:-2]
    else:
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    """
    The feature extraction through gradient ascent. The image is sent through
    the given filter, a loss is calculated and the image is adjusted to the
    outcome

    :param img: The current state of the created feature image
    :param filter_index: the index of the filter that should be visualised
    :param learning_rate: The rate of how much the image should be adjusted in
      each training step
    """
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def set_learning_rate(rate):
    """Sets a new learning rate"""
    global_constants.learning_rate = rate


def set_iterations(num):
    """Sets a new value for the amount of iterations"""
    global_constants.iterations = num


def set_noise(use_noise):
    """Sets a new value for the amount of iterations"""
    global_constants.noise = use_noise


def set_blur(use_blur):
    """Sets a new value for the amount of iterations"""
    global_constants.blur = use_blur


# vgl https://github.com/gabrielpierobon/cnnshapes/blob/master/README.md
def show_activations(img, images_per_row=10):
    """
    A currently unused function for visualising the activations
    of diefferent layers in the network when forward passing a generated
    feature image into the net

    :param img: The generated feature visualisation
    :param images_per_row: Changes the structure of the visualised plot
    """
    layer_outputs = [
        layer.output for layer in global_constants.model.layers[:]]
    # Extracts the outputs of the top 12 layers
    activation_model = keras.models.Model(inputs=global_constants.model.input,
                                          outputs=layer_outputs)
    # Creates a model that will return these outputs, given the model input
    layer_names = []
    for layer in global_constants.model.layers[:]:
        layer_names.append(layer.name)

    for layer_name, layer_activation \
            in zip(layer_names, activation_model.predict(img)):
        if not layer_name.startswith("mixed"):
            print(layer_name, " has been skipped")
            continue
        size = layer_activation.shape[1]
        n_cols = layer_activation.shape[-1] // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                             row * size: (row + 1) * size] = channel_image
        plt.figure(figsize=(1. / size * display_grid.shape[1],
                            1. / size * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='inferno')
    plt.show()
