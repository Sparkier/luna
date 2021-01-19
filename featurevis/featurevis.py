"""
The main file for the feature vis process
"""
import tensorflow as tf
from tensorflow import keras

from luna.featurevis import images as imgs


def add_noise(img, noise, pctg, channels_first=False):
    """Adds noise to the image to be manipulated.

    Args:
        img (list): the image data to which noise should be added
        noise (boolean): whether noise should be added at all
        pctg (number): how much noise should be added to the image
        channels_first (bool, optional): whether the image is encoded channels
        first. Defaults to False.

    Returns:
        list: the modified image
    """
    if noise:
        if channels_first:
            img_noise = tf.random.uniform((1, 3, len(img[2]), len(img[3])),
                                          dtype=tf.dtypes.float32)
        else:
            img_noise = tf.random.uniform((1, len(img[1]), len(img[2]), 3),
                                          dtype=tf.dtypes.float32)
        img_noise = (img_noise - 0.5) * 0.25 * ((100 - pctg) / 100)
        img = img + img_noise
        img = tf.clip_by_value(img, -1, 1)
    return img


def blur_image(img, blur, pctg):
    """Gaussian blur the image to be modified.

    Args:
        img (list): the image to be blurred
        blur (boolean): whether to blur the image
        pctg (number): how much blur should be applied

    Returns:
        list: the blurred image
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


def visualize_filter(image, model, layer, filter_index, iterations,
                     learning_rate, noise, blur, scale, channels_first=False):
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
    feature_extractor = get_feature_extractor(model, layer)
    print('Starting Feature Vis Process')
    for iteration in range(iterations):
        pctg = int(iteration / iterations * 100)
        image = add_noise(image, noise, pctg, channels_first)
        image = blur_image(image, blur, pctg)
        image = rescale_image(image, scale, pctg)
        loss, image = gradient_ascent_step(
            image, feature_extractor, filter_index, learning_rate)
        print('>>', pctg, '%', end="\r", flush=True)

    print('>> 100 %')
    # Decode the resulting input image
    image = imgs.deprocess_image(image[0].numpy())
    return loss, image


def compute_loss(input_image, model, filter_index, channels_first=False):
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
    activation = model(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    if channels_first:
        filter_activation = activation[:, filter_index, 2:-2, 2:-2]
    else:
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, model, filter_index, learning_rate, channels_first=False):
    """Performing one step of gradient ascend.

    Args:
        img (array): the image to be changed by the gradiend ascend
        model (object): the model with which to perform the image change
        filter_index (number): which filter to optimize for
        learning_rate (number): how much to change the image per iteration
        channels_first (bool, optional): Whether the image is channels first.
        Defaults to False.

    Returns:
        tuple: the loss and the modified image
    """
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, model, filter_index, channels_first)
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
