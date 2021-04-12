"""
The transformations for feature vis process
"""
import random
import tensorflow as tf
import tensorflow_addons as tfa



def add_noise(img, noise, pctg):
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
        if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
            img_noise = tf.random.uniform((img.shape),
                                          dtype=tf.dtypes.float32)
        else:
            img_noise = tf.random.uniform((img.shape),
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


def rescale_image(img, scale):
    """
    Will rescale the current state of the image
    :param img: the current state of the feature vis image
    :param scale: true, if image should be randomly scaled
    :param pctg: the amount of scaling in percentage
    :return: the altered image
    """
    if scale:
        scale_factor = [1, 0.975, 1.025, 0.95, 1.05]
        factor = random.choice(scale_factor)
        img = img * factor
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

def jitter(image, jitter_size, seed=None):
    """Jitters the image

    Args:
        image (array): the image
        d (int): shape of cropping
        seed (int, optional): Seed. Defaults to None.

    Returns:
        array: Jittered image
    """
    image = tf.convert_to_tensor(value=image, dtype_hint=tf.float32)
    t_shp = tf.shape(input=image)
    crop_shape = tf.concat([t_shp[:-3], t_shp[-3:-1] - jitter_size, t_shp[-1:]], 0)
    crop = tf.image.random_crop(image, crop_shape, seed=seed)
    shp = image.get_shape().as_list()
    mid_shp_changed = [
        shp[-3] - jitter_size if shp[-3] is not None else None,
        shp[-2] - jitter_size if shp[-3] is not None else None,
    ]
    crop.set_shape(shp[:-3] + mid_shp_changed + shp[-1:])
    return crop

#pylint: disable=E1123
def pad(img, pad_size, pad_mode="REFLECT", constant_value=0.5):
    """Pads the image

    Args:
        img (list): the image
        w (int): dimension of padding
        mode (str, optional): Defaults to "REFLECT".
        constant_value (float, optional): Defaults to 0.5.

    Returns:
        [type]: [description]
    """
    if constant_value == "uniform":
        constant_value_ = tf.random.uniform([], 0, 1)
    else:
        constant_value_ = constant_value
    return tf.pad(img, paddings=[(0, 0), (pad_size, pad_size), (pad_size, pad_size),
                                (0, 0)], mode=pad_mode, constant_values=constant_value_)

def random_rotate(img, rotation):
    """Randomly rotates the image

    Args:
        img (list): image
        rotation (int): degree of rotation

    Returns:
        img (list): rotated image
    """
    return tfa.image.rotate(img, rotation)
