"""
The transformations for feature vis process
"""
import random
import tensorflow as tf


def add_noise(img, noise):
    """Adds noise to the image to be manipulated.

    Args:
        img (list): the image data to which noise should be added.
        noise (boolean): whether noise should be added at all.
    Returns:
        list: the modified image.
    """
    if noise:
        if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
            img_noise = tf.random.uniform((1, 3, img.shape[2], len(img[3])),
                                          dtype=tf.dtypes.float32)
        else:
            img_noise = tf.random.uniform((img.shape),
                                          dtype=tf.dtypes.float32)
        img_noise = (img_noise - 0.5) * 0.25 * random.random()
        img = img + img_noise
        img = tf.clip_by_value(img, -1, 1)
    return img


def blur_image(img, blur):
    """Gaussian blur the image to be modified.

    Args:
        img (list): the image to be blurred.
        blur (boolean): whether to blur the image.
    Returns:
        list: the blurred image.
    """
    if blur:
        img = gaussian_blur(img, sigma=0.001 + random.random())
        img = tf.clip_by_value(img, -1, 1)
    return img


def rescale_image(img, scale):
    """Rescales the image.

    Args:
       img (list) the current state of the feature vis image.
       scale (bool) true, if image should be randomly scaled.
    Returns:
        list: the rescaled image.
    """
    if scale:
        # factors are selected based on the optimized factors reported by lucid
        # https://distill.pub/2017/feature-visualization/
        scale_factor = [1, 0.975, 1.025, 0.95, 1.05]
        factor = random.choice(scale_factor)
        img = img * factor
    return img


def crop_or_pad(image, trans):
    """Randomly crop or pad the image.

    Args:
        image (list): Image.
        trans (bool): True if image needs to crop or pad.

    Returns:
        list: the cropped/padded image.
    """
    if trans:
        trans_size = random.randint(-6, 6)
        image = tf.image.resize_with_crop_or_pad(
            image, image.shape[1]-trans_size, image.shape[2]-trans_size)
    return image


def vert_rotation(img, rotation):
    """Randomly rotates the image.

    Args:
        img (list): image.
        rotation (bool): True if image needs to rotate.

    Returns:
        img (list): rotated image.
    """
    if rotation:
        img = tf.image.rot90(img)
    return img


def random_flip(img, flip):
    """Fliping the image up, down, left, right.

    Args:
        img (list): image.
        flip (bool): True if image needs to flip.
    """
    if flip:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
    return img


# https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319
def gaussian_blur(img, kernel_size=3, sigma=5):
    """
    Helper function for blurring the image, will soon be replaced by tfa.image.gaussian_filter2d.

    Args:
        img: the current state of the image.
        kernel_size: size of the convolution kernel used for blurring.
        sigma: gaussian blurring constant.

    Returns:
        The altered image.
    """
    def gauss_kernel(channels, kernel_size, sigma):
        """
        Calculates the gaussian convolution kernel for the blurring process.

        Args:
            channels: amount of feature channels.
            kernel_size: size of the kernel.
            sigma: gaussian blurring constant.

        Returns:
            the kernel for the given values.
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


def color_augmentation(image, color_aug):
    """Augmenting the color.

    Args:
        image (list): Image.
        color_aug (bool): True if image needs to augment.

    Returns:
        img (list): Augmented image.
    """
    if color_aug:
        image = tf.image.random_saturation(image, 0, 0.5)
        image = tf.image.random_contrast(image, 0, 0.8)
        image = tf.image.random_brightness(image, 0.01)
    return image
