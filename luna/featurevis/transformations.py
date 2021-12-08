"""
The transformations for feature vis process
"""
import random
import tensorflow as tf
import tensorflow_addons as tfa


def noise(img):
    """Adds noise to the image to be manipulated.

    Args:
        img (list): the image data to which noise should be added.
    Returns:
        list: the modified image.
    """
    if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
        img_noise = tf.random.uniform(
            (1, 3, img.shape[2], len(img[3])), dtype=tf.dtypes.float32
        )
    else:
        img_noise = tf.random.uniform((img.shape), dtype=tf.dtypes.float32)
    img_noise = (img_noise - 0.5) * 0.25 * random.random()
    img = img + img_noise
    img = tf.clip_by_value(img, -1, 1)
    return img


def pad(img, pad_size, pad_mode="REFLECT", constant_value=0.5):
    """Pad the image

    Args:
        img (list): the image data to be padded.
        pad_size (int): size of pad.
        mode (str, optional): mode for padding; can be "CONSTANT", "REFLECT" or "SYMMETRIC".
        constant_value (float, optional): The scalar pad value. Defaults to 0.5.

    Returns:
        list: the padded image.
    """
    img = tf.pad(
        img,
        [(0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)],
        pad_mode,
        constant_value,
    )
    return img


def jitter(img, jitter_dist, seed=None):
    """Jitter the image & add an artistic effect to the image.

    Args:
        img (list): the image data to be jittered.
        jitter_dist (int): the distance of effective neighborhoods.
        seed (int, optional): Defaults to None.

    Returns:
        list: the jittered image.
    """
    img = tf.convert_to_tensor(img)
    img_shape = tf.shape(img)
    crop_shape = tf.concat(
        [img_shape[:-3], img_shape[-3:-1] - jitter_dist, img_shape[-1:]], 0
    )
    crop = tf.image.random_crop(img, crop_shape, seed=seed)
    img_shape_list = img.get_shape().as_list()
    mid_shp_changed = [
        img_shape_list[-3] -
        jitter_dist if img_shape_list[-3] is not None else None,
        img_shape_list[-2] -
        jitter_dist if img_shape_list[-3] is not None else None,
    ]
    crop.set_shape(img_shape_list[:-3] + mid_shp_changed + img_shape_list[-1:])

    return crop


def blur(img):
    """Gaussian blur the image to be modified.

    Args:
        img (list): the image to be blurred.
    Returns:
        list: the blurred image.
    """
    img = gaussian_blur(img, sigma=0.001 + random.random())
    img = tf.clip_by_value(img, -1, 1)
    return img


def random_select(rand_values, seed=None):
    """Generate the upper bound on the range of random value.

    Args:
        rand_scale (list): random selection for upper bound
        seed (int, optional): for reproducibility. Defaults to None.

    Returns:
        list: list of random values
    """
    scale_list = list(rand_values)
    rand_n = tf.random.uniform((), 0, len(scale_list), "int32", seed)
    return tf.constant(scale_list)[rand_n]


def scale_values(img):
    """Rescales the image.

    Args:
       img (list): the current state of the feature vis image.

    Returns:
        list: the scaled image.
    """
    # factors are selected based on the optimized factors reported by lucid
    # https://distill.pub/2017/feature-visualization/
    scale_factor = [1, 0.975, 1.025, 0.95, 1.05]
    factor = random.choice(scale_factor)
    img = img * factor
    return img


def bilinear_rescale(img, *args, seed=None):
    """rescaling image by bilinear interpolation

    Args:
        img (list): the current state of the feature vis image.
        *args (tuple): list of arguments i.e. rescale values

    Returns:
        list: the rescaled image.
    """
    rescale_val = args[0]

    if not isinstance(args[0], list):
        rescale_val = list(args)

    if isinstance(rescale_val, int):
        rescale_val = list(rescale_val)

    img = tf.convert_to_tensor(img)
    scale = random_select(rescale_val, seed=seed)
    img_shape = tf.shape(img)
    scale_shape = tf.cast(
        scale * tf.cast(img_shape[-3:-1], "float32"), "int32")
    img = tf.compat.v1.image.resize_bilinear(img, scale_shape)
    return img


def crop_or_pad(img):
    """Randomly crop or pad the image.

    Args:
        image (list): Image.

    Returns:
        list: the cropped/padded image.
    """
    trans_size = random.randint(-6, 6)
    img = tf.image.resize_with_crop_or_pad(
        img, img.shape[1] - trans_size, img.shape[2] - trans_size
    )
    return img


def ninety_degree_rotation(img):
    """Rotating the image by ninety degree.

    Args:
        img (list): image.

    Returns:
        img (list): rotated image.
    """
    img = tf.image.rot90(img)
    return img


def angle_conversion(angle, units):
    """Converting the angle to the desirable format

    Args:
        angle (list): list of random angle values
        units (str): conversion unit

    Returns:
        list: converted angle values to the desirable form
    """
    angle_f = tf.cast(angle, "float32")
    if units.lower() == "degrees":
        angle = 3.14 * angle_f / 180.0
    elif units.lower() in ["radians", "rads", "rad"]:
        angle = angle_f
    return angle


def rotation(img, *args, units="degrees", seed=None):
    """Rotating the image with random angle

    Args:
        img (list): the image data
        *args (tuple): list of arguments i.e. angles of rotatation
        units (str, optional): the unit of rotation move. Defaults to "degrees".
        seed ([type], optional): for reproducibility. Defaults to None.

    Returns:
        list: rotated image with selected values
    """
    angles = args[0]

    if not isinstance(args[0], list):
        angles = list(args)

    if isinstance(angles, int):
        angles = list(angles)

    img = tf.convert_to_tensor(img)
    angle = random_select(angles, seed=seed)
    angle = angle_conversion(angle, units)
    return tfa.image.rotate(img, angle)


def flip(img):
    """Fliping the image up, down, left, right.

    Args:
        img (list): image.

    Returns:
        list: the flipped image.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


# https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319
def gaussian_blur(img, kernel_size=3, sigma=5):
    """
    Calculates the gaussian convolution kernel for the blurring process.

    Args:
        img: the current state of the image.
        kernel_size: size of the convolution kernel used for blurring.
        sigma: gaussian blurring constant.

    Returns:
        The altered image.
    """

    axis = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xvals, yvals = tf.meshgrid(axis, axis)
    kernel = tf.exp(-(xvals ** 2 + yvals ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, tf.shape(img)[-1]])
    gaussian_kernel = kernel[..., tf.newaxis]

    return tf.nn.depthwise_conv2d(
        img, gaussian_kernel, [1, 1, 1, 1], padding="SAME", data_format="NHWC"
    )


def color_augmentation(img):
    """Augmenting the color.

    Args:
        image (list): Image.

    Returns:
        list: Augmented image.
    """
    img = tf.image.random_saturation(img, 0, 0.5)
    img = tf.image.random_contrast(img, 0, 0.8)
    img = tf.image.random_brightness(img, 0.01)
    return img


def standard_transformation(img):
    """Standard transformations if no transformations were chosen by user (Suggested by Lucid)

    Args:
        img (list): image.

    Returns:
        list: transformed imaged with standard modes.
    """
    img = pad(img, 12, pad_mode="constant")
    img = jitter(img, 8)
    img = bilinear_rescale(
        img, [1 + (i - 5) / 50.0 for i in range(11)], seed=None)
    img = rotation(img, list(range(-10, 11)) + 5 * [0])
    img = jitter(img, 4)
    return img
