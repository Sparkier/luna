"""
The utility functions for creating and processing the images
for the feature visualisation process
"""
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from luna.featurevis import global_constants


def initialize_image(width, height):
    """
    Creates an initial randomized image to start feature vis process.
    This could be subject to optimization in the future.

    :param width: The width of the image
    :param height: The height of the image

    :return: A randomly generated image
    """
    print('initializing image')
    # We start from a gray image with some random noise
    if tf.compat.v1.keras.backend.image_data_format() == 'channels_first':
        img = tf.random.uniform((1, 3, width, height), dtype=tf.dtypes.float32)
    else:
        img = tf.random.uniform((1, width, height, 3), dtype=tf.dtypes.float32)
    # rescale values to be in the middle quarter of possible values
    img = (img - 0.5) * 0.25 + 0.5
    val_range = global_constants.model_info["range_top"] - \
        global_constants.model_info["range_bot"]
    # rescale values to be in the given range of values
    img = global_constants.model_info["range_bot"] + img * val_range
    return img


def deprocess_image(img):
    """
    Takes the values of an image array and normalizes them to be in the
    standard 0-255 RGB range

    :param img: The generated image array

    :return: A rescaled version of the image
    """
    print('Deprocessing image')
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    if global_constants.model_info["name"] == global_constants.INCEPTIONV1["name"]:
        img = img[:, 25:-25, 25:-25]
    else:
        img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def save_image(img, name=None):
    """
    Saves a generated image array as a numpy array in a file

    :param img: The generated image
    :param name: A possible name, if none given it is auto generated
    """
    if global_constants.model_info["name"] == global_constants.INCEPTIONV1["name"]:
        img = tf.transpose(img, [2, 0, 1])
    arr = keras.preprocessing.image.img_to_array(img)
    name = datetime.now().isoformat().replace("-", "").replace(":", "")\
        .replace("+", "").replace(".", "") if name is None else name
    np.save("test/input/{0}.npy".format(name), arr)
