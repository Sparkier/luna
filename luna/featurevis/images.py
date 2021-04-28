"""
The utility functions for creating and processing the images
for the feature visualisation process
"""
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras


def initialize_image(width, height, val_range_top=1.0, val_range_bottom=-1.0):
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
    val_range = val_range_top - val_range_bottom
    # rescale values to be in the given range of values
    img = val_range_bottom + img * val_range
    return img


def deprocess_image(img):
    """
    Takes the values of an image array and normalizes them to be in the
    standard 0-255 RGB range
    :param img: The generated image array
    :return: A rescaled version of the image
    """
    print('Deprocessing image')
    # compute the normal scores (z scores) and add little noise for uncertainty
    img = ((img - img.mean()) / img.std()) + 1e-5
    # ensure that the variance is 0.15
    img *= 0.15
    #croping the center adn clip the values between 0 and 1
    img = img[25:-25, 25:-25, :]
    img += 0.5
    img = np.clip(img, 0, 1)

    #convert to RGB
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")

    return img


def save_image(img, name=None):
    """
    Saves a generated image array as a numpy array in a file
    :param img: The generated image
    :param name: A possible name, if none given it is auto generated
    """
    arr = keras.preprocessing.image.img_to_array(img)
    if name is None:
        name = datetime.now().isoformat()
        name = name.replace("-", "")
        name = name.replace(":", "")
        name = name.replace("+", "")
        name = name.replace(".", "")
    np.save("{0}.npy".format(name), arr)
