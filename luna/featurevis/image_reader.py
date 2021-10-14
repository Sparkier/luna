"""
A utility file that takes all numpy arrays in the input folder and saves
them as pngs in the output folder
"""

from pathlib import Path
import numpy as np
from tensorflow import keras
import tensorflow as tf


def save_npy_as_png(input_path, output_path):
    """Saves a numpy image as a png given an image path.

    Args:
        path (string): the path to the numpy file containing the image data.
    """
    filepath = Path(input_path)
    filename = filepath.stem
    out_path = Path(f"{output_path}/{filename}.png")
    arr = np.load(filepath)
    np.moveaxis(arr, 0, -1)
    if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
        arr = tf.transpose(arr, [0, 2, 1])
    img = keras.preprocessing.image.array_to_img(
        arr, data_format="channels_last")
    keras.preprocessing.image.save_img(
        out_path, img, data_format="channels_last")
