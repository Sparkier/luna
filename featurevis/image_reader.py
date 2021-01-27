"""
A utility file that takes all numpy arrays in the input folder and saves
them as pngs in the output folder
"""

from pathlib import Path
import numpy as np
from tensorflow import keras


def save_npy_as_png(input_path, output_path = "lune/outputs"):
    """Saves a numpy image as a png given an image path.

    Args:
        path (string): the path to the numpy file containing the image data
    """
    filepath = Path(input_path)
    filename = filepath.stem
    out_path = Path("{}/{}.png".format(output_path, filename))
    arr = np.load(filepath)
    np.moveaxis(arr, 0, -1)
    img = keras.preprocessing.image.array_to_img(arr)
    keras.preprocessing.image.save_img(out_path, img)
