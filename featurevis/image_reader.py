"""
A utility file that takes all numpy arrays in the input folder and saves
them as pngs in the output folder
"""
import re
from os import listdir
import numpy as np
from tensorflow import keras


INPUT_DIR = "../test/input/"
OUTPUT_DIR = "../test/output/"
images = []


def main():
    """The main function of this file, lists all files in the given directories"""
    for file in listdir(OUTPUT_DIR):
        images.append(file)
    for file in listdir(INPUT_DIR):
        save_npy_as_image(file)


def save_npy_as_image(name):
    """
    This function takes a given file (name), reads its contents and saves
    them as an image file in the output directory

    :param name: The filename of the current file
    """
    subname = re.search('.*?(\\w+)\\.npy', name)
    if subname is None:
        return
    img_name = "../test/output/{0}.png".format(subname.group(1))
    if img_name in images:
        return
    arr = np.load(INPUT_DIR + name)
    np.moveaxis(arr, 0, -1)
    img = keras.preprocessing.image.array_to_img(arr)
    keras.preprocessing.image.save_img(img_name, img)


if __name__ == '__main__':
    main()
