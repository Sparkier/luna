"""
This file is for setting up the models used for the feature vis
process, the objects for the global model and layer values are set here
"""
import tensorflow as tf
from tensorflow import keras
from tf_slim import nets as slim_nets

from luna.pretrained_models import googlenet


def model_resnet50v2():
    """
    Instantiates ResNet50V2 architecture using keras

    Returns:
        keras.application: ResNet50V2 Architecture
    """
    tf.compat.v1.keras.backend.set_image_data_format('channels_last')
    return keras.applications.ResNet50V2(weights="imagenet", include_top=False)


def model_inceptionv3():
    """
    Instantiates InceptionV3 architecture using keras

    Returns:
        keras.application: InceptionV3 Architecture
    """
    tf.compat.v1.keras.backend.set_image_data_format('channels_last')
    return keras.applications.InceptionV3(
        weights="imagenet", include_top=False)


def model_inceptionv1():
    """
    Instantiates InceptionV1 architecture using googlnet

    Returns:
        googlenet: InceptionV1 Architecture
    """
    tf.compat.v1.keras.backend.set_image_data_format('channels_first')
    return googlenet.create_googlenet()


def model_inceptionv1_slim():
    """
    Instantiates InceptionV1 architecture using tensorflow slim

    Returns:
        slim_net: InceptionV1 Architecture
    """
    inputs = tf.random.uniform((1, 224, 224, 3), dtype=tf.dtypes.float32)
    return slim_nets.inception.inception_v1(inputs)


def model_vgg16():
    """
    Instantiates vgg16 architecture using keras

    Returns:
        keras.application: vgg16 Architecture
    """
    tf.compat.v1.keras.backend.set_image_data_format('channels_last')
    return keras.applications.VGG16(weights="imagenet", include_top=False)


def model_vgg19():
    """
    Instantiates vgg19 architecture using keras

    Returns:
        keras.applications: vgg19 Architecture
    """
    tf.compat.v1.keras.backend.set_image_data_format('channels_last')
    return keras.applications.VGG19(weights="imagenet", include_top=False)
