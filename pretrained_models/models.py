"""
This file is for setting up the models used for the feature vis
process, the objects for the global model and layer values are set here
"""
import tensorflow as tf
from tensorflow import keras
from tf_slim import nets as slim_nets

from luna.pretrained_models import googlenet


def get_model(model_name):
    """
    Sets the model to a specified architecture

    :param model_info: The collective information for the model
    """
    tf.compat.v1.keras.backend.set_image_data_format('channels_last')
    if model_name == "resnet50v2":
        return keras.applications.ResNet50V2(weights="imagenet", include_top=False)
    if model_name == "vgg16":
        return keras.applications.VGG16(weights="imagenet", include_top=False)
    if model_name == "inceptionV3":
        return keras.applications.InceptionV3(weights="imagenet", include_top=False)
    if model_name == "googlenet":
        tf.compat.v1.keras.backend.set_image_data_format('channels_first')
        return googlenet.create_googlenet()
    if model_name == "inceptionV1slim":
        inputs = tf.random.uniform((1, 224, 224, 3), dtype=tf.dtypes.float32)
        return slim_nets.inception.inception_v1(inputs)
