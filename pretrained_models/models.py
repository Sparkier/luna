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
    elif model_name == "vgg16":
        return keras.applications.VGG16(weights="imagenet", include_top=False)
    elif model_name == "inceptionV3":
        return keras.applications.InceptionV3(weights="imagenet", include_top=False)
    elif model_name == "googlenet":
        tf.compat.v1.keras.backend.set_image_data_format('channels_first')
        return googlenet.create_googlenet()
    elif model_name == "inceptionV1slim":
        inputs = tf.random.uniform((1, 224, 224, 3), dtype=tf.dtypes.float32)
        return slim_nets.inception.inception_v1(inputs)
    else:
        return


# def set_layer(name):
#     """
#     Sets the layer to a specified name, it should checked if the layer is
#     actually present in the given model architecture
#
#     :param name: The name of the given layer
#     """
#     global_constants.layer_name = name
#     # Set up a model that returns the activation values for our target layer
#     global_constants.layer = global_constants.model.get_layer(
#         name=global_constants.LAYER_NAME)
#     global_constants.feature_extractor = keras.Model(
#         inputs=global_constants.model.inputs,
#         outputs=global_constants.layer.output)
