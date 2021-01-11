"""
This file is for setting up the models used for the feature vis
process, the objects for the global model and layer values are set here
"""
import tensorflow as tf
from tensorflow import keras
from tf_slim import nets as slim_nets

from luna.featurevis import global_constants
from luna.pretrained_models import googlenet


def set_model(model_info):
    """
    Sets the model to a specified architecture

    :param model_info: The collective information for the model
    """
    tf.compat.v1.keras.backend.set_image_data_format('channels_last')
    if model_info["name"] == global_constants.RESNET50V2["name"]:
        global_constants.model = keras.applications.ResNet50V2(
            weights="imagenet", include_top=False)
    elif model_info["name"] == global_constants.VGG16["name"]:
        global_constants.model = keras.applications.VGG16(
            weights="imagenet", include_top=False)
    elif model_info["name"] == global_constants.INCEPTIONV3["name"]:
        global_constants.model = keras.applications.InceptionV3(
            weights="imagenet", include_top=False)
    elif model_info["name"] == global_constants.INCEPTIONV1["name"]:
        tf.compat.v1.keras.backend.set_image_data_format('channels_first')
        global_constants.model = googlenet.create_googlenet()
    elif model_info["name"] == global_constants.INCEPTIONV1SLIM["name"]:
        inputs = tf.random.uniform(
            (1, global_constants.IMG_HEIGHT, global_constants.IMG_WIDTH, 3),
            dtype=tf.dtypes.float32)
        global_constants.model = slim_nets.inception.inception_v1(inputs)
    else:
        return
    global_constants.model_info = model_info
    print(model_info["name"], global_constants.model)


def set_layer(name):
    """
    Sets the layer to a specified name, it should checked if the layer is
    actually present in the given model architecture

    :param name: The name of the given layer
    """
    global_constants.layer_name = name
    # Set up a model that returns the activation values for our target layer
    global_constants.layer = global_constants.model.get_layer(
        name=global_constants.LAYER_NAME)
    global_constants.feature_extractor = keras.Model(
        inputs=global_constants.model.inputs,
        outputs=global_constants.layer.output)
