from luna.pretrained_models import models
from luna.featurevis import featurevis, images, image_reader
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from luna.featurevis.transformations import *
from luna.featurevis.regularizers import *
from tensorflow import keras
import numpy as np

dirname = os.path.dirname(__file__)
output_path = r'C:\Users\lucaz\Documents\Fuzhi\GitHub\featurevis_experimentation\output_tests\refactoring'
image_size = 32
model_name = "daniel"#"resnet50v2"#"vgg16" #inceptionV3"#"vgg19" #"inceptionV3"
layer_name = "conv2d_4" #"conv2d_4" #"mixed5"#"conv2_block1_1_conv" #"block3_conv1"#"mixed5" #"block3_conv4"#"mixed5"
channel_num = 15 #5 #5 #40 #30
model = keras.models.load_model(r"C:\Users\lucaz\Documents\Fuzhi\GitHub\luna\nws_main_00001")
model.summary()

tf.keras.backend.clear_session()

def change_model(model, new_input_shape=(None, 40, 40, 3),custom_objects=None):
    # replace input shape of first layer
    config = model.layers[0].get_config()
    print(config)
    config['batch_input_shape']=new_input_shape
    model.layers[0]=model.layers[0].from_config(config)

    # rebuild model architecture by exporting and importing via json
    new_model = tf.keras.models.model_from_json(model.to_json(),custom_objects=custom_objects)

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            print("Loaded layer {}".format(layer.name))
        except:
            print("Could not transfer weights for layer {}".format(layer.name))
    return new_model

new_model = change_model(model, new_input_shape=[None] + [299,299,3])

new_model.summary()