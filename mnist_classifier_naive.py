import os
dirname = os.path.dirname(__file__)
from luna.featurevis import featurevis, images, image_reader
import matplotlib.pyplot as plt

import tensorflow as tf
from luna.featurevis.transformations import *
from luna.featurevis.regularizers import *
from tensorflow import keras
import json

model_path = r"C:\Users\lucaz\OneDrive\Fuzhi\Uni Ulm\luna\mnist_classifier\mnist_classifier"

output_path = r'C:\Users\lucaz\Documents\Fuzhi\GitHub\featurevis_experimentation\output_tests\mnist_luna'
image_size = 28

model_name = "mnist"#"resnet50v2"#"vgg16" #inceptionV3"#"vgg19" #"inceptionV3"
#layer_name = "conv2d" #"conv2d_4" #"mixed5"#"conv2_block1_1_conv" #"block3_conv1"#"mixed5" #"block3_conv4"#"mixed5"
#channel_num = 5 #5 #5 #40 #30

danielNet_layers = ["conv2d", "conv2d_1"]
danielNet_channels = [32,64]
# Optimizer
optimizer = tf.keras.optimizers.Adam(epsilon=1e-08)
# optimizer params
iterations = 512
learning_rate = 0.05
opt_param = featurevis.OptimizationParameters(iterations, learning_rate, optimizer=optimizer)
model = keras.models.load_model(model_path)
print(model.summary())

for layer_name, num_of_channel in zip(danielNet_layers, danielNet_channels):
    for channel_num in range(num_of_channel):
        image = images.initialize_image_ref(image_size,image_size, fft=False, decorrelate=False)
        #print(model.summary())
        image= featurevis.visualize_filter(image, model, layer_name, channel_num,
                                        opt_param, transformation=None, threshold= None)

        img_name = f"{model_name}_{layer_name}_{channel_num}"
        images.save_image(image, name=img_name)
        image_reader.save_npy_as_png(f"raw_image_mnist/{img_name}.npy", output_path, GREY=True)



