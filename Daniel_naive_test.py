import os
dirname = os.path.dirname(__file__)
#from luna.pretrained_models import models

from luna.featurevis import featurevis, images, image_reader
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from luna.featurevis.transformations import *
from luna.featurevis.regularizers import *
from tensorflow import keras
import json
INFERENCE = True
output_path = r'C:\Users\lucaz\Documents\Fuzhi\GitHub\featurevis_experimentation\output_tests\DanielNet_luna'
image_size = 32

model_name = "DanielNet"#"resnet50v2"#"vgg16" #inceptionV3"#"vgg19" #"inceptionV3"
#layer_name = "conv2d" #"conv2d_4" #"mixed5"#"conv2_block1_1_conv" #"block3_conv1"#"mixed5" #"block3_conv4"#"mixed5"
#channel_num = 5 #5 #5 #40 #30
activation_score_all = {}
danielNet_layers = ["conv2d_1"]#["conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4"]
danielNet_channels = [1]#[8, 16, 16, 32, 32]
# Optimizer
optimizer = tf.keras.optimizers.Adam(epsilon=1e-08)
# optimizer params
iterations =12
learning_rate = 0.05
opt_param = featurevis.OptimizationParameters(iterations, learning_rate, optimizer=optimizer)
model = keras.models.load_model(r"C:\Users\lucaz\Documents\Fuzhi\GitHub\luna\nws_main_00001")

for layer_name, num_of_channel in zip(danielNet_layers, danielNet_channels):
    for channel_num in range(num_of_channel):
        image = images.initialize_image_ref(image_size,image_size, fft=False, decorrelate=False)
        # im= Image.fromarray(image[0].numpy())
        # im.save("conv2d_0_saveing_method.png")
        #print(model.summary())
        image= featurevis.visualize_filter(image, model, layer_name, channel_num,
                                        opt_param, transformation=None, threshold= None)


        # if INFERENCE:
        #     model_activation = keras.Model(inputs=model.inputs, outputs=model.get_layer(name=layer_name).output)
        #     x = np.expand_dims(image, axis=0)
        #     pred = model_activation(x)
        #     activation_score = tf.reduce_mean(pred[:,:,:,channel_num])
        #     print(f"activation_score from inference is {activation_score.numpy()}")
        #     activation_score_all.update({str(f"{layer_name}_{channel_num}"): activation_score.numpy()})

        img_name = f"{model_name}_{layer_name}_{channel_num}"
        #images.save_image(image, name=img_name)
        #image_reader.save_npy_as_png(f"raw_image_test/{img_name}.npy", output_path)



