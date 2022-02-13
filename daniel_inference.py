
import os
dirname = os.path.dirname(__file__)
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
from scipy import misc
from tensorflow.keras.preprocessing import image
model = keras.models.load_model(r"C:\Users\lucaz\Documents\Fuzhi\GitHub\luna\nws_main_00001")
print(model.summary())

danielNet_layers_luna = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4"]
danielNet_layers_lucid = ["Conv2D", "Conv2D_1", "Conv2D_2", "Conv2D_3", "Conv2D_4"]
danielNet_channels = [8, 16, 16, 32, 32]

path_png_images = r"C:\Users\lucaz\Documents\Fuzhi\GitHub\featurevis_experimentation\output_tests\DanielNet_luna"
# danielNet_layers_luna = ["conv2d"]
# danielNet_layers_lucid = ["Conv2D"]
# danielNet_channels = [3]
activation_score_all = {}

for layer_name, layer_name_lucid, num_of_channel in zip(danielNet_layers_luna, danielNet_layers_lucid,danielNet_channels):
    for channel_num in range(num_of_channel):
        raw_image_array= np.load(f"raw_image_test/DanielNet_{layer_name}_{channel_num}.npy")
        #png_image = image.load_img(f"{path_png_images}/DanielNet_{layer_name}_{channel_num}.png")
        #raw_image_array = image.img_to_array(png_image)
        #raw_image_array = np.load(rf"C:\Users\lucaz\Documents\Fuzhi\GitHub\featurevis_experimentation\output_tests\DanielNet_raw_images_lucid\{layer_name_lucid}_{channel_num}.npy")
        print(f"the shape of loaded image is {raw_image_array.shape} and the values are {raw_image_array}")
        x = np.expand_dims(raw_image_array, axis=0)
        model_activation = keras.Model(inputs=model.inputs, outputs=model.get_layer(name=layer_name).output)

        pred = model_activation.predict(x)
        activation_score = -1 *tf.reduce_mean(pred[:,:,:,channel_num])
        print(activation_score)
        activation_score_all.update({str(f"{layer_name}_{channel_num}"): activation_score.numpy()})

file = open(f'danielNet_score_from_inference_lucid_2.json', 'w')
file= json.dump(str(activation_score_all), file)