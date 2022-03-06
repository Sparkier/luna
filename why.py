import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
dirname = os.path.dirname(__file__)
# layer: conv2d, channel:5
from PIL import Image



#image initialization
np.random.seed(1)
image_f = np.random.normal(size=[1, 32, 32, 3], scale=0.01).astype(np.float32)
image = tf.nn.sigmoid(image_f)
image_test = tf.nn.sigmoid(image_f)
np.save("luna_0.npy", image_test)

#im = tf.keras.utils.img_to_array(image_test[0])
Image.fromarray((image_test[0].numpy() *255).astype(np.uint8)).save("test_test_test.png")

#Daniel model
model = keras.models.load_model(r"C:\Users\lucaz\Documents\Fuzhi\GitHub\luna\nws_main_00001")
optimizer = tf.keras.optimizers.Adam(epsilon=1e-08, learning_rate=0.05)

layer_weight =keras.Model(inputs=model.inputs, outputs=model.get_layer(name="conv2d").output)



def compute_activation():
    filter_activation = layer_weight(img)[:,:,:,5]
    print(f" filter activation should not change {filter_activation}")
    score = -1 *tf.reduce_mean(filter_activation)
    print(score)
    return score

img = tf.Variable(image)
for i in range(512):
    filter_activation_test = layer_weight(image_test)[:,:,:,5]
    print(f"test image should give us same values at every round {filter_activation_test}")
    print(layer_weight.get_weights()[0])
    print(img)
    optimizer.minimize(compute_activation, [img])
    print(f"after step {i} image is {img}")

# compare the gradients






# import numpy as np
# import tensorflow as tf
# from tensorflow import keras

# np.random.seed(1)
# image_f = tf.Variable(np.random.normal(size=[1, 32, 32, 3], scale=0.01).astype(np.float32))
# img = tf.nn.sigmoid(image_f)
# tf.compat.v1.keras.backend.set_image_data_format('channels_last')
# model = keras.applications.VGG16(weights="imagenet", include_top=False)
# optimizer = tf.keras.optimizers.Adam(epsilon=1e-08, learning_rate=0.05)
# layer_weight =keras.Model(inputs=model.inputs, outputs=model.get_layer(name="block3_conv1").output)

# for i in range(5):
#     img = tf.Variable(img)
#     print(img)
#     filter_activation = layer_weight(img)[:,:,:,5]
#     def compute_activation():
#         score = -1 * tf.reduce_mean(filter_activation)
#         print(score)
#         return score
#     optimizer.minimize(compute_activation, img)
#     print(img)

