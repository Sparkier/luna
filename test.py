from tensorflow import keras
from luna.pretrained_models import models
from luna.featurevis import featurevis, images, image_reader

from luna.featurevis.transformations import *
import matplotlib.pyplot as plt
import numpy as np


model = models.model_inceptionv3()
model.trainable = False
image = images.initialize_image_ref(299, 299, fft=True, decorrelate=True, seed=1)
#image = images.initialize_image(299, 299)
iterations = 500
learning_rate = 0.05

optimizer = tf.keras.optimizers.Adam(epsilon=1e-08, learning_rate=learning_rate)

# Define a function containing all the transformations that you would like to apply
# At the moment scaling and blur yield the best results.
# Nonetheless, all other lucid transformations are implemented in featurevis.transformations and can be added too.
def my_trans(img):
    """Function containing all the desired transformations
    """
    # img = pad(img, 16)
    # img = jitter(img, 16)
    img = scale_values(img)
    # img = rotation(img, list(range(-5, 6)))
    # img = jitter(img, 8)
    img = blur(img)
    # img = standard_transformation(img)
    return img
opt_param = featurevis.OptimizationParameters(iterations, learning_rate, optimizer=optimizer)
activation, image= featurevis.visualize(image = image, model = model, layer ="mixed5", optimization_parameters=opt_param,
                                        filter_index= 30, transformation=my_trans)

#opt_param = featurevis.OptimizationParameters(iterations, learning_rate, optimizer=optimizer)
# Visualize filter 30 within the mixed5 layer. Note that we use filter, while some use channel, to denote a unit within a layer.
#objective = featurevis.objectives.FilterObjective(model, layer="mixed5", filter_index=30)
#print(objective)
# activation, image= featurevis.visualize(image=image, objective=objective,
#                                         #filter_index=30,
#                                         optimization_parameters=opt_param, transformation=my_trans)

plt.imshow(image)
plt.savefig("image.svg")
plt.clf()

images.save_image(image, name="test")
image_reader.save_npy_as_png("test.npy", ".")


# # Daniel model
# #model = keras.models.load_model(r"C:\Users\lucaz\Documents\Fuzhi\GitHub\luna\nws_main_00001")
# model = models.model_inceptionv3()
# #model.trainable = False
# #image = images.initialize_image(299, 299)
# image = images.initialize_image_ref(32, 32, fft=False, decorrelate=False, seed=1)
# np.save("june.npy", image)
# print(f"initial image is {image}")
# iterations = 512
# #learning_rate = 0.7
# PROCESS = "no"
# INITIALIZED = "ref"
# TRNS= "no"
# AD_TYPE = "gradient"
# optimizer = tf.keras.optimizers.Adam(epsilon=1e-08, learning_rate=0.05)

# # Define a function containing all the transformations that you would like to apply
# # At the moment scaling and blur yield the best results.
# # Nonetheless, all other lucid transformations are implemented in featurevis.transformations and can be added too.
# def my_trans(img):
#     """Function containing all the desired transformations
#     """
#     img = scale_values(img)
#     img = blur(img)
#     return img

# opt_param = featurevis.OptimizationParameters(iterations, learning_rate=None, optimizer=optimizer)
# activation, image= featurevis.visualize_filter(image, model, "mixed5", 30, opt_param, transformation=None, minimize=False)
# # for daniel model : "conv2d", 1,

# plt.imshow(image)
# plt.axis("off")
# plt.savefig(fr"results_june\{INITIALIZED}_initialized_{TRNS}_trans_{AD_TYPE}_{PROCESS}_deprocess_{iterations}_mixed5_30_june.svg")
# plt.clf()

# images.save_image(image, name=f"{INITIALIZED}_initialized_{TRNS}_trans_{AD_TYPE}_{PROCESS}_deprocess_{iterations}_mixed5_30")



# different_image_saving = np.load(f"results_june\{INITIALIZED}_initialized_{TRNS}_trans_{AD_TYPE}_{PROCESS}_deprocess_{iterations}_mixed5_30_june.npy")
# plt.imshow(different_image_saving)
# plt.savefig(r"results_june\different_30_june.png")
# plt.clf()

# image_reader.save_npy_as_png(f"{INITIALIZED}_initialized_{TRNS}_trans_{AD_TYPE}_{PROCESS}_deprocess_{iterations}_mixed5_30_june.npy", ".")

