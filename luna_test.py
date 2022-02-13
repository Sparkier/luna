from luna.pretrained_models import models
from luna.featurevis import featurevis, images, image_reader
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from luna.featurevis.transformations import *
from luna.featurevis.regularizers import *
from tensorflow import keras
import json
dirname = os.path.dirname(__file__)
output_path = r'C:\Users\lucaz\Documents\Fuzhi\GitHub\featurevis_experimentation\output_tests\DanielNet'
image_size = 32
model_name = "DanielNet"#"resnet50v2"#"vgg16" #inceptionV3"#"vgg19" #"inceptionV3"
layer_name = "conv2d_1" #"conv2d_4" #"mixed5"#"conv2_block1_1_conv" #"block3_conv1"#"mixed5" #"block3_conv4"#"mixed5"
channel_num = 5 #5 #5 #40 #30

# resnet v250 layer: conv2_block1_1_conv , channel: 3
# vgg16 layer: block3_conv1 channel : 2

# Aux

#scale = True
#pad_crop = False
#flip = False
#vert_rotation = False
#noise = False
#olor_aug = False

#Aug
pad_size = 8
pad_mode = "CONSTANT"
jitter_value = 8
bilinear = [0.9, 0.97, 1.1, 1.05]
rotation_value = [21]

#Custom
custom = None

#regularizer
reg = None # {"l1_regularizer":-0.05, "tv_regularizer": -0.25, "blur_regularizer": -1.0}

# Optimizer
optimizer = tf.keras.optimizers.Adam(epsilon=1e-08)
# optimizer params
iterations = 512
learning_rate = 0.05

# Thrashold
threshold=[1]

#image = images.initialize_image(image_size,image_size)
image = images.initialize_image_ref(image_size,image_size, fft=False, decorrelate=False)
#model = models.model_vgg19()
#model = models.model_inceptionv3()
#model = models.model_vgg16()
#model = models.model_resnet50v2()
model = keras.models.load_model(r"C:\Users\lucaz\Documents\Fuzhi\GitHub\luna\nws_main_00001")
print(model.get_layer(layer_name).get_weights()[0][:,:,:,channel_num])
print(model.summary())

def my_trans(img):
    img = pad(img, 12, pad_mode="constant")
    img = jitter(img, 8)
    #img = scale_values(img)
    #img = blur(img)
    #img = pad(img, 4)


    img = bilinear_rescale(img, [1 + (i - 5) / 50.0 for i in range(11)])
    #img = rotation(img, range(-10, 11))
    #img = rotation(img, range(-12,12))
    img = jitter(img, 4)


    return img

# def my_regs(activation, activation_score):
#     activation_score += l1_regularization(activation, l1_value=-0.05)
#     activation_score += total_variation(activation)

#     return activation_score

opt_param = featurevis.OptimizationParameters(iterations, learning_rate, optimizer=optimizer)


image= featurevis.visualize_filter(image, model, layer_name, channel_num,
                                       opt_param, transformation=None, threshold= threshold)#, regularization=my_regs)



img_name = f"{model_name}_{layer_name}_{channel_num}"


images.save_image(image, name=img_name)
image_reader.save_npy_as_png(f"{img_name}.npy", output_path)




# Model: "vgg19"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, None, None, 3)]   0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, None, None, 64)    1792
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, None, None, 64)    36928
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, None, None, 64)    0
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, None, None, 128)   73856
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, None, None, 128)   147584
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, None, None, 128)   0
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, None, None, 256)   295168
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, None, None, 256)   590080
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, None, None, 256)   590080
# _________________________________________________________________
# block3_conv4 (Conv2D)        (None, None, None, 256)   590080
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, None, None, 256)   0
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, None, None, 512)   1180160
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, None, None, 512)   2359808
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, None, None, 512)   2359808
# _________________________________________________________________
# block4_conv4 (Conv2D)        (None, None, None, 512)   2359808
# _________________________________________________________________
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, None, None, 512)   2359808
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, None, None, 512)   2359808
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, None, None, 512)   2359808
# _________________________________________________________________
# block5_conv4 (Conv2D)        (None, None, None, 512)   2359808
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, None, None, 512)   0
# =================================================================
# Total params: 20,024,384
# Trainable params: 20,024,384
# Non-trainable params: 0
# _________________________________________________________________



# Model: "resnet50v2"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            [(None, None, None,  0
# __________________________________________________________________________________________________
# conv1_pad (ZeroPadding2D)       (None, None, None, 3 0           input_1[0][0]
# __________________________________________________________________________________________________
# conv1_conv (Conv2D)             (None, None, None, 6 9472        conv1_pad[0][0]
# __________________________________________________________________________________________________
# pool1_pad (ZeroPadding2D)       (None, None, None, 6 0           conv1_conv[0][0]
# __________________________________________________________________________________________________
# pool1_pool (MaxPooling2D)       (None, None, None, 6 0           pool1_pad[0][0]
# __________________________________________________________________________________________________
# conv2_block1_preact_bn (BatchNo (None, None, None, 6 256         pool1_pool[0][0]
# __________________________________________________________________________________________________
# conv2_block1_preact_relu (Activ (None, None, None, 6 0           conv2_block1_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv2_block1_1_conv (Conv2D)    (None, None, None, 6 4096        conv2_block1_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv2_block1_1_bn (BatchNormali (None, None, None, 6 256         conv2_block1_1_conv[0][0]
# __________________________________________________________________________________________________
# conv2_block1_1_relu (Activation (None, None, None, 6 0           conv2_block1_1_bn[0][0]
# __________________________________________________________________________________________________
# conv2_block1_2_pad (ZeroPadding (None, None, None, 6 0           conv2_block1_1_relu[0][0]
# __________________________________________________________________________________________________
# conv2_block1_2_conv (Conv2D)    (None, None, None, 6 36864       conv2_block1_2_pad[0][0]
# __________________________________________________________________________________________________
# conv2_block1_2_bn (BatchNormali (None, None, None, 6 256         conv2_block1_2_conv[0][0]
# __________________________________________________________________________________________________
# conv2_block1_2_relu (Activation (None, None, None, 6 0           conv2_block1_2_bn[0][0]
# __________________________________________________________________________________________________
# conv2_block1_0_conv (Conv2D)    (None, None, None, 2 16640       conv2_block1_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv2_block1_3_conv (Conv2D)    (None, None, None, 2 16640       conv2_block1_2_relu[0][0]
# __________________________________________________________________________________________________
# conv2_block1_out (Add)          (None, None, None, 2 0           conv2_block1_0_conv[0][0]
#                                                                  conv2_block1_3_conv[0][0]
# __________________________________________________________________________________________________
# conv2_block2_preact_bn (BatchNo (None, None, None, 2 1024        conv2_block1_out[0][0]
# __________________________________________________________________________________________________
# conv2_block2_preact_relu (Activ (None, None, None, 2 0           conv2_block2_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv2_block2_1_conv (Conv2D)    (None, None, None, 6 16384       conv2_block2_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv2_block2_1_bn (BatchNormali (None, None, None, 6 256         conv2_block2_1_conv[0][0]
# __________________________________________________________________________________________________
# conv2_block2_1_relu (Activation (None, None, None, 6 0           conv2_block2_1_bn[0][0]
# __________________________________________________________________________________________________
# conv2_block2_2_pad (ZeroPadding (None, None, None, 6 0           conv2_block2_1_relu[0][0]
# __________________________________________________________________________________________________
# conv2_block2_2_conv (Conv2D)    (None, None, None, 6 36864       conv2_block2_2_pad[0][0]
# __________________________________________________________________________________________________
# conv2_block2_2_bn (BatchNormali (None, None, None, 6 256         conv2_block2_2_conv[0][0]
# __________________________________________________________________________________________________
# conv2_block2_2_relu (Activation (None, None, None, 6 0           conv2_block2_2_bn[0][0]
# __________________________________________________________________________________________________
# conv2_block2_3_conv (Conv2D)    (None, None, None, 2 16640       conv2_block2_2_relu[0][0]
# __________________________________________________________________________________________________
# conv2_block2_out (Add)          (None, None, None, 2 0           conv2_block1_out[0][0]
#                                                                  conv2_block2_3_conv[0][0]
# __________________________________________________________________________________________________
# conv2_block3_preact_bn (BatchNo (None, None, None, 2 1024        conv2_block2_out[0][0]
# __________________________________________________________________________________________________
# conv2_block3_preact_relu (Activ (None, None, None, 2 0           conv2_block3_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv2_block3_1_conv (Conv2D)    (None, None, None, 6 16384       conv2_block3_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv2_block3_1_bn (BatchNormali (None, None, None, 6 256         conv2_block3_1_conv[0][0]
# __________________________________________________________________________________________________
# conv2_block3_1_relu (Activation (None, None, None, 6 0           conv2_block3_1_bn[0][0]
# __________________________________________________________________________________________________
# conv2_block3_2_pad (ZeroPadding (None, None, None, 6 0           conv2_block3_1_relu[0][0]
# __________________________________________________________________________________________________
# conv2_block3_2_conv (Conv2D)    (None, None, None, 6 36864       conv2_block3_2_pad[0][0]
# __________________________________________________________________________________________________
# conv2_block3_2_bn (BatchNormali (None, None, None, 6 256         conv2_block3_2_conv[0][0]
# __________________________________________________________________________________________________
# conv2_block3_2_relu (Activation (None, None, None, 6 0           conv2_block3_2_bn[0][0]
# __________________________________________________________________________________________________
# max_pooling2d (MaxPooling2D)    (None, None, None, 2 0           conv2_block2_out[0][0]
# __________________________________________________________________________________________________
# conv2_block3_3_conv (Conv2D)    (None, None, None, 2 16640       conv2_block3_2_relu[0][0]
# __________________________________________________________________________________________________
# conv2_block3_out (Add)          (None, None, None, 2 0           max_pooling2d[0][0]
#                                                                  conv2_block3_3_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block1_preact_bn (BatchNo (None, None, None, 2 1024        conv2_block3_out[0][0]
# __________________________________________________________________________________________________
# conv3_block1_preact_relu (Activ (None, None, None, 2 0           conv3_block1_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv3_block1_1_conv (Conv2D)    (None, None, None, 1 32768       conv3_block1_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block1_1_bn (BatchNormali (None, None, None, 1 512         conv3_block1_1_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block1_1_relu (Activation (None, None, None, 1 0           conv3_block1_1_bn[0][0]
# __________________________________________________________________________________________________
# conv3_block1_2_pad (ZeroPadding (None, None, None, 1 0           conv3_block1_1_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block1_2_conv (Conv2D)    (None, None, None, 1 147456      conv3_block1_2_pad[0][0]
# __________________________________________________________________________________________________
# conv3_block1_2_bn (BatchNormali (None, None, None, 1 512         conv3_block1_2_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block1_2_relu (Activation (None, None, None, 1 0           conv3_block1_2_bn[0][0]
# __________________________________________________________________________________________________
# conv3_block1_0_conv (Conv2D)    (None, None, None, 5 131584      conv3_block1_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block1_3_conv (Conv2D)    (None, None, None, 5 66048       conv3_block1_2_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block1_out (Add)          (None, None, None, 5 0           conv3_block1_0_conv[0][0]
#                                                                  conv3_block1_3_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block2_preact_bn (BatchNo (None, None, None, 5 2048        conv3_block1_out[0][0]
# __________________________________________________________________________________________________
# conv3_block2_preact_relu (Activ (None, None, None, 5 0           conv3_block2_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv3_block2_1_conv (Conv2D)    (None, None, None, 1 65536       conv3_block2_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block2_1_bn (BatchNormali (None, None, None, 1 512         conv3_block2_1_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block2_1_relu (Activation (None, None, None, 1 0           conv3_block2_1_bn[0][0]
# __________________________________________________________________________________________________
# conv3_block2_2_pad (ZeroPadding (None, None, None, 1 0           conv3_block2_1_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block2_2_conv (Conv2D)    (None, None, None, 1 147456      conv3_block2_2_pad[0][0]
# __________________________________________________________________________________________________
# conv3_block2_2_bn (BatchNormali (None, None, None, 1 512         conv3_block2_2_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block2_2_relu (Activation (None, None, None, 1 0           conv3_block2_2_bn[0][0]
# __________________________________________________________________________________________________
# conv3_block2_3_conv (Conv2D)    (None, None, None, 5 66048       conv3_block2_2_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block2_out (Add)          (None, None, None, 5 0           conv3_block1_out[0][0]
#                                                                  conv3_block2_3_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block3_preact_bn (BatchNo (None, None, None, 5 2048        conv3_block2_out[0][0]
# __________________________________________________________________________________________________
# conv3_block3_preact_relu (Activ (None, None, None, 5 0           conv3_block3_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv3_block3_1_conv (Conv2D)    (None, None, None, 1 65536       conv3_block3_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block3_1_bn (BatchNormali (None, None, None, 1 512         conv3_block3_1_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block3_1_relu (Activation (None, None, None, 1 0           conv3_block3_1_bn[0][0]
# __________________________________________________________________________________________________
# conv3_block3_2_pad (ZeroPadding (None, None, None, 1 0           conv3_block3_1_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block3_2_conv (Conv2D)    (None, None, None, 1 147456      conv3_block3_2_pad[0][0]
# __________________________________________________________________________________________________
# conv3_block3_2_bn (BatchNormali (None, None, None, 1 512         conv3_block3_2_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block3_2_relu (Activation (None, None, None, 1 0           conv3_block3_2_bn[0][0]
# __________________________________________________________________________________________________
# conv3_block3_3_conv (Conv2D)    (None, None, None, 5 66048       conv3_block3_2_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block3_out (Add)          (None, None, None, 5 0           conv3_block2_out[0][0]
#                                                                  conv3_block3_3_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block4_preact_bn (BatchNo (None, None, None, 5 2048        conv3_block3_out[0][0]
# __________________________________________________________________________________________________
# conv3_block4_preact_relu (Activ (None, None, None, 5 0           conv3_block4_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv3_block4_1_conv (Conv2D)    (None, None, None, 1 65536       conv3_block4_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block4_1_bn (BatchNormali (None, None, None, 1 512         conv3_block4_1_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block4_1_relu (Activation (None, None, None, 1 0           conv3_block4_1_bn[0][0]
# __________________________________________________________________________________________________
# conv3_block4_2_pad (ZeroPadding (None, None, None, 1 0           conv3_block4_1_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block4_2_conv (Conv2D)    (None, None, None, 1 147456      conv3_block4_2_pad[0][0]
# __________________________________________________________________________________________________
# conv3_block4_2_bn (BatchNormali (None, None, None, 1 512         conv3_block4_2_conv[0][0]
# __________________________________________________________________________________________________
# conv3_block4_2_relu (Activation (None, None, None, 1 0           conv3_block4_2_bn[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, None, None, 5 0           conv3_block3_out[0][0]
# __________________________________________________________________________________________________
# conv3_block4_3_conv (Conv2D)    (None, None, None, 5 66048       conv3_block4_2_relu[0][0]
# __________________________________________________________________________________________________
# conv3_block4_out (Add)          (None, None, None, 5 0           max_pooling2d_1[0][0]
#                                                                  conv3_block4_3_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block1_preact_bn (BatchNo (None, None, None, 5 2048        conv3_block4_out[0][0]
# __________________________________________________________________________________________________
# conv4_block1_preact_relu (Activ (None, None, None, 5 0           conv4_block1_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block1_1_conv (Conv2D)    (None, None, None, 2 131072      conv4_block1_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block1_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block1_1_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block1_1_relu (Activation (None, None, None, 2 0           conv4_block1_1_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block1_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block1_1_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block1_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block1_2_pad[0][0]
# __________________________________________________________________________________________________
# conv4_block1_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block1_2_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block1_2_relu (Activation (None, None, None, 2 0           conv4_block1_2_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block1_0_conv (Conv2D)    (None, None, None, 1 525312      conv4_block1_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block1_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block1_2_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block1_out (Add)          (None, None, None, 1 0           conv4_block1_0_conv[0][0]
#                                                                  conv4_block1_3_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block2_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block1_out[0][0]
# __________________________________________________________________________________________________
# conv4_block2_preact_relu (Activ (None, None, None, 1 0           conv4_block2_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block2_1_conv (Conv2D)    (None, None, None, 2 262144      conv4_block2_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block2_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block2_1_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block2_1_relu (Activation (None, None, None, 2 0           conv4_block2_1_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block2_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block2_1_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block2_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block2_2_pad[0][0]
# __________________________________________________________________________________________________
# conv4_block2_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block2_2_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block2_2_relu (Activation (None, None, None, 2 0           conv4_block2_2_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block2_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block2_2_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block2_out (Add)          (None, None, None, 1 0           conv4_block1_out[0][0]
#                                                                  conv4_block2_3_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block3_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block2_out[0][0]
# __________________________________________________________________________________________________
# conv4_block3_preact_relu (Activ (None, None, None, 1 0           conv4_block3_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block3_1_conv (Conv2D)    (None, None, None, 2 262144      conv4_block3_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block3_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block3_1_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block3_1_relu (Activation (None, None, None, 2 0           conv4_block3_1_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block3_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block3_1_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block3_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block3_2_pad[0][0]
# __________________________________________________________________________________________________
# conv4_block3_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block3_2_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block3_2_relu (Activation (None, None, None, 2 0           conv4_block3_2_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block3_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block3_2_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block3_out (Add)          (None, None, None, 1 0           conv4_block2_out[0][0]
#                                                                  conv4_block3_3_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block4_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block3_out[0][0]
# __________________________________________________________________________________________________
# conv4_block4_preact_relu (Activ (None, None, None, 1 0           conv4_block4_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block4_1_conv (Conv2D)    (None, None, None, 2 262144      conv4_block4_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block4_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block4_1_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block4_1_relu (Activation (None, None, None, 2 0           conv4_block4_1_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block4_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block4_1_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block4_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block4_2_pad[0][0]
# __________________________________________________________________________________________________
# conv4_block4_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block4_2_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block4_2_relu (Activation (None, None, None, 2 0           conv4_block4_2_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block4_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block4_2_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block4_out (Add)          (None, None, None, 1 0           conv4_block3_out[0][0]
#                                                                  conv4_block4_3_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block5_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block4_out[0][0]
# __________________________________________________________________________________________________
# conv4_block5_preact_relu (Activ (None, None, None, 1 0           conv4_block5_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block5_1_conv (Conv2D)    (None, None, None, 2 262144      conv4_block5_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block5_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block5_1_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block5_1_relu (Activation (None, None, None, 2 0           conv4_block5_1_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block5_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block5_1_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block5_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block5_2_pad[0][0]
# __________________________________________________________________________________________________
# conv4_block5_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block5_2_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block5_2_relu (Activation (None, None, None, 2 0           conv4_block5_2_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block5_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block5_2_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block5_out (Add)          (None, None, None, 1 0           conv4_block4_out[0][0]
#                                                                  conv4_block5_3_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block6_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block5_out[0][0]
# __________________________________________________________________________________________________
# conv4_block6_preact_relu (Activ (None, None, None, 1 0           conv4_block6_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block6_1_conv (Conv2D)    (None, None, None, 2 262144      conv4_block6_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block6_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block6_1_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block6_1_relu (Activation (None, None, None, 2 0           conv4_block6_1_bn[0][0]
# __________________________________________________________________________________________________
# conv4_block6_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block6_1_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block6_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block6_2_pad[0][0]
# __________________________________________________________________________________________________
# conv4_block6_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block6_2_conv[0][0]
# __________________________________________________________________________________________________
# conv4_block6_2_relu (Activation (None, None, None, 2 0           conv4_block6_2_bn[0][0]
# __________________________________________________________________________________________________
# max_pooling2d_2 (MaxPooling2D)  (None, None, None, 1 0           conv4_block5_out[0][0]
# __________________________________________________________________________________________________
# conv4_block6_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block6_2_relu[0][0]
# __________________________________________________________________________________________________
# conv4_block6_out (Add)          (None, None, None, 1 0           max_pooling2d_2[0][0]
#                                                                  conv4_block6_3_conv[0][0]
# __________________________________________________________________________________________________
# conv5_block1_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block6_out[0][0]
# __________________________________________________________________________________________________
# conv5_block1_preact_relu (Activ (None, None, None, 1 0           conv5_block1_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv5_block1_1_conv (Conv2D)    (None, None, None, 5 524288      conv5_block1_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv5_block1_1_bn (BatchNormali (None, None, None, 5 2048        conv5_block1_1_conv[0][0]
# __________________________________________________________________________________________________
# conv5_block1_1_relu (Activation (None, None, None, 5 0           conv5_block1_1_bn[0][0]
# __________________________________________________________________________________________________
# conv5_block1_2_pad (ZeroPadding (None, None, None, 5 0           conv5_block1_1_relu[0][0]
# __________________________________________________________________________________________________
# conv5_block1_2_conv (Conv2D)    (None, None, None, 5 2359296     conv5_block1_2_pad[0][0]
# __________________________________________________________________________________________________
# conv5_block1_2_bn (BatchNormali (None, None, None, 5 2048        conv5_block1_2_conv[0][0]        
# __________________________________________________________________________________________________
# conv5_block1_2_relu (Activation (None, None, None, 5 0           conv5_block1_2_bn[0][0]
# __________________________________________________________________________________________________
# conv5_block1_0_conv (Conv2D)    (None, None, None, 2 2099200     conv5_block1_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv5_block1_3_conv (Conv2D)    (None, None, None, 2 1050624     conv5_block1_2_relu[0][0]
# __________________________________________________________________________________________________
# conv5_block1_out (Add)          (None, None, None, 2 0           conv5_block1_0_conv[0][0]
#                                                                  conv5_block1_3_conv[0][0]
# __________________________________________________________________________________________________
# conv5_block2_preact_bn (BatchNo (None, None, None, 2 8192        conv5_block1_out[0][0]
# __________________________________________________________________________________________________
# conv5_block2_preact_relu (Activ (None, None, None, 2 0           conv5_block2_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv5_block2_1_conv (Conv2D)    (None, None, None, 5 1048576     conv5_block2_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv5_block2_1_bn (BatchNormali (None, None, None, 5 2048        conv5_block2_1_conv[0][0]
# __________________________________________________________________________________________________
# conv5_block2_1_relu (Activation (None, None, None, 5 0           conv5_block2_1_bn[0][0]
# __________________________________________________________________________________________________
# conv5_block2_2_pad (ZeroPadding (None, None, None, 5 0           conv5_block2_1_relu[0][0]
# __________________________________________________________________________________________________
# conv5_block2_2_conv (Conv2D)    (None, None, None, 5 2359296     conv5_block2_2_pad[0][0]
# __________________________________________________________________________________________________
# conv5_block2_2_bn (BatchNormali (None, None, None, 5 2048        conv5_block2_2_conv[0][0]
# __________________________________________________________________________________________________
# conv5_block2_2_relu (Activation (None, None, None, 5 0           conv5_block2_2_bn[0][0]
# __________________________________________________________________________________________________
# conv5_block2_3_conv (Conv2D)    (None, None, None, 2 1050624     conv5_block2_2_relu[0][0]
# __________________________________________________________________________________________________
# conv5_block2_out (Add)          (None, None, None, 2 0           conv5_block1_out[0][0]
#                                                                  conv5_block2_3_conv[0][0]
# __________________________________________________________________________________________________
# conv5_block3_preact_bn (BatchNo (None, None, None, 2 8192        conv5_block2_out[0][0]
# __________________________________________________________________________________________________
# conv5_block3_preact_relu (Activ (None, None, None, 2 0           conv5_block3_preact_bn[0][0]
# __________________________________________________________________________________________________
# conv5_block3_1_conv (Conv2D)    (None, None, None, 5 1048576     conv5_block3_preact_relu[0][0]
# __________________________________________________________________________________________________
# conv5_block3_1_bn (BatchNormali (None, None, None, 5 2048        conv5_block3_1_conv[0][0]
# __________________________________________________________________________________________________
# conv5_block3_1_relu (Activation (None, None, None, 5 0           conv5_block3_1_bn[0][0]
# __________________________________________________________________________________________________
# conv5_block3_2_pad (ZeroPadding (None, None, None, 5 0           conv5_block3_1_relu[0][0]
# __________________________________________________________________________________________________
# conv5_block3_2_conv (Conv2D)    (None, None, None, 5 2359296     conv5_block3_2_pad[0][0]
# __________________________________________________________________________________________________
# conv5_block3_2_bn (BatchNormali (None, None, None, 5 2048        conv5_block3_2_conv[0][0]
# conv5_block3_2_relu (Activation (None, None, None, 5 0           conv5_block3_2_bn[0][0]
# __________________________________________________________________________________________________
# conv5_block3_3_conv (Conv2D)    (None, None, None, 2 1050624     conv5_block3_2_relu[0][0]
# __________________________________________________________________________________________________
# conv5_block3_out (Add)          (None, None, None, 2 0           conv5_block2_out[0][0]
#                                                                  conv5_block3_3_conv[0][0]
# __________________________________________________________________________________________________
# post_bn (BatchNormalization)    (None, None, None, 2 8192        conv5_block3_out[0][0]
# __________________________________________________________________________________________________
# post_relu (Activation)          (None, None, None, 2 0           post_bn[0][0]
# ==================================================================================================
# Total params: 23,564,800
# Trainable params: 23,519,360
# Non-trainable params: 45,440
# __________________________________________________________________________________________________
# >>>


# Model: "vgg16"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_4 (InputLayer)         [(None, None, None, 3)]   0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, None, None, 64)    1792
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, None, None, 64)    36928
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, None, None, 64)    0
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, None, None, 128)   73856
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, None, None, 128)   147584
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, None, None, 128)   0
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, None, None, 256)   295168
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, None, None, 256)   590080
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, None, None, 256)   590080
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, None, None, 256)   0
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, None, None, 512)   1180160
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, None, None, 512)   2359808
# _________________________________________________________________
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, None, None, 512)   0
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, None, None, 512)   2359808
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, None, None, 512)   2359808
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, None, None, 512)   2359808
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, None, None, 512)   0
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________