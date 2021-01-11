"""
This file acts as a central storage for all global variables within the scope
the application
"""
import sys
import keras

this = sys.modules[__name__]

SESS = None

# Image constants
IMG_WIDTH = 224
IMG_HEIGHT = 224
OUTPUT_WIDTH = 174
OUTPUT_HEIGHT = 174

# Used Models for FV
RESNET50V2 = {"name": "resnet50v2",
              "range_bot": -1,
              "range_top": 1}
INCEPTIONV1 = {"name": "inceptionV1",
               "range_bot": -1,
               "range_top": 1}
INCEPTIONV1SLIM = {"name": "inceptionV1slim",
                   "range_bot": -1,
                   "range_top": 1}
INCEPTIONV3 = {"name": "inceptionV3",
               "range_bot": -1,
               "range_top": 1}
VGG16 = {"name": "VGG16",
         "range_bot": 0,
         "range_top": 255}

# basic global variables
LAYER_NAME = None
MODEL_INFO = RESNET50V2
MAIN_IMG = None
MODEL = None
LAYER = None
FEATURE_EXTRACTOR = keras.Model()
ITERATIONS = None
LEARNING_RATE = None
WRITER = None
MARGIN = 5
ITERATION = 0
BLUR = False
NOISE = False
SCALE = False
