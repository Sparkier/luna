from pretrained_models import models
from featurevis import featurevis, images, image_reader
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-a", "--architecture", type = str, default = "inceptionV3", help = "The model architecture")
args = parser.parse_args()

arch = args.architecture
#architecture = "inceptionV3"

model = models.get_model(arch)
# model.summary()
image = images.initialize_image(224, 224)
loss, image, layer_name, channel_num = featurevis.visualize_filter(image, model, "mixed6", 1, 500, 2, 0, 0, 0)

name = "feature_vis_{}_{}_{}".format(arch, layer_name, channel_num)
print(loss)

images.save_image(image, name= name)
image_reader.save_npy_as_png("{}.npy".format(name))

