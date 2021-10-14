Usage
=====

The following provides an example of how Luna can be used for feature visualization::

    from luna.pretrained_models import models
    from luna.featurevis import featurevis, images, image_reader
    
    
    model = models.model_inceptionv3()
    image = images.initialize_image(224, 224)
    
    iterations = 10
    learning_rate = 200
    
    blur = False
    scale = False
    pad_crop = True
    flip = False
    rotation = True
    noise = False
    color_aug = True
    
    opt_param = featurevis.OptimizationParameters(iterations, learning_rate)
    aug_param = featurevis.AugmentationParameters(blur, scale, pad_crop, flip, rotation, noise, color_aug)
    
    loss, image = featurevis.visualize_filter(image, model, "mixed6", 10, opt_param, aug_param)
    images.save_image(image, name="test")
    image_reader.save_npy_as_png("test.npy", ".")

