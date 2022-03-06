Usage
=====

Before using luna, you need to install it from pip as follows: ``pip install luna_fviz``.

The following provides an example of how Luna can be used for feature visualization::

    from luna.pretrained_models import models
    from luna.featurevis import featurevis, images, image_reader
    from luna.featurevis.transformations import *
    
    
    model = models.model_inceptionv3()
    image = images.initialize_image(224, 224)
    
    iterations = 2500
    learning_rate = 0.7
    
    # Define a function containing all the transformations that you would like to apply
    # At the moment scaling and blur yield the best results.
    # Nonetheless, all other lucid transformations are implemented in featurevis.transformations and can be added too.
    def my_trans(img):
        """Function containing all the desired transformations
        """
        img = scale_values(img)
        img = blur(img)
    return img

    opt_param = featurevis.OptimizationParameters(iterations, learning_rate)
    loss, image = featurevis.visualize_filter(image, model, "mixed5", 30, opt_param, transformation=my_trans)
    images.save_image(image, name="test")
    image_reader.save_npy_as_png("test.npy", ".")
