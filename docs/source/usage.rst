Usage
=====

Before using luna, you need to install it from pip as follows: ``pip install luna_fviz``.

The following provides an example of how Luna can be used for feature visualization::

    from luna.pretrained_models import models
    from luna.featurevis import featurevis, images, image_reader
    from luna.featurevis.transformations import *
    import tensorflow as tf
    import matplotlib.pyplot as plt

    model = models.model_inceptionv3()
    model.trainable = False
    image = images.initialize_image_ref(299, 299, fft=False, decorrelate=False, seed=1)

    iterations = 512
    learning_rate = 0.05

    optimizer = tf.keras.optimizers.Adam(epsilon=1e-08, learning_rate=learning_rate)

    # Define a function containing all the transformations that you would like to apply
    # At the moment scaling and blur yield the best results.
    # Nonetheless, all other lucid transformations are implemented in featurevis.transformations and can be added too.
    def my_trans(img):
        """Function containing all the desired transformations
        """
        img = scale_values(img)
        img = blur(img)
        return img

    opt_param = featurevis.OptimizationParameters(iterations, learning_rate, optimizer=optimizer)
    activation, image= featurevis.visualize_filter(image, model, layer="mixed5", filter_index=30, optimization_parameters=opt_param, transformation=my_trans)

    plt.imshow(image)
    plt.savefig("image.svg")
    plt.clf()

    images.save_image(image, name="test")
    image_reader.save_npy_as_png("test.npy", ".")
