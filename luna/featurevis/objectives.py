import tensorflow as tf

class FilterObjective(object):
    """ Feature visualization of a given channel in a layer of a model.
        Computes the mean activation of the specified channel.
    """
        
    def __init__(self, filter_index, regularization=None):
        """ Feature visualization of a given channel in a layer of a model.
            Computes the mean activation of the channel.
            Args: 
                filter_index (number): the index of the filter to be visualized.
                regularization (function): customized regularizers to be applied. Defaults to None.
        """
        self.filter_index = filter_index
        self.regularization = regularization

    def loss(self, input_image, model):
        """Computes the loss for the feature visualization process.

        Args:
            input_image (array): the image that is used to compute the loss.
            model (object): the model on which to compute the loss.
        Returns:
            number: the loss for the specified setting
        """

        activation = model(input_image)
        if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
            filter_activation = activation[:, self.filter_index, :, :]
        else:
            filter_activation = activation[:, :, :, self.filter_index]
        activation_score = tf.reduce_mean(filter_activation)
        if self.regularization:
            if not callable(self.regularization):
                raise ValueError("The regularizations need to be a function.")
            activation_score = self.regularization(activation, activation_score)
        return -activation_score

class LayerObjective(object):
    """ Deepdream visualization of a layer, see Mordvintsev et al. 2015.
        Computes (mean activation)^2 of the input.
    """
    def __init__(self, regularization=None):
        """ Deepdream visualization of a layer, see Mordvintsev et al. 2015.
            Computes (mean activation)^2 of the input.
            Args: 
                regularization (function): customized regularizers to be applied. Defaults to None.
        """
        self.regularization = regularization

    def loss(self, input_image, model):
        """Computes the layer loss for the feature visualization process.

        Args:
            input_image (array): the image that is used to compute the loss.
            model (object): the model on which to compute the loss.
        Returns:
            number: the activation for the specified setting
        """

        activation = model(input_image)
        activation_score = tf.reduce_mean(activation**2)
        if self.regularization:
            if not callable(self.regularization):
                raise ValueError("The regularizations need to be a function.")
            activation_score = self.regularization(activation, activation_score)
        return -activation_score



