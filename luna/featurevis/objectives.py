"""
Methods for computing the loss for which to optimize the visualization for.
"""
import tensorflow as tf


class FilterObjective:
    """ Feature visualization of a given filter in a layer of a model.
        Computes the mean activation of the specified filter.
    """

    def __init__(self, model, layer, filter_index, regularization=None):
        """ Feature visualization of a given filter in a layer of a model.
            Computes the mean activation of the filter.
            Args:
                model (object): the model to be used for the feature visualization.
                layer (string): the name of the layer to be used in the visualization.
                filter_index (number): the index of the filter to be visualized.
                regularization (function): customized regularizers to be applied. Defaults to None.
        """
        self.model = get_feature_extractor(model, layer)
        self.filter_index = filter_index
        self.regularization = regularization

    def loss(self, input_image):
        """Computes the loss for the feature visualization process.
        Args:
            input_image (array): the image that is used to compute the loss.
            model (object): the model on which to compute the loss.
        Returns:
            number: the loss for the specified setting
        """
        activation = self.model(input_image)
        if tf.compat.v1.keras.backend.image_data_format() == "channels_first":
            filter_activation = activation[:, self.filter_index, :, :]
        else:
            filter_activation = activation[:, :, :, self.filter_index]
        activation_score = tf.reduce_mean(filter_activation)
        if self.regularization:
            if not callable(self.regularization):
                raise ValueError("The regularizations need to be a function.")
            activation_score = self.regularization(
                activation, activation_score)
        return -activation_score

    def __repr__(self) -> str:
        return f"FilterObjective({self.model}, {self.filter_index}, {self.regularization})"


class LayerObjective:
    """ Deepdream visualization of a layer, see Mordvintsev et al. 2015.
        Computes (mean activation)^2 of the input.
    """

    def __init__(self, model, layer, regularization=None):
        """ Deepdream visualization of a layer, see Mordvintsev et al. 2015.
            Computes (mean activation)^2 of the input.
            Args:
                model (object): the model to be used for the feature visualization.
                layer (string): the name of the layer to be used in the visualization.
                regularization (function): customized regularizers to be applied. Defaults to None.
        """
        self.model = get_feature_extractor(model, layer)
        self.regularization = regularization

    def loss(self, input_image):
        """Computes the layer loss for the feature visualization process.
        Args:
            input_image (array): the image that is used to compute the loss.
            model (object): the model on which to compute the loss.
        Returns:
            number: the loss for the specified setting
        """
        activation = self.model(input_image)
        activation_score = tf.reduce_mean(activation**2)
        if self.regularization:
            if not callable(self.regularization):
                raise ValueError("The regularizations need to be a function.")
            activation_score = self.regularization(
                activation, activation_score)
        return -activation_score

    def __repr__(self) -> str:
        return f"LayerObjective({self.model}, {self.regularization})"

class LayerActivationObjective:
    """ Targets a provided layer activation.
        Computes (mean(target_activation - activation))^2.
    """

    def __init__(self, model, layer, target_activation, regularization=None):
        """ For visualization of a specific target activation of a layer.
            Computes (mean(target_activation - activation))^2 of the input.
            Args:
                model (object): the model to be used for the feature visualization.
                layer (string): the name of the layer to be used in the visualization.
                target_activation (array): the target activation.
                regularization (function): customized regularizers to be applied. Defaults to None.
        """
        self.model = get_feature_extractor(model, layer)
        self.regularization = regularization
        self.target_activation = target_activation

    def loss(self, input_image):
        """Computes the layer loss for the feature visualization process.
        Args:
            input_image (array): the image that is used to compute the loss.
            model (object): the model on which to compute the loss.
        Returns:
            number: the loss for the specified setting
        """
        activation = self.model(input_image)
        activation_score = -tf.reduce_mean((self.target_activation - activation)**2)
        if self.regularization:
            if not callable(self.regularization):
                raise ValueError("The regularizations need to be a function.")
            activation_score = self.regularization(
                activation, activation_score)
        return -activation_score

    def __repr__(self) -> str:
        return f"ActivationObjective({self.model}, {self.regularization}, {self.target_activation})"


def get_feature_extractor(model, layer_name):
    """Builds a model that that returns the activation of the specified layer.

    Args:
        model (object): the model used as a basis for the feature extractor.
        layer (string): the layer at which to cap the original model.
    """
    layer = model.get_layer(name=layer_name)
    return tf.keras.Model(inputs=model.inputs, outputs=layer.output)
