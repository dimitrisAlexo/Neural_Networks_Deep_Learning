import numpy as np

from layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        """
        Initializes an Activation layer.

        Parameters:
        - activation: The activation function.
        - activation_prime: The derivative of the activation function.
        """
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        """
        Performs forward propagation through the activation layer.

        Parameters:
        - input: The input to the activation layer.

        Returns:
        The output after applying the activation function.
        """
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        """
        Performs backward propagation through the activation layer.

        Parameters:
        - output_gradient: The gradient of the loss with respect to the layer's output.
        - learning_rate: The learning rate for updating parameters during backpropagation.

        Returns:
        The gradient of the loss with respect to the layer's input.
        """
        return np.multiply(output_gradient, self.activation_prime(self.input))
