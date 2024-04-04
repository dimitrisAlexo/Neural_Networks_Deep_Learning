import numpy as np

from layer import Layer  # Assuming that 'Layer' is a custom class defined in another file


class Dense(Layer):
    def __init__(self, input_size, output_size):
        """
        Initializes a Dense layer with random weights and biases.

        Parameters:
        - input_size: Number of input neurons.
        - output_size: Number of output neurons.
        """
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        """
        Performs forward propagation through the Dense layer.

        Parameters:
        - input: Input data.

        Returns:
        The output after applying the layer's transformation.
        """
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        """
        Performs backward propagation through the Dense layer and updates parameters.

        Parameters:
        - output_gradient: Gradient of the loss with respect to the layer's output.
        - learning_rate: Learning rate for updating parameters during backpropagation.

        Returns:
        Gradient of the loss with respect to the layer's input.
        """
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
