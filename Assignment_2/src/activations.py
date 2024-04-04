import numpy as np

from activation import Activation


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class LeLeLu(Activation):
    def __init__(self, alpha=0.01):
        def lelelu(x):
            return np.where(x > 0, x, alpha * (np.exp(x) - 1))

        def lelelu_prime(x):
            return np.where(x > 0, 1, alpha * np.exp(x))

        super().__init__(lelelu, lelelu_prime)


class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)


class Softmax(Activation):
    def __init__(self):
        super().__init__(self.softmax, self.softmax_prime)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Avoid numerical instability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def softmax_prime(self, x):
        s = self.softmax(x)
        return s * (1 - s)

