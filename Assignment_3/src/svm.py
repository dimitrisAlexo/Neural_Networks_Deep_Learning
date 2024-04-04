# IMPORTS
import numpy as np

class SVM:

    def __init__(self, kernel='poly', degree=3, sigma=100, epoches=1000, learning_rate=0.001):
        """
        Constructor for SVM class.

        Parameters:
        - kernel (str): Type of kernel function ('poly' for polynomial, 'rbf' for Gaussian).
        - degree (int): Degree of the polynomial kernel (only applicable if kernel is 'poly').
        - sigma (float): Parameter for the Gaussian kernel.
        - epoches (int): Number of training epochs.
        - learning_rate (float): Learning rate for the optimization algorithm.
        """
        self.alpha = None
        self.b = 0
        self.degree = degree
        self.c = 1
        self.C = 100
        self.sigma = sigma
        self.epoches = epoches
        self.learning_rate = learning_rate

        if kernel == 'poly':
            self.kernel = self.polynomial_kernel  # for polynomial kernel
        elif kernel == 'rbf':
            self.kernel = self.gaussian_kernel  # for Gaussian kernel

    def polynomial_kernel(self, X, Z):
        """
        Polynomial kernel function.

        Parameters:
        - X (numpy.ndarray): Input data matrix.
        - Z (numpy.ndarray): Another input data matrix.

        Returns:
        - numpy.ndarray: Kernel matrix computed using the polynomial kernel.
        """
        return (self.c + X.dot(Z.T)) ** self.degree  # (c + X.y)^degree

    def gaussian_kernel(self, X, Z):
        """
        Gaussian (RBF) kernel function.

        Parameters:
        - X (numpy.ndarray): Input data matrix.
        - Z (numpy.ndarray): Another input data matrix.

        Returns:
        - numpy.ndarray: Kernel matrix computed using the Gaussian kernel.
        """
        return np.exp(-(1 / self.sigma ** 2) * np.linalg.norm(X[:, np.newaxis] - Z[np.newaxis, :], axis=2) ** 2)

    def train(self, X, y):
        """
        Train the SVM model.

        Parameters:
        - X (numpy.ndarray): Training data matrix.
        - y (numpy.ndarray): Labels for the training data.

        Returns:
        - None
        """
        self.X = X
        self.y = y
        self.alpha = np.random.random(X.shape[0])
        self.b = 0
        self.ones = np.ones(X.shape[0])

        y_mul_kernel = np.outer(y, y) * self.kernel(X, X)

        for i in range(self.epoches):
            gradient = self.ones - y_mul_kernel.dot(self.alpha)

            self.alpha += self.learning_rate * gradient
            self.alpha[self.alpha > self.C] = self.C
            self.alpha[self.alpha < 0] = 0

            loss = np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * y_mul_kernel)

        alpha_index = np.where((self.alpha) > 0 & (self.alpha < self.C))[0]

        # For intercept b, we will only consider α which are 0 < α < C
        b_list = []
        for index in alpha_index:
            b_list.append(y[index] - (self.alpha * y).dot(self.kernel(X, X[index])))

        self.b = np.mean(b_list)

    def predict(self, X):
        """
        Predict labels for input data.

        Parameters:
        - X (numpy.ndarray): Input data matrix.

        Returns:
        - numpy.ndarray: Predicted labels.
        """
        return np.sign(self.decision_function(X))

    def calculate_accuracy(self, X, y):
        correct_predictions = 0

        output = self.predict(X)

        for x, y_true in zip(output, y):
            predicted_label = x
            true_label = y_true

            # print(predicted_label, true_label)

            if predicted_label == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / len(y)
        return accuracy

    def decision_function(self, X):
        """
        Compute the decision function values.

        Parameters:
        - X (numpy.ndarray): Input data matrix.

        Returns:
        - numpy.ndarray: Decision function values.
        """
        return (self.alpha * self.y).dot(self.kernel(self.X, X)) + self.b
