from sklearn.cluster import KMeans

from helpers import *


class RBF:

    def __init__(self, input_dimension, hidden_dimension, output_dimension, gamma):
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.gamma = gamma
        self.centers = None
        self.weights = np.random.rand(self.hidden_dimension, self.output_dimension) - 0.5

    def gaussian_rbf(self, x, center):
        return np.exp(-self.gamma * np.linalg.norm(x - center) ** 2)

    def calculate_hidden_outputs(self, x):
        num_samples = x.shape[0]
        hidden_outputs = np.zeros((num_samples, self.hidden_dimension))

        # Loop through each data point
        for i in range(num_samples):
            # Calculate the output of each RBF neuron for the current data point
            for j in range(self.hidden_dimension):
                hidden_outputs[i, j] = self.gaussian_rbf(x[i], self.centers[j])

        return hidden_outputs

    def random_centers(self, x):
        num_samples = x.shape[0]

        # Shuffle the input data
        shuffled_indices = np.random.permutation(num_samples)

        # Select the first 'hidden_dimension' points as the initial RBF centers
        initial_centers = x[shuffled_indices[:self.hidden_dimension]]

        return initial_centers

    def kmeans_centers(self, x):
        # Use KMeans to calculate RBF centers
        kmeans = KMeans(n_clusters=self.hidden_dimension, n_init='auto')
        kmeans.fit(x)
        return kmeans.cluster_centers_

    def initialize_centers(self, x, centers_alg):
        if centers_alg == 'kmeans':
            # Initialize centers using k-means clustering
            return self.kmeans_centers(x)
        elif centers_alg == 'random':
            # Initialize centers randomly from the input data
            return self.random_centers(x)
        else:
            raise ValueError("Invalid centers_alg. Choose 'kmeans' or 'random'.")

    def fit(self, x_train, y_train, x_test, y_test, learning_rate, epochs, centers_alg='kmeans'):

        self.centers = self.initialize_centers(x_train, centers_alg)

        # Initialize lists to store data for plotting
        train_accuracies = []
        test_accuracies = []
        mse_losses = []

        # Calculate RBF layer
        hidden_outputs = self.calculate_hidden_outputs(x_train)
        val_hidden_outputs = self.calculate_hidden_outputs(x_test)

        # Training
        for epoch in range(epochs):
            # Forward propagation
            output = np.dot(hidden_outputs, self.weights)
            output = self.softmax(output)

            # Backward propagation
            error = output - y_train
            gradient = np.dot(hidden_outputs.T, error)

            # Update weights (delta rule)
            self.weights -= learning_rate * gradient

            # Calculate Mean Squared Error (MSE) loss
            mse_loss = np.mean(np.sum((y_train - output) ** 2, axis=1))
            mse_losses.append(mse_loss)

            # Calculate accuracy
            train_accuracy = accuracy_score(np.argmax(y_train, axis=1), np.argmax(output, axis=1))

            hidden_output = np.dot(val_hidden_outputs, self.weights)
            val_output = self.softmax(hidden_output)
            test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(val_output, axis=1))

            # Store accuracy values
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            # Print epoch stats
            print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_accuracy:.2f}, Test Accuracy: "
                  f"{test_accuracy:.2f}, " f" MSE Loss: {mse_loss:.2f}")

        # Plot the data
        plot_accuracy(train_accuracies, test_accuracies)
        plot_mse(mse_losses)

    def predict(self, x):
        rbf_layer_output = self.calculate_hidden_outputs(x)
        hidden_output = np.dot(rbf_layer_output, self.weights)
        output = self.softmax(hidden_output)
        return output

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
