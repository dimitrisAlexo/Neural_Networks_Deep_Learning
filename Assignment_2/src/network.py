import numpy as np

from helpers import plot_accuracy
from helpers import plot_loss


def predict(network, input):
    """
    Predicts the output of the neural network given an input through forward propagation.

    Parameters:
    - network: The neural network.
    - input: The input data.

    Returns:
    The predicted output.
    """
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def shuffle_data(x_train, y_train):
    """
    Shuffles the input and output data to introduce randomness during training.

    Parameters:
    - x_train: Input data.
    - y_train: Output data.

    Returns:
    Shuffled input and output data.
    """
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    return x_train[indices], y_train[indices]


def create_mini_batches(x_train, y_train, batch_size):
    """
    Creates mini-batches from input and output data.

    Parameters:
    - x_train: Input data.
    - y_train: Output data.
    - batch_size: Size of each mini-batch.

    Returns:
    List of mini-batches, each containing a tuple of input and output data.
    """
    mini_batches = []
    for i in range(0, len(x_train), batch_size):
        x_mini_batch = x_train[i:i + batch_size]
        y_mini_batch = y_train[i:i + batch_size]
        mini_batches.append((x_mini_batch, y_mini_batch))
    return mini_batches


def calculate_accuracy(network, x_test, y_test):
    """
    Calculates the accuracy of the neural network on a test dataset.

    Parameters:
    - network: The neural network.
    - x_test: Test input data.
    - y_test: Test output data.

    Returns:
    Accuracy of the network on the test dataset.
    """
    correct_predictions = 0

    for x, y_true in zip(x_test, y_test):
        output = predict(network, x)
        predicted_label = np.argmax(output)
        true_label = np.argmax(y_true)

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(y_test)
    return accuracy


def train_mini_batch(network, loss, loss_prime, x_train, y_train, x_test, y_test,
                     epochs=100, learning_rate=0.63, batch_size=20):
    """
    Trains the neural network using mini-batch gradient descent.

    Parameters:
    - network: The neural network.
    - loss: Loss function.
    - loss_prime: Derivative of the loss function.
    - x_train: Training input data.
    - y_train: Training output data.
    - x_test: Test input data.
    - y_test: Test output data.
    - epochs: Number of training epochs.
    - learning_rate: Learning rate for gradient descent.
    - batch_size: Size of each mini-batch.
    - verbose: If True, prints training progress.
    """

    train_accuracy = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)
    mse_matrix = np.zeros(epochs)

    for e in range(epochs):
        x_train, y_train = shuffle_data(x_train, y_train)
        error = 0

        mini_batches = create_mini_batches(x_train, y_train, batch_size)

        for x_mini_batch, y_mini_batch in mini_batches:
            # Initialize gradients for this mini-batch
            total_grad = 0

            for x, y in zip(x_mini_batch, y_mini_batch):
                # forward
                output = predict(network, x)

                # error
                error += loss(y, output)

                # backward
                grad = loss_prime(y, output)
                total_grad += grad

            # Update parameters after processing the mini-batch
            avg_grad = total_grad / len(x_mini_batch)
            for layer in reversed(network):
                avg_grad = layer.backward(avg_grad, learning_rate)

        error /= len(x_train)
        train_accuracy[e] = calculate_accuracy(network, x_train, y_train)
        test_accuracy[e] = calculate_accuracy(network, x_test, y_test)
        mse_matrix[e] = error

        print(f"{e + 1}/{epochs}, accuracy={train_accuracy[e]}")

    best_train_accuracy = np.max(train_accuracy)
    best_test_accuracy = np.max(test_accuracy)

    plot_accuracy(train_accuracy, test_accuracy, batch_size, learning_rate, 40)
    plot_loss(mse_matrix, batch_size, learning_rate, 40)

    return best_train_accuracy, best_test_accuracy
