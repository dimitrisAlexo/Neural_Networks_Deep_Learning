# IMPORTS
import os
import pickle

import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar10(data_path):
    x_train, y_train = [], []
    x_test, y_test = [], []

    # Load training data
    for i in range(1, 6):
        file_path = os.path.join(data_path, f'data_batch_{i}')
        batch_data = unpickle(file_path)
        x_train.append(batch_data[b'data'])
        y_train.extend(batch_data[b'labels'])

    # Load test data
    test_file_path = os.path.join(data_path, 'test_batch')
    test_batch_data = unpickle(test_file_path)
    x_test.append(test_batch_data[b'data'])
    y_test.extend(test_batch_data[b'labels'])

    # Concatenate batches
    x_train = np.concatenate(x_train)
    x_test = np.concatenate(x_test)

    # Reshape and normalize pixel values
    x_train = x_train.reshape((len(x_train), 3, 32, 32)).transpose(0, 2, 3, 1)
    x_test = x_test.reshape((len(x_test), 3, 32, 32)).transpose(0, 2, 3, 1)

    # Convert to numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Convert y_train and y_test to categorical
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_test, y_test)


def find_num_of_components(x_train, x_train_standardized, retain_percentile):
    # Perform PCA
    pca = PCA()
    pca.fit_transform(x_train_standardized)
    # Find the number of components to retain 99% of the variance
    cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_explained_variance_ratio >= retain_percentile) + 1
    return num_components


def plot_accuracy(train_accuracies, test_accuracies):
    """
    Plot training and test accuracies in the same graph.

    Parameters:
    - train_accuracies: List of training accuracies for each epoch.
    - test_accuracies: List of test accuracies for each epoch.
    """
    epochs = range(1, len(train_accuracies) + 1)

    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy vs. Epochs')
    plt.legend()
    plt.show()


def plot_mse(mse_losses):
    """
    Plot Mean Squared Error (MSE) in a graph.

    Parameters:
    - mse_losses: List of MSE losses for each epoch.
    """
    epochs = range(1, len(mse_losses) + 1)

    plt.plot(epochs, mse_losses, label='MSE Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Mean Squared Error (MSE) vs. Epochs')
    plt.legend()
    plt.show()


def train_and_evaluate_knn(X_train, y_train, X_test, y_test, k=1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, knn.predict(X_train))
    test_accuracy = accuracy_score(y_test, knn.predict(X_test))
    return train_accuracy, test_accuracy


def train_and_evaluate_nearest_centroid(X_train, y_train, X_test, y_test):
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, nearest_centroid.predict(X_train))
    test_accuracy = accuracy_score(y_test, nearest_centroid.predict(X_test))
    return train_accuracy, test_accuracy
