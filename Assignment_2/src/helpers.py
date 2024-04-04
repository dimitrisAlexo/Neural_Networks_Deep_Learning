# IMPORTS
import pickle
from os import listdir
from os.path import isfile, join

import numpy as np
from matplotlib import pyplot as plt


# Function to unpickle the whole of the dataset
def unpickle_all_data(directory):
    train = dict()
    test = dict()
    train_x = []
    test_x = []

    for filename in listdir(directory):
        if isfile(join(directory, filename)):

            if 'data_batch' in filename:

                with open(directory + '/' + filename, 'rb') as fo:
                    data = pickle.load(fo, encoding='bytes')

                if 'data' not in train:
                    train['data'] = data[b'data']
                    train['labels'] = np.array(data[b'labels'])
                else:
                    train['data'] = np.concatenate((train['data'], data[b'data']))
                    train['labels'] = np.concatenate((train['labels'], data[b'labels']))

            elif 'test_batch' in filename:

                with open(directory + '/' + filename, 'rb') as fo:
                    data = pickle.load(fo, encoding='bytes')

                test['data'] = data[b'data']
                test['labels'] = data[b'labels']

    # Manipulate the data to the proper format
    for image in train['data']:
        train_x.append(np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0)))
    train_y = [label for label in train['labels']]

    for image in test['data']:
        test_x.append(np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0)))
    test_y = [label for label in test['labels']]

    # Transform the data to np array format
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return (train_x, train_y), (test_x, test_y)


def plot_accuracy(train_accuracy_matrix, test_accuracy_matrix, batch_size, learning_rate, num_hidden_neurons):
    """
    Plots both train and test accuracies against epoch and includes parameters in the title.

    Parameters:
    - train_accuracy_matrix: Matrix containing train accuracy values for each epoch.
    - test_accuracy_matrix: Matrix containing test accuracy values for each epoch.
    - batch_size: Size of each mini-batch.
    - learning_rate: Learning rate for gradient descent.
    - num_hidden_neurons: Number of hidden neurons in the network.
    """

    epochs = len(train_accuracy_matrix)

    plt.figure()
    plt.plot(range(1, epochs + 1), train_accuracy_matrix, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracy_matrix, label='Test Accuracy')
    plt.title(f'Accuracy Plot (Batch Size: {batch_size}, Learning Rate: {learning_rate}, '
              f'Hidden Neurons: {num_hidden_neurons})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss(mse_matrix, batch_size, learning_rate, num_hidden_neurons):
    """
    Plots MSE against epoch and includes parameters in the title.

    Parameters:
    - mse_matrix: Matrix containing MSE values for each epoch.
    - batch_size: Size of each mini-batch.
    - learning_rate: Learning rate for gradient descent.
    - num_hidden_neurons: Number of hidden neurons in the network.
    """

    epochs = len(mse_matrix)

    plt.figure()
    plt.plot(range(1, epochs + 1), mse_matrix, label='Mean Squared Error (MSE)')
    plt.title(f'MSE Plot (Batch Size: {batch_size}, Learning Rate: {learning_rate}, '
              f'Hidden Neurons: {num_hidden_neurons})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()
