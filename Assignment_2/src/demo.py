# IMPORTS
import time

from keras.utils import np_utils

from activations import *
from dense import *
from helpers import *
from losses import *
from network import *


def preprocess_data(x, y, limit, flag=False):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 32 * 32 * 3, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    if flag:
        return x[:limit], y[:limit]
    else:
        return x, y


def calculate_accuracy(network, x_test, y_test):
    correct_predictions = 0

    for x, y_true in zip(x_test, y_test):
        output = predict(network, x)
        predicted_label = np.argmax(output)
        true_label = np.argmax(y_true)

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / len(y_test)
    return accuracy


# Define the path to your CIFAR-10 data file
data_file = r'C:\Code\Python\Neural\cifar-10-batches-py'

# Load CIFAR-10 data using the provided function
(x_train, y_train), (x_test, y_test) = unpickle_all_data(data_file)

start_time = time.time()

x_train, y_train = preprocess_data(x_train, y_train, 1000, False)
x_test, y_test = preprocess_data(x_test, y_test, 100, False)

# neural network
network = [
    Dense(32 * 32 * 3, 40),
    Sigmoid(),
    Dense(40, 10),
    Sigmoid()
]

# train
best_train_accuracy, best_test_accuracy = train_mini_batch(network, mse, mse_prime, x_train, y_train, x_test, y_test,
                                                           epochs=35, learning_rate=0.63, batch_size=1)

# # demonstrate
# for x, y in zip(x_test, y_test):
#     output = predict(network, x)
#     print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

print('Optimal Training Accuracy:', best_train_accuracy)
print('Optimal Testing Accuracy:', best_test_accuracy)

print("Elapsed time: ", time.time() - start_time, "sec")
