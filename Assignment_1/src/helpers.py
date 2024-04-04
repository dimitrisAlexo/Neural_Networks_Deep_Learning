import pickle
import numpy as np
from os import listdir
from os.path import isfile, join


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
