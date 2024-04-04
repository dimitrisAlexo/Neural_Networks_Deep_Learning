# IMPORTS
import numpy as np
import os
import pickle
import time
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from helpers import *
from svm import *

start_time = time.time()

# Suppress the LOKY warning about the number of physical cores
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Set the desired number of cores

# Define the path to your CIFAR-10 data file
data_file = r'C:\Code\Python\Neural\cifar-10-batches-py'

# Load CIFAR-10 data using the provided function
(x_train, y_train), (x_test, y_test) = load_cifar10(data_file)

# Filter only the samples with labels 0 or 1
train_mask = (y_train == 0) | (y_train == 1)
test_mask = (y_test == 0) | (y_test == 1)

x_train = x_train[train_mask]
y_train = y_train[train_mask]
x_test = x_test[test_mask]
y_test = y_test[test_mask]

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten images
x_train_flatten = x_train.reshape(-1, 3072).astype('float32')
x_test_flatten = x_test.reshape(-1, 3072).astype('float32')

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
x_train_standardized = scaler.fit_transform(x_train_flatten)
x_test_standardized = scaler.transform(x_test_flatten)

# num_components = find_num_of_components(x_train, x_train_standardized, 0.90)
num_components = 103

# Perform PCA
pca = PCA(num_components)
x_train = pca.fit_transform(x_train_standardized)
x_test = pca.transform(x_test_standardized)

# Randomly select 1000 samples for training
random_indices_train = np.random.choice(x_train.shape[0], 2000, replace=False)
x_train = x_train[random_indices_train].astype('float32')
y_train = y_train[random_indices_train]

# Randomly select 100 samples for testing
random_indices_test = np.random.choice(x_test.shape[0], 500, replace=False)
x_test = x_test[random_indices_test].astype('float32')
y_test = y_test[random_indices_test]

print("----------------------------------------------------------------")

print("Support Vector Machine\n")

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

clf = SVM(kernel="rbf")
clf.train(x_train, y_train)

print("Train accuracy: ", clf.calculate_accuracy(x_train, y_train))
print("Test accuracy: ", clf.calculate_accuracy(x_test, y_test))

t1 = time.time()
print("Elapsed time: ", t1 - start_time, "sec")

print("----------------------------------------------------------------")

print("KNN (k=1)\n")

# Train and evaluate the K-Nearest Neighbors classifiers
train_accuracy_knn1, test_accuracy_knn1 = train_and_evaluate_knn(x_train, y_train, x_test, y_test, k=1)
print("K-Nearest Neighbors (k=1) Train Accuracy: {:.4f}".format(train_accuracy_knn1))
print("K-Nearest Neighbors (k=1) Test Accuracy: {:.4f}".format(test_accuracy_knn1))

t2 = time.time()
print("Elapsed time: ", t2 - t1, "sec")

print("----------------------------------------------------------------")

print("KNN (k=3)\n")

train_accuracy_knn3, test_accuracy_knn3 = train_and_evaluate_knn(x_train, y_train, x_test, y_test, k=3)
print("K-Nearest Neighbors (k=3) Train Accuracy: {:.4f}".format(train_accuracy_knn3))
print("K-Nearest Neighbors (k=3) Test Accuracy: {:.4f}".format(test_accuracy_knn3))

t3 = time.time()
print("Elapsed time: ", t3 - t2, "sec")

print("----------------------------------------------------------------")

print("Nearest Centroid\n")

# Train and evaluate the Nearest Centroid classifier
train_accuracy_kmeans, test_accuracy_kmeans = train_and_evaluate_nearest_centroid(x_train, y_train, x_test, y_test)
print("Nearest Centroid Clustering Train Accuracy: {:.4f}".format(train_accuracy_kmeans))
print("Nearest Centroid Clustering Test Accuracy: {:.4f}".format(test_accuracy_kmeans))

print("Elapsed time: ", time.time() - t3, "sec")

print("----------------------------------------------------------------")
