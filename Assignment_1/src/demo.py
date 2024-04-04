from helpers import *
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import time

# Define the path to your CIFAR-10 data file
data_file = r'C:\Code\Python\Neural\cifar-10-batches-py'

# Load CIFAR-10 data using the provided function
(X_train, y_train), (X_test, y_test) = unpickle_all_data(data_file)

# Flatten the image data
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Suppress the LOKY warning about the number of physical cores
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Set the desired number of cores


def train_and_evaluate_knn(X_train, y_train, X_test, y_test, k=1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def train_and_evaluate_nearest_centroid(X_train, y_train, X_test, y_test):
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(X_train, y_train)
    y_pred = nearest_centroid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


start_time = time.time()

# Train and evaluate the K-Nearest Neighbors classifiers
accuracy_knn1 = train_and_evaluate_knn(X_train, y_train, X_test, y_test, k=1)
print("K-Nearest Neighbors (k=1) Accuracy: {:.4f}".format(accuracy_knn1))

accuracy_knn3 = train_and_evaluate_knn(X_train, y_train, X_test, y_test, k=3)
print("K-Nearest Neighbors (k=3) Accuracy: {:.4f}".format(accuracy_knn3))

# Train and evaluate the Nearest Centroid classifier
accuracy_kmeans = train_and_evaluate_nearest_centroid(X_train, y_train, X_test, y_test)
print("Nearest Centroid Clustering Accuracy: {:.4f}".format(accuracy_kmeans))

print("Elapsed time: ", time.time() - start_time, "sec")
