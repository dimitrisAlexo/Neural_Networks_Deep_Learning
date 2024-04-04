# IMPORTS
import time

from sklearn.preprocessing import StandardScaler

from rbf import *

# Suppress the LOKY warning about the number of physical cores
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # Set the desired number of cores

# Define the path to your CIFAR-10 data file
data_file = r'C:\Code\Python\Neural\cifar-10-batches-py'

# Load CIFAR-10 data using the provided function
(x_train, y_train), (x_test, y_test) = load_cifar10(data_file)

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

print("----------------------------------------------------------------")

print("RBF Neural Network\n")

input_dimension = num_components
hidden_dimension = 80
output_dimension = 10

gamma = 0.001
learning_rate = 0.0001
epochs = 200

network = RBF(input_dimension, hidden_dimension, output_dimension, gamma=gamma)

start_time = time.time()

network.fit(x_train, y_train, x_test, y_test, learning_rate=learning_rate, epochs=epochs, centers_alg='kmeans')

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
