from numpy import random as rnd
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from common import common

# Fetch the MNIST database
mnist = fetch_mldata('MNIST original')

# Separate the features (X) and the targets (y)
X, y = mnist['data'], mnist['target']

# Split the data into a training set and a testing set
TRAIN_TEST_THRESHOLD = 60000
X_train, y_train = X[:TRAIN_TEST_THRESHOLD], y[:TRAIN_TEST_THRESHOLD]
X_test, y_test = X[TRAIN_TEST_THRESHOLD:], y[TRAIN_TEST_THRESHOLD:]
print(X_train.shape)
print(X_test.shape)

# Generate random noise
SMALLEST_NOISE_COLOR = 0
HIGHEST_NOISE_COLOR = 250
FLATTENED_IMAGE_SIZE = 784
noise_train = rnd.randint(SMALLEST_NOISE_COLOR,
                          HIGHEST_NOISE_COLOR,
                          (len(X_train), FLATTENED_IMAGE_SIZE))
noise_test = rnd.randint(SMALLEST_NOISE_COLOR,
                         HIGHEST_NOISE_COLOR,
                         (len(X_test), FLATTENED_IMAGE_SIZE))

# Create new train and test "noised" sets
X_train_noised = X_train + noise_train
X_test_noised = X_test + noise_test

# Create and train the classifier
classifier = KNeighborsClassifier()
classifier.fit(X_train_noised, X_train)

TEST_INDEX = 2
noised_test_image = X_test_noised[TEST_INDEX]
predicted_clean_digit = classifier.predict([noised_test_image])

common.show_mnist_image(noised_test_image)
common.show_mnist_image(predicted_clean_digit)
