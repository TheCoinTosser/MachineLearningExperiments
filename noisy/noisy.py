##############################################################
# Multioutput classification problem - Clearing noisy images #
#                                                            #
# This code gets the MNIST images, add some artificial noise #
# to them, and train a model where the input are the noised  #
# images and the output are the original (clean) images.     #
##############################################################

from numpy import random as rnd
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from common import c_image, c_io, c_timer

# The training takes about 12 seconds on an Intel i7, 2.9Ghz. Therefore, if you don't want to wait that much every time,
# you should set this flag to True (Warning: the trained model is over 1GB)
SHOULD_PERSIST_MODEL = False

# Fetch the MNIST database
mnist = fetch_mldata('MNIST original')

# Separate the features (X) and the targets (y)
X, y = mnist['data'], mnist['target']

# Split the data into a training set and a testing set
TRAIN_TEST_THRESHOLD = 60000
X_train, y_train = X[:TRAIN_TEST_THRESHOLD], y[:TRAIN_TEST_THRESHOLD]
X_test, y_test = X[TRAIN_TEST_THRESHOLD:], y[TRAIN_TEST_THRESHOLD:]

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

# Read existing classifier from disk, so we don't need to re-train it.
# If first time, create the classifier, train it and persist it.
MODEL_FILE_NAME = 'train/model.md'
(classifier_from_disk_exists, classifier) = c_io.read(MODEL_FILE_NAME)
if not classifier_from_disk_exists:

    classifier = KNeighborsClassifier()

    timer = c_timer.Timer()
    print("Training classifier...")
    classifier.fit(X_train_noised, X_train)
    print("Classifier successfully trained! Took %d seconds." % timer.stop())

    if SHOULD_PERSIST_MODEL:
        print("Persisting model...")
        c_io.persist(MODEL_FILE_NAME, classifier)
        print("Model successfully persisted!")


RANDOM_TEST_INDEX = 3
noised_test_image = X_test_noised[RANDOM_TEST_INDEX]
predicted_clean_digit = classifier.predict([noised_test_image])

c_image.show_mnist_image(noised_test_image)
c_image.show_mnist_image(predicted_clean_digit)
