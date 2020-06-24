import os
import random
import cv2
import numpy as np
import pickle

DIRECTORY = "shopee-product-detection-dataset"
TRAIN_PATH = os.path.join(DIRECTORY, "train", "train")
TEST_PATH = os.path.join(DIRECTORY, "test", "test")

IMG_SIZE = 100

# Generate the dataset from the training images.
training_data = []
for category in os.listdir(TRAIN_PATH):
    if category == ".DS_Store":
        continue
    path = os.path.join(TRAIN_PATH, category)
    for image in os.listdir(path):
        try:
            image_file = os.path.join(path, image)
            image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([resized_array, int(category)])
        except Exception as e:
            pass

# Shuffle the dataset.
random.shuffle(training_data)

# Separate the dataset by its respective image arrays and labels.
images = []
labels = []
for image_array, category in training_data:
    images.append(image_array)
    labels.append(category)

# Split the arrays into 9:1, training data and testing data respectively.
test_images = []
test_labels = []
ratio = int(len(images) * 0.1)
for _ in range(ratio):
    test_images.append(images.pop())
    test_labels.append(labels.pop())

# Reshape image arrays.
images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Store all the preprocessed data onto a file.
dataset = (images, labels, test_images, test_labels)
dataset_pickle = open("dataset.pickle", "wb")
pickle.dump(dataset, dataset_pickle)
dataset_pickle.close()
