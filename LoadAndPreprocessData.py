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

# Convert into numpy arrays and reshape.
images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Store all the preprocessed data onto a file.
image_pickle = open("training_images.pickle", "wb")
pickle.dump(images, image_pickle)
image_pickle.close()

label_pickle = open("training_labels.pickle", "wb")
pickle.dump(labels, label_pickle)
label_pickle.close()
