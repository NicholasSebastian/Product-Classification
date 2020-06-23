import tensorflow as tf
import pickle

# Read the preprocessed images data.
image_pickle = open("training_images.pickle", "rb")
images = pickle.load(image_pickle)
image_pickle.close()

label_pickle = open("training_labels.pickle", "rb")
labels = pickle.load(label_pickle)
label_pickle.close()

# Code goes here...
