import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D
import numpy as np
import pickle

# Read the preprocessed images data.
image_pickle = open("training_images.pickle", "rb")
images = pickle.load(image_pickle)
image_pickle.close()

label_pickle = open("training_labels.pickle", "rb")
labels = pickle.load(label_pickle)
label_pickle.close()

# Convert to numpy arrays and scale.
images = np.array(images / 255.0)
labels = np.array(labels)

# Create the sequential model.
model = Sequential()

# Layer 1
model.add(Conv2D(64, (3, 3), input_shape=images.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Conv2D(64, (3, 3), input_shape=images.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Flatten())
model.add(Dense(64))

# Output Layer
model.add(Dense(42))
model.add(Activation("sigmoid"))

# Compile model.
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Train model.
model.fit(images, labels, batch_size=32, epochs=3, validation_split=0.1)
