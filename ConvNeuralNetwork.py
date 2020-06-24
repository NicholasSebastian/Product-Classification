import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D
import numpy as np
import pickle

# Read the preprocessed images data.
dataset_pickle = open("dataset.pickle", "rb")
images, labels, test_images, test_labels = pickle.load(dataset_pickle)
dataset_pickle.close()

# Convert to numpy arrays and scale.
images = np.array(images / 255.0)
labels = np.array(labels)
test_images = np.array(test_images / 255.0)
test_labels = np.array(test_labels)

# Create the sequential model.
model = Sequential()

# Layer 1
model.add(Conv2D(64, (5, 5), input_shape=images.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Conv2D(64, (5, 5), input_shape=images.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Flatten())
model.add(Dense(64))

# Output Layer
model.add(Dense(42))
model.add(Activation("sigmoid"))

# Compile model.
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Train model.
model.fit(images, labels, batch_size=128, epochs=3, validation_split=0.1)

# Evaluate model.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Accuracy: {test_acc}\nTest Loss: {test_loss}")
