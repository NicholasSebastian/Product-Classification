import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
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

# Model.
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=images.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(42))
model.add(Activation("softmax"))

# Compile model.
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# Train model.
model.fit(images, labels, batch_size=64, epochs=10,
          verbose=1, validation_split=0.1)

# Evaluate model.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Accuracy: {test_acc}\nTest Loss: {test_loss}")

# Save the model.
model.save("ProductDetection.model")
