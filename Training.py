import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
import os

from ConvModel import create_conv_model

print("Reading dataset...")
dataset_pickle = open("dataset.pickle", "rb")
images, labels, test_images, test_labels = pickle.load(dataset_pickle)
dataset_pickle.close()

labels = to_categorical(labels)
test_labels = to_categorical(test_labels)

print("Creating model...")
model = create_conv_model(images.shape[1:])
model.summary()

checkpoint_path = "checkpoints/epoch-{epoch:03d}.ckpt"
callback = ModelCheckpoint(filepath=checkpoint_path,
                           save_weights_only=True, verbose=1)

print("Training model...")
model.fit(images, labels, batch_size=256, epochs=15,
          verbose=1, validation_data=(test_images, test_labels), callbacks=[callback])

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Accuracy: {test_acc}\nTest Loss: {test_loss}")

print("Saving model...")
model.save("ConvModel_Trained.h5")
