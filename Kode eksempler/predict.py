import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import datasets
from tensorflow.keras.models import load_model


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def find_target_layer(model):
    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
    for layer in reversed(model.layers):
        # check to see if the layer has a 4D output
        if len(layer.output.shape) == 4:
            print(layer.name)
            return layer.name
    # otherwise, we could not find a 4D layer so the GradCAM
    # algorithm cannot be applied
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1

# train_images, test_images = train_images / 255.0, test_images / 255.0

saved_model_path = "CPP/Computer Vision/2025 January/NN/savedModels/simpleCnn.keras"
model = load_model(saved_model_path)
model.summary()

image = cv2.resize(train_images[0], (32, 32))
image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=0)
prediction = model.predict(image)
print(type(prediction), prediction.shape)
print(prediction)
