import numpy as np
from tensorflow import keras

# needs to know number of output classes for shape
def load_data(num_classes):
  # load training and testing img dataset
  # https://keras.io/api/datasets/mnist/#load_data-function
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

  # preserve shape for prediction data
  (x_predict, y_predict) = (x_test, y_test)

  # squish data flat for use with model_create_functional() input shape
  # x_train = x_train.reshape(60000, 784).astype("float32") / 255
  # x_test = x_test.reshape(10000, 784).astype("float32") / 255

  # Scale images to the [0, 1] range
  x_train = x_train.astype("float32") / 255
  x_test = x_test.astype("float32") / 255

  # Make sure images have shape (28, 28, 1)
  x_train = np.expand_dims(x_train, -1)
  x_test = np.expand_dims(x_test, -1)

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)

  return (x_train, y_train), (x_test, y_test), (x_predict, y_predict)