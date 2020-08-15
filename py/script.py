import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Using keras functional API 
# https://keras.io/guides/functional_api/
# https://keras.io/examples/vision/mnist_convnet/

# specify amount of layers beyond input and ouput
def model_create_functional(num_layers):
  # input node
  inputs = keras.Input(shape=(784,))

  # create node in graph of layers
  dense = layers.Dense(64, activation="relu")
  x = dense(inputs)

  # add aditional layers
  for _ in range(num_layers):
    x = layers.Dense(64, activation="relu")(x)

  # create output layer
  outputs = layers.Dense(10)(x)

  # create model
  model = keras.Model(inputs=inputs, outputs=outputs, name='minst_model')

  # retun the model
  return model

model = model_create_functional(1)

# print model
model.summary()




# TODO export dataset in tfjs format

# TODO create tf model