import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter(indent=2)

# load training and testing img dataset
# https://keras.io/api/datasets/mnist/#load_data-function
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# print(np.matrix(x_test[0]))

# save data for predictions
(x_predict, y_predict) = (x_test, y_test)

# reshape data into binary 
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

# TODO export dataset in tfjs format

# Using keras functional API to create model
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

def train_model(model, x_train, y_train, batch_size, epochs):
  # compile model
  model.compile(
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=keras.optimizers.RMSprop(),
      metrics=["accuracy"],
  )

  # train model on data
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

  # return 
  return model, history

# model and training consts
NUM_LAYERS = 10
EPOCHS = 10
BATCH_SIZE = 128

model = model_create_functional(NUM_LAYERS)

# print model
model.summary()

model, history = train_model(model, x_train, y_train, BATCH_SIZE, EPOCHS)

# print training history
# pp.pprint(history.history)

# plot history
# plt.plot(history.history['loss'], 'C2', label='loss')
plt.plot(history.history['accuracy'], 'C1', label='accuracy')
plt.legend()
plt.ylabel('epochs')
plt.show()

# use test data
test_scores = model.evaluate(x_test, y_test, verbose=2)
print(f'Test loss: {test_scores[0]} Test accuracy: {test_scores[1]}')

# model((x_test[0], y_test[0]))
