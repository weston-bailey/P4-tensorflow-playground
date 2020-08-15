import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# for training and testing models. returns trainded model and training history data
def train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs):
  # train model on data
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

  # test model on test dataset, print the evaluation
  test_scores = model.evaluate(x_test, y_test, verbose=1)

  # return 
  return model, history, test_scores

# returns prediciton of input img
def predict_model(model, img):
  # format img
  # TODO variable shape input
  img = img.reshape(1, 28, 28, 1).astype("float32") / 255

  # array of predictions
  predictions = model.predict(img)

  # return index of highest prediction
  return np.argmax(predictions)

# Sequential syntax models
# https://keras.io/guides/sequential_model/

# specify amount of hidden layers beyond input and ouput, input shape, number of classes for ouput
def model_create_sequential_arbitrary(num_hidden_layers, input_shape, num_classes):
  # create new model
  model = keras.Sequential()
  
  # using core input and dense layers 
  # https://keras.io/api/layers/core_layers/
  # using flatten reshaping later
  # https://keras.io/api/layers/reshaping_layers/

  # input layer, and flatten method for CNN
  model.add(keras.Input(shape=input_shape))
  model.add(layers.Flatten())

  # create hidden dense layers
  for _ in range(num_hidden_layers):
    model.add(layers.Dense(128, activation='relu'))
  
  # output layer 
  model.add(layers.Dense(num_classes, activation='softmax'))  

  # compile model
  model.compile(
    loss="categorical_crossentropy", 
    optimizer="adam", 
    metrics=["accuracy"]
  )

  # print model summary in console
  model.summary()

  # welcome to the world, my friend
  return model

# static hidden layers, specify input shape, number of classes for ouput
def model_create_sequential_advanced_1(input_shape, num_classes):
  # create new model
  model = keras.Sequential()

  # alt model
  # using conv2d convolution layers
  # https://keras.io/api/layers/convolution_layers/
  # maxpooling2d maxpooling layers
  # https://keras.io/api/layers/pooling_layers/
  # using core dense layers 
  # https://keras.io/api/layers/core_layers/
  # using flatten reshaping later
  # https://keras.io/api/layers/reshaping_layers/
  # using dropout regularizarion layer
  # https://keras.io/api/layers/regularization_layers/

  # 2D spatial convolution input layer 
  model.add(layers.Conv2D(30, (5, 5), input_shape=(input_shape), activation='relu'))
  model.add(layers.Conv2D(15, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D())
  model.add(layers.Dropout(0.2))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dense(50, activation='relu'))
  model.add(layers.Dense(num_classes, activation='softmax'))

  # compile model
  model.compile(
    loss="categorical_crossentropy", 
    optimizer="adam", 
    metrics=["accuracy"]
  )

  # print model summary in console
  model.summary()

  return model

# static hidden layers, specify input shape, number of classes for ouput
def model_create_sequential_advanced_2(input_shape, num_classes):
  # create new model
  model = keras.Sequential()

  # alt model
  # using conv2d convolution layers
  # https://keras.io/api/layers/convolution_layers/
  # maxpooling2d maxpooling layers
  # https://keras.io/api/layers/pooling_layers/
  # using core dense layers 
  # https://keras.io/api/layers/core_layers/
  # using flatten reshaping later
  # https://keras.io/api/layers/reshaping_layers/
  # using dropout regularizarion layer
  # https://keras.io/api/layers/regularization_layers/

  # core input layer
  model.add(keras.Input(shape=input_shape))
  model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_classes, activation="softmax"))

  # compile model
  model.compile(
    loss="categorical_crossentropy", 
    optimizer="adam", 
    metrics=["accuracy"]
  )

  # print model summary in console
  model.summary()

  return model



# Using keras functional API to create model
# to make a CNN with an arbitrary amount of hidden layers
# https://keras.io/guides/functional_api/
# https://keras.io/examples/vision/mnist_convnet/

# specify amount of layers beyond input and ouput (input shape hard coded)
# this function is unused in the playground
def model_create_functional(num_hidden_layers):
  # input node
  inputs = keras.Input(shape=(784,))

  # create node in graph of layers
  dense = layers.Dense(64, activation="relu")
  x = dense(inputs)

  # add aditional layers
  for _ in range(num_hidden_layers):
    x = layers.Dense(64, activation="relu")(x)

  # create output layer
  outputs = layers.Dense(10)(x)

  # create model
  model = keras.Model(inputs=inputs, outputs=outputs, name='minst_model')

  # print model
  model.summary()

  # compile model
  model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
  )

  # retun the model
  return model