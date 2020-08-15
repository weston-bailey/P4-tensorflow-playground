import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter(indent=2)

# https://keras.io/guides/sequential_model/

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

(x_predict, y_predict) = (x_test, y_test)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()

# model 
# model.add(keras.Input(shape=input_shape))
# model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(num_classes, activation="softmax"))

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
# model.add(layers.Conv2D(30, (5, 5), input_shape=(input_shape), activation='relu'))
# model.add(layers.Conv2D(15, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.2))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(50, activation='relu'))
# model.add(layers.Dense(num_classes, activation='softmax'))

# like the first -- this seems faster and basically as good?
model.add(keras.Input(shape=input_shape))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

EPOCHS = 10
BATCH_SIZE = 128

def train_model(model, x_train, y_train, batch_size, epochs):
  # train model on data
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

  # return 
  return model, history

model, history = train_model(model, x_train, y_train, BATCH_SIZE, EPOCHS)

# save the model
model.save('model.h5')

# plt.plot(history.history['accuracy'], 'C1', label='accuracy')
# plt.legend()
# plt.ylabel('epochs')
# plt.draw()

# test model on test dataset, print the evaluation
test_scores = model.evaluate(x_test, y_test, verbose=1)
print(f'Test loss: {test_scores[0]} Test accuracy: {test_scores[1]}')

# index of test img for prediciton
prediction_index = 1

# reshape a test img for prediction
predict = x_predict[prediction_index].reshape(1, 28, 28, 1).astype("float32") / 255

plt.imshow(x_predict[prediction_index], cmap=plt.get_cmap('gray'))

predict = model.predict(predict)
# find the largerest pridection from the softmax array
predict = np.argmax(predict)

print(f'evalutation of input: {predict}')

# call show so plot doesn't dissapear
plt.show()