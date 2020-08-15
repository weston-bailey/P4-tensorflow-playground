import matplotlib.pyplot as plt

# show accuracy over epochs
def plot_accuracy(history):
  plt.plot(history.history['accuracy'], 'C1', label='accuracy')
  plt.legend()
  plt.ylabel('epochs')
  plt.show()

# show loss over epochs
def plot_loss(history):
  plt.plot(history.history['loss'], 'C2', label='loss')
  plt.legend()
  plt.ylabel('epochs')
  plt.show()

# display img used for prediction
def plot_predict(img):
  plt.imshow(img, cmap=plt.get_cmap('gray'))
  plt.show()
