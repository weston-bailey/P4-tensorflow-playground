from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[4], cmap=plt.get_cmap('gray'))

plt.show()