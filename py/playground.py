from lib.__init__ import *


def main():
  # layers for model_create_sequential_arbitrary()
  num_hidden_layers = 10
  # output classes (0 - 9) numberset
  num_classes = 10
  # input shape = 28 x 28 px img 1 dimension matrix
  input_shape = (28, 28, 1)
  # num of training sessions
  epochs = 10
  # training batch size
  batch_size = 128
  # index of test img for prediciton
  prediction_index = 1  

  # availible models
  model = model_create_sequential_arbitrary(num_hidden_layers, input_shape, num_classes)
  # model = model_create_sequential_advanced_1(input_shape, num_classes)
  # model = model_create_sequential_advanced_2(input_shape, num_classes)

  # get and shape data for training, returns training data, testing data and prediction data
  (x_train, y_train), (x_test, y_test), (x_predict, y_predict) = load_data(num_classes)

  # make a prediction on input 
  prediction = predict_model(model, x_predict[prediction_index])
  print(f'evalutation of input: {prediction}')

  # show prediction in plot
  plot_predict(x_predict[prediction_index])

  # train and test model
  model, history, test_scores = train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs)

  # print test data
  print(f'Test loss: {test_scores[0]} Test accuracy: {test_scores[1]}')

  # display plot of accuracy
  plot_accuracy(history)

  # make a prediction on input 
  prediction = predict_model(model, x_predict[prediction_index])
  print(f'evalutation of input: {prediction}')

  # show prediction in plot
  plot_predict(x_predict[prediction_index])


if __name__=="__main__":
  main()
