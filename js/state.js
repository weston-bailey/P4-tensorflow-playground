let state = {
  // current dataset
  dataSet: 'numbers',
  // current model type
  modelType: 'dense',
  // dense layer config
  hiddenLayers: 1,
  units: 128,
  // the model
  model: undefined,
  // training config
  batchSize: 128,
  epochs: 1,
  learningRate: .001,
  // image tensor of canvas TODO use 
  inputImg: undefined,
  // TODO may be deprecated
  predictionIndex: 1,
  // dataset specifcs
  numbers: {
    // formatted tensors
    data: {
      test: [],
      train: []
    },
    // constants
    dataSetLength: 65000,
    trainTestRatio: 5 / 6,
    imgPath: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
    labelPath: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8',
  },
  fashion: {
    data: {
      test: [],
      train: []
    },
    dataSetLength: 70000,
    trainTestRatio: 6 / 7,
    imgPath: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_images.png',
    labelPath: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_labels_uint8',
  },
  predictImageSet: {},
  currentEpoch: 0,
  // below are constants
  inputShape: [28, 28, 1],
  outputClasses: 10,
  trainDataSize: 55000,
  testDataSize: 15000,
  imageHeight: 28,
  imageWidth: 28,
  imageSize: 28 * 28,
}