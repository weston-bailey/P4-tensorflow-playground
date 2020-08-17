let state = {
  hiddenLayers: 5,
  inputShape: [28, 28, 1],
  outputClasses: 10,
  epochs: 7,
  units: 128,
  batchSize: 64,
  predictionIndex: 1,
  trainDataSize: 55000,
  testDataSize: 15000,
  imageHeight: 28,
  imageWidth: 28,
  imageSize: 28 * 28,
  inputImg: undefined,
  numbers: {
    dataSetLength: 65000,
    trainTestRatio: 5 / 6,
    imgPath: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
    labelPath: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8',
    data: {
      test: [],
      train: []
    },
  },
  fashion: {
    dataSetLength: 70000,
    trainTestRatio: 6 / 7,
    imgPath: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_images.png',
    labelPath: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_labels_uint8',
    data: {
      test: [],
      train: []
    },
  },
}