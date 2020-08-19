async function getData(args) {  
  const data = new MnistData(args);
  await data.load()
  return data
}

function formatTestData(data) {
  // iamges 
  let xTest = tf.tensor4d(data.testImages, [data.testImages.length / state.imageSize, state.imageHeight, state.imageWidth, 1])
  // labels
  let yTest = tf.tensor2d( data.testLabels, [data.testLabels.length / state.outputClasses, state.outputClasses])
  return [xTest, yTest]
}

function formatTrainData(data) {
  // images
  let xTrain = tf.tensor4d(data.trainImages, [data.trainImages.length / state.imageSize, state.imageHeight, state.imageWidth, 1])
  // labels
  let yTrain = tf.tensor2d(data.trainLabels, [data.trainLabels.length / state.outputClasses, state.outputClasses])
  return [xTrain, yTrain]
}

// Creates a convolution nueral network model
function createConvNetModel(){
  //Create the model
  const model = tf.sequential();
  // 2d convolution input
  model.add(tf.layers.conv2d({ inputShape: state.inputShape, filters: 32, kernelSize: [3, 3], activation: 'relu'}))
  // pooling layer to downsample 
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }))
  // 2d convolution with double the filters
  model.add(tf.layers.conv2d({ filters: 32, kernelSize: [3, 3], activation: 'relu'}))
  // pooling layer to downsample 
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }))
  // flatten layer before output
  model.add(tf.layers.flatten({}));
  // add hidden layers
  for(let i = 0; i < state.hiddenLayers; i++){
    model.add(tf.layers.dense({ units: state.units, activation: 'relu' })); 
  }
  // softmax output layer
  model.add(tf.layers.dense({ units: state.outputClasses, activation: 'softmax' })); 

  let optimizer = tf.train.adam(state.learningRate)
  // compile
  model.compile({loss: 'categoricalCrossentropy', optimizer: optimizer, metrics:['accuracy']});
  // summary in console
  model.summary();
  showModelSummary(model);

  return model
}

// create a sequential desnse model with an arbitrary
// amount of hidden layers
function createArbitraryDenseModel() {
  //Create the model
  const model = tf.sequential();
  // add input layer that flattens shape
  model.add(tf.layers.flatten({ inputShape: state.inputShape }));
  // add hidden layers
  for(let i = 0; i < state.hiddenLayers; i++){
    model.add(tf.layers.dense({ units: state.units, activation: 'relu' })); 
  }
  // softmax output layer
  model.add(tf.layers.dense({ units: state.outputClasses, activation: 'softmax' })); 

  let optimizer = tf.train.adam(state.learningRate)
  // compile
  model.compile({loss: 'categoricalCrossentropy', optimizer: optimizer, metrics:['accuracy']});
  // summary in console
  model.summary()
  showModelSummary(model);

  return model
}

async function modelPredictCanvas(model, imageTensor){
  const testImage = imageTensor;
  const preds = await model.predict(testImage).argMax(-1);
  showPrediction(preds)
  // imageTensor.dispose()
}

// called at end of fitting epochs
async function onBatchEnd(batch, logs) {
  if(logs.batch == 0) {
    state.currentEpoch++
    showFittingTrainingStatus()
  }
  if(logs.batch % 10 == 0) showEpochTrainingStatus(logs)
  await tf.nextFrame();
}

// train model against data
async function fitModel(model, xTrain, yTrain, xTest, yTest) {
  state.currentEpoch = 0;
  state.batchAcc = [];
  state.batchLoss = [];
  info = await model.fit(xTrain, yTrain, {
    batchSize: state.batchSize,
    validationData: [xTest, yTest],
    epochs: state.epochs,
    shuffle: true,
    callbacks: { onBatchEnd }
  });
  await evaluateModel(model, xTest, yTest)

  return model, info
}

// evaluate model's accuracy TODO breakout ui stuff
async function evaluateModel(model, xTest, yTest) {
  // evaluate model
  const classNames = state.dataSet === 'numbers' ? state.numbers.classNames : state.fashion.classNames;

  // const [xTest, yTest] = state.dataSet === 'numbers' ? state.numbers.data.test : state.fashion.data.test;

  let xPredict = model.predict(xTest).argMax(1)
  let yPredict = yTest.argMax(1)
  let evaluation = await tfvis.metrics.accuracy(yPredict, xPredict);
  EVAL_ACC.innerText = `Overall Accuracy: ${evaluation.toFixed(2)}`;
  const classaAcc = await tfvis.metrics.perClassAccuracy(yPredict,xPredict);
  tfvis.show.perClassAccuracy(MODEL_EVAL_TABLE, classaAcc, classNames)
  return evaluation, [xPredict, yPredict]
}

