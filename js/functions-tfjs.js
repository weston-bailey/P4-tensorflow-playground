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
  // compile
  model.compile({loss: 'categoricalCrossentropy', optimizer: 'adam', metrics:['accuracy']});
  // summary in console
  model.summary()

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
  // compile
  model.compile({loss: 'categoricalCrossentropy', optimizer: 'adam', metrics:['accuracy']});
  // summary in console
  model.summary()
  return model
}

function modelPredictCanvas(model, imageTensor){

  console.log(imageTensor)
  const testImage = imageTensor;
  const preds = model.predict(testImage).argMax(-1);
  console.log(preds)
  console.log(model)
  let p = document.getElementById("pedict-num")
  p.innerText = `I predicted that the image your drew is a ${preds.arraySync()}`
  // imageTensor.dispose()
}

function modelPredict(model, data){
  // get test data
  const [xTest, yTest] = data
  // x_predict = xTest.slice([0, 0, 0, 0], [1, state.imageHeight, state.imageWidth, 1])
  let rand = Math.floor(Math.random() * 1000)
  x_predict = xTest.slice([rand, 0, 0, 0], [1, state.imageHeight, state.imageWidth, 1])
  y_predict = yTest.slice([rand, 0], [1, state.outputClasses]).argMax(-1)


  const preds = model.predict(x_predict).argMax(-1);
  console.log(preds)

  console.log(preds.arraySync()) // this is the prediction value

  // show prediction on canvas
  let div = document.getElementById('predict')
  const canvas = document.createElement('canvas');
  ctx = canvas.getContext('2d')
  let p = document.getElementById("pedict-num")
  p.innerText = `I predicted the above image is a ${preds.arraySync()}\n~~~~~~~~~~~~~~\naccording to the dataset it is a ${y_predict.arraySync()}`
  // canvas.style = 'margin: 4px;';
  x_predict = x_predict.reshape([28, 28, 1])
  
  
  tf.browser.toPixels(x_predict, canvas)
  div.appendChild(canvas);
}
// called at end of fitting epochs
async function onBatchEnd(batch, logs) {
  // TODO .toFixed(1)
  let show = document.getElementById("learn-logs")
  if(logs.batch % 10 == 0) show.innerText = `batch: ${logs.batch}\nloss: ${logs.loss}\naccuracy: ${logs.acc}`
  console.log('logs', logs);

  // TODO all the graphing goes in this function

  // returns a promise that resolve when a requestAnimationFrame has completed
  await tf.nextFrame();
}

// train model against data
async function fitModel(model, xTrain, yTrain, xTest, yTest) {
  info = await model.fit(xTrain, yTrain, {
    batchSize: state.batchSize,
    validationData: [xTest, yTest],
    epochs: state.epochs,
    shuffle: true,
    callbacks: { onBatchEnd }
  });
  return model, info
}