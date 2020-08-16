document.addEventListener('DOMContentLoaded', () => init())

let state = {
  hiddenLayers: 1,
  inputShape: [28, 28, 1],
  outputClasses: 10,
  epochs: 7,
  units: 128,
  batchSize: 128,
  predictionIndex: 1,
  trainDataSize: 55000,
  testDataSize: 15000,
  imageHeight: 28,
  imageWidth: 28,
  imageSize: 28 * 28,
  numbers: {
    dataSetLength: 65000,
    trainTestRatio: 5 / 6,
    imgPath: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
    labelPath: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8',
  },
  fashion: {
    dataSetLength: 70000,
    trainTestRatio: 6 / 7,
    imgPath: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_images.png',
    labelPath: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_labels_uint8'
  }
}

// TODOS 

// port another model

// resize displayed images

// graph data

// add canvas to draw on



async function init() {
  // load data
  const numbersData = await getData({
    imageSize: state.imageSize,
    outputClasses: state.outputClasses,
    dataSetLength: state.numbers.dataSetLength,
    trainTestRatio: state.numbers.trainTestRatio,
    imgPath: state.numbers.imgPath,
    labelPath: state.numbers.labelPath
  });
  const fashionData = await getData({
    imageSize: state.imageSize,
    outputClasses: state.outputClasses,
    dataSetLength: state.fashion.dataSetLength,
    trainTestRatio: state.fashion.trainTestRatio,
    imgPath: state.fashion.imgPath,
    labelPath: state.fashion.labelPath
  });
  // groom data
  const [x_train, y_train] = formatTrainData(numbersData)  
  const [x_test, y_test] = formatTestData(numbersData)  

  console.log([x_train, y_train])
  console.log([x_test, y_test])

  // create model
  model = createArbitraryDenseModel()
  // model = createConvNetModel()

  // fit model
  // model, history = fitModel(model, x_train, y_train)
  model, info = await fitModel(model, x_train, y_train, x_test, y_test)

  console.log('Final accuracy', info.history.acc);
  console.log(model)
  console.log(info)
  modelPredict([x_test, y_test])
  
}

function modelPredict(data){
  // get test data
  const [x_test, y_test] = data
  // x_predict = x_test.slice([0, 0, 0, 0], [1, state.imageHeight, state.imageWidth, 1])
  let rand = Math.floor(Math.random() * 1000)
  x_predict = x_test.slice([rand, 0, 0, 0], [1, state.imageHeight, state.imageWidth, 1])
  y_predict = y_test.slice([rand, 0], [1, state.outputClasses]).argMax(-1)


  const preds = model.predict(x_predict).argMax(-1);
  console.log(preds)

  console.log(preds.arraySync()) // this is the prediction value

  // show prediction on canvas
  var canvas = document.getElementById('predict')
  var p = document.getElementById("pedict-num")
  p.innerText = `I predicted the above image is a ${preds.arraySync()}\n~~~~~~~~~~~~~~\naccording to the dataset it is a ${y_predict.arraySync()}`
  ctx = canvas.getContext('2d')
  canvas.width = 28;
  canvas.height = 28;
  canvas.style = 'margin: 4px;';
  x_predict = x_predict.reshape([28, 28, 1]);
  tf.browser.toPixels(x_predict, canvas)
}

function formatTestData(data) {
  // iamges 
  let x_test = tf.tensor4d(data.testImages, [data.testImages.length / state.imageSize, state.imageHeight, state.imageWidth, 1])
  // labels
  let y_test = tf.tensor2d( data.testLabels, [data.testLabels.length / state.outputClasses, state.outputClasses])
  return [x_test, y_test]
}

function formatTrainData(data) {
  // images
  let x_train = tf.tensor4d(data.trainImages, [data.trainImages.length / state.imageSize, state.imageHeight, state.imageWidth, 1])
  // labels
  let y_train = tf.tensor2d(data.trainLabels, [data.trainLabels.length / state.outputClasses, state.outputClasses])
  return [x_train, y_train]
}

// called at end of fitting epochs
function onBatchEnd(batch, logs) {
  var show = document.getElementById("learn-logs")
  if(logs.batch % 10 == 0) show.innerText = `batch: ${logs.batch}\nloss: ${logs.loss}\naccuracy: ${logs.acc}`
  console.log('logs', logs);
}

// train model against data
async function fitModel(model, x_train, y_train, x_test, y_test) {
  info = await model.fit(x_train, y_train, {
    batchSize: state.batchSize,
    validationData: [x_test, y_test],
    epochs: state.epochs,
    shuffle: true,
    callbacks: { onBatchEnd }
  });
  return model, info
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

async function getData(args) {  
  const data = new MnistData(args);
  await data.load()
  return data
}

/* CODE GRAVEYARD 

async function run() {  
  const data = new MnistData();
  await data.load();
  await showExamples(data);
}

async function showExamples(data) {
  // Create a container in the visor
  const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const arr_x = [-1, -2,  0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 6]
  const arr_y = [-1, -2, -1, 1, 1, 0, 2, 3, 1, 3, 2, 4, 3, 6, 5]

 function weave(arrX, arrY) { 
    return arrX.map((x, i) => { 
      return { 'x':x, 'y':arrY[i] }
    }); 
  }
  const toy_data = weave(arr_x, arr_y)
  console.log(toy_data)
  const label = 'toy data'

  // CHART.JS VIS
  var canvas = document.getElementById('scatter-chartjs')
  var scatterChart = new Chart(canvas, {
      type: 'bubble',
      data: {
          datasets: [{
              data: toy_data,
              label: label,
              backgroundColor: 'blue'}]
      },
      options: {
          responsive: false
      }
  })

})

*/