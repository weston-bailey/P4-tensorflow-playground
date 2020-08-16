document.addEventListener('DOMContentLoaded', () => init())

let state = {
  hiddenLayers: 1,
  inputShape: [28, 28, 1],
  outputClasses: 10,
  epochs: 1,
  batchSize: 128,
  predictionIndex: 1,
  trainDataSize: 55000,
  testDataSize: 15000
}

console.log('hello front end')
async function init() {
  // load data
  const data = await getData()
  console.log(data)
  // groom data
  const [x_train, y_train] = formatTrainData(data)  
  const [x_test, y_test] = formatTestData(data)  

  console.log([x_train, y_train])
  console.log([x_test, y_test])

  // create model
  model = modelCreateArbitraryHidden()

  // fit model
  // model, history = fitModel(model, x_train, y_train)
  model.fit(x_train, y_train, {
    batchSize: state.batchSize,
    validationData: [x_test, y_test],
    epochs: state.epochs,
    shuffle: true,
    callbacks: { onBatchEnd }
  })
  .then(info => {
    console.log('Final accuracy', info.history.acc);
    console.log(model)
    console.log(info)
  });
  // console.log(model)
  // console.log(history)
  // test model

  // cast prediction

  // ????????

  // PROFIT!!!
  
}

function formatTestData(data) {
  const IMAGE_H = 28
  const IMAGE_W = 28
  const IMAGE_SIZE = IMAGE_H * IMAGE_W
  const N_CLASSES = 10
  const N_DATA  = 65000

  let x_test = tf.tensor4d(data.testImages, [data.testImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1])

  let y_test = tf.tensor2d( data.testLabels, [data.testLabels.length / N_CLASSES, N_CLASSES])

  return [x_test, y_test]
}

function formatTrainData(data) {

  const IMAGE_H = 28
  const IMAGE_W = 28
  const IMAGE_SIZE = IMAGE_H * IMAGE_W
  const N_CLASSES = 10
  const N_DATA  = 65000

  let x_test = tf.tensor4d(data.testImages, [data.testImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1])

  let y_test = tf.tensor2d( data.testLabels, [data.testLabels.length / N_CLASSES, N_CLASSES])

  let x_train = tf.tensor4d(data.trainImages, [data.trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1])

  let y_train = tf.tensor2d(data.trainLabels, [data.trainLabels.length / N_CLASSES, N_CLASSES])

  return [x_train, y_train]
}

function onBatchEnd(batch, logs) {
  console.log('logs', logs);
}


async function fitModel(model, x_train, y_train) {
}

function modelCreateArbitraryHidden() {
  //Create the model
  const model = tf.sequential();
  // add input layer 
  model.add(tf.layers.flatten({ inputShape: state.inputShape }));
  // add hidden layers
  for(let i = 0; i < state.hiddenLayers; i++){
    model.add(tf.layers.dense({ units: 128, activation: 'relu' })); 
  }
  // softmax output layer
  model.add(tf.layers.dense({ units: state.outputClasses, activation: 'softmax' })); 
  // compile
  model.compile({loss: 'categoricalCrossentropy', optimizer: 'adam', metrics:['accuracy']});
  model.summary()
  return model
}

async function getData() {  
  const data = new MnistData();
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
  var ctx = document.getElementById('scatter-chartjs')
  var scatterChart = new Chart(ctx, {
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