document.addEventListener('DOMContentLoaded', () => init())

let state = {
  hiddenLayers: 10,
  inputShape: [28, 28, 1],
  outputClasses: 10,
  epochs: 10,
  batchSize: 128,
  predictionIndex: 1
}

console.log('hello front end')
async function init() {
  // load data
  const data = await getData()

  // groom data

  // create model
  model = modelCreateArbitraryHidden()

  // fit model

  // test model

  // cast prediction

  // ????????

  // PROFIT!!!
  
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
  model.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'});
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