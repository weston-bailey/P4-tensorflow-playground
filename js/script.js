document.addEventListener('DOMContentLoaded', () => init());
/* ~~~~~~~~~~~~~~~~~~~~~~ event listeners ~~~~~~~~~~~~~~~~~~~~~~ */
// select dataset
const DATA_SELECT = document.getElementById('data-select');
DATA_SELECT.addEventListener('change', e => handleDataSelect(e));
// model creation
const MODEL_SELECT = document.getElementById('model-select');
const UNITS_NUMBER = document.getElementById('units-number');
const HIDDEN_LAYERS_NUMBER = document.getElementById('hidden-layers-number');
const MODEL_CREATE_DESTROY = document.getElementById('model-create-destory');
MODEL_SELECT.addEventListener('change', e => handleModelSelect(e));
UNITS_NUMBER.addEventListener('change', e => handleUnitsNumber(e));
HIDDEN_LAYERS_NUMBER.addEventListener('change', e => handleHiddenLayersNumber(e))
MODEL_CREATE_DESTROY.addEventListener('click', () => handleModelCreateDestroy());
// train form
const BATCH_NUMBER = document.getElementById('batch-number');
const EPOCHS_NUMBER = document.getElementById('epochs-number');
const LEARNING_RATE_NUMBER = document.getElementById('learning-rate-number');
const TRAINING_START_PAUSE = document.getElementById('training-start-pause');
const TRAINING_STOP = document.getElementById('training-stop');
BATCH_NUMBER.addEventListener('change', e => handleBatchNumber(e));
EPOCHS_NUMBER.addEventListener('change', e => handleEpochsNumber(e));
LEARNING_RATE_NUMBER.addEventListener('change', e => handleLearningRateNumber(e));
TRAINING_START_PAUSE.addEventListener('click', () => handleTrainingStartPause());
TRAINING_STOP.addEventListener('click', () => handleTrainingStop());



// input canvas 
const CAST_TO_IMAGE = document.getElementById("cast-to-image");
const CLEAR_INPUT_CANVAS = document.getElementById("clear-input-canvas");
CAST_TO_IMAGE.addEventListener('click', () => handlePredict());
CLEAR_INPUT_CANVAS.addEventListener('click', () => inputCanvas.clear());


// TODOS 

// resize displayed images

// graph data

// 


let inputCanvas

let model

async function init() {
  inputCanvas = new InputCanvas({
    canvas: 'input-canvas',
    width: 400,
    height: 400,
    bgColor: '#000000',
    strokeStyle: '#FFFFFF'
  })
  inputCanvas.init()
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
  state.numbers.data.train = [xTrainNumbers, yTrainNumbers] = formatTrainData(numbersData);
  state.numbers.data.test = [xTestNumbers, yTestNumbers] = formatTestData(numbersData);

  state.fashion.data.train = [xTrainFashion, yTrainFashion] = formatTrainData(fashionData);
  state.fashion.data.test = [xTestFashion, yTestFashion] = formatTestData(fashionData);

  // const [xTrain, yTrain] = state.fashion.data.train;
  // const [xTest, yTest] = state.fashion.data.test;
  // console.log([xTrainNumbers, yTrainNumbers])
  // console.log([xTestNumbers, yTestNumbers])
  // console.log([xTrain, yTrain])
  // console.log([xTest, yTest])

  // create model
  // model = createArbitraryDenseModel()
  // model = createConvNetModel()

  // fit model
  // model, history = fitModel(model, xTrainNumbers, yTrainNumbers)
  // model, info = await fitModel(model, xTrain, yTrain, xTest, yTest)

  // console.log('Final accuracy', info.history.acc);
  // console.log(model)
  // console.log(info)
  // modelPredict(model, [xTest, yTest])
  
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