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
const EVAL_MODEL = document.getElementById('eval-model');
// const TRAINING_STOP = document.getElementById('training-stop');
BATCH_NUMBER.addEventListener('change', e => handleBatchNumber(e));
EPOCHS_NUMBER.addEventListener('change', e => handleEpochsNumber(e));
LEARNING_RATE_NUMBER.addEventListener('change', e => handleLearningRateNumber(e));
TRAINING_START_PAUSE.addEventListener('click', () => handleTrainingStartPause());
EVAL_MODEL.addEventListener('click', () => evaluateModel(state.model));
// TRAINING_STOP.addEventListener('click', () => handleTrainingStop());
// select image
const IMAGE_SELECT = document.getElementById('image-select');
const POPLULATE_IMAGE_SELECT = document.getElementById("populate-image-select");
POPLULATE_IMAGE_SELECT.addEventListener('click', () => populateImageSelect(state.dataSet === 'numbers' ? state.numbers.data.train : state.fashion.data.train));
// input canvas 
const CAST_TO_IMAGE = document.getElementById("cast-to-image");
const CLEAR_INPUT_CANVAS = document.getElementById("clear-input-canvas");
CAST_TO_IMAGE.addEventListener('click', () => handlePredict());
CLEAR_INPUT_CANVAS.addEventListener('click', () => inputCanvas.clear());
// right column info area
const DEMO_DATA = document.getElementById('demo-data');
const MODEL_LAYER_DETAILS = document.getElementById('model-layer-details');
const EPOCH_TRAINING_STATUS = document.getElementById('epoch-training-status');
const FITTING_TRAINING_STATUS = document.getElementById('fitting-training-status');
const BATCH_LOSS_STATUS = document.getElementById('batch-loss-status');
const BATCH_ACC_STATUS = document.getElementById('batch-acc-status');
const BATCH_TRAINING_GRAPH = document.getElementById('batch-training-graph');
const MODEL_EVAL_TABLE = document.getElementById('model-eval-table');
const EVAL_ACC = document.getElementById('eval-acc');
const MODEL_PREDICTION = document.getElementById('model-prediction');
const MODEL_UNCERTAINTY = document.getElementById('model-uncertainty');
const MODEL_FIND_SIMILAR = document.getElementById('model-find-simmilar');


// TODO check if there is a reason this is not in state
let inputCanvas

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

  showDemoData(state.numbers.data.test)
  populateImageSelect(state.numbers.data.test)

}
