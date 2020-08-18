/* ~~~~~~~~~~~~~~~~~~~~~~ event handlers ~~~~~~~~~~~~~~~~~~~~~~ */

// model creation form
function handleDataSelect(e) {
  state.dataSet = e.target.value;
  clearDivChildren(DEMO_DATA)
  state.dataSet === 'numbers' ? showDemoData(state.numbers.data.train) : showDemoData(state.fashion.data.train);
}
function handleModelSelect(e) {
  state.modelType = e.target.value;
}

function handleUnitsNumber(e) {
  state.units = parseInt(e.target.value);
}

function handleHiddenLayersNumber(e) {
  state.hiddenLayers = parseInt(e.target.value);
}

function modelCreateFormControl(btnClass, btnText, disabled) {
  // change button
  MODEL_CREATE_DESTROY.className = btnClass;
  MODEL_CREATE_DESTROY.innerText = btnText;
  // lock/unlock form
  DATA_SELECT.disabled = disabled;
  MODEL_SELECT.disabled = disabled;
  UNITS_NUMBER.disabled = disabled;
  HIDDEN_LAYERS_NUMBER.disabled = disabled;
}

function handleModelCreateDestroy() {
  if(!state.model) {
    // create model
    switch(state.modelType){
      case 'dense':
        state.model = createArbitraryDenseModel();
        break;
      case 'convolutional':
        state.model = createConvNetModel();
        break;
      default:
        console.log('oh no!');
    }
    // update form
    return modelCreateFormControl("form-control btn btn-danger", "Destory Model", true)
  }
  // destroy model
  state.model.dispose()
  state.model = undefined;
  clearDivChildren(MODEL_LAYER_DETAILS)
  // update form
  return modelCreateFormControl("form-control btn btn-primary", "Create Model", false)
}

// training form
function handleBatchNumber(e) {
  console.log('called')
  state.batchSize = parseInt(e.target.value)
  console.log(state.batchSize);
}

function handleEpochsNumber(e) {
  state.epochs = e.target.value;
  console.log(state.epochs);
}

function handleLearningRateNumber(e) {
  state.learningRate = parseFloat(e.target.value);
  console.log(state.learningRate);
}

async function handleTrainingStartPause() {
  const [xTrain, yTrain] = state.dataSet === 'numbers' ? state.numbers.data.train : state.fashion.data.train;
  const [xTest, yTest] = state.dataSet === 'numbers' ? state.numbers.data.test : state.fashion.data.test;
  console.log(xTrain, yTrain, xTest, yTest, state.model)
  let info
  state.model, info = await fitModel(state.model, xTrain, yTrain, xTest, yTest)
  console.log(info)
}

function handleTrainingStop() {
  console.log('stop')
}

// input canvas
function handlePredict() {
  if(!state.model) return;
  const imageTensor = inputCanvas.castToImage();
  modelPredictCanvas(state.model, imageTensor)
}

// model feedback column

// clear a div of all children
function clearDivChildren(id) {
  while (id.firstChild) {
    id.removeChild(id.lastChild)
  }
}

function showDemoData(data) {
  const [xTest, yTest] = data;
  for(let i = 0; i < 10; i++){
    let rand = Math.floor(Math.random() * 1000);
    let img = xTest.slice([rand, 0, 0, 0], [1, state.imageHeight, state.imageWidth, 1]);
    const canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d');
    canvas.style = 'margin: 4px;';
    img = img.reshape([28, 28, 1]).resizeBilinear([45, 45]);
    tf.browser.toPixels(img, canvas)
    DEMO_DATA.appendChild(canvas);
    canvas.className = 'demo-data-canvas'
  }

}

function showModelSummary(model) {
  // console.log(MODEL_LAYER_DETAILS)
  tfvis.show.modelSummary(MODEL_LAYER_DETAILS, model)
}

function showEpochTrainingStatus(logs){
  let percent = 103 * (logs.batch / (state.trainDataSize / state.batchSize));
  percent = `${percent < 100 ? Math.ceil(percent) : 100}%`; 
  EPOCH_TRAINING_STATUS.style.width = percent;
  EPOCH_TRAINING_STATUS.innerText = percent;
  BATCH_LOSS_STATUS.style.width = `${logs.loss * 100.}%`;
  BATCH_LOSS_STATUS.innerText = `${logs.loss.toFixed(3)}`;
  BATCH_ACC_STATUS.style.width = `${logs.acc * 100.}%`;
  BATCH_ACC_STATUS.innerText = `${logs.acc.toFixed(3)}`;
  console.log(logs)
}

function showFittingTrainingStatus(){
  let percent = 100 * (state.currentEpoch / state.epochs);
  FITTING_TRAINING_STATUS.style.width = `${Math.ceil(percent)}%`;
  FITTING_TRAINING_STATUS.innerText = state.currentEpoch;
  // console.log(percent)
}

