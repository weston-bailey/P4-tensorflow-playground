/* ~~~~~~~~~~~~~~~~~~~~~~ event handlers ~~~~~~~~~~~~~~~~~~~~~~ */

// model creation form
function handleDataSelect(e) {
  state.dataSet = e.target.value;
  console.log(state.dataSet)
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

function handlePredict() {
  // if(!state.model) return;
  const imageTensor = inputCanvas.castToImage();
  modelPredictCanvas(state.model, imageTensor)
  // modelPredict(state.model, state.numbers.data.test)
  // console.log(imageTensor)
}