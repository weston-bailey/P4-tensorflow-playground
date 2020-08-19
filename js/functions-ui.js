/* ~~~~~~~~~~~~~~~~~~~~~~ event handlers ~~~~~~~~~~~~~~~~~~~~~~ */

// model creation form 
// TODO tidy this function up
function handleDataSelect(e) {
  state.dataSet = e.target.value;
  clearDivChildren(DEMO_DATA)
  state.dataSet === 'numbers' ? showDemoData(state.numbers.data.train) : showDemoData(state.fashion.data.train);
  state.dataSet === 'numbers' ? populateImageSelect(state.numbers.data.train) : populateImageSelect(state.fashion.data.train);
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
  state.batchSize = parseInt(e.target.value)
}

function handleEpochsNumber(e) {
  state.epochs = e.target.value;
}

function handleLearningRateNumber(e) {
  state.learningRate = parseFloat(e.target.value);
}

async function handleTrainingStartPause() {
  const [xTrain, yTrain] = state.dataSet === 'numbers' ? state.numbers.data.train : state.fashion.data.train;
  const [xTest, yTest] = state.dataSet === 'numbers' ? state.numbers.data.test : state.fashion.data.test;
  let info;
  state.model, info = await fitModel(state.model, xTrain, yTrain, xTest, yTest)
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

function populateImageSelect(data) {
  const [xTest, yTest] = data;
  clearDivChildren(IMAGE_SELECT);
  for(let i = 0; i < 10; i++){
    let rand = Math.floor(Math.random() * 1000);
    let img = xTest.slice([rand, 0, 0, 0], [1, state.imageHeight, state.imageWidth, 1]);
    const canvas = document.createElement('canvas');
    let ctx = canvas.getContext('2d');
    canvas.style = 'margin: 4px;';
    img = img.reshape([28, 28, 1]).resizeBilinear([45, 45]);
    tf.browser.toPixels(img, canvas);
    canvas.setAttribute('id', rand);
    canvas.addEventListener('click', (e) => {
      let index = parseInt(e.target.id);
      const [xTest, yTest] =  state.dataSet === 'numbers' ? state.numbers.data.test : state.fashion.data.test;
      const imageTensor = xTest.slice([index, 0, 0, 0], [1, state.imageHeight, state.imageWidth, 1]);
      modelPredictCanvas(state.model, imageTensor);
    })
    IMAGE_SELECT.appendChild(canvas);
    canvas.className = 'image-select-canvas'
  }
}

function showModelSummary(model) {
  tfvis.show.modelSummary(MODEL_LAYER_DETAILS, model)
}

function showEpochTrainingStatus(logs){
  // progress bars
  let percent = 103 * (logs.batch / (state.trainDataSize / state.batchSize));
  percent = `${percent < 100 ? Math.ceil(percent) : 100}%`; 
  EPOCH_TRAINING_STATUS.style.width = percent;
  EPOCH_TRAINING_STATUS.innerText = percent;
  BATCH_LOSS_STATUS.style.width = `${logs.loss * 100.}%`;
  BATCH_LOSS_STATUS.innerText = `${logs.loss.toFixed(3)}`;
  BATCH_ACC_STATUS.style.width = `${logs.acc * 100.}%`;
  BATCH_ACC_STATUS.innerText = `${logs.acc.toFixed(3)}`;
  // graph
  state.batchAcc.push({ x: state.batchAcc.length, y: logs.acc })
  state.batchLoss.push({ x: state.batchLoss.length, y: logs.loss >= .999 ? .999 : logs.loss })
  const options = {
    xLabel: 'batch 10x',
    yLabel: 'Value',
    yAxisDomain: [0, 1],
    seriesColors: ['#28a745', '#dc3545']
  }; // Prep the data
  const data = { values: [state.batchAcc, state.batchLoss], series: ['Accuracy', 'Loss'] }
  tfvis.render.linechart(BATCH_TRAINING_GRAPH, data, options)
}

function showFittingTrainingStatus(){
  let percent = 100 * (state.currentEpoch / state.epochs);
  FITTING_TRAINING_STATUS.style.width = `${Math.ceil(percent)}%`;
  FITTING_TRAINING_STATUS.innerText = state.currentEpoch;
}

function showPrediction(preds){
  let classNames = state.dataSet === 'numbers' ? state.numbers.classNames : state.fashion.classNames;
  let uncert = Array.from(preds.dataSync());
  preds = preds.argMax(-1);
  let index = preds.arraySync();
  preds.dispose();
  // show prediction
  MODEL_PREDICTION.innerText = `I think that is an image of a(n) ${classNames[index]}.\nHere is a bar graph of my other guesses:`;
  let data = []
  for(let i = 0; i < uncert.length; i++) {
    data.push({ index: i, value: uncert[i] });
  }
  tfvis.render.barchart(MODEL_UNCERTAINTY, data);
}

