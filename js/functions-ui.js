/* ~~~~~~~~~~~~~~~~~~~~~~ event handlers ~~~~~~~~~~~~~~~~~~~~~~ */
function handleModelSelect(e) {
  state.modelType = e.target.value;
}

function handleUnitsNumber(e) {
  state.units = parseInt(e.target.value);
}

function handleHiddenLayersNumber(e) {
  // update state
  state.hiddenLayers = parseInt(e.target.value);
}

// change form after model is made 
function modelCreateFormControl(btnClass, btnText, disabled) {
  // change button
  MODEL_CREATE_DESTROY.className = btnClass;
  MODEL_CREATE_DESTROY.innerText = btnText;
  // lock/unlock form
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