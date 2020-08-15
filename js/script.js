document.addEventListener('DOMContentLoaded', () => init())

function init() {
  console.log('hello front end')
  model = testModel()
  console.log(model)
}

function testModel() {
  //Create the model
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 128, inputShape: [1]})); // layer 1
  model.add(tf.layers.dense({units: 128, inputShape: [128], activation: 'sigmoid'})); // layer 2
  model.add(tf.layers.dense({units: 1, inputShape: [128]})); // output layer
  model.compile({loss: 'meanSquaredError', optimizer: 'adam'}); // compile with params

  return model
}