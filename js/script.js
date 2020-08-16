document.addEventListener('DOMContentLoaded', () => init())

console.log('hello front end')
async function init() {
  model = testModel()
  console.log(model)
  console.log(tf)
  const data = await getData()
  console.log(data)
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

async function getData() {  
  const data = new MnistData();
  await data.load()
  return data
}


// load data

// groom data

// create model

// fit model

// test model

// cast prediction

// ????????

// PROFIT!!!

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