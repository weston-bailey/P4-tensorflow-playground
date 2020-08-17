/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// this file has been modified by weston bailey
// the original work was retrived from: https://raw.githubusercontent.com/tensorflow/tfjs-examples/master/mnist-core/data.js

class MnistData {
  constructor(args) {
    this.imageSize = args.imageSize;
    this.outputClasses = args.outputClasses;
    this.dataSetLength = args.dataSetLength;
    this.trainTestRatio = args.trainTestRatio;
    this.imgPath = args.imgPath;
    this.labelPath = args.labelPath;
    this.datasetImages = null;
    this.datasetLabels = null;
    this.trainIndices = null;
    this.testIndices = null;
    this.trainImages = null;
    this.testImages = null;
    this.trainLabels = null;
    this.testLabels = null;
  }

  async load() {

    const NUM_TRAIN_ELEMENTS = Math.floor(this.trainTestRatio * this.dataSetLength);
    const NUM_TEST_ELEMENTS = this.dataSetLength - NUM_TRAIN_ELEMENTS;
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer =
            new ArrayBuffer(this.dataSetLength * this.imageSize * 4);

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < this.dataSetLength / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
              datasetBytesBuffer, i * this.imageSize * chunkSize * 4,
              this.imageSize * chunkSize);
          ctx.drawImage(
              img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
              chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = this.imgPath;
    });

    const labelsRequest = fetch(this.labelPath);
    const [imgResponse, labelsResponse] =
        await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    this.trainImages = this.datasetImages.slice(0, this.imageSize * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(this.imageSize * NUM_TRAIN_ELEMENTS);
    this.trainLabels = this.datasetLabels.slice(0, this.outputClasses * NUM_TRAIN_ELEMENTS);
    this.testLabels = this.datasetLabels.slice(this.outputClasses * NUM_TRAIN_ELEMENTS);
  }
}
