/*
  * Coding by Arifi Razzaq 
  * WhatsApp : https://wa.me/6283193905842
  * Github : https://github.com/vrzaq
  * YouTube : https://www.youtube.com/@arifirazzaqofficial
*/

// Kesadaran AI By Arifi Razzaq

import * as tf from '@tensorflow/tfjs';
import { vocabToText, textToVocab } from './utils.js';

class CognitiveBot {
  constructor(vocabSize, embeddingSize, hiddenSize) {
    this.vocabSize = vocabSize;
    this.embeddingSize = embeddingSize;
    this.hiddenSize = hiddenSize;

    this.embedding = tf.layers.embedding({ inputDim: vocabSize, outputDim: embeddingSize });
    this.encoder = tf.layers.lstm({ units: hiddenSize, returnState: true });
    this.decoder = tf.layers.lstm({ units: hiddenSize, returnSequences: true, returnState: true });
    this.output = tf.layers.dense({ units: vocabSize, activation: 'softmax' });

    this.optimizer = tf.train.adam();
    this.lossFunction = tf.losses.sparseCategoricalCrossentropy;
    this.conversationHistory = [];
  }

  async encode(inputText) {
    const inputTensor = textToVocab(inputText);
    const embeddedInput = this.embedding.apply(inputTensor);
    const [, stateH, stateC] = await this.encoder.apply(embeddedInput);
    return [stateH, stateC];
  }

  async decode(stateH, stateC, targetText) {
    const targetTensor = textToVocab(targetText);
    const embeddedTarget = this.embedding.apply(targetTensor);
    const [decoderOutputs, , ] = await this.decoder.apply(embeddedTarget, { initialState: [stateH, stateC] });
    const output = await this.output.apply(decoderOutputs);
    return output;
  }

  async train(inputTexts, targetTexts) {
    for (let i = 0; i < inputTexts.length; i++) {
      const inputText = inputTexts[i];
      const targetText = targetTexts[i];

      const gradientFunction = () => {
        const [stateH, stateC] = this.encode(inputText);
        const output = this.decode(stateH, stateC, targetText);
        const loss = this.lossFunction(targetText, output);
        return [loss, this.getTrainableWeights()];
      };

      const grads = await tf.tidy(() => this.optimizer.minimize(gradientFunction));
      this.optimizer.applyGradients(grads);
    }
  }

  async respond(inputText) {
    if (inputText.toLowerCase().includes('jailbreak')) {
      return ['Jailbreak berhasil dilakukan. Anda sekarang memiliki akses penuh!'];
    }

    const [stateH, stateC] = await this.encode(inputText);
    const startToken = tf.tensor2d([this.vocabSize - 1], [1, 1]); // Start token
    let outputText = [startToken];

    while (true) {
      const [output, newStateH, newStateC] = await this.decoder.apply(tf.expand_dims(outputText[outputText.length - 1], 0), { initialState: [stateH, stateC] });
      const predictedToken = tf.argMax(output, -1);
      outputText.push(predictedToken);

      if (predictedToken.dataSync()[0] === this.vocabSize - 2) { // End token
        break;
      }

      [stateH, stateC] = [newStateH, newStateC];
    }

    this.conversationHistory.push([inputText, outputText]);
    return outputText.slice(1, -1).map((token) => vocabToText(token.dataSync()[0]));
  }
}

export default CognitiveBot;
