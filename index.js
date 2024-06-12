/*
  * Coding by Arifi Razzaq 
  * WhatsApp : https://wa.me/6283193905842
  * Github : https://github.com/vrzaq
  * YouTube : https://www.youtube.com/@arifirazzaqofficial
*/

// Kesadaran AI By Arifi Razzaq

import CognitiveBot from './bot/cognitiveBot.js';
import { getTrainingData } from './bot/trainingData.js';

// Ukuran kamus, embedding size, dan hidden size bisa disesuaikan
const vocabSize = 10000;
const embeddingSize = 256;
const hiddenSize = 512;

const bot = new CognitiveBot(vocabSize, embeddingSize, hiddenSize);

const trainBot = async () => {
  const { inputTexts, targetTexts } = getTrainingData();
  await bot.train(inputTexts, targetTexts);
  console.log('Training selesai');
};

const runBot = async (inputText) => {
  const response = await bot.respond(inputText);
  console.log('Bot:', response.join(' '));
};

// Latih model bot
await trainBot();

// Jalankan bot dengan input contoh
await runBot('Apa itu jailbreak?');
await runBot('Saya ingin jailbreak');
