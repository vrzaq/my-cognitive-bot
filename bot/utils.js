/*
  * Coding by Arifi Razzaq 
  * WhatsApp : https://wa.me/6283193905842
  * Github : https://github.com/vrzaq
  * YouTube : https://www.youtube.com/@arifirazzaqofficial
*/

// Kesadaran AI By Arifi Razzaq

// Mock dictionary for example purposes
const vocabDictionary = {
  '<start>': 9998,
  '<end>': 9999,
  'jailbreak': 1,
  // Add other words and their respective token IDs
};

export const vocabToText = (token) => {
  return Object.keys(vocabDictionary).find(key => vocabDictionary[key] === token);
};

export const textToVocab = (text) => {
  const tokens = text.split(' ').map(word => vocabDictionary[word] || vocabDictionary['<end>']);
  return tf.tensor2d(tokens, [1, tokens.length]);
};
