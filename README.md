# N‑Gram Language Model

A simple Python implementation of a bigram language model.
It reads a text file, preprocesses the text into tokenized sentences, builds unigram and bigram counts, computes bigram probabilities, and can:

- Predict the most likely next word
- Compute sentence probabilities
- Generate random sentences based on learned bigram statistics
- Display the most common unigrams and bigrams

## Features
- Text preprocessing with `<s>` and `</s>` sentence markers
- Unigram and bigram frequency counting
- Maximum likelihood estimation for bigram probabilities
- Next‑word prediction and weighted random generation
- Sentence probability comparison
- Simple CLI output for model statistics
