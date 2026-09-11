"""
N-gram Language Model

Name: Trevor Tourdot
Date: 09/09/2026

"""

import re
import random
from collections import Counter


def read_text_file(filename):
    """
    Read and return the contents of a text file.
    """
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def preprocess_text(text):
    """
    Lowercase text, split it into sentences, and tokenize each sentence.

    Add:
    - <s> at the beginning of each sentence
    - </s> at the end of each sentence

    Return:
    A list of tokenized sentences, where each sentence is a list of tokens.

    Example:
    [
        ["<s>", "i", "want", "food", "</s>"],
        ["<s>", "sam", "likes", "ham", "</s>"]
    ]
    """
    # Convert text to lowercase
    text = text.lower()

    # Split text into sentences
    raw_sentences = re.split(r"[.!?]+", text)

    sentences = []

    #tokenize each sentence
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        #keep words and apostrophes together
        tokens = re.findall(r"[a-z0-9']+", sentence)
        if tokens:
            sentences.append(["<s>"] + tokens + ["</s>"])

    return sentences


def build_unigrams(sentences):
    """
    Build unigram counts from tokenized sentences.

    Return:
    A Counter object containing token counts.
    """
    # Count all tokens in all sentences
    unigrams = Counter()

    for sentence in sentences:
        unigrams.update(sentence)

    return unigrams


def build_bigrams(sentences):
    """
    Build bigram counts from tokenized sentences.

    Return:
    A Counter object where keys are tuples like (word1, word2)
    and values are counts.
    """
    bigrams = Counter()

    for sentence in sentences:
        for i in range(len(sentence) -1):
            bigrams[(sentence[i], sentence[i+1])] +=1

    return bigrams


def compute_bigram_probabilities(unigrams, bigrams):
    """
    Compute bigram probabilities using Maximum Likelihood Estimation (MLE).

    Formula:
    P(word2 | word1) = count(word1, word2) / count(word1)

    Return:
    A dictionary where keys are bigram tuples and values are probabilities.
    """
    # Compute probabilities from counts
    probabilities = {}

    for (word1, word2), count in bigrams.items():
        probabilities[(word1, word2)] = count / unigrams[word1]

    return probabilities


def get_next_word_candidates(bigram_probs, current_word):
    """
    Return all possible next words and their probabilities for a given word.

    Example return value:
    {"to": 0.6, "food": 0.2, "eat": 0.2}
    """
    # Find all bigrams that begin with current_word
    candidates = {}

    for (word1, word2), prob in bigram_probs.items():
        if word1 == current_word:
            candidates[word2] = prob

    return candidates


def predict_next_word(bigram_probs, current_word):
    """
    Return the most likely next word after current_word.

    If there are no candidates, return None.
    """
    # Get candidates
    candidates = get_next_word_candidates(bigram_probs, current_word)

    if not candidates:
        return None

    # Return the word with the highest probability
    return max(candidates, key=candidates.get)


def sentence_probability(sentence, bigram_probs):
    """
    Compute the probability of a sentence using the bigram model.

    Steps:
    - tokenize the sentence
    - add <s> and </s>
    - multiply the bigram probabilities

    If any bigram is missing, return 0.0
    """
    # Tokenize the sentence
    # Add sentence markers
    tokens = re.findall(r"[a-z0-9']+", sentence.lower())

    probability = 1.0

    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])

        if bigram not in bigram_probs:
            return 0.0

        probability *= bigram_probs[bigram]

    return probability


def weighted_choice(candidates):
    """
    Randomly choose one next word based on probability weights.

    Example:
    If candidates = {"eat": 0.7, "sleep": 0.3}
    then "eat" should be chosen more often.
    """
    # Use random.choices() with weights
    words = list(candidates.keys())
    weights = list(candidates.values())
    return random.choices(words, weights = weights, k=1)[0]


def generate_sentence(bigram_probs, max_length=15):
    """
    Generate a random sentence using the bigram model.

    Start with <s>
    Continue choosing words until:
    - </s> is reached, or
    - max_length is reached
    """
    # Start at <s>
    current_word = "<s>"

    generated_words = []

    for _ in range(max_length):
        candidates = get_next_word_candidates(bigram_probs, current_word)

        if not candidates:
            break

        # Repeatedly choose the next word
        next_word = weighted_choice(candidates)

        # Stop if </s> is reached
        if next_word == "</s>":
            break

        generated_words.append(next_word)
        current_word = next_word

    return " ".join(generated_words)


def print_top_unigrams(unigrams, top_n=10):
    """
    Print the most common unigrams.
    """
    print("\nTop Unigrams:")
    # Print the top_n most common tokens
    for word, count in unigrams.most_common(top_n):
        print(f"{word}: {count}")


def print_top_bigrams(bigrams, top_n=10):
    """
    Print the most common bigrams.
    """
    print("\nTop Bigrams:")
    # Print the top_n most common bigrams
    for (word1, word2), count in bigrams.most_common(top_n):
        print(f"({word1}, {word2}): {count}")


def compare_sentence_probabilities(sentence1, sentence2, bigram_probs):
    """
    Compare the probabilities of two sentences and print the results.
    """
    # Compute sentence probabilities
    prob1 = sentence_probability(sentence1, bigram_probs)
    prob2 = sentence_probability(sentence2, bigram_probs)



    print(f"\nSentence probability: '{sentence1}' = {prob1}")
    print(f"Sentence probability: '{sentence2}' = {prob2}")


def main():
    filename = "Cthulhu.txt"

    try:
        text = read_text_file(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # Build the model
    sentences = preprocess_text(text)
    unigrams = build_unigrams(sentences)
    bigrams = build_bigrams(sentences)
    bigram_probs = compute_bigram_probabilities(unigrams, bigrams)

    print("Language Model Built Successfully")
    print(f"Number of sentences: {len(sentences)}")
    print(f"Number of unique unigrams: {len(unigrams)}")
    print(f"Number of unique bigrams: {len(bigrams)}")

    print_top_unigrams(unigrams)
    print_top_bigrams(bigrams)

    # Example next-word prediction
    test_word = "I"
    predicted = predict_next_word(bigram_probs, test_word)
    print(f"\nMost likely word after '{test_word}': {predicted}")

    # Example sentence comparison
    sentence1 = "As my"
    sentence2 = "of the"
    compare_sentence_probabilities(sentence1, sentence2, bigram_probs)

    # Generate a few random sentences
    print("\nGenerated Sentences:")
    for i in range(3):
        print(f"{i + 1}. {generate_sentence(bigram_probs)}")


if __name__ == "__main__":
    main()