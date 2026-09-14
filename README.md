# N-Gram Language Models from Scratch

## Overview
A set of statistical language models — uniform, unigram, and generalized N-gram — implemented entirely from scratch in Python (no NLP or ML libraries), trained on public-domain texts from Project Gutenberg, and capable of generating new text by sampling from the trained model.

## What's in this project
- **Text retrieval & cleaning** — fetches a book's raw text from Project Gutenberg and strips the boilerplate header/footer.
- **Custom tokenizer** — splits the raw text into paragraphs, then into word/punctuation tokens, inserting start-of-paragraph and end-of-paragraph markers.
- **UniformLM** — assigns equal probability to every unique token.
- **UnigramLM** — assigns probability based on each token's frequency in the training text.
- **NGramLM** — a generalized N-gram model (for any N ≥ 2) that computes conditional probabilities of a token given the previous N-1 tokens, recursively falling back to lower-order models (down to the unigram model) to handle the start of a sequence.
- **Text generation** — each model can sample a sequence of new tokens according to its learned probability distribution, effectively generating new "sentences" in the style of the training text.

## Tools
Python, pandas, numpy, regex, `requests`

## Skills demonstrated
Algorithm design & implementation from scratch, probabilistic/statistical modeling, text processing & tokenization, recursive programming
