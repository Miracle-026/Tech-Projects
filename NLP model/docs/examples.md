# Examples

This document provides end-to-end examples for each task type in the `nlp-continual-learning` repository: sentiment analysis, text classification, language modeling, named entity recognition (NER), and intent classification.

## 1. Sentiment Analysis

### Data Preparation
Create a dataset in `datasets/sentiment/movies/train.json`:
```json
[
    {"text": "This movie was fantastic!", "label": 1},
    {"text": "Really disappointing film.", "label": 0}
]