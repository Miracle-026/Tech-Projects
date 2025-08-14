# Usage Guide

This document provides instructions for using the training scripts, demo notebooks, and API in the `nlp-continual-learning` repository.

## Training Scripts

The repository includes training scripts for each model in the `scripts/` directory. Each script uses a configuration file to specify model and dataset parameters.

### 1. Train EWC Sentiment Analyzer
```bash
python scripts/train_ewc_sentiment.py --config configs/ewc_config.yaml --data-dir datasets/sentiment/ --save-dir results/models/ --verbose