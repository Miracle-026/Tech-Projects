# Deployment Guide

This document outlines strategies for deploying the Flask API (`src/api/server.py`) in the `nlp-continual-learning` repository for production use.

## Local Deployment

### Prerequisites
- Follow `docs/installation.md` to set up the environment.
- Ensure model checkpoints are available in `results/models/`.

### Running the API
```bash
python src/api/server.py