```markdown
# Methodology

This document outlines the continual learning approaches implemented in the `nlp-continual-learning` repository, focusing on strategies to mitigate catastrophic forgetting in NLP tasks.

## 1. Elastic Weight Consolidation (EWC)
**Model**: `EWCSentimentAnalyzer` (`src/models/ewc_sentiment.py`)

- **Description**: EWC mitigates forgetting by adding a regularization term to the loss function, penalizing changes to weights critical for previous tasks. It uses a Fisher information matrix to identify important parameters.
- **Application**: Applied to sentiment analysis, adapting to new domains (e.g., movie reviews to product reviews) while preserving performance on earlier domains.
- **Implementation Details**:
  - Uses PyTorch with Transformers (BERT).
  - Regularization strength controlled by `ewc_lambda`.
  - Maintains task-specific Fisher matrices and reference weights.

**Reference**: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS, 2017).

## 2. Experience Replay
**Model**: `ReplayClassifier` (`src/models/replay_classifier.py`)

- **Description**: Experience Replay stores a subset of previous task examples in a memory buffer and replays them during training on new tasks to maintain performance.
- **Application**: Used for text classification, learning new categories without forgetting prior ones.
- **Implementation Details**:
  - Uses TensorFlow with BERT.
  - Employs a `MemoryBuffer` to store and sample past examples.
  - Balances new and replayed data during training.

**Reference**: Rolnick et al., "Experience Replay for Continual Learning" (NeurIPS, 2019).

## 3. Progressive Neural Networks
**Model**: `ProgressiveLMAnalyzer` (`src/models/progressive_lm.py`)

- **Description**: Progressive Neural Networks grow the model architecture by adding new columns for each task, with lateral connections to leverage knowledge from previous tasks.
- **Application**: Applied to language modeling, adapting to new domains while retaining prior knowledge.
- **Implementation Details**:
  - Uses PyTorch with GPT-2.
  - Adds task-specific layers while preserving frozen weights from earlier tasks.
  - Supports dynamic architecture expansion.

**Reference**: Rusu et al., "Progressive Neural Networks" (arXiv, 2016).

## 4. Gradient Episodic Memory (GEM)
**Model**: `GEMNERAnalyzer` (`src/models/gem_ner.py`)

- **Description**: GEM projects gradients to ensure they do not interfere with previous tasks, using a memory buffer to store reference gradients and examples.
- **Application**: Used for Named Entity Recognition (NER), adapting to new entity types and domains.
- **Implementation Details**:
  - Combines PyTorch, spaCy, and Transformers (BERT).
  - Uses `GradientBuffer` for storing task-specific gradients.
  - Employs gradient projection with a margin parameter to balance learning and preservation.

**Reference**: Lopez-Paz and Ranzato, "Gradient Episodic Memory for Continual Learning" (NeurIPS, 2017).

## 5. Online Learning
**Model**: `OnlineIntentAnalyzer` (`src/models/online_intent.py`)

- **Description**: Online learning incrementally updates the model with streaming data, using a memory buffer for experience replay to mitigate forgetting.
- **Application**: Applied to intent classification for chatbots, adapting to new user intents in real-time.
- **Implementation Details**:
  - Uses scikit-learn’s `SGDClassifier` for incremental updates.
  - Integrates NLTK for text preprocessing (tokenization, stemming).
  - Maintains a `BalancedBuffer` to ensure fair sampling across tasks.

**Reference**: Shalev-Shwartz, "Online Learning and Online Convex Optimization" (Foundations and Trends in Machine Learning, 2012).

## Common Components
- **Base Learner** (`src/core/base_learner.py`): Provides a unified interface for training, evaluation, and checkpointing across models.
- **Memory Buffers** (`src/core/memory_buffer.py`): Includes `MemoryBuffer`, `BalancedBuffer`, and `GradientBuffer` for storing and sampling past examples or gradients.
- **Regularization** (`src/core/regularization.py`): Implements EWC and GEM regularization strategies.
- **Utilities** (`src/utils/`): Includes `data_loader.py` for loading datasets and `visualization.py` for plotting metrics.

## Evaluation Metrics
- **Accuracy**: Task-specific classification or tagging accuracy.
- **F1 Score**: Harmonic mean of precision and recall, used for NER and intent classification.
- **Backward Transfer**: Measures performance degradation on previous tasks (negative indicates forgetting).
- **Forward Transfer**: Measures improvement on new tasks due to prior knowledge (positive indicates transfer learning).
- **Loss**: Task-specific loss (e.g., cross-entropy for classification, NER).

## Dataset Structure
- Data is stored in `datasets/<task_type>/<domain>/<split>.json` (e.g., `datasets/ner/news/train.json`).
- Supported tasks: NER, sentiment analysis, text classification, language modeling, intent classification.
- JSON formats are defined in `src/utils/data_loader.py`.

This methodology ensures robust continual learning across diverse NLP tasks, balancing adaptation to new tasks with preservation of prior knowledge.