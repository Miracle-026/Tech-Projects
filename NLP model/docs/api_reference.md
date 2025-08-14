# API Reference

This document provides an API reference for the model classes and utilities in the `nlp-continual-learning` repository.

## Model Classes

### EWCSentimentAnalyzer
**Module**: `src.models.ewc_sentiment`

**Description**: Sentiment analysis model using Elastic Weight Consolidation (EWC) for continual learning.

**Methods**:
- `__init__(model_name: str, memory_size: int, learning_rate: float, batch_size: int, max_length: int, ewc_lambda: float, device: str, save_dir: str, verbose: bool)`: Initialize the model with BERT and EWC parameters.
- `train_task(train_data: List[SentimentExample], task_id: int, domain: str, epochs: int, validation_data: Optional[List[SentimentExample]]) -> List[TaskMetrics]`: Train on a new sentiment analysis task.
- `evaluate_task(test_data: List[SentimentExample], task_id: int, domain: str) -> TaskMetrics`: Evaluate on a specific task.
- `predict_single(text: str, task_id: int, return_probability: bool) -> SentimentPrediction`: Predict sentiment for a single text.
- `analyze_forgetting(test_datasets: Dict[str, List[SentimentExample]]) -> Dict[str, float]`: Measure forgetting across domains.
- `get_domain_summary() -> Dict[str, Any]`: Get performance summary across domains.
- `save_checkpoint(path: str) -> str`: Save model checkpoint.
- `load_checkpoint(path: str)`: Load model from checkpoint.

### ReplayClassifier
**Module**: `src.models.replay_classifier`

**Description**: Text classification model using experience replay.

**Methods**:
- Similar to `EWCSentimentAnalyzer`, with methods adapted for text classification using TensorFlow and BERT.

### ProgressiveLMAnalyzer
**Module**: `src.models.progressive_lm`

**Description**: Language modeling model using progressive neural networks.

**Methods**:
- Similar to `EWCSentimentAnalyzer`, with methods adapted for language modeling using PyTorch and GPT-2.

### GEMNERAnalyzer
**Module**: `src.models.gem_ner`

**Description**: Named Entity Recognition (NER) model using Gradient Episodic Memory (GEM).

**Methods**:
- `__init__(model_name: str, memory_size: int, learning_rate: float, batch_size: int, max_length: int, device: str, save_dir: str, verbose: bool)`: Initialize with BERT and GEM parameters.
- `train_task(train_data: List[NERExample], task_id: int, domain: str, epochs: int, validation_data: Optional[List[NERExample]]) -> List[TaskMetrics]`: Train on a new NER task.
- `evaluate_task(test_data: List[NERExample], task_id: int, domain: str) -> TaskMetrics`: Evaluate on a specific task.
- `predict_single(text: str, task_id: int, return_labels: bool) -> List[Tuple[int, int, str]]`: Predict entities for a single text.
- `analyze_forgetting(test_datasets: Dict[str, List[NERExample]]) -> Dict[str, float]`: Measure forgetting across domains.
- `get_domain_summary() -> Dict[str, Any]`: Get performance summary across domains.
- `save_checkpoint(path: str) -> str`: Save model checkpoint.
- `load_checkpoint(path: str)`: Load model from checkpoint.

### OnlineIntentAnalyzer
**Module**: `src.models.online_intent`

**Description**: Intent classification model using online learning.

**Methods**:
- `__init__(memory_size: int, learning_rate: float, batch_size: int, device: str, save_dir: str, verbose: bool)`: Initialize with scikit-learn’s SGDClassifier.
- `train_task(train_data: List[IntentExample], task_id: int, domain: str, epochs: int, validation_data: Optional[List[IntentExample]]) -> List[TaskMetrics]`: Train on a new intent classification task.
- `evaluate_task(test_data: List[IntentExample], task_id: int, domain: str) -> TaskMetrics`: Evaluate on a specific task.
- `predict_single(text: str, task_id: int, return_probability: bool) -> IntentPrediction`: Predict intent for a single text.
- `analyze_forgetting(test_datasets: Dict[str, List[IntentExample]]) -> Dict[str, float]`: Measure forgetting across domains.
- `get_domain_summary() -> Dict[str, Any]`: Get performance summary across domains.
- `save_checkpoint(path: str) -> str`: Save model checkpoint.
- `load_checkpoint(path: str)`: Load model from checkpoint.

## Core Components

### ContinualLearner
**Module**: `src.core.base_learner`

**Description**: Base class for continual learning models.

**Methods**:
- `__init__(model: Any, device: str, save_dir: str, verbose: bool)`: Initialize with a model and configuration.
- `train_task(...)`: Abstract method for training.
- `evaluate_task(...)`: Abstract method for evaluation.
- `predict(...)`: Abstract method for prediction.
- `analyze_forgetting(...)`: Measure forgetting across tasks.
- `get_domain_summary() -> Dict[str, Any]`: Get performance summary.
- `save_checkpoint(path: str) -> str`: Save checkpoint.
- `load_checkpoint(path: str)`: Load checkpoint.

### MemoryBuffer, BalancedBuffer, GradientBuffer
**Module**: `src.core.memory_buffer`

**Description**: Classes for storing and sampling past examples or gradients.

**Methods**:
- `add(item: MemoryItem)`: Add an item to the buffer.
- `get_random_batch(batch_size: int) -> List[MemoryItem]`: Sample a random batch.
- `get_all() -> List[MemoryItem]`: Get all items.

### GradientProjection
**Module**: `src.core.regularization`

**Description**: Implements gradient projection for GEM.

**Methods**:
- `__init__(model: nn.Module, device: str, memory_buffer: GradientBuffer)`: Initialize with model and buffer.
- `update_task_knowledge(task_id: int)`: Update reference gradients for a task.
- `project_gradients(gradients: Dict[str, torch.Tensor])`: Project gradients to avoid interference.

## Utilities

### Data Loader
**Module**: `src.utils.data_loader`

**Description**: Functions for loading datasets.

**Functions**:
- `load_ner_data(file_path: str, task_id: int, domain: str) -> List[NERExample]`: Load NER data.
- `load_sentiment_data(file_path: str, task_id: int, domain: str) -> List[SentimentExample]`: Load sentiment data.
- `load_classification_data(file_path: str, task_id: int, domain: str) -> List[ClassificationExample]`: Load classification data.
- `load_language_modeling_data(file_path: str, task_id: int, domain: str) -> List[LanguageModelingExample]`: Load language modeling data.
- `load_intent_data(file_path: str, task_id: int, domain: str) -> List[IntentExample]`: Load intent data.
- `load_dataset(data_dir: str, task_type: str, domains: List[str], split: str) -> Dict[str, List[Any]]`: Load datasets for multiple domains.

### Visualization
**Module**: `src.utils.visualization`

**Description**: Functions for plotting metrics.

**Functions**:
- `plot_training_metrics(metrics_history: Dict[str, List[TaskMetrics]], output_path: str, metric_types: List[str], title_prefix: str)`: Plot training metrics.
- `plot_forgetting_scores(forgetting_scores: Dict[str, float], output_path: str, title: str)`: Plot forgetting scores.
- `plot_domain_summary(summary: Dict[str, Any], output_path: str, metric: str, title: str)`: Plot domain performance summary.

## API Endpoints
**Module**: `src.api.server`

**Description**: Flask API for model inference.

**Endpoints**:
- `GET /health`: Check server status and available models.
- `POST /predict/<model_name>`: Perform inference with the specified model. Expects JSON with `text` and `task_id`.