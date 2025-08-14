# Changelog

All notable changes to the `nlp-continual-learning` repository are documented in this file.

## [1.0.0] - 2025-08-09

### Added
- **Core Models**:
  - `EWCSentimentAnalyzer` for sentiment analysis with Elastic Weight Consolidation.
  - `ReplayClassifier` for classification with experience replay.
  - `ProgressiveLMAnalyzer` for language modeling with progressive networks.
  - `GEMNERAnalyzer` for NER with Gradient Episodic Memory.
  - `OnlineIntentAnalyzer` for intent detection with online learning.
  - `MultiTaskModel` for simultaneous training on sentiment, classification, NER, and intent tasks.
- **Scripts**:
  - `scripts/evaluate_models.py`: Compare model performance across tasks.
  - `scripts/tune_hyperparameters.py`: Optimize model hyperparameters via grid search.
  - `scripts/augment_data.py`: Generate synthetic data using synonym replacement.
  - `scripts/monitor_models.py`: Detect data drift in production.
  - `scripts/transfer_learning.py`: Fine-tune models on new domains.
  - `scripts/cross_task_eval.py`: Evaluate models on non-primary tasks.
  - `scripts/active_learning.py`: Select informative samples using uncertainty sampling.
  - `scripts/compress_models.py`: Apply pruning, quantization, and distillation.
  - `scripts/retrain_models.py`: Periodically retrain models with new and active learning data.
  - `scripts/version_data.py`: Manage dataset versions for reproducibility.
- **Utilities**:
  - `src/utils/data_loader.py`: Load datasets for all task types.
  - `src/utils/visualization.py`: Plot training metrics, confusion matrices, and t-SNE embeddings.
  - `src/utils/explainability.py`: Provide SHAP and token-level explanations.
- **API**:
  - `src/api/server.py`: Flask API with `/health`, `/predict`, `/batch_predict`, and WebSocket `predict` endpoints.
  - Added authentication and input validation for security.
- **Documentation**:
  - `docs/installation.md`: Setup instructions.
  - `docs/usage.md`: Model usage guide.
  - `docs/methodology.md`: Continual learning methodologies.
  - `docs/api_reference.md`: API endpoint details.
  - `docs/deployment.md`: Deployment strategies (Docker, AWS).
  - `docs/performance_tuning.md`: Optimization techniques (mixed precision, pruning).
  - `docs/security.md`: Security practices for the API.
  - `docs/maintenance.md`: Maintenance procedures.
- **Testing**:
  - `tests/test_models.py`: Unit tests with robustness and out-of-domain checks.
  - `.github/workflows/ci.yml`: CI/CD pipeline for linting and testing.

### Changed
- Updated `src/api/server.py` to include batch prediction and WebSocket support.
- Enhanced `src/utils/visualization.py` with confusion matrices and t-SNE embeddings.
- Updated `tests/test_models.py` with adversarial robustness and out-of-domain tests.

### Fixed
- Ensured compatibility across all models with consistent data loaders and configurations.
- Addressed potential security vulnerabilities in API endpoints.

## [0.1.0] - 2025-07-01
- Initial implementation of core models and basic training scripts.
- Added data preprocessing and initial visualization utilities.
- Set up Flask API with single prediction endpoint.
- Created initial documentation (`installation.md`, `usage.md`).