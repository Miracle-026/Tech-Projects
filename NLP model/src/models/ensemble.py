"""
Ensemble model combining predictions from multiple continual learning models.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from src.models.ewc_sentiment import EWCSentimentAnalyzer
from src.models.replay_classifier import ReplayClassifier
from src.models.progressive_lm import ProgressiveLMAnalyzer
from src.models.gem_ner import GEMNERAnalyzer
from src.models.online_intent import OnlineIntentAnalyzer

class EnsembleModel:
    """Ensemble model for combining predictions from multiple models."""
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None, save_dir: str = "results/models/", verbose: bool = False):
        """Initialize ensemble with a list of models."""
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.save_dir = save_dir
        self.logger = logging.getLogger("EnsembleModel")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        # Validate models
        self.model_types = [model.__class__.__name__ for model in models]
        self.logger.info(f"Initialized ensemble with models: {self.model_types}")
    
    def predict_single(self, text: str, task_id: int, task_type: str, return_probability: bool = False) -> Union[Dict[str, Any], str, List]:
        """Predict for a single text using ensemble."""
        predictions = []
        probabilities = []
        
        for model in self.models:
            try:
                if isinstance(model, (EWCSentimentAnalyzer, ReplayClassifier, OnlineIntentAnalyzer)):
                    pred = model.predict_single(text, task_id, return_probability=True)
                    predictions.append(pred.label if hasattr(pred, 'label') else pred.intent)
                    probabilities.append(pred.probability)
                elif isinstance(model, ProgressiveLMAnalyzer):
                    pred = model.predict_single(text, task_id)
                    predictions.append(pred)  # Text generation output
                elif isinstance(model, GEMNERAnalyzer):
                    pred = model.predict_single(text, task_id, return_labels=True)
                    predictions.append(pred)  # List of entities
            except Exception as e:
                self.logger.warning(f"Prediction failed for {model.__class__.__name__}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from any model")
        
        # Handle different task types
        if task_type in ['sentiment', 'classification', 'intent']:
            # Majority voting for classification tasks
            unique_preds, counts = np.unique(predictions, return_counts=True)
            majority_pred = unique_preds[np.argmax(counts)]
            if return_probability:
                # Weighted average of probabilities for majority class
                prob = np.mean([p for pred, p in zip(predictions, probabilities) if pred == majority_pred])
                return {'label': majority_pred, 'probability': float(prob)}
            return majority_pred
        elif task_type == 'ner':
            # Merge entities (take union of non-overlapping entities)
            merged_entities = []
            seen_spans = set()
            for pred in predictions:
                for start, end, label in pred:
                    span = (start, end)
                    if span not in seen_spans:
                        merged_entities.append((start, end, label))
                        seen_spans.add(span)
            return merged_entities
        elif task_type == 'language_modeling':
            # Return first model's prediction (or implement custom aggregation)
            return predictions[0]
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def save_checkpoint(self, path: str) -> str:
        """Save ensemble model checkpoints."""
        import os
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            model_path = f"{path}_model_{i}_{model.__class__.__name__}.pt"
            model.save_checkpoint(model_path)
            self.logger.info(f"Saved checkpoint for {model.__class__.__name__}: {model_path}")
        # Save ensemble metadata
        metadata = {'model_types': self.model_types, 'weights': self.weights}
        with open(path + '_ensemble.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        return path
    
    def load_checkpoint(self, path: str):
        """Load ensemble model checkpoints."""
        import json
        with open(path + '_ensemble.json', 'r') as f:
            metadata = json.load(f)
        self.model_types = metadata['model_types']
        self.weights = metadata['weights']
        self.models = []
        for i, model_type in enumerate(self.model_types):
            model_path = f"{path}_model_{i}_{model_type}.pt"
            if model_type == 'EWCSentimentAnalyzer':
                model = EWCSentimentAnalyzer.load_from_checkpoint(model_path)
            elif model_type == 'ReplayClassifier':
                model = ReplayClassifier.load_from_checkpoint(model_path)
            elif model_type == 'ProgressiveLMAnalyzer':
                model = ProgressiveLMAnalyzer.load_from_checkpoint(model_path)
            elif model_type == 'GEMNERAnalyzer':
                model = GEMNERAnalyzer.load_from_checkpoint(model_path)
            elif model_type == 'OnlineIntentAnalyzer':
                model = OnlineIntentAnalyzer.load_from_checkpoint(model_path)
            self.models.append(model)
            self.logger.info(f"Loaded checkpoint for {model_type}: {model_path}")