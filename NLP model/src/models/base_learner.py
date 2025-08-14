"""
Base class for continual learning models.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

class BaseLearner(nn.Module):
    """Base class for continual learning models."""
    
    def __init__(self, model_name: str, learning_rate: float, max_length: int, device: str, save_dir: str, verbose: bool = False):
        super().__init__()
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = device
        self.save_dir = save_dir
        self.verbose = verbose
        self.task_id_to_domain = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
    
    def train_task(self, train_data: List[Dict[str, Any]], task_id: int, domain: str, task_type: str, epochs: int, validation_data: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
        """Train the model on a specific task (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement train_task")
    
    def predict_single(self, text: str, task_id: int, task_type: str, return_probability: bool = False) -> Any:
        """Predict for a single input (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement predict_single")
    
    def evaluate_all_tasks(self, test_datasets: Dict[str, List[Dict[str, Any]]]) -> Dict[int, Any]:
        """Evaluate the model on all tasks (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement evaluate_all_tasks")
    
    def save_checkpoint(self, path: str) -> str:
        """Save model checkpoint."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'state_dict': self.state_dict(), 'task_id_to_domain': self.task_id_to_domain}, path)
        self.logger.info(f"Saved checkpoint to {path}")
        return path
    
    @classmethod
    def load_from_checkpoint(cls, path: str, **kwargs):
        """Load model from checkpoint."""
        model = cls(**kwargs)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        model.task_id_to_domain = checkpoint['task_id_to_domain']
        return model
    
    def compute_f1(self, predictions: List[int], labels: List[int]) -> float:
        """Compute macro F1 score."""
        return f1_score(labels, predictions, average='macro') if predictions and labels else 0.0