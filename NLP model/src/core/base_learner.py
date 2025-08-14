"""
Base class for continual learning models.
Provides common interface and utilities for all continual learning approaches.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

@dataclass
class TaskMetrics:
    """Metrics for a single task."""
    accuracy: float
    loss: float
    precision: float
    recall: float
    f1_score: float
    task_id: int
    epoch: int

class ContinualLearner(ABC):
    """Abstract base class for continual learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        save_dir: Optional[Path] = None,
        verbose: bool = True
    ):
        self.device = self._setup_device(device)
        self.model = model.to(self.device)
        self.save_dir = Path(save_dir) if save_dir else Path("./checkpoints")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Continual learning state
        self.current_task = 0
        self.task_history: List[int] = []
        self.task_metrics: Dict[int, List[TaskMetrics]] = {}
        self.task_data_info: Dict[int, Dict[str, Any]] = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def train_task(
        self,
        train_data: Any,
        task_id: int,
        epochs: int = 10,
        **kwargs
    ) -> List[TaskMetrics]:
        """Train on a new task."""
        pass
    
    @abstractmethod
    def evaluate_task(
        self,
        test_data: Any,
        task_id: int,
        **kwargs
    ) -> TaskMetrics:
        """Evaluate on a specific task."""
        pass
    
    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """Make predictions on new data."""
        pass
    
    def evaluate_all_tasks(
        self,
        test_datasets: Dict[int, Any]
    ) -> Dict[int, TaskMetrics]:
        """Evaluate model on all previously seen tasks."""
        results = {}
        
        for task_id, test_data in test_datasets.items():
            if task_id in self.task_history:
                results[task_id] = self.evaluate_task(test_data, task_id)
        
        return results
    
    def compute_forgetting(
        self,
        current_results: Dict[int, TaskMetrics],
        baseline_results: Optional[Dict[int, TaskMetrics]] = None
    ) -> Dict[int, float]:
        """Compute backward transfer (forgetting) for each task."""
        if baseline_results is None:
            # Use best performance from task history as baseline
            baseline_results = {}
            for task_id in current_results.keys():
                if task_id in self.task_metrics:
                    best_metric = max(
                        self.task_metrics[task_id],
                        key=lambda x: x.accuracy
                    )
                    baseline_results[task_id] = best_metric
        
        forgetting = {}
        for task_id, current_metric in current_results.items():
            if task_id in baseline_results:
                baseline_acc = baseline_results[task_id].accuracy
                current_acc = current_metric.accuracy
                forgetting[task_id] = baseline_acc - current_acc
            else:
                forgetting[task_id] = 0.0
        
        return forgetting
    
    def compute_forward_transfer(
        self,
        current_task_id: int,
        current_performance: float,
        baseline_performance: Optional[float] = None
    ) -> float:
        """Compute forward transfer for current task."""
        if baseline_performance is None:
            return 0.0
        return current_performance - baseline_performance
    
    def save_checkpoint(self, checkpoint_name: str = None) -> Path:
        """Save model checkpoint."""
        if checkpoint_name is None:
            checkpoint_name = f"task_{self.current_task}_checkpoint.pt"
        
        checkpoint_path = self.save_dir / checkpoint_name
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "current_task": self.current_task,
            "task_history": self.task_history,
            "task_metrics": self.task_metrics,
            "task_data_info": self.task_data_info,
        }
        
        # Add model-specific state
        checkpoint.update(self.get_model_specific_state())
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_task = checkpoint["current_task"]
        self.task_history = checkpoint["task_history"]
        self.task_metrics = checkpoint["task_metrics"]
        self.task_data_info = checkpoint["task_data_info"]
        
        # Load model-specific state
        self.load_model_specific_state(checkpoint)
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    @abstractmethod
    def get_model_specific_state(self) -> Dict[str, Any]:
        """Get model-specific state for checkpointing."""
        return {}
    
    @abstractmethod
    def load_model_specific_state(self, checkpoint: Dict[str, Any]) -> None:
        """Load model-specific state from checkpoint."""
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of continual learning performance."""
        if not self.task_metrics:
            return {"message": "No tasks completed yet"}
        
        summary = {
            "total_tasks": len(self.task_history),
            "current_task": self.current_task,
            "task_history": self.task_history,
        }
        
        # Average metrics across all tasks
        all_accuracies = []
        all_f1_scores = []
        
        for task_id, metrics_list in self.task_metrics.items():
            if metrics_list:
                best_metric = max(metrics_list, key=lambda x: x.accuracy)
                all_accuracies.append(best_metric.accuracy)
                all_f1_scores.append(best_metric.f1_score)
        
        if all_accuracies:
            summary.update({
                "average_accuracy": np.mean(all_accuracies),
                "average_f1_score": np.mean(all_f1_scores),
                "accuracy_std": np.std(all_accuracies),
                "f1_std": np.std(all_f1_scores),
            })
        
        return summary
    
    def visualize_learning_curve(self, task_id: int = None, save_path: Path = None):
        """Visualize learning curves for tasks."""
        try:
            import matplotlib.pyplot as plt
            
            if task_id is None:
                # Plot all tasks
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.ravel()
                
                for i, (tid, metrics_list) in enumerate(self.task_metrics.items()):
                    if i >= 4:  # Limit to 4 subplots
                        break
                    
                    epochs = [m.epoch for m in metrics_list]
                    accuracies = [m.accuracy for m in metrics_list]
                    losses = [m.loss for m in metrics_list]
                    
                    ax = axes[i]
                    ax2 = ax.twinx()
                    
                    line1 = ax.plot(epochs, accuracies, 'b-', label='Accuracy')
                    line2 = ax2.plot(epochs, losses, 'r-', label='Loss')
                    
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy', color='b')
                    ax2.set_ylabel('Loss', color='r')
                    ax.set_title(f'Task {tid} Learning Curve')
                    
                    # Combined legend
                    lines = line1 + line2
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='center right')
                
                plt.tight_layout()
            
            else:
                # Plot specific task
                if task_id not in self.task_metrics:
                    raise ValueError(f"Task {task_id} not found in metrics")
                
                metrics_list = self.task_metrics[task_id]
                epochs = [m.epoch for m in metrics_list]
                accuracies = [m.accuracy for m in metrics_list]
                losses = [m.loss for m in metrics_list]
                
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax2 = ax1.twinx()
                
                line1 = ax1.plot(epochs, accuracies, 'b-', label='Accuracy')
                line2 = ax2.plot(epochs, losses, 'r-', label='Loss')
                
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy', color='b')
                ax2.set_ylabel('Loss', color='r')
                ax1.set_title(f'Task {task_id} Learning Curve')
                
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='center right')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Learning curve saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib not available for visualization")
    
    def __str__(self) -> str:
        """String representation of the learner."""
        return f"{self.__class__.__name__}(tasks_completed={len(self.task_history)})"
    
    def __repr__(self) -> str:
        """Detailed representation of the learner."""
        return (f"{self.__class__.__name__}("
                f"current_task={self.current_task}, "
                f"total_tasks={len(self.task_history)}, "
                f"device={self.device})")