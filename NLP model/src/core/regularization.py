"""
Regularization techniques for continual learning to prevent catastrophic forgetting.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
from collections import defaultdict
import math

class RegularizationTechnique(ABC):
    """Abstract base class for regularization techniques."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.task_id = 0
    
    @abstractmethod
    def compute_penalty(self, current_loss: torch.Tensor) -> torch.Tensor:
        """Compute regularization penalty."""
        pass
    
    @abstractmethod
    def update_task_knowledge(self, task_id: int, dataloader: Any = None) -> None:
        """Update task-specific knowledge after completing a task."""
        pass

class ElasticWeightConsolidation(RegularizationTechnique):
    """Elastic Weight Consolidation (EWC) regularization."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lambda_reg: float = 1000.0,
        sample_size: int = 1000
    ):
        super().__init__(model, device)
        self.lambda_reg = lambda_reg
        self.sample_size = sample_size
        
        # Store Fisher information and optimal parameters
        self.fisher_matrices: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optimal_params: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def compute_penalty(self, current_loss: torch.Tensor) -> torch.Tensor:
        """Compute EWC regularization penalty."""
        penalty = 0.0
        
        for task_id in self.fisher_matrices.keys():
            fisher = self.fisher_matrices[task_id]
            optimal = self.optimal_params[task_id]
            
            for name, param in self.model.named_parameters():
                if name in fisher and param.requires_grad:
                    penalty += (fisher[name] * (param - optimal[name]) ** 2).sum()
        
        return self.lambda_reg / 2 * penalty
    
    def update_task_knowledge(self, task_id: int, dataloader: Any = None) -> None:
        """Compute and store Fisher information matrix."""
        if dataloader is None:
            raise ValueError("DataLoader required for EWC Fisher computation")
        
        # Store optimal parameters
        optimal_params = {}
        for name, param in self.model.named_parameters():
            optimal_params[name] = param.detach().clone()
        self.optimal_params[task_id] = optimal_params
        
        # Compute Fisher information matrix
        fisher = self._compute_fisher_information(dataloader)
        self.fisher_matrices[task_id] = fisher
    
    def _compute_fisher_information(self, dataloader: Any) -> Dict[str, torch.Tensor]:
        """Compute Fisher information matrix using samples from dataloader."""
        fisher = {}
        
        # Initialize Fisher matrix
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
        
        self.model.eval()
        sample_count = 0
        
        for batch in dataloader:
            if sample_count >= self.sample_size:
                break
            
            # Unpack batch (assuming standard format)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, None
            
            inputs = inputs.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Use log-likelihood for Fisher computation
            if targets is not None:
                targets = targets.to(self.device)
                log_likelihood = F.cross_entropy(outputs, targets)
            else:
                # Use model's own predictions as pseudo-targets
                pseudo_targets = outputs.max(1)[1]
                log_likelihood = F.cross_entropy(outputs, pseudo_targets)
            
            # Compute gradients
            gradients = grad(log_likelihood, self.model.parameters(), 
                           retain_graph=False, create_graph=False)
            
            # Accumulate Fisher information (squared gradients)
            for (name, param), gradient in zip(self.model.named_parameters(), gradients):
                if param.requires_grad and gradient is not None:
                    fisher[name] += gradient ** 2
            
            sample_count += inputs.size(0)
        
        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= sample_count
        
        self.model.train()
        return fisher

class SynapticIntelligence(RegularizationTechnique):
    """Synaptic Intelligence (SI) regularization."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        si_c: float = 0.1,
        xi: float = 1.0
    ):
        super().__init__(model, device)
        self.si_c = si_c
        self.xi = xi
        
        # Track parameter changes and gradients
        self.omega: Dict[str, torch.Tensor] = {}
        self.previous_params: Dict[str, torch.Tensor] = {}
        self.accumulated_gradients: Dict[str, torch.Tensor] = {}
        
        self._initialize_si_variables()
    
    def _initialize_si_variables(self) -> None:
        """Initialize SI tracking variables."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.omega[name] = torch.zeros_like(param.data)
                self.previous_params[name] = param.detach().clone()
                self.accumulated_gradients[name] = torch.zeros_like(param.data)
    
    def compute_penalty(self, current_loss: torch.Tensor) -> torch.Tensor:
        """Compute SI regularization penalty."""
        penalty = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.omega and param.requires_grad:
                penalty += (self.omega[name] * 
                          (param - self.previous_params[name]) ** 2).sum()
        
        return self.si_c * penalty
    
    def update_gradients(self) -> None:
        """Update accumulated gradients (call during training)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.accumulated_gradients[name] += param.grad.detach().clone()
    
    def update_task_knowledge(self, task_id: int, dataloader: Any = None) -> None:
        """Update omega values after task completion."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Compute parameter change
                param_change = param.detach() - self.previous_params[name]
                
                # Update omega (importance weights)
                if param_change.abs().sum() > 0:
                    self.omega[name] += (self.accumulated_gradients[name] * 
                                       param_change) / (param_change ** 2 + self.xi)
                
                # Update previous parameters
                self.previous_params[name] = param.detach().clone()
                
                # Reset accumulated gradients
                self.accumulated_gradients[name].zero_()

class MemoryAwareRegularization(RegularizationTechnique):
    """Memory-Aware Regularization (MAR) technique."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        alpha: float = 0.5,
        memory_strength: float = 1.0
    ):
        super().__init__(model, device)
        self.alpha = alpha
        self.memory_strength = memory_strength
        
        # Store task-specific attention weights
        self.attention_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self.reference_params: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def compute_penalty(self, current_loss: torch.Tensor) -> torch.Tensor:
        """Compute MAR penalty based on attention weights."""
        penalty = 0.0
        
        for task_id in self.attention_weights.keys():
            attention = self.attention_weights[task_id]
            reference = self.reference_params[task_id]
            
            for name, param in self.model.named_parameters():
                if name in attention and param.requires_grad:
                    penalty += (attention[name] * 
                              (param - reference[name]) ** 2).sum()
        
        return self.memory_strength * penalty
    
    def update_task_knowledge(self, task_id: int, dataloader: Any = None) -> None:
        """Update attention weights based on parameter importance."""
        # Store reference parameters
        reference_params = {}
        for name, param in self.model.named_parameters():
            reference_params[name] = param.detach().clone()
        self.reference_params[task_id] = reference_params
        
        # Compute attention weights (simplified - could use more sophisticated methods)
        attention_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Use parameter magnitude as importance measure
                importance = torch.abs(param.detach())
                attention_weights[name] = importance / (importance.sum() + 1e-8)
        
        self.attention_weights[task_id] = attention_weights

class L2Regularization(RegularizationTechnique):
    """Simple L2 regularization on parameter changes."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        lambda_l2: float = 0.01
    ):
        super().__init__(model, device)
        self.lambda_l2 = lambda_l2
        self.reference_params: Dict[str, torch.Tensor] = {}
    
    def compute_penalty(self, current_loss: torch.Tensor) -> torch.Tensor:
        """Compute L2 penalty on parameter changes."""
        penalty = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.reference_params and param.requires_grad:
                penalty += ((param - self.reference_params[name]) ** 2).sum()
        
        return self.lambda_l2 * penalty
    
    def update_task_knowledge(self, task_id: int, dataloader: Any = None) -> None:
        """Store current parameters as reference."""
        for name, param in self.model.named_parameters():
            self.reference_params[name] = param.detach().clone()

class GradientProjection(RegularizationTechnique):
    """Gradient Episodic Memory (GEM) style gradient projection."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        memory_buffer: Any = None
    ):
        super().__init__(model, device)
        self.memory_buffer = memory_buffer
        self.reference_gradients: Dict[int, torch.Tensor] = {}
    
    def compute_penalty(self, current_loss: torch.Tensor) -> torch.Tensor:
        """GEM doesn't use penalty - it projects gradients."""
        return torch.tensor(0.0, device=self.device)
    
    def project_gradients(self, task_id: int) -> None:
        """Project current gradients to not interfere with previous tasks."""
        if not self.reference_gradients:
            return
        
        # Get current gradients
        current_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.view(-1))
        
        if not current_grads:
            return
        
        current_grad_vec = torch.cat(current_grads)
        
        # Project against all previous task gradients
        for prev_task_id, ref_grad in self.reference_gradients.items():
            if prev_task_id != task_id:
                # Check if current gradient violates constraint
                dot_product = torch.dot(current_grad_vec, ref_grad)
                
                if dot_product < 0:
                    # Project gradient
                    projection = (dot_product / torch.dot(ref_grad, ref_grad)) * ref_grad
                    current_grad_vec = current_grad_vec - projection
        
        # Update model gradients with projected values
        idx = 0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_size = param.grad.numel()
                param.grad.data = current_grad_vec[idx:idx + grad_size].view(param.grad.shape)
                idx += grad_size
    
    def update_task_knowledge(self, task_id: int, dataloader: Any = None) -> None:
        """Store reference gradients for the completed task."""
        if self.memory_buffer is None or dataloader is None:
            return
        
        # Sample from memory buffer for this task
        memory_samples = self.memory_buffer.sample(min(100, len(self.memory_buffer)))
        
        if not memory_samples:
            return
        
        # Compute reference gradients on memory samples
        self.model.eval()
        total_loss = 0.0
        
        for sample in memory_samples:
            # Forward pass (implementation depends on data format)
            # This is a placeholder - actual implementation would depend on specific data format
            loss = self._compute_sample_loss(sample)
            total_loss += loss
        
        total_loss /= len(memory_samples)
        
        # Compute gradients
        gradients = grad(total_loss, self.model.parameters(), 
                        retain_graph=False, create_graph=False)
        
        # Store flattened gradients
        grad_vec = []
        for g in gradients:
            if g is not None:
                grad_vec.append(g.view(-1))
        
        if grad_vec:
            self.reference_gradients[task_id] = torch.cat(grad_vec)
        
        self.model.train()
    
    def _compute_sample_loss(self, sample: Any) -> torch.Tensor:
        """Compute loss for a single memory sample (placeholder)."""
        # This should be implemented based on specific model and data format
        return torch.tensor(0.0, device=self.device)

class RegularizationManager:
    """Manager for multiple regularization techniques."""
    
    def __init__(
        self,
        model: nn.Module,
        techniques: List[RegularizationTechnique],
        weights: Optional[List[float]] = None
    ):
        self.model = model
        self.techniques = techniques
        self.weights = weights or [1.0] * len(techniques)
        
        if len(self.weights) != len(self.techniques):
            raise ValueError("Number of weights must match number of techniques")
    
    def compute_total_penalty(self, current_loss: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of all regularization penalties."""
        total_penalty = torch.tensor(0.0, device=current_loss.device)
        
        for technique, weight in zip(self.techniques, self.weights):
            penalty = technique.compute_penalty(current_loss)
            total_penalty += weight * penalty
        
        return total_penalty
    
    def update_all_techniques(self, task_id: int, dataloader: Any = None) -> None:
        """Update all regularization techniques after task completion."""
        for technique in self.techniques:
            technique.update_task_knowledge(task_id, dataloader)
    
    def project_gradients(self, task_id: int) -> None:
        """Apply gradient projection if any technique supports it."""
        for technique in self.techniques:
            if hasattr(technique, 'project_gradients'):
                technique.project_gradients(task_id)
    
    def get_technique_by_type(self, technique_type: type) -> Optional[RegularizationTechnique]:
        """Get first technique of specified type."""
        for technique in self.techniques:
            if isinstance(technique, technique_type):
                return technique
        return None