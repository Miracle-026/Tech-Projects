"""
Memory buffer implementations for experience replay in continual learning.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import random
from collections import deque, defaultdict
from dataclasses import dataclass
import pickle
from pathlib import Path

@dataclass
class MemoryItem:
    """Single item in memory buffer."""
    input_data: Any
    target: Any
    task_id: int
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class MemoryBuffer(ABC):
    """Abstract base class for memory buffers."""
    
    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.buffer: List[MemoryItem] = []
        self.current_size = 0
    
    @abstractmethod
    def add(self, item: MemoryItem) -> None:
        """Add item to buffer."""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> List[MemoryItem]:
        """Sample batch from buffer."""
        pass
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self.current_size >= self.capacity
    
    def __len__(self) -> int:
        return self.current_size
    
    def clear(self) -> None:
        """Clear all items from buffer."""
        self.buffer.clear()
        self.current_size = 0

class RandomBuffer(MemoryBuffer):
    """Random sampling memory buffer with uniform replacement."""
    
    def add(self, item: MemoryItem) -> None:
        """Add item with random replacement if buffer is full."""
        if not self.is_full():
            self.buffer.append(item)
            self.current_size += 1
        else:
            # Random replacement
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = item
    
    def sample(self, batch_size: int) -> List[MemoryItem]:
        """Sample random batch from buffer."""
        if self.current_size == 0:
            return []
        
        sample_size = min(batch_size, self.current_size)
        return random.sample(self.buffer, sample_size)

class FIFOBuffer(MemoryBuffer):
    """First-In-First-Out memory buffer."""
    
    def __init__(self, capacity: int, device: str = "cpu"):
        super().__init__(capacity, device)
        self.buffer = deque(maxlen=capacity)
    
    def add(self, item: MemoryItem) -> None:
        """Add item with FIFO replacement."""
        if len(self.buffer) < self.capacity:
            self.current_size += 1
        self.buffer.append(item)
    
    def sample(self, batch_size: int) -> List[MemoryItem]:
        """Sample random batch from buffer."""
        if self.current_size == 0:
            return []
        
        sample_size = min(batch_size, self.current_size)
        return random.sample(list(self.buffer), sample_size)

class BalancedBuffer(MemoryBuffer):
    """Balanced memory buffer that maintains equal samples per task."""
    
    def __init__(self, capacity: int, device: str = "cpu"):
        super().__init__(capacity, device)
        self.task_buffers: Dict[int, List[MemoryItem]] = defaultdict(list)
        self.task_counts: Dict[int, int] = defaultdict(int)
        self.seen_tasks: set = set()
    
    def add(self, item: MemoryItem) -> None:
        """Add item with balanced sampling across tasks."""
        task_id = item.task_id
        self.seen_tasks.add(task_id)
        
        # Calculate capacity per task
        capacity_per_task = self.capacity // len(self.seen_tasks)
        
        # Add to task-specific buffer
        if len(self.task_buffers[task_id]) < capacity_per_task:
            self.task_buffers[task_id].append(item)
            self.task_counts[task_id] += 1
            self.current_size += 1
        else:
            # Random replacement within task
            idx = random.randint(0, capacity_per_task - 1)
            self.task_buffers[task_id][idx] = item
        
        # Rebalance if needed
        self._rebalance()
    
    def _rebalance(self) -> None:
        """Rebalance buffer across tasks."""
        if not self.seen_tasks:
            return
        
        capacity_per_task = self.capacity // len(self.seen_tasks)
        
        # Trim oversized task buffers
        for task_id in self.seen_tasks:
            task_buffer = self.task_buffers[task_id]
            if len(task_buffer) > capacity_per_task:
                # Keep random subset
                self.task_buffers[task_id] = random.sample(
                    task_buffer, capacity_per_task
                )
                self.task_counts[task_id] = capacity_per_task
        
        # Update current size
        self.current_size = sum(self.task_counts.values())
    
    def sample(self, batch_size: int) -> List[MemoryItem]:
        """Sample balanced batch across all tasks."""
        if self.current_size == 0:
            return []
        
        # Sample proportionally from each task
        samples = []
        samples_per_task = batch_size // len(self.seen_tasks) if self.seen_tasks else 0
        remaining_samples = batch_size % len(self.seen_tasks) if self.seen_tasks else 0
        
        for task_id in self.seen_tasks:
            task_samples = samples_per_task
            if remaining_samples > 0:
                task_samples += 1
                remaining_samples -= 1
            
            task_buffer = self.task_buffers[task_id]
            if task_buffer:
                task_sample_size = min(task_samples, len(task_buffer))
                samples.extend(random.sample(task_buffer, task_sample_size))
        
        return samples

class GradientBuffer(MemoryBuffer):
    """Memory buffer that stores gradients for GEM-style replay."""
    
    def __init__(self, capacity: int, device: str = "cpu"):
        super().__init__(capacity, device)
        self.gradients: Dict[int, torch.Tensor] = {}
        self.task_buffers: Dict[int, List[MemoryItem]] = defaultdict(list)
    
    def add(self, item: MemoryItem) -> None:
        """Add item to task-specific buffer."""
        task_id = item.task_id
        self.task_buffers[task_id].append(item)
        
        # Maintain capacity per task
        if len(self.task_buffers[task_id]) > self.capacity // 10:  # Max 10% per task
            self.task_buffers[task_id].pop(0)  # FIFO
        
        self.current_size = sum(len(buffer) for buffer in self.task_buffers.values())
    
    def sample(self, batch_size: int) -> List[MemoryItem]:
        """Sample from all tasks for gradient computation."""
        samples = []
        for task_buffer in self.task_buffers.values():
            if task_buffer:
                sample_size = min(batch_size // len(self.task_buffers), len(task_buffer))
                samples.extend(random.sample(task_buffer, sample_size))
        return samples
    
    def store_gradients(self, task_id: int, gradients: torch.Tensor) -> None:
        """Store reference gradients for a task."""
        self.gradients[task_id] = gradients.detach().clone()
    
    def get_reference_gradients(self, task_id: int) -> Optional[torch.Tensor]:
        """Get stored reference gradients for a task."""
        return self.gradients.get(task_id)

class SemanticBuffer(MemoryBuffer):
    """Memory buffer that uses semantic similarity for intelligent replacement."""
    
    def __init__(
        self, 
        capacity: int, 
        device: str = "cpu",
        embedding_dim: int = 768,
        similarity_threshold: float = 0.8
    ):
        super().__init__(capacity, device)
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.embeddings: List[torch.Tensor] = []
    
    def add(self, item: MemoryItem) -> None:
        """Add item with semantic-aware replacement."""
        # Extract or compute embedding
        if hasattr(item, 'embedding') and item.embedding is not None:
            embedding = item.embedding
        else:
            embedding = self._compute_embedding(item.input_data)
        
        if not self.is_full():
            self.buffer.append(item)
            self.embeddings.append(embedding)
            self.current_size += 1
        else:
            # Find most similar item and replace if above threshold
            similarities = [
                torch.cosine_similarity(embedding.unsqueeze(0), emb.unsqueeze(0))
                for emb in self.embeddings
            ]
            max_sim_idx = torch.argmax(torch.tensor(similarities))
            max_similarity = similarities[max_sim_idx]
            
            if max_similarity > self.similarity_threshold:
                # Replace similar item
                self.buffer[max_sim_idx] = item
                self.embeddings[max_sim_idx] = embedding
            else:
                # Random replacement if no similar item found
                idx = random.randint(0, self.capacity - 1)
                self.buffer[idx] = item
                self.embeddings[idx] = embedding
    
    def _compute_embedding(self, input_data: Any) -> torch.Tensor:
        """Compute embedding for input data (placeholder)."""
        # In practice, this would use a pre-trained encoder
        if isinstance(input_data, torch.Tensor):
            return torch.randn(self.embedding_dim)
        else:
            return torch.randn(self.embedding_dim)
    
    def sample(self, batch_size: int) -> List[MemoryItem]:
        """Sample diverse batch based on embeddings."""
        if self.current_size == 0:
            return []
        
        sample_size = min(batch_size, self.current_size)
        
        # Use k-means clustering or diversity-based sampling
        if sample_size == self.current_size:
            return list(self.buffer)
        
        # Greedy diversity sampling
        selected_indices = []
        remaining_indices = list(range(self.current_size))
        
        # Start with random item
        first_idx = random.choice(remaining_indices)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Greedily select most diverse items
        while len(selected_indices) < sample_size and remaining_indices:
            max_min_distance = -1
            best_idx = None
            
            for idx in remaining_indices:
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    distance = 1 - torch.cosine_similarity(
                        self.embeddings[idx].unsqueeze(0),
                        self.embeddings[selected_idx].unsqueeze(0)
                    )
                    min_distance = min(min_distance, distance.item())
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        return [self.buffer[i] for i in selected_indices]

class HierarchicalBuffer(MemoryBuffer):
    """Hierarchical memory buffer with multiple levels of storage."""
    
    def __init__(
        self,
        capacity: int,
        device: str = "cpu",
        levels: int = 3,
        level_ratios: List[float] = None
    ):
        super().__init__(capacity, device)
        self.levels = levels
        self.level_ratios = level_ratios or [0.5, 0.3, 0.2]  # L1, L2, L3
        
        # Create level-specific buffers
        self.level_buffers: List[List[MemoryItem]] = []
        self.level_capacities: List[int] = []
        
        for i, ratio in enumerate(self.level_ratios):
            level_capacity = int(capacity * ratio)
            self.level_capacities.append(level_capacity)
            self.level_buffers.append([])
    
    def add(self, item: MemoryItem) -> None:
        """Add item to appropriate level based on importance."""
        # Start at level 0 (most important)
        level = 0
        
        # Add to level 0 first
        if len(self.level_buffers[level]) < self.level_capacities[level]:
            self.level_buffers[level].append(item)
        else:
            # Move oldest item to next level and add new item
            self._cascade_down(level)
            self.level_buffers[level].append(item)
        
        self.current_size = sum(len(buffer) for buffer in self.level_buffers)
    
    def _cascade_down(self, from_level: int) -> None:
        """Move oldest item from one level to the next."""
        if from_level >= len(self.level_buffers) - 1:
            # Last level, just remove oldest
            if self.level_buffers[from_level]:
                self.level_buffers[from_level].pop(0)
            return
        
        # Move oldest item to next level
        if self.level_buffers[from_level]:
            item = self.level_buffers[from_level].pop(0)
            next_level = from_level + 1
            
            if len(self.level_buffers[next_level]) < self.level_capacities[next_level]:
                self.level_buffers[next_level].append(item)
            else:
                # Cascade further down
                self._cascade_down(next_level)
                self.level_buffers[next_level].append(item)
    
    def sample(self, batch_size: int) -> List[MemoryItem]:
        """Sample from all levels with bias toward higher levels."""
        if self.current_size == 0:
            return []
        
        samples = []
        remaining_batch_size = batch_size
        
        # Sample from each level proportionally
        for level, buffer in enumerate(self.level_buffers):
            if not buffer or remaining_batch_size == 0:
                continue
            
            # Higher levels get more samples
            level_weight = 2 ** (len(self.level_buffers) - level - 1)
            total_weight = sum(2 ** (len(self.level_buffers) - i - 1) 
                             for i in range(len(self.level_buffers)))
            
            level_samples = int(batch_size * level_weight / total_weight)
            level_samples = min(level_samples, len(buffer), remaining_batch_size)
            
            if level_samples > 0:
                samples.extend(random.sample(buffer, level_samples))
                remaining_batch_size -= level_samples
        
        return samples

class MemoryBufferManager:
    """Manager for multiple memory buffers with different strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.buffers: Dict[str, MemoryBuffer] = {}
        self._initialize_buffers()
    
    def _initialize_buffers(self) -> None:
        """Initialize buffers based on configuration."""
        buffer_configs = self.config.get('buffers', {})
        
        for name, buffer_config in buffer_configs.items():
            buffer_type = buffer_config.get('type', 'random')
            capacity = buffer_config.get('capacity', 1000)
            device = buffer_config.get('device', 'cpu')
            
            if buffer_type == 'random':
                self.buffers[name] = RandomBuffer(capacity, device)
            elif buffer_type == 'fifo':
                self.buffers[name] = FIFOBuffer(capacity, device)
            elif buffer_type == 'balanced':
                self.buffers[name] = BalancedBuffer(capacity, device)
            elif buffer_type == 'gradient':
                self.buffers[name] = GradientBuffer(capacity, device)
            elif buffer_type == 'semantic':
                embedding_dim = buffer_config.get('embedding_dim', 768)
                similarity_threshold = buffer_config.get('similarity_threshold', 0.8)
                self.buffers[name] = SemanticBuffer(
                    capacity, device, embedding_dim, similarity_threshold
                )
            elif buffer_type == 'hierarchical':
                levels = buffer_config.get('levels', 3)
                level_ratios = buffer_config.get('level_ratios', None)
                self.buffers[name] = HierarchicalBuffer(
                    capacity, device, levels, level_ratios
                )
    
    def get_buffer(self, name: str) -> MemoryBuffer:
        """Get buffer by name."""
        if name not in self.buffers:
            raise ValueError(f"Buffer '{name}' not found")
        return self.buffers[name]
    
    def add_to_buffer(self, buffer_name: str, item: MemoryItem) -> None:
        """Add item to specific buffer."""
        self.buffers[buffer_name].add(item)
    
    def sample_from_buffer(self, buffer_name: str, batch_size: int) -> List[MemoryItem]:
        """Sample from specific buffer."""
        return self.buffers[buffer_name].sample(batch_size)
    
    def save_buffers(self, save_dir: Path) -> None:
        """Save all buffers to disk."""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, buffer in self.buffers.items():
            buffer_path = save_dir / f"{name}_buffer.pkl"
            with open(buffer_path, 'wb') as f:
                pickle.dump(buffer, f)
    
    def load_buffers(self, load_dir: Path) -> None:
        """Load all buffers from disk."""
        for name in self.buffers.keys():
            buffer_path = load_dir / f"{name}_buffer.pkl"
            if buffer_path.exists():
                with open(buffer_path, 'rb') as f:
                    self.buffers[name] = pickle.load(f)
    
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all buffers."""
        summary = {}
        for name, buffer in self.buffers.items():
            summary[name] = {
                'type': buffer.__class__.__name__,
                'capacity': buffer.capacity,
                'current_size': len(buffer),
                'utilization': len(buffer) / buffer.capacity if buffer.capacity > 0 else 0,
            }
        return summary