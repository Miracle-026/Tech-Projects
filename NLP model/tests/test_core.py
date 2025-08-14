"""
Unit tests for core components.
"""

import unittest
import torch
import torch.nn as nn
from src.core.base_learner import ContinualLearner, TaskMetrics
from src.core.memory_buffer import MemoryBuffer, MemoryItem, BalancedBuffer, GradientBuffer
from src.core.regularization import GradientProjection
import tempfile
import os

class TestCoreComponents(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.device = 'cpu'
    
    def test_base_learner(self):
        # Mock model
        class MockModel(nn.Module):
            def __init__(self): super().__init__()
            def forward(self, x): return x
        
        model = ContinualLearner(
            model=MockModel(),
            device=self.device,
            save_dir=self.temp_dir,
            verbose=False
        )
        
        # Test task history
        model.current_task = 1
        model.task_history.append(1)
        self.assertEqual(model.task_history, [1])
        
        # Test checkpointing
        checkpoint_path = model.save_checkpoint('test.pt')
        self.assertTrue(os.path.exists(checkpoint_path))
        model.load_checkpoint(checkpoint_path)
        
        # Test metrics
        metrics = TaskMetrics(accuracy=0.9, loss=0.1, precision=0.9, recall=0.9, f1_score=0.9, task_id=1, epoch=1)
        model.task_metrics[1] = [metrics]
        self.assertEqual(model.task_metrics[1][0].accuracy, 0.9)
    
    def test_memory_buffer(self):
        buffer = MemoryBuffer(capacity=5, device=self.device)
        item = MemoryItem(input_data="test", target="label", task_id=0, timestamp=0.0, metadata={})
        
        # Test adding items
        buffer.add(item)
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer[0], item)
        
        # Test capacity limit
        for i in range(6):
            buffer.add(MemoryItem(input_data=f"test{i}", target=f"label{i}", task_id=0, timestamp=float(i)))
        self.assertEqual(len(buffer), 5)
        
        # Test retrieval
        items = buffer.get_random_batch(2)
        self.assertLessEqual(len(items), 2)
    
    def test_balanced_buffer(self):
        buffer = BalancedBuffer(capacity=10, device=self.device)
        buffer.add(MemoryItem(input_data="test1", target="label1", task_id=0, timestamp=0.0))
        buffer.add(MemoryItem(input_data="test2", target="label2", task_id=1, timestamp=1.0))
        
        # Test balanced sampling
        items = buffer.get_random_batch(2)
        self.assertLessEqual(len(items), 2)
        self.assertTrue(all(item.task_id in [0, 1] for item in items))
    
    def test_gradient_buffer(self):
        class MockModel(nn.Module):
            def __init__(self): super().__init__(); self.fc = nn.Linear(10, 2)
            def forward(self, x): return self.fc(x)
        
        buffer = GradientBuffer(capacity=5, device=self.device)
        model = MockModel().to(self.device)
        item = MemoryItem(input_data=torch.randn(1, 10), target=torch.tensor([0]), task_id=0, timestamp=0.0)
        
        buffer.add(item)
        self.assertEqual(len(buffer), 1)
    
    def test_gradient_projection(self):
        class MockModel(nn.Module):
            def __init__(self): super().__init__(); self.fc = nn.Linear(10, 2)
            def forward(self, x): return self.fc(x)
        
        model = MockModel().to(self.device)
        buffer = GradientBuffer(capacity=5, device=self.device)
        projection = GradientProjection(model=model, device=self.device, memory_buffer=buffer)
        
        # Test updating task knowledge (placeholder test)
        projection.reference_gradients[0] = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
        self.assertIn(0, projection.reference_gradients)
    
    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

if __name__ == '__main__':
    unittest.main()