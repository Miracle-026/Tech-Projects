"""
Experience Replay-based Text Classification with BERT.
Maintains a memory buffer of previous examples to prevent catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import time
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from ..core.base_learner import ContinualLearner, TaskMetrics
from ..core.memory_buffer import MemoryBuffer, MemoryItem, BalancedBuffer


@dataclass
class ClassificationExample:
    """Single text classification example."""
    text: str
    label: int
    category: str
    task_id: int = 0


class TextClassificationDataset(Dataset):
    """Dataset for text classification."""
    
    def __init__(
        self, 
        examples: List[ClassificationExample], 
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            example.text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(example.label, dtype=torch.long),
            'task_id': torch.tensor(example.task_id, dtype=torch.long),
            'category': example.category
        }


class DynamicTextClassifier(nn.Module):
    """Text classifier that can adapt to new classes."""
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        initial_num_classes: int = 2,
        dropout_rate: float = 0.1,
        hidden_size: Optional[int] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dynamic classification head
        bert_hidden_size = self.bert.config.hidden_size
        self.hidden_size = hidden_size or bert_hidden_size // 2
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(bert_hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Start with initial classifier
        self.classifier = nn.Linear(self.hidden_size, initial_num_classes)
        self.num_classes = initial_num_classes
        
        # Keep track of class mappings
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
    
    def expand_classifier(self, new_num_classes: int) -> None:
        """Expand classifier to accommodate new classes."""
        if new_num_classes <= self.num_classes:
            return
        
        # Create new classifier with more outputs
        old_classifier = self.classifier
        self.classifier = nn.Linear(self.hidden_size, new_num_classes)
        
        # Copy old weights
        with torch.no_grad():
            self.classifier.weight[:self.num_classes] = old_classifier.weight
            self.classifier.bias[:self.num_classes] = old_classifier.bias
            
            # Initialize new weights
            nn.init.xavier_uniform_(self.classifier.weight[self.num_classes:])
            nn.init.zeros_(self.classifier.bias[self.num_classes:])
        
        self.num_classes = new_num_classes
    
    def add_new_class(self, class_name: str) -> int:
        """Add a new class and return its index."""
        if class_name in self.class_to_idx:
            return self.class_to_idx[class_name]
        
        new_idx = len(self.class_to_idx)
        self.class_to_idx[class_name] = new_idx
        self.idx_to_class[new_idx] = class_name
        
        # Expand classifier if needed
        if new_idx >= self.num_classes:
            self.expand_classifier(new_idx + 1)
        
        return new_idx
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # Extract features
        features = self.feature_extractor(pooled_output)
        
        # Classification
        logits = self.classifier(features)
        return logits


class ExperienceReplayClassifier(ContinualLearner):
    """Experience replay-based continual text classifier."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        memory_size: int = 1000,
        replay_batch_size: int = 32,
        replay_frequency: int = 1,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "auto",
        save_dir: Optional[str] = None,
        verbose: bool = True
    ):
        # Initialize model
        model = DynamicTextClassifier(model_name=model_name)
        super().__init__(model, device, save_dir, verbose)
        
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.replay_batch_size = replay_batch_size
        self.replay_frequency = replay_frequency
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize memory buffer
        self.memory_buffer = BalancedBuffer(capacity=memory_size, device=self.device)
        
        # Track categories and performance
        self.category_to_task_id: Dict[str, int] = {}
        self.task_categories: Dict[int, List[str]] = {}
    
    def train_task(
        self,
        train_data: List[ClassificationExample],
        task_id: int,
        categories: List[str],
        epochs: int = 3,
        validation_data: Optional[List[ClassificationExample]] = None,
        **kwargs
    ) -> List[TaskMetrics]:
        """Train on a new classification task with experience replay."""
        
        self.logger.info(
            f"Training task {task_id} with categories {categories} for {epochs} epochs"
        )
        
        # Update category mappings
        for category in categories:
            self.category_to_task_id[category] = task_id
            if task_id not in self.task_categories:
                self.task_categories[task_id] = []
            if category not in self.task_categories[task_id]:
                self.task_categories[task_id].append(category)
        
        # Add new classes to model
        label_mapping = {}
        for example in train_data:
            if example.category not in label_mapping:
                class_idx = self.model.add_new_class(example.category)
                label_mapping[example.category] = class_idx
        
        # Update labels in training data
        for example in train_data:
            example.label = label_mapping[example.category]
            example.task_id = task_id
        
        # Create dataset and dataloader
        train_dataset = TextClassificationDataset(train_data, self.tokenizer, self.max_length)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        task_metrics = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            replay_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            all_predictions = []
            all_labels = []
            
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                # Current task batch
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass on current task
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                current_loss = criterion(logits, labels)
                
                total_loss_batch = current_loss
                
                # Experience replay
                if len(self.memory_buffer) > 0 and batch_idx % self.replay_frequency == 0:
                    replay_items = self.memory_buffer.sample(self.replay_batch_size)
                    
                    if replay_items:
                        replay_loss_batch = self._replay_step(replay_items, criterion)
                        total_loss_batch += replay_loss_batch
                        replay_loss += replay_loss_batch.item()
                
                # Backward pass
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Compute metrics
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total_loss += current_loss.item()
                
                if self.verbose and batch_idx % 100 == 0:
                    avg_replay_loss = replay_loss / max(1, (batch_idx // self.replay_frequency + 1))
                    self.logger.info(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                        f"Current Loss: {current_loss.item():.4f}, "
                        f"Replay Loss: {avg_replay_loss:.4f}"
                    )
            
            # Add current task examples to memory buffer
            self._update_memory_buffer(train_data)
            
            # Compute epoch metrics
            epoch_accuracy = correct_predictions / total_samples
            epoch_loss = total_loss / len(train_loader)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted', zero_division=0
            )
            
            epoch_metrics = TaskMetrics(
                accuracy=epoch_accuracy,
                loss=epoch_loss,
                precision=precision,
                recall=recall,
                f1_score=f1,
                task_id=task_id,
                epoch=epoch
            )
            
            task_metrics.append(epoch_metrics)
            
            epoch_time = time.time() - epoch_start_time
            avg_replay_loss = replay_loss / len(train_loader)
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - "
                f"Loss: {epoch_loss:.4f}, Replay: {avg_replay_loss:.4f}, "
                f"Accuracy: {epoch_accuracy:.4f}, F1: {f1:.4f}"
            )
            
            # Validation
            if validation_data:
                val_metrics = self.evaluate_task(validation_data, task_id, categories)
                self.logger.info(
                    f"Validation - Accuracy: {val_metrics.accuracy:.4f}, "
                    f"F1: {val_metrics.f1_score:.4f}"
                )
        
        # Update task information
        self.current_task = task_id
        if task_id not in self.task_history:
            self.task_history.append(task_id)
        self.task_metrics[task_id] = task_metrics
        self.task_data_info[task_id] = {
            'categories': categories,
            'num_samples': len(train_data),
            'epochs': epochs,
            'memory_size_after': len(self.memory_buffer)
        }
        
        self.logger.info(f"Task {task_id} training completed. Memory buffer size: {len(self.memory_buffer)}")
        return task_metrics
    
    def _replay_step(self, replay_items: List[MemoryItem], criterion: nn.Module) -> torch.Tensor:
        """Perform one replay step with memory buffer items."""
        if not replay_items:
            return torch.tensor(0.0, device=self.device)
        
        # Prepare replay batch
        replay_texts = []
        replay_labels = []
        
        for item in replay_items:
            if hasattr(item.input_data, 'text'):
                replay_texts.append(item.input_data.text)
            else:
                replay_texts.append(str(item.input_data))
            replay_labels.append(item.target)
        
        # Tokenize replay batch
        encoding = self.tokenizer(
            replay_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        replay_input_ids = encoding['input_ids'].to(self.device)
        replay_attention_mask = encoding['attention_mask'].to(self.device)
        replay_labels_tensor = torch.tensor(replay_labels, dtype=torch.long, device=self.device)
        
        # Forward pass on replay batch
        replay_logits = self.model(replay_input_ids, replay_attention_mask)
        replay_loss = criterion(replay_logits, replay_labels_tensor)
        
        return replay_loss
    
    def _update_memory_buffer(self, train_data: List[ClassificationExample]) -> None:
        """Add training examples to memory buffer."""
        for example in train_data:
            memory_item = MemoryItem(
                input_data=example,
                target=example.label,
                task_id=example.task_id,
                timestamp=time.time(),
                metadata={'category': example.category}
            )
            self.memory_buffer.add(memory_item)
    
    def evaluate_task(
        self,
        test_data: List[ClassificationExample],
        task_id: int,
        categories: List[str] = None,
        **kwargs
    ) -> TaskMetrics:
        """Evaluate on classification test data."""
        
        # Update labels with current class mapping
        for example in test_data:
            if example.category in self.model.class_to_idx:
                example.label = self.model.class_to_idx[example.category]
            else:
                # Unknown category - this shouldn't happen in proper evaluation
                example.label = -1
        
        # Filter out examples with unknown categories
        valid_test_data = [ex for ex in test_data if ex.label != -1]
        
        if not valid_test_data:
            self.logger.warning(f"No valid test examples for task {task_id}")
            return TaskMetrics(0.0, float('inf'), 0.0, 0.0, 0.0, task_id, -1)
        
        # Create dataset and dataloader
        test_dataset = TextClassificationDataset(valid_test_data, self.tokenizer, self.max_length)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Collect predictions and labels
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        avg_loss = total_loss / len(test_loader)
        
        metrics = TaskMetrics(
            accuracy=accuracy,
            loss=avg_loss,
            precision=precision,
            recall=recall,
            f1_score=f1,
            task_id=task_id,
            epoch=-1  # Evaluation metric
        )
        
        categories_str = ', '.join(categories) if categories else 'N/A'
        self.logger.info(
            f"Task {task_id} ({categories_str}) evaluation - "
            f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Loss: {avg_loss:.4f}"
        )
        
        return metrics
    
    def predict(
        self, 
        texts: List[str], 
        return_probabilities: bool = False,
        return_categories: bool = True,
        **kwargs
    ) -> List[Any]:
        """Make predictions on new text data."""
        
        self.model.eval()
        predictions = []
        probabilities = []
        categories = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                batch_probs = torch.softmax(logits, dim=1)
                batch_predictions = torch.argmax(logits, dim=1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(batch_probs.cpu().numpy())
                
                # Convert indices to categories
                for pred_idx in batch_predictions.cpu().numpy():
                    category = self.model.idx_to_class.get(pred_idx, f"unknown_{pred_idx}")
                    categories.extend([category])
        
        # Prepare return format
        results = []
        for i in range(len(texts)):
            result = predictions[i]
            if return_categories:
                result = categories[i]
            
            if return_probabilities:
                result = (result, probabilities[i])
            
            results.append(result)
        
        return results
    
    def predict_single(
        self, 
        text: str, 
        return_probability: bool = False,
        return_category: bool = True
    ) -> Any:
        """Predict class for a single text."""
        results = self.predict(
            [text], 
            return_probabilities=return_probability,
            return_categories=return_category
        )
        return results[0]
    
    def get_model_specific_state(self) -> Dict[str, Any]:
        """Get replay-specific state for checkpointing."""
        return {
            'memory_buffer': self.memory_buffer,
            'category_to_task_id': self.category_to_task_id,
            'task_categories': self.task_categories,
            'model_class_mappings': {
                'class_to_idx': self.model.class_to_idx,
                'idx_to_class': self.model.idx_to_class,
                'num_classes': self.model.num_classes
            },
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'replay_batch_size': self.replay_batch_size,
            'replay_frequency': self.replay_frequency,
        }
    
    def load_model_specific_state(self, checkpoint: Dict[str, Any]) -> None:
        """Load replay-specific state from checkpoint."""
        self.memory_buffer = checkpoint.get('memory_buffer', self.memory_buffer)
        self.category_to_task_id = checkpoint.get('category_to_task_id', {})
        self.task_categories = checkpoint.get('task_categories', {})
        
        # Restore model class mappings
        model_mappings = checkpoint.get('model_class_mappings', {})
        if model_mappings:
            self.model.class_to_idx = model_mappings.get('class_to_idx', {})
            self.model.idx_to_class = model_mappings.get('idx_to_class', {})
            self.model.num_classes = model_mappings.get('num_classes', 2)
            
            # Expand classifier if needed
            if self.model.num_classes > self.model.classifier.out_features:
                self.model.expand_classifier(self.model.num_classes)
        
        # Update other parameters
        self.model_name = checkpoint.get('model_name', self.model_name)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.batch_size = checkpoint.get('batch_size', self.batch_size)
        self.max_length = checkpoint.get('max_length', self.max_length)
        self.replay_batch_size = checkpoint.get('replay_batch_size', self.replay_batch_size)
        self.replay_frequency = checkpoint.get('replay_frequency', self.replay_frequency)
    
    def analyze_memory_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of examples in memory buffer."""
        if not hasattr(self.memory_buffer, 'task_buffers'):
            return {'message': 'Memory buffer does not support task-wise analysis'}
        
        distribution = {}
        total_items = 0
        
        for task_id, task_buffer in self.memory_buffer.task_buffers.items():
            task_distribution = {}
            task_count = len(task_buffer)
            total_items += task_count
            
            # Count by category
            category_counts = {}
            for item in task_buffer:
                category = item.metadata.get('category', 'unknown') if item.metadata else 'unknown'
                category_counts[category] = category_counts.get(category, 0) + 1
            
            task_distribution = {
                'total_items': task_count,
                'categories': category_counts,
                'task_categories': self.task_categories.get(task_id, [])
            }
            
            distribution[f'task_{task_id}'] = task_distribution
        
        distribution['total_memory_items'] = total_items
        distribution['memory_utilization'] = total_items / self.memory_buffer.capacity
        
        return distribution
    
    def get_classification_report(self, test_datasets: Dict[str, List[ClassificationExample]]) -> str:
        """Generate detailed classification report across all categories."""
        all_true_labels = []
        all_predicted_labels = []
        all_label_names = []
        
        for category_set, test_data in test_datasets.items():
            # Predict on test data
            texts = [example.text for example in test_data]
            predictions = self.predict(texts, return_categories=True)
            
            # Collect true and predicted labels
            for example, prediction in zip(test_data, predictions):
                all_true_labels.append(example.category)
                all_predicted_labels.append(prediction)
                if example.category not in all_label_names:
                    all_label_names.append(example.category)
                if prediction not in all_label_names:
                    all_label_names.append(prediction)
        
        # Generate classification report
        report = classification_report(
            all_true_labels,
            all_predicted_labels,
            labels=all_label_names,
            target_names=all_label_names,
            zero_division=0
        )
        
        return report
    
    def get_category_summary(self) -> Dict[str, Any]:
        """Get summary of learned categories and their performance."""
        summary = {
            'total_categories': len(self.model.class_to_idx),
            'categories_by_task': {},
            'category_mappings': {
                'class_to_idx': dict(self.model.class_to_idx),
                'idx_to_class': dict(self.model.idx_to_class)
            },
            'memory_distribution': self.analyze_memory_distribution()
        }
        
        for task_id, categories in self.task_categories.items():
            summary['categories_by_task'][f'task_{task_id}'] = categories
        
        return summary