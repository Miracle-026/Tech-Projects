"""
Progressive Neural Network for Language Modeling.
Implements a continual learning approach with task-specific columns to prevent catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

from ..core.base_learner import ContinualLearner, TaskMetrics
from ..core.memory_buffer import MemoryBuffer, MemoryItem, BalancedBuffer

@dataclass
class LanguageModelingExample:
    """Single language modeling example."""
    text: str
    task_id: int
    domain: str
    label: Optional[int] = None  # Optional for supervised tasks

class LanguageModelingDataset(Dataset):
    """Dataset for language modeling."""
    
    def __init__(
        self, 
        examples: List[LanguageModelingExample], 
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
        
        # Tokenize text for language modeling
        encoding = self.tokenizer(
            example.text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For masked language modeling, create labels as input_ids
        # For supervised tasks, use provided label
        labels = input_ids.clone() if example.label is None else torch.tensor(example.label, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'task_id': torch.tensor(example.task_id, dtype=torch.long),
            'domain': example.domain
        }

class ProgressiveTransformerColumn(nn.Module):
    """Task-specific transformer column for progressive neural network."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_size: int = 384,  # DistilBERT hidden size / 2
        dropout_rate: float = 0.1,
        num_classes: Optional[int] = None  # None for language modeling
    ):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Task-specific head
        self.hidden_size = hidden_size
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer (language modeling or classification)
        self.is_classification = num_classes is not None
        if self.is_classification:
            self.output_layer = nn.Linear(hidden_size, num_classes)
        else:
            self.output_layer = nn.Linear(hidden_size, self.transformer.config.vocab_size)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        features = self.feature_extractor(pooled_output)
        features = self.dropout(features)
        return self.output_layer(features)

class ProgressiveLanguageModel(nn.Module):
    """Progressive Neural Network for Language Modeling."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        initial_num_classes: Optional[int] = None,
        dropout_rate: float = 0.1,
        hidden_size: int = 384
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # Shared backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Task-specific columns
        self.columns: Dict[int, ProgressiveTransformerColumn] = {}
        self.task_num_classes: Dict[int, int] = {}
        
        # Initial column (optional classification head)
        if initial_num_classes is not None:
            self.add_column(task_id=0, num_classes=initial_num_classes)
    
    def add_column(self, task_id: int, num_classes: Optional[int] = None) -> None:
        """Add a new task-specific column."""
        column = ProgressiveTransformerColumn(
            model_name=self.model_name,
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            num_classes=num_classes
        )
        self.columns[task_id] = column.to(self.backbone.device)
        if num_classes is not None:
            self.task_num_classes[task_id] = num_classes
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        """Forward pass using task-specific column."""
        if task_id not in self.columns:
            raise ValueError(f"Task {task_id} not found in columns")
        
        # Get shared backbone features
        backbone_outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pass through task-specific column
        column_output = self.columns[task_id](input_ids, attention_mask)
        return column_output

class ProgressiveLMAnalyzer(ContinualLearner):
    """Progressive Neural Network-based Language Model for Continual Learning."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        memory_size: int = 1000,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "auto",
        save_dir: Optional[str] = None,
        verbose: bool = True
    ):
        # Initialize model
        model = ProgressiveLanguageModel(model_name=model_name)
        super().__init__(model, device, save_dir, verbose)
        
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize memory buffer for optional replay
        self.memory_buffer = BalancedBuffer(capacity=memory_size, device=self.device)
        
        # Track domains and task configurations
        self.domain_to_task_id: Dict[str, int] = {}
        self.task_id_to_domain: Dict[int, str] = {}
        self.task_configs: Dict[int, Dict[str, Any]] = {}
    
    def train_task(
        self,
        train_data: List[LanguageModelingExample],
        task_id: int,
        domain: str,
        epochs: int = 3,
        validation_data: Optional[List[LanguageModelingExample]] = None,
        is_classification: bool = False,
        num_classes: Optional[int] = None,
        **kwargs
    ) -> List[TaskMetrics]:
        """Train on a new language modeling task."""
        
        self.logger.info(f"Training task {task_id} ({domain}) for {epochs} epochs")
        
        # Update domain mappings
        self.domain_to_task_id[domain] = task_id
        self.task_id_to_domain[task_id] = domain
        
        # Add new task-specific column
        if task_id not in self.model.columns:
            self.model.add_column(task_id, num_classes if is_classification else None)
        
        # Create dataset and dataloader
        train_dataset = LanguageModelingDataset(train_data, self.tokenizer, self.max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss() if is_classification else nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Training loop
        self.model.train()
        task_metrics = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            all_predictions = []
            all_labels = []
            
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask, task_id)
                
                # Reshape logits and labels for loss computation
                if is_classification:
                    loss = criterion(logits, labels)
                    predictions = torch.argmax(logits, dim=1)
                else:
                    # For language modeling, reshape for vocab size
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)
                    loss = criterion(logits, labels)
                    predictions = torch.argmax(logits, dim=1)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Compute metrics
                if is_classification:
                    correct_predictions += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:
                    # For language modeling, compute perplexity
                    total_samples += (labels != self.tokenizer.pad_token_id).sum().item()
                    correct_predictions += ((predictions == labels) & (labels != self.tokenizer.pad_token_id)).sum().item()
                
                total_loss += loss.item()
                
                if self.verbose and batch_idx % 100 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}"
                    )
            
            # Compute epoch metrics
            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            
            if is_classification:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels, all_predictions, average='weighted', zero_division=0
                )
            else:
                # For language modeling, use perplexity as primary metric
                precision = recall = f1 = np.exp(epoch_loss)  # Perplexity
                
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
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - "
                f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
                f"{'F1' if is_classification else 'Perplexity'}: {f1:.4f}"
            )
            
            # Validation
            if validation_data:
                val_metrics = self.evaluate_task(validation_data, task_id, domain, is_classification)
                self.logger.info(
                    f"Validation - Accuracy: {val_metrics.accuracy:.4f}, "
                    f"{'F1' if is_classification else 'Perplexity'}: {val_metrics.f1_score:.4f}"
                )
            
            # Add examples to memory buffer
            self._update_memory_buffer(train_data)
        
        # Update task information
        self.current_task = task_id
        if task_id not in self.task_history:
            self.task_history.append(task_id)
        self.task_metrics[task_id] = task_metrics
        self.task_data_info[task_id] = {
            'domain': domain,
            'num_samples': len(train_data),
            'epochs': epochs,
            'is_classification': is_classification,
            'num_classes': num_classes,
            'memory_size_after': len(self.memory_buffer)
        }
        
        self.logger.info(f"Task {task_id} ({domain}) training completed")
        return task_metrics
    
    def _update_memory_buffer(self, train_data: List[LanguageModelingExample]) -> None:
        """Add training examples to memory buffer."""
        for example in train_data:
            memory_item = MemoryItem(
                input_data=example,
                target=example.label if example.label is not None else example.text,
                task_id=example.task_id,
                timestamp=time.time(),
                metadata={'domain': example.domain}
            )
            self.memory_buffer.add(memory_item)
    
    def evaluate_task(
        self,
        test_data: List[LanguageModelingExample],
        task_id: int,
        domain: str = None,
        is_classification: bool = False,
        **kwargs
    ) -> TaskMetrics:
        """Evaluate on language modeling test data."""
        
        if domain is None:
            domain = self.task_id_to_domain.get(task_id, "unknown")
        
        # Create dataset and dataloader
        test_dataset = LanguageModelingDataset(test_data, self.tokenizer, self.max_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss() if is_classification else nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask, task_id)
                
                # Compute loss
                if is_classification:
                    loss = criterion(logits, labels)
                    predictions = torch.argmax(logits, dim=1)
                else:
                    logits = logits.view(-1, logits.size(-1))
                    labels_flat = labels.view(-1)
                    loss = criterion(logits, labels_flat)
                    predictions = torch.argmax(logits, dim=1)
                
                # Collect metrics
                if is_classification:
                    correct_predictions += (predictions == labels).sum().item()
                    total_samples += labels.size(0)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:
                    total_samples += (labels_flat != self.tokenizer.pad_token_id).sum().item()
                    correct_predictions += ((predictions == labels_flat) & (labels_flat != self.tokenizer.pad_token_id)).sum().item()
                
                total_loss += loss.item()
        
        # Compute metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        if is_classification:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted', zero_division=0
            )
        else:
            # For language modeling, use perplexity
            precision = recall = f1 = np.exp(avg_loss)
        
        metrics = TaskMetrics(
            accuracy=accuracy,
            loss=avg_loss,
            precision=precision,
            recall=recall,
            f1_score=f1,
            task_id=task_id,
            epoch=-1
        )
        
        self.logger.info(
            f"Task {task_id} ({domain}) evaluation - "
            f"Accuracy: {accuracy:.4f}, {'F1' if is_classification else 'Perplexity'}: {f1:.4f}, "
            f"Loss: {avg_loss:.4f}"
        )
        
        return metrics
    
    def predict(
        self,
        texts: List[str],
        task_id: int,
        return_probabilities: bool = False,
        is_classification: bool = False,
        **kwargs
    ) -> List[Any]:
        """Make predictions on new text data."""
        
        self.model.eval()
        predictions = []
        probabilities = []
        
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
                logits = self.model(input_ids, attention_mask, task_id)
                
                if is_classification:
                    batch_probs = torch.softmax(logits, dim=1)
                    batch_predictions = torch.argmax(logits, dim=1)
                else:
                    # For language modeling, return predicted token IDs
                    batch_probs = torch.softmax(logits.view(-1, logits.size(-1)), dim=-1)
                    batch_predictions = torch.argmax(logits, dim=-1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(batch_probs.cpu().numpy())
        
        if return_probabilities:
            return list(zip(predictions, probabilities))
        return predictions
    
    def predict_single(
        self,
        text: str,
        task_id: int,
        return_probability: bool = False,
        is_classification: bool = False
    ) -> Any:
        """Predict for a single text."""
        results = self.predict(
            [text],
            task_id,
            return_probabilities=return_probability,
            is_classification=is_classification
        )
        return results[0]
    
    def get_model_specific_state(self) -> Dict[str, Any]:
        """Get progressive-specific state for checkpointing."""
        return {
            'memory_buffer': self.memory_buffer,
            'domain_to_task_id': self.domain_to_task_id,
            'task_id_to_domain': self.task_id_to_domain,
            'task_configs': self.task_configs,
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'task_num_classes': self.model.task_num_classes
        }
    
    def load_model_specific_state(self, checkpoint: Dict[str, Any]) -> None:
        """Load progressive-specific state from checkpoint."""
        self.memory_buffer = checkpoint.get('memory_buffer', self.memory_buffer)
        self.domain_to_task_id = checkpoint.get('domain_to_task_id', {})
        self.task_id_to_domain = checkpoint.get('task_id_to_domain', {})
        self.task_configs = checkpoint.get('task_configs', {})
        
        # Restore task-specific columns
        task_num_classes = checkpoint.get('task_num_classes', {})
        for task_id, num_classes in task_num_classes.items():
            if task_id not in self.model.columns:
                self.model.add_column(task_id, num_classes)
        
        # Update other parameters
        self.model_name = checkpoint.get('model_name', self.model_name)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.batch_size = checkpoint.get('batch_size', self.batch_size)
        self.max_length = checkpoint.get('max_length', self.max_length)
    
    def analyze_forgetting(self, test_datasets: Dict[str, List[LanguageModelingExample]]) -> Dict[str, float]:
        """Analyze catastrophic forgetting across domains."""
        forgetting_scores = {}
        
        for domain, test_data in test_datasets.items():
            if domain in self.domain_to_task_id:
                task_id = self.domain_to_task_id[domain]
                is_classification = self.task_configs.get(task_id, {}).get('is_classification', False)
                
                # Get current performance
                current_metrics = self.evaluate_task(test_data, task_id, domain, is_classification)
                
                # Get best historical performance
                if task_id in self.task_metrics:
                    best_historical = max(
                        self.task_metrics[task_id],
                        key=lambda x: x.accuracy
                    )
                    forgetting = best_historical.accuracy - current_metrics.accuracy
                    forgetting_scores[domain] = forgetting
                else:
                    forgetting_scores[domain] = 0.0
        
        return forgetting_scores
    
    def get_domain_summary(self) -> Dict[str, Any]:
        """Get summary of performance across domains."""
        summary = {
            'domains': list(self.domain_to_task_id.keys()),
            'num_domains': len(self.domain_to_task_id),
            'domain_performance': {}
        }
        
        for domain, task_id in self.domain_to_task_id.items():
            if task_id in self.task_metrics:
                best_metrics = max(
                    self.task_metrics[task_id],
                    key=lambda x: x.accuracy
                )
                summary['domain_performance'][domain] = {
                    'best_accuracy': best_metrics.accuracy,
                    'best_f1_or_perplexity': best_metrics.f1_score,
                    'task_id': task_id,
                    'is_classification': self.task_configs.get(task_id, {}).get('is_classification', False)
                }
        
        return summary