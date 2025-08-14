"""
Gradient Episodic Memory (GEM) for Named Entity Recognition.
Adapts to new entity types and domains while preserving performance on previous tasks.
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
import spacy
from spacy.tokens import Doc
import logging

from ..core.base_learner import ContinualLearner, TaskMetrics
from ..core.memory_buffer import MemoryBuffer, MemoryItem, GradientBuffer
from ..core.regularization import GradientProjection

@dataclass
class NERExample:
    """Single Named Entity Recognition example."""
    text: str
    entities: List[Tuple[int, int, str]]  # (start, end, label)
    task_id: int
    domain: str

class NERDataset(Dataset):
    """Dataset for Named Entity Recognition."""
    
    def __init__(
        self,
        examples: List[NERExample],
        tokenizer: AutoTokenizer,
        spacy_nlp: spacy.language.Language,
        max_length: int = 512,
        label_map: Optional[Dict[str, int]] = None
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.spacy_nlp = spacy_nlp
        self.max_length = max_length
        self.label_map = label_map or {}
        
        # Initialize label mapping if not provided
        if not self.label_map:
            unique_labels = set()
            for example in examples:
                for _, _, label in example.entities:
                    unique_labels.add(label)
            self.label_map = {label: idx for idx, label in enumerate(sorted(unique_labels), start=1)}
            self.label_map['O'] = 0  # Outside entity
        
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        # Tokenize with transformers
        encoding = self.tokenizer(
            example.text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Align tokens with spaCy for entity labels
        doc = self.spacy_nlp(example.text)
        token_offsets = encoding['offset_mapping'].squeeze()
        labels = ['O'] * len(token_offsets)
        
        # Map character-based entity spans to token-based labels
        for start_char, end_char, label in example.entities:
            for idx, (start, end) in enumerate(token_offsets):
                if start >= start_char and end <= end_char and end > 0:
                    labels[idx] = label
        
        # Convert labels to indices
        label_ids = [self.label_map.get(label, 0) for label in labels]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'task_id': torch.tensor(example.task_id, dtype=torch.long),
            'domain': example.domain,
            'text': example.text
        }

class NERClassifier(nn.Module):
    """Transformer-based NER classifier."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

class GEMNERAnalyzer(ContinualLearner):
    """Gradient Episodic Memory-based NER analyzer."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        memory_size: int = 1000,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "auto",
        save_dir: Optional[str] = None,
        verbose: bool = True
    ):
        # Initialize model
        model = NERClassifier(model_name=model_name, num_labels=2)  # Initial num_labels
        super().__init__(model, device, save_dir, verbose)
        
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Initialize tokenizer and spaCy
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.spacy_nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        
        # Initialize memory buffer and GEM
        self.memory_buffer = GradientBuffer(capacity=memory_size, device=self.device)
        self.gem = GradientProjection(model=self.model, device=self.device, memory_buffer=self.memory_buffer)
        
        # Track domains and labels
        self.domain_to_task_id: Dict[str, int] = {}
        self.task_id_to_domain: Dict[int, str] = {}
        self.label_map: Dict[str, int] = {'O': 0}
        self.idx_to_label: Dict[int, str] = {0: 'O'}
    
    def update_label_map(self, new_labels: List[str]) -> None:
        """Update label map with new entity types."""
        for label in new_labels:
            if label not in self.label_map:
                new_idx = len(self.label_map)
                self.label_map[label] = new_idx
                self.idx_to_label[new_idx] = label
        
        # Update classifier if needed
        if len(self.label_map) > self.model.num_labels:
            old_classifier = self.model.classifier
            self.model.num_labels = len(self.label_map)
            self.model.classifier = nn.Linear(self.model.transformer.config.hidden_size, self.model.num_labels)
            
            # Copy existing weights
            with torch.no_grad():
                self.model.classifier.weight[:old_classifier.out_features] = old_classifier.weight
                self.model.classifier.bias[:old_classifier.out_features] = old_classifier.bias
                # Initialize new weights
                nn.init.xavier_uniform_(self.model.classifier.weight[old_classifier.out_features:])
                nn.init.zeros_(self.model.classifier.bias[old_classifier.out_features:])
    
    def train_task(
        self,
        train_data: List[NERExample],
        task_id: int,
        domain: str,
        epochs: int = 3,
        validation_data: Optional[List[NERExample]] = None,
        **kwargs
    ) -> List[TaskMetrics]:
        """Train on a new NER task."""
        
        self.logger.info(f"Training task {task_id} ({domain}) for {epochs} epochs")
        
        # Update domain mappings
        self.domain_to_task_id[domain] = task_id
        self.task_id_to_domain[task_id] = domain
        
        # Update label map
        new_labels = set(label for example in train_data for _, _, label in example.entities)
        self.update_label_map(new_labels)
        
        # Create dataset and dataloader
        train_dataset = NERDataset(train_data, self.tokenizer, self.spacy_nlp, self.max_length, self.label_map)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens
        
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
                logits = self.model(input_ids, attention_mask)
                
                # Reshape for loss
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                
                # Apply GEM gradient projection
                self.gem.project_gradients(task_id)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Compute metrics
                predictions = torch.argmax(logits, dim=1)
                valid_mask = labels != -100
                correct_predictions += (predictions[valid_mask] == labels[valid_mask]).sum().item()
                total_samples += valid_mask.sum().item()
                all_predictions.extend(predictions[valid_mask].cpu().numpy())
                all_labels.extend(labels[valid_mask].cpu().numpy())
                
                total_loss += loss.item()
                
                if self.verbose and batch_idx % 100 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}"
                    )
            
            # Add examples to memory buffer
            self._update_memory_buffer(train_data, task_id)
            
            # Update GEM reference gradients
            self.gem.update_task_knowledge(task_id, train_loader)
            
            # Compute epoch metrics
            epoch_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
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
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - "
                f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, F1: {f1:.4f}"
            )
            
            # Validation
            if validation_data:
                val_metrics = self.evaluate_task(validation_data, task_id, domain)
                self.logger.info(
                    f"Validation - Accuracy: {val_metrics.accuracy:.4f}, F1: {val_metrics.f1_score:.4f}"
                )
        
        # Update task information
        self.current_task = task_id
        if task_id not in self.task_history:
            self.task_history.append(task_id)
        self.task_metrics[task_id] = task_metrics
        self.task_data_info[task_id] = {
            'domain': domain,
            'num_samples': len(train_data),
            'epochs': epochs,
            'labels': list(new_labels),
            'memory_size_after': len(self.memory_buffer)
        }
        
        self.logger.info(f"Task {task_id} ({domain}) training completed")
        return task_metrics
    
    def _update_memory_buffer(self, train_data: List[NERExample], task_id: int) -> None:
        """Add training examples to memory buffer."""
        for example in train_data:
            memory_item = MemoryItem(
                input_data=example,
                target=example.entities,
                task_id=task_id,
                timestamp=time.time(),
                metadata={'domain': example.domain}
            )
            self.memory_buffer.add(memory_item)
    
    def evaluate_task(
        self,
        test_data: List[NERExample],
        task_id: int,
        domain: str = None,
        **kwargs
    ) -> TaskMetrics:
        """Evaluate on NER test data."""
        
        if domain is None:
            domain = self.task_id_to_domain.get(task_id, "unknown")
        
        # Create dataset and dataloader
        test_dataset = NERDataset(test_data, self.tokenizer, self.spacy_nlp, self.max_length, self.label_map)
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
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                logits = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                # Compute loss
                loss = criterion(logits, labels_flat)
                
                # Collect metrics
                predictions = torch.argmax(logits, dim=1)
                valid_mask = labels_flat != -100
                correct_predictions += (predictions[valid_mask] == labels_flat[valid_mask]).sum().item()
                total_samples += valid_mask.sum().item()
                all_predictions.extend(predictions[valid_mask].cpu().numpy())
                all_labels.extend(labels_flat[valid_mask].cpu().numpy())
                
                total_loss += loss.item()
        
        # Compute metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
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
            f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Loss: {avg_loss:.4f}"
        )
        
        return metrics
    
    def predict(
        self,
        texts: List[str],
        task_id: int,
        return_labels: bool = True,
        **kwargs
    ) -> List[Any]:
        """Predict entities for new text data."""
        
        self.model.eval()
        predictions = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt',
                return_offsets_mapping=True
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            offsets = encoding['offset_mapping']
            
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                batch_predictions = torch.argmax(logits, dim=2).cpu().numpy()
            
            # Convert token-level predictions to entity spans
            for text_idx, (text, pred, offset) in enumerate(zip(batch_texts, batch_predictions, offsets)):
                entities = []
                current_entity = None
                for token_idx, (label_idx, (start, end)) in enumerate(zip(pred, offset)):
                    label = self.idx_to_label.get(label_idx, 'O')
                    if label == 'O':
                        if current_entity:
                            entities.append(current_entity)
                            current_entity = None
                    else:
                        if current_entity and current_entity[2] == label:
                            current_entity = (current_entity[0], end, label)
                        else:
                            if current_entity:
                                entities.append(current_entity)
                            current_entity = (start, end, label)
                
                if current_entity:
                    entities.append(current_entity)
                
                if return_labels:
                    predictions.append([(start, end, label) for start, end, label in entities])
                else:
                    predictions.append(pred)
        
        return predictions
    
    def predict_single(
        self,
        text: str,
        task_id: int,
        return_labels: bool = True
    ) -> Any:
        """Predict entities for a single text."""
        results = self.predict([text], task_id, return_labels)
        return results[0]
    
    def get_model_specific_state(self) -> Dict[str, Any]:
        """Get GEM-specific state for checkpointing."""
        return {
            'memory_buffer': self.memory_buffer,
            'domain_to_task_id': self.domain_to_task_id,
            'task_id_to_domain': self.task_id_to_domain,
            'label_map': self.label_map,
            'idx_to_label': self.idx_to_label,
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'gem_reference_gradients': self.gem.reference_gradients
        }
    
    def load_model_specific_state(self, checkpoint: Dict[str, Any]) -> None:
        """Load GEM-specific state from checkpoint."""
        self.memory_buffer = checkpoint.get('memory_buffer', self.memory_buffer)
        self.domain_to_task_id = checkpoint.get('domain_to_task_id', {})
        self.task_id_to_domain = checkpoint.get('task_id_to_domain', {})
        self.label_map = checkpoint.get('label_map', {'O': 0})
        self.idx_to_label = checkpoint.get('idx_to_label', {0: 'O'})
        
        # Update classifier if needed
        if len(self.label_map) > self.model.num_labels:
            self.model.num_labels = len(self.label_map)
            self.model.classifier = nn.Linear(self.model.transformer.config.hidden_size, self.model.num_labels)
            nn.init.xavier_uniform_(self.model.classifier.weight)
            nn.init.zeros_(self.model.classifier.bias)
        
        self.gem.reference_gradients = checkpoint.get('gem_reference_gradients', {})
        
        # Update other parameters
        self.model_name = checkpoint.get('model_name', self.model_name)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.batch_size = checkpoint.get('batch_size', self.batch_size)
        self.max_length = checkpoint.get('max_length', self.max_length)
    
    def analyze_forgetting(self, test_datasets: Dict[str, List[NERExample]]) -> Dict[str, float]:
        """Analyze catastrophic forgetting across domains."""
        forgetting_scores = {}
        
        for domain, test_data in test_datasets.items():
            if domain in self.domain_to_task_id:
                task_id = self.domain_to_task_id[domain]
                
                # Get current performance
                current_metrics = self.evaluate_task(test_data, task_id, domain)
                
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
            'domain_performance': {},
            'label_map': dict(self.label_map)
        }
        
        for domain, task_id in self.domain_to_task_id.items():
            if task_id in self.task_metrics:
                best_metrics = max(
                    self.task_metrics[task_id],
                    key=lambda x: x.accuracy
                )
                summary['domain_performance'][domain] = {
                    'best_accuracy': best_metrics.accuracy,
                    'best_f1': best_metrics.f1_score,
                    'task_id': task_id,
                    'labels': self.task_data_info.get(task_id, {}).get('labels', [])
                }
        
        return summary