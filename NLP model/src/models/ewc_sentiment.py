"""
Elastic Weight Consolidation (EWC) for Sentiment Analysis.
Demonstrates continual learning across different domains (movies, products, social media).
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

from ..core.base_learner import ContinualLearner, TaskMetrics
from ..core.regularization import ElasticWeightConsolidation


@dataclass
class SentimentExample:
    """Single sentiment analysis example."""
    text: str
    label: int  # 0: negative, 1: positive
    domain: str
    

class SentimentDataset(Dataset):
    """Dataset for sentiment analysis."""
    
    def __init__(
        self, 
        examples: List[SentimentExample], 
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
            'domain': example.domain
        }


class SentimentClassifier(nn.Module):
    """BERT-based sentiment classifier."""
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits


class EWCSentimentAnalyzer(ContinualLearner):
    """EWC-based continual learning sentiment analyzer."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        ewc_lambda: float = 1000.0,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "auto",
        save_dir: Optional[str] = None,
        verbose: bool = True
    ):
        # Initialize model
        model = SentimentClassifier(model_name=model_name)
        super().__init__(model, device, save_dir, verbose)
        
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize EWC regularization
        self.ewc = ElasticWeightConsolidation(
            model=self.model,
            device=self.device,
            lambda_reg=ewc_lambda
        )
        
        # Domain mapping for tasks
        self.domain_to_task_id: Dict[str, int] = {}
        self.task_id_to_domain: Dict[int, str] = {}
    
    def train_task(
        self,
        train_data: List[SentimentExample],
        task_id: int,
        domain: str,
        epochs: int = 3,
        validation_data: Optional[List[SentimentExample]] = None,
        **kwargs
    ) -> List[TaskMetrics]:
        """Train on a new sentiment analysis task/domain."""
        
        self.logger.info(f"Training task {task_id} ({domain}) for {epochs} epochs")
        
        # Update domain mappings
        self.domain_to_task_id[domain] = task_id
        self.task_id_to_domain[task_id] = domain
        
        # Create dataset and dataloader
        train_dataset = SentimentDataset(train_data, self.tokenizer, self.max_length)
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
            correct_predictions = 0
            total_samples = 0
            all_predictions = []
            all_labels = []
            
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                
                # Compute loss
                task_loss = criterion(logits, labels)
                
                # Add EWC regularization penalty
                ewc_penalty = self.ewc.compute_penalty(task_loss)
                total_loss_with_reg = task_loss + ewc_penalty
                
                # Backward pass
                total_loss_with_reg.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Compute metrics
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                total_loss += task_loss.item()
                
                if self.verbose and batch_idx % 100 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {task_loss.item():.4f}, EWC: {ewc_penalty.item():.4f}"
                    )
            
            # Compute epoch metrics
            epoch_accuracy = correct_predictions / total_samples
            epoch_loss = total_loss / len(train_loader)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
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
                    f"Validation - Accuracy: {val_metrics.accuracy:.4f}, "
                    f"F1: {val_metrics.f1_score:.4f}"
                )
        
        # Update task information
        self.current_task = task_id
        if task_id not in self.task_history:
            self.task_history.append(task_id)
        self.task_metrics[task_id] = task_metrics
        self.task_data_info[task_id] = {
            'domain': domain,
            'num_samples': len(train_data),
            'epochs': epochs
        }
        
        # Update EWC after task completion
        self.ewc.update_task_knowledge(task_id, train_loader)
        
        self.logger.info(f"Task {task_id} ({domain}) training completed")
        return task_metrics
    
    def evaluate_task(
        self,
        test_data: List[SentimentExample],
        task_id: int,
        domain: str = None,
        **kwargs
    ) -> TaskMetrics:
        """Evaluate on sentiment analysis test data."""
        
        if domain is None:
            domain = self.task_id_to_domain.get(task_id, "unknown")
        
        # Create dataset and dataloader
        test_dataset = SentimentDataset(test_data, self.tokenizer, self.max_length)
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
            all_labels, all_predictions, average='weighted'
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
        
        self.logger.info(
            f"Task {task_id} ({domain}) evaluation - "
            f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Loss: {avg_loss:.4f}"
        )
        
        return metrics
    
    def predict(
        self, 
        texts: List[str], 
        return_probabilities: bool = False,
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
                logits = self.model(input_ids, attention_mask)
                batch_probs = torch.softmax(logits, dim=1)
                batch_predictions = torch.argmax(logits, dim=1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(batch_probs.cpu().numpy())
        
        if return_probabilities:
            return list(zip(predictions, probabilities))
        return predictions
    
    def predict_single(self, text: str, return_probability: bool = False) -> Any:
        """Predict sentiment for a single text."""
        results = self.predict([text], return_probabilities=return_probability)
        return results[0]
    
    def get_model_specific_state(self) -> Dict[str, Any]:
        """Get EWC-specific state for checkpointing."""
        return {
            'ewc_fisher_matrices': self.ewc.fisher_matrices,
            'ewc_optimal_params': self.ewc.optimal_params,
            'domain_to_task_id': self.domain_to_task_id,
            'task_id_to_domain': self.task_id_to_domain,
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
        }
    
    def load_model_specific_state(self, checkpoint: Dict[str, Any]) -> None:
        """Load EWC-specific state from checkpoint."""
        self.ewc.fisher_matrices = checkpoint.get('ewc_fisher_matrices', {})
        self.ewc.optimal_params = checkpoint.get('ewc_optimal_params', {})
        self.domain_to_task_id = checkpoint.get('domain_to_task_id', {})
        self.task_id_to_domain = checkpoint.get('task_id_to_domain', {})
        
        # Update other parameters if available
        self.model_name = checkpoint.get('model_name', self.model_name)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.batch_size = checkpoint.get('batch_size', self.batch_size)
        self.max_length = checkpoint.get('max_length', self.max_length)
    
    def analyze_forgetting(self, test_datasets: Dict[str, List[SentimentExample]]) -> Dict[str, float]:
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
                    
                    # Compute forgetting (negative of backward transfer)
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
                    'best_f1': best_metrics.f1_score,
                    'task_id': task_id
                }
        
        return summary