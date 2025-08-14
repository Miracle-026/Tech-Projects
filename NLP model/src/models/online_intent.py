"""
Online Intent Classification for Continual Learning.
Uses online learning algorithms with scikit-learn and NLTK for real-time adaptation to new intents.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import pickle
import time
from dataclasses import dataclass

from ..core.base_learner import ContinualLearner, TaskMetrics
from ..core.memory_buffer import MemoryBuffer, MemoryItem, BalancedBuffer
from ..utils.data_loader import IntentExample

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class IntentPrediction:
    """Prediction result for intent classification."""
    intent: str
    probability: float

class OnlineIntentAnalyzer(ContinualLearner):
    """Online learning-based intent classifier for continual learning."""
    
    def __init__(
        self,
        memory_size: int = 1000,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        device: str = "cpu",  # scikit-learn runs on CPU
        save_dir: Optional[str] = None,
        verbose: bool = True
    ):
        # Initialize base learner with None model (will be set per task)
        super().__init__(model=None, device=device, save_dir=save_dir, verbose=verbose)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Initialize vectorizer and stemmer
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.stemmer = PorterStemmer()
        
        # Initialize memory buffer
        self.memory_buffer = BalancedBuffer(capacity=memory_size, device=device)
        
        # Track domains, intents, and classifiers
        self.domain_to_task_id: Dict[str, int] = {}
        self.task_id_to_domain: Dict[int, str] = {}
        self.intent_map: Dict[str, int] = {}  # Maps intent strings to indices
        self.idx_to_intent: Dict[int, str] = {}
        self.task_classifiers: Dict[int, SGDClassifier] = {}
        self.task_intents: Dict[int, List[str]] = {}
        self.task_vectorizers: Dict[int, TfidfVectorizer] = {}
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text using NLTK tokenization and stemming."""
        tokens = word_tokenize(text.lower())
        stemmed = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed)
    
    def update_intent_map(self, new_intents: List[str]) -> None:
        """Update intent map with new intent labels."""
        for intent in new_intents:
            if intent not in self.intent_map:
                new_idx = len(self.intent_map)
                self.intent_map[intent] = new_idx
                self.idx_to_intent[new_idx] = intent
    
    def train_task(
        self,
        train_data: List[IntentExample],
        task_id: int,
        domain: str,
        epochs: int = 1,  # Single pass for online learning
        validation_data: Optional[List[IntentExample]] = None,
        **kwargs
    ) -> List[TaskMetrics]:
        """Train on a new intent classification task."""
        
        self.logger.info(f"Training task {task_id} ({domain}) for {epochs} epochs")
        
        # Update domain mappings
        self.domain_to_task_id[domain] = task_id
        self.task_id_to_domain[task_id] = domain
        
        # Update intent map
        new_intents = list(set(example.intent for example in train_data))
        self.update_intent_map(new_intents)
        self.task_intents[task_id] = new_intents
        
        # Initialize or update classifier for this task
        if task_id not in self.task_classifiers:
            self.task_classifiers[task_id] = SGDClassifier(
                loss='log',  # Logistic regression
                learning_rate='constant',
                eta0=self.learning_rate,
                max_iter=1000,
                tol=1e-3,
                n_jobs=-1
            )
            self.task_vectorizers[task_id] = TfidfVectorizer(
                max_features=5000,
                stop_words='english'
            )
        
        # Preprocess and vectorize training data
        texts = [self._preprocess_text(example.text) for example in train_data]
        labels = [self.intent_map[example.intent] for example in train_data]
        
        # Fit vectorizer
        X = self.task_vectorizers[task_id].fit_transform(texts) if task_id not in self.task_vectorizers else \
            self.task_vectorizers[task_id].transform(texts)
        
        # Train classifier incrementally
        task_metrics = []
        for epoch in range(epochs):
            start_time = time.time()
            total_samples = 0
            correct_predictions = 0
            all_predictions = []
            all_labels = []
            total_loss = 0.0
            
            # Process in mini-batches
            for i in range(0, len(texts), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = labels[i:i + self.batch_size]
                
                # Partial fit for online learning
                self.task_classifiers[task_id].partial_fit(batch_X, batch_y, classes=list(self.intent_map.values()))
                
                # Compute predictions and loss
                batch_predictions = self.task_classifiers[task_id].predict(batch_X)
                batch_probs = self.task_classifiers[task_id].predict_proba(batch_X)
                
                # Approximate loss using log loss
                for probs, y in zip(batch_probs, batch_y):
                    total_loss += -np.log(probs[y] + 1e-10)
                
                correct_predictions += (batch_predictions == batch_y).sum()
                total_samples += len(batch_y)
                all_predictions.extend(batch_predictions)
                all_labels.extend(batch_y)
                
                if self.verbose and i % (self.batch_size * 100) == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{epochs}, Batch {i//self.batch_size}, "
                        f"Batch Accuracy: {(batch_predictions == batch_y).mean():.4f}"
                    )
            
            # Update memory buffer
            self._update_memory_buffer(train_data)
            
            # Compute metrics
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted', zero_division=0
            )
            
            epoch_metrics = TaskMetrics(
                accuracy=accuracy,
                loss=avg_loss,
                precision=precision,
                recall=recall,
                f1_score=f1,
                task_id=task_id,
                epoch=epoch
            )
            task_metrics.append(epoch_metrics)
            
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s - "
                f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
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
            'intents': new_intents,
            'memory_size_after': len(self.memory_buffer)
        }
        
        self.logger.info(f"Task {task_id} ({domain}) training completed")
        return task_metrics
    
    def _update_memory_buffer(self, train_data: List[IntentExample]) -> None:
        """Add training examples to memory buffer."""
        for example in train_data:
            memory_item = MemoryItem(
                input_data=example,
                target=example.intent,
                task_id=example.task_id,
                timestamp=time.time(),
                metadata={'domain': example.domain}
            )
            self.memory_buffer.add(memory_item)
    
    def evaluate_task(
        self,
        test_data: List[IntentExample],
        task_id: int,
        domain: str = None,
        **kwargs
    ) -> TaskMetrics:
        """Evaluate on intent classification test data."""
        
        if domain is None:
            domain = self.task_id_to_domain.get(task_id, "unknown")
        
        if task_id not in self.task_classifiers:
            self.logger.error(f"No classifier found for task {task_id}")
            return TaskMetrics(accuracy=0.0, loss=0.0, precision=0.0, recall=0.0, f1_score=0.0, task_id=task_id, epoch=-1)
        
        # Preprocess and vectorize
        texts = [self._preprocess_text(example.text) for example in test_data]
        labels = [self.intent_map[example.intent] for example in test_data]
        X = self.task_vectorizers[task_id].transform(texts)
        
        # Predict and compute metrics
        predictions = self.task_classifiers[task_id].predict(X)
        probs = self.task_classifiers[task_id].predict_proba(X)
        
        # Compute loss
        total_loss = 0.0
        for prob, y in zip(probs, labels):
            total_loss += -np.log(prob[y] + 1e-10)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
        avg_loss = total_loss / len(labels) if len(labels) > 0 else 0.0
        
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
        return_probabilities: bool = False,
        **kwargs
    ) -> List[Any]:
        """Predict intents for new text data."""
        
        if task_id not in self.task_classifiers:
            self.logger.error(f"No classifier found for task {task_id}")
            return []
        
        # Preprocess and vectorize
        processed_texts = [self._preprocess_text(text) for text in texts]
        X = self.task_vectorizers[task_id].transform(processed_texts)
        
        # Predict
        predictions = self.task_classifiers[task_id].predict(X)
        probabilities = self.task_classifiers[task_id].predict_proba(X) if return_probabilities else None
        
        # Convert indices to intent labels
        intent_predictions = [
            IntentPrediction(
                intent=self.idx_to_intent[pred],
                probability=np.max(prob) if probabilities is not None else None
            ) for pred, prob in zip(predictions, probabilities if probabilities else [None] * len(predictions))
        ]
        
        return [pred.intent for pred in intent_predictions] if not return_probabilities else intent_predictions
    
    def predict_single(
        self,
        text: str,
        task_id: int,
        return_probability: bool = False
    ) -> Any:
        """Predict intent for a single text."""
        results = self.predict([text], task_id, return_probabilities=return_probability)
        return results[0]
    
    def get_model_specific_state(self) -> Dict[str, Any]:
        """Get online intent-specific state for checkpointing."""
        # Serialize scikit-learn models and vectorizers
        serialized_classifiers = {
            task_id: pickle.dumps(classifier) for task_id, classifier in self.task_classifiers.items()
        }
        serialized_vectorizers = {
            task_id: pickle.dumps(vectorizer) for task_id, vectorizer in self.task_vectorizers.items()
        }
        
        return {
            'memory_buffer': self.memory_buffer,
            'domain_to_task_id': self.domain_to_task_id,
            'task_id_to_domain': self.task_id_to_domain,
            'intent_map': self.intent_map,
            'idx_to_intent': self.idx_to_intent,
            'task_intents': self.task_intents,
            'task_classifiers': serialized_classifiers,
            'task_vectorizers': serialized_vectorizers,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
    
    def load_model_specific_state(self, checkpoint: Dict[str, Any]) -> None:
        """Load online intent-specific state from checkpoint."""
        self.memory_buffer = checkpoint.get('memory_buffer', self.memory_buffer)
        self.domain_to_task_id = checkpoint.get('domain_to_task_id', {})
        self.task_id_to_domain = checkpoint.get('task_id_to_domain', {})
        self.intent_map = checkpoint.get('intent_map', {})
        self.idx_to_intent = checkpoint.get('idx_to_intent', {})
        self.task_intents = checkpoint.get('task_intents', {})
        
        # Deserialize classifiers and vectorizers
        serialized_classifiers = checkpoint.get('task_classifiers', {})
        self.task_classifiers = {
            int(task_id): pickle.loads(classifier) for task_id, classifier in serialized_classifiers.items()
        }
        serialized_vectorizers = checkpoint.get('task_vectorizers', {})
        self.task_vectorizers = {
            int(task_id): pickle.loads(vectorizer) for task_id, vectorizer in serialized_vectorizers.items()
        }
        
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        self.batch_size = checkpoint.get('batch_size', self.batch_size)
        
        # Set model to None since we use task-specific classifiers
        self.model = None
    
    def analyze_forgetting(self, test_datasets: Dict[str, List[IntentExample]]) -> Dict[str, float]:
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
            'intent_map': dict(self.intent_map)
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
                    'intents': self.task_intents.get(task_id, [])
                }
        
        return summary
