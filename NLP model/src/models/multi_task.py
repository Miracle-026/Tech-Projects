"""
Multi-task learning model for simultaneous training on multiple NLP tasks.
"""
import numpy as np
import logging
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from src.models.base_learner import BaseLearner
from src.utils.data_loader import load_sentiment_data, load_classification_data, load_ner_data, load_intent_data

class MultiTaskModel(BaseLearner):
    """Multi-task learning model for sentiment, classification, NER, and intent tasks."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', task_types: List[str] = ['sentiment', 'classification', 'ner', 'intent'],
                 learning_rate: float = 2e-5, max_length: int = 128, device: str = 'cpu', save_dir: str = 'results/models/', verbose: bool = False):
        super().__init__(model_name, learning_rate, max_length, device, save_dir, verbose)
        self.task_types = task_types
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name).to(device)
        self.heads = nn.ModuleDict({
            'sentiment': nn.Linear(self.bert.config.hidden_size, 2).to(device),  # Binary sentiment
            'classification': nn.Linear(self.bert.config.hidden_size, 2).to(device),  # Binary classification
            'intent': nn.Linear(self.bert.config.hidden_size, 10).to(device),  # Assume 10 intent classes
            'ner': nn.Linear(self.bert.config.hidden_size, 9).to(device)  # Assume 9 NER tags (IOB scheme)
        })
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.task_id_to_domain = {}
        self.logger = logging.getLogger("MultiTaskModel")
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
    
    def forward(self, inputs, task_type: str):
        outputs = self.bert(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        if task_type == 'ner':
            token_outputs = outputs.last_hidden_state  # All tokens
            return self.heads[task_type](token_outputs)
        return self.heads[task_type](cls_output)
    
    def train_task(self, train_data: List[Dict[str, Any]], task_id: int, domain: str, task_type: str, epochs: int = 3, validation_data: Optional[List[Dict[str, Any]]] = None):
        """Train on a specific task."""
        self.task_id_to_domain[task_id] = domain
        self.train()
        metrics = []
        
        for epoch in range(epochs):
            total_loss = 0
            predictions, labels = [], []
            for item in train_data:
                inputs = self.tokenizer(item['text'], return_tensors='pt', max_length=self.max_length, padding=True, truncation=True).to(self.device)
                if task_type == 'ner':
                    label = torch.tensor(item['labels'], dtype=torch.long).to(self.device)
                else:
                    label = torch.tensor(item.get('label', item.get('intent', 0)), dtype=torch.long).to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.forward(inputs, task_type)
                loss = self.criterion(outputs.squeeze(0) if task_type != 'ner' else outputs.squeeze(0), label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                if task_type != 'ner':
                    pred = torch.argmax(outputs, dim=-1).item()
                    predictions.append(pred)
                    labels.append(label.item())
            
            avg_loss = total_loss / len(train_data)
            f1 = self.compute_f1(predictions, labels) if task_type != 'ner' else 0
            metrics.append(type('Metrics', (), {'epoch': epoch, 'loss': avg_loss, 'f1_score': f1})())
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def predict_single(self, text: str, task_id: int, task_type: str, return_probability: bool = False):
        """Predict for a single text."""
        self.eval()
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.forward(inputs, task_type)
            if task_type == 'ner':
                preds = torch.argmax(outputs, dim=-1).squeeze().cpu().numpy()
                tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
                entities = [(i, i+1, f"ENTITY_{pred}") for i, pred in enumerate(preds) if pred != 0]
                return entities
            else:
                probs = torch.softmax(outputs, dim=-1).squeeze().cpu().numpy()
                pred = np.argmax(probs)
                return type('Prediction', (), {'label': pred, 'probability': probs.tolist() if return_probability else None})()
    
    def evaluate_all_tasks(self, test_datasets: Dict[str, List[Dict[str, Any]]]):
        """Evaluate on all tasks."""
        results = {}
        for task_id, domain in self.task_id_to_domain.items():
            task_type = 'sentiment' if 'sentiment' in domain else 'classification' if 'classification' in domain else 'ner' if 'ner' in domain else 'intent'
            test_data = test_datasets.get(domain, [])
            predictions, labels = [], []
            total_loss = 0
            for item in test_data:
                inputs = self.tokenizer(item['text'], return_tensors='pt', max_length=self.max_length, padding=True, truncation=True).to(self.device)
                if task_type == 'ner':
                    label = torch.tensor(item['labels'], dtype=torch.long).to(self.device)
                else:
                    label = torch.tensor(item.get('label', item.get('intent', 0)), dtype=torch.long).to(self.device)
                
                outputs = self.forward(inputs, task_type)
                loss = self.criterion(outputs.squeeze(0) if task_type != 'ner' else outputs.squeeze(0), label)
                total_loss += loss.item()
                
                if task_type != 'ner':
                    pred = torch.argmax(outputs, dim=-1).item()
                    predictions.append(pred)
                    labels.append(label.item())
            
            avg_loss = total_loss / len(test_data) if test_data else 0
            f1 = self.compute_f1(predictions, labels) if task_type != 'ner' else 0
            results[task_id] = type('Metrics', (), {'loss': avg_loss, 'f1_score': f1})()
        return results
    
    def save_checkpoint(self, path: str) -> str:
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'bert_state_dict': self.bert.state_dict(),
            'heads_state_dict': self.heads.state_dict(),
            'task_id_to_domain': self.task_id_to_domain
        }, path)
        self.logger.info(f"Saved checkpoint to {path}")
        return path
    
    @classmethod
    def load_from_checkpoint(cls, path: str, **kwargs):
        """Load model from checkpoint."""
        model = cls(**kwargs)
        checkpoint = torch.load(path)
        model.bert.load_state_dict(checkpoint['bert_state_dict'])
        model.heads.load_state_dict(checkpoint['heads_state_dict'])
        model.task_id_to_domain = checkpoint['task_id_to_domain']
        return model