"""
Visualization utilities for plotting training metrics, forgetting scores, confusion matrices, and t-SNE embeddings.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import torch
from transformers import AutoModel, AutoTokenizer

def plot_training_metrics(metrics_history: Dict[str, List[Any]], output_path: str, metric_types: List[str], title_prefix: str):
    """Plot training metrics (e.g., accuracy, F1, perplexity) for each domain."""
    plt.figure(figsize=(12, 5))
    
    for i, metric_type in enumerate(metric_types, 1):
        plt.subplot(1, len(metric_types), i)
        for domain, metrics in metrics_history.items():
            values = [getattr(m, metric_type) for m in metrics]
            epochs = [m.epoch + 1 for m in metrics]
            plt.plot(epochs, values, label=f"{domain} {metric_type.capitalize()}")
        plt.xlabel('Epoch')
        plt.ylabel(metric_type.capitalize())
        plt.title(f"{title_prefix} {metric_type.capitalize()} per Domain")
        plt.legend()
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_forgetting_scores(forgetting_scores: Dict[str, float], output_path: str, title: str):
    """Plot forgetting scores across domains."""
    plt.figure(figsize=(8, 6))
    domains = list(forgetting_scores.keys())
    scores = list(forgetting_scores.values())
    plt.bar(domains, scores)
    plt.xlabel('Domain')
    plt.ylabel('Forgetting Score')
    plt.title(title)
    plt.xticks(rotation=45)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_domain_summary(summary: Dict[str, Any], output_path: str, metric: str, title: str):
    """Plot domain performance summary for a specific metric."""
    plt.figure(figsize=(8, 6))
    domains = []
    values = []
    for domain, perf in summary['domain_performance'].items():
        domains.append(domain)
        values.append(perf[f"best_{metric}"])
    plt.bar(domains, values)
    plt.xlabel('Domain')
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.xticks(rotation=45)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(labels: List[Any], predictions: List[Any], output_path: str, title: str):
    """Plot a confusion matrix for classification tasks."""
    unique_labels = sorted(set(labels))
    cm = confusion_matrix(labels, predictions, labels=unique_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_tsne_embeddings(texts: List[str], labels: List[Any], model_name: str, output_path: str, title: str, max_texts: int = 1000):
    """Plot t-SNE embeddings of text representations."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Get embeddings
    embeddings = []
    for text in texts[:max_texts]:
        inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
    embeddings = np.array(embeddings)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels[:max_texts]))
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels[:max_texts]) if l == label]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, alpha=0.6)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(title)
    plt.legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()