"""
Utility functions for loading datasets for continual learning tasks.
Supports NER, sentiment analysis, text classification, and language modeling.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union
import logging
from dataclasses import dataclass

from ..models.gem_ner import NERExample
from ..models.progressive_lm import LanguageModelingExample

logger = logging.getLogger(__name__)

@dataclass
class SentimentExample:
    """Single sentiment analysis example."""
    text: str
    label: int
    task_id: int
    domain: str

@dataclass
class ClassificationExample:
    """Single text classification example."""
    text: str
    label: int
    task_id: int
    domain: str

@dataclass
class IntentExample:
    """Single intent classification example."""
    text: str
    intent: str
    task_id: int
    domain: str

def load_ner_data(file_path: str, task_id: int = 0, domain: str = "default") -> List[NERExample]:
    """
    Load NER data from a JSON file.
    
    Expected JSON format:
    [
        {
            "text": "Sample text",
            "entities": [[start, end, label], ...]
        },
        ...
    ]
    
    Args:
        file_path: Path to the JSON file.
        task_id: Task identifier.
        domain: Domain name for the dataset.
    
    Returns:
        List of NERExample objects.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            if not isinstance(item, dict) or 'text' not in item or 'entities' not in item:
                logger.warning(f"Skipping invalid item in {file_path}: {item}")
                continue
            
            entities = []
            for entity in item['entities']:
                if not (isinstance(entity, list) and len(entity) == 3 and
                        isinstance(entity[0], int) and isinstance(entity[1], int) and
                        isinstance(entity[2], str)):
                    logger.warning(f"Skipping invalid entity in {file_path}: {entity}")
                    continue
                entities.append((entity[0], entity[1], entity[2]))
            
            examples.append(NERExample(
                text=item['text'],
                entities=entities,
                task_id=task_id,
                domain=domain
            ))
        
        logger.info(f"Loaded {len(examples)} NER examples from {file_path}")
        return examples
    
    except Exception as e:
        logger.error(f"Error loading NER data from {file_path}: {str(e)}")
        return []

def load_sentiment_data(file_path: str, task_id: int = 0, domain: str = "default") -> List[SentimentExample]:
    """
    Load sentiment analysis data from a JSON file.
    
    Expected JSON format:
    [
        {
            "text": "Sample text",
            "label": 0  // Integer label (e.g., 0=negative, 1=positive)
        },
        ...
    ]
    
    Args:
        file_path: Path to the JSON file.
        task_id: Task identifier.
        domain: Domain name for the dataset.
    
    Returns:
        List of SentimentExample objects.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            if not isinstance(item, dict) or 'text' not in item or 'label' not in item:
                logger.warning(f"Skipping invalid item in {file_path}: {item}")
                continue
            
            if not isinstance(item['label'], int):
                logger.warning(f"Skipping invalid label in {file_path}: {item['label']}")
                continue
            
            examples.append(SentimentExample(
                text=item['text'],
                label=item['label'],
                task_id=task_id,
                domain=domain
            ))
        
        logger.info(f"Loaded {len(examples)} sentiment examples from {file_path}")
        return examples
    
    except Exception as e:
        logger.error(f"Error loading sentiment data from {file_path}: {str(e)}")
        return []

def load_classification_data(file_path: str, task_id: int = 0, domain: str = "default") -> List[ClassificationExample]:
    """
    Load text classification data from a JSON file.
    
    Expected JSON format:
    [
        {
            "text": "Sample text",
            "label": 0  // Integer label
        },
        ...
    ]
    
    Args:
        file_path: Path to the JSON file.
        task_id: Task identifier.
        domain: Domain name for the dataset.
    
    Returns:
        List of ClassificationExample objects.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            if not isinstance(item, dict) or 'text' not in item or 'label' not in item:
                logger.warning(f"Skipping invalid item in {file_path}: {item}")
                continue
            
            if not isinstance(item['label'], int):
                logger.warning(f"Skipping invalid label in {file_path}: {item['label']}")
                continue
            
            examples.append(ClassificationExample(
                text=item['text'],
                label=item['label'],
                task_id=task_id,
                domain=domain
            ))
        
        logger.info(f"Loaded {len(examples)} classification examples from {file_path}")
        return examples
    
    except Exception as e:
        logger.error(f"Error loading classification data from {file_path}: {str(e)}")
        return []

def load_language_modeling_data(file_path: str, task_id: int = 0, domain: str = "default") -> List[LanguageModelingExample]:
    """
    Load language modeling data from a JSON file.
    
    Expected JSON format:
    [
        {
            "text": "Sample text",
            "label": null  // Optional integer label for supervised tasks
        },
        ...
    ]
    
    Args:
        file_path: Path to the JSON file.
        task_id: Task identifier.
        domain: Domain name for the dataset.
    
    Returns:
        List of LanguageModelingExample objects.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            if not isinstance(item, dict) or 'text' not in item:
                logger.warning(f"Skipping invalid item in {file_path}: {item}")
                continue
            
            label = item.get('label', None)
            if label is not None and not isinstance(label, int):
                logger.warning(f"Skipping invalid label in {file_path}: {label}")
                continue
            
            examples.append(LanguageModelingExample(
                text=item['text'],
                task_id=task_id,
                domain=domain,
                label=label
            ))
        
        logger.info(f"Loaded {len(examples)} language modeling examples from {file_path}")
        return examples
    
    except Exception as e:
        logger.error(f"Error loading language modeling data from {file_path}: {str(e)}")
        return []

def load_intent_data(file_path: str, task_id: int = 0, domain: str = "default") -> List[IntentExample]:
    """
    Load intent classification data from a JSON file.
    
    Expected JSON format:
    [
        {
            "text": "Sample text",
            "intent": "book_flight"  // String intent label
        },
        ...
    ]
    
    Args:
        file_path: Path to the JSON file.
        task_id: Task identifier.
        domain: Domain name for the dataset.
    
    Returns:
        List of IntentExample objects.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            if not isinstance(item, dict) or 'text' not in item or 'intent' not in item:
                logger.warning(f"Skipping invalid item in {file_path}: {item}")
                continue
            
            if not isinstance(item['intent'], str):
                logger.warning(f"Skipping invalid intent in {file_path}: {item['intent']}")
                continue
            
            examples.append(IntentExample(
                text=item['text'],
                intent=item['intent'],
                task_id=task_id,
                domain=domain
            ))
        
        logger.info(f"Loaded {len(examples)} intent examples from {file_path}")
        return examples
    
    except Exception as e:
        logger.error(f"Error loading intent data from {file_path}: {str(e)}")
        return []

def load_dataset(
    data_dir: str,
    task_type: str,
    domains: List[str],
    split: str = "train"
) -> Dict[str, List[Any]]:
    """
    Load datasets for multiple domains and a specific task type.
    
    Args:
        data_dir: Base directory containing datasets (e.g., 'datasets/').
        task_type: Type of task ('ner', 'sentiment', 'classification', 'language_modeling', 'intent').
        domains: List of domain names.
        split: Dataset split ('train', 'test', etc.).
    
    Returns:
        Dictionary mapping domains to lists of examples.
    """
    data_dir = Path(data_dir)
    datasets = {}
    
    loader_functions = {
        'ner': load_ner_data,
        'sentiment': load_sentiment_data,
        'classification': load_classification_data,
        'language_modeling': load_language_modeling_data,
        'intent': load_intent_data
    }
    
    if task_type not in loader_functions:
        logger.error(f"Unsupported task type: {task_type}")
        return datasets
    
    loader = loader_functions[task_type]
    
    for task_id, domain in enumerate(domains):
        file_path = data_dir / task_type / domain / f"{split}.json"
        examples = loader(str(file_path), task_id=task_id, domain=domain)
        if examples:
            datasets[domain] = examples
    
    return datasets
