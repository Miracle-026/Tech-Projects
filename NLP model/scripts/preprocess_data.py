"""
Script to preprocess and clean datasets for the nlp-continual-learning repository.
"""

import argparse
import json
from pathlib import Path
import logging
import nltk
import spacy
from typing import List, Dict, Any, Tuple

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("DataPreprocessor")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def load_nlp_tools():
    """Load NLTK and spaCy tools."""
    nltk.download('punkt', quiet=True)
    return spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])

def clean_text(text: str, nlp: spacy.language.Language) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = ' '.join(text.strip().split())
    # Tokenize and remove special characters
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_punct]
    return ' '.join(tokens)

def preprocess_sentiment(data: List[Dict[str, Any]], domain: str, task_id: int, nlp: spacy.language.Language) -> List[Dict[str, Any]]:
    """Preprocess sentiment data."""
    cleaned_data = []
    for item in data:
        if not isinstance(item.get('text'), str) or not isinstance(item.get('label'), (int, str)):
            continue
        cleaned_item = {
            'text': clean_text(item['text'], nlp),
            'label': item['label'],
            'task_id': task_id,
            'domain': domain
        }
        cleaned_data.append(cleaned_item)
    return cleaned_data

def preprocess_classification(data: List[Dict[str, Any]], domain: str, task_id: int, nlp: spacy.language.Language) -> List[Dict[str, Any]]:
    """Preprocess classification data (same as sentiment)."""
    return preprocess_sentiment(data, domain, task_id, nlp)

def preprocess_language_modeling(data: List[Dict[str, Any]], domain: str, task_id: int, nlp: spacy.language.Language) -> List[Dict[str, Any]]:
    """Preprocess language modeling data."""
    cleaned_data = []
    for item in data:
        if not isinstance(item.get('text'), str):
            continue
        cleaned_item = {
            'text': clean_text(item['text'], nlp),
            'task_id': task_id,
            'domain': domain
        }
        if 'label' in item and item['label'] is not None:
            cleaned_item['label'] = item['label']
        cleaned_data.append(cleaned_item)
    return cleaned_data

def preprocess_ner(data: List[Dict[str, Any]], domain: str, task_id: int, nlp: spacy.language.Language) -> List[Dict[str, Any]]:
    """Preprocess NER data."""
    cleaned_data = []
    for item in data:
        if not isinstance(item.get('text'), str) or not isinstance(item.get('entities'), list):
            continue
        # Validate entities format: [(start, end, label), ...]
        valid_entities = []
        for entity in item['entities']:
            if (isinstance(entity, (list, tuple)) and len(entity) == 3 and
                isinstance(entity[0], int) and isinstance(entity[1], int) and isinstance(entity[2], str)):
                valid_entities.append(entity)
        cleaned_item = {
            'text': clean_text(item['text'], nlp),
            'entities': valid_entities,
            'task_id': task_id,
            'domain': domain
        }
        cleaned_data.append(cleaned_item)
    return cleaned_data

def preprocess_intent(data: List[Dict[str, Any]], domain: str, task_id: int, nlp: spacy.language.Language) -> List[Dict[str, Any]]:
    """Preprocess intent data."""
    cleaned_data = []
    for item in data:
        if not isinstance(item.get('text'), str) or not isinstance(item.get('intent'), str):
            continue
        cleaned_item = {
            'text': clean_text(item['text'], nlp),
            'intent': item['intent'],
            'task_id': task_id,
            'domain': domain
        }
        cleaned_data.append(cleaned_item)
    return cleaned_data

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for nlp-continual-learning")
    parser.add_argument('--data-dir', type=str, default='datasets/', help='Input dataset directory')
    parser.add_argument('--output-dir', type=str, default='datasets/processed/', help='Output directory for processed datasets')
    parser.add_argument('--task-type', choices=['sentiment', 'classification', 'language_modeling', 'ner', 'intent'],
                        required=True, help='Task type to preprocess')
    parser.add_argument('--domains', nargs='+', default=['general'], help='Domains to process')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load NLP tools
    nlp = load_nlp_tools()
    
    # Setup directories
    data_dir = Path(args.data_dir) / args.task_type
    output_dir = Path(args.output_dir) / args.task_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Preprocessing function mapping
    preprocess_functions = {
        'sentiment': preprocess_sentiment,
        'classification': preprocess_classification,
        'language_modeling': preprocess_language_modeling,
        'ner': preprocess_ner,
        'intent': preprocess_intent
    }
    
    preprocess_fn = preprocess_functions[args.task_type]
    
    # Process each domain and split
    for domain in args.domains:
        for split in ['train', 'test']:
            input_file = data_dir / domain / f'{split}.json'
            output_file = output_dir / domain / f'{split}.json'
            
            if not input_file.exists():
                logger.warning(f"Skipping {input_file}: File not found")
                continue
            
            # Load data
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            # Preprocess data
            logger.info(f"Processing {input_file}")
            cleaned_data = preprocess_fn(data, domain, task_id=args.domains.index(domain), nlp=nlp)
            
            # Save processed data
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(cleaned_data, f, indent=2)
            logger.info(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    main()