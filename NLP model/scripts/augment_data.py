"""
Script to augment datasets for low-resource tasks using synonym replacement.
"""

import argparse
import json
from pathlib import Path
import logging
import nltk
from nltk.corpus import wordnet
import random
from typing import List, Dict, Any

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("DataAugmentor")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def load_wordnet():
    """Download WordNet for synonym replacement."""
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

def get_synonyms(word: str) -> List[str]:
    """Get synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.add(synonym)
    return list(synonyms)

def augment_text(text: str, replace_prob: float = 0.3) -> str:
    """Augment text by replacing words with synonyms."""
    words = nltk.word_tokenize(text)
    new_words = words.copy()
    for i, word in enumerate(words):
        if random.random() < replace_prob:
            synonyms = get_synonyms(word)
            if synonyms:
                new_words[i] = random.choice(synonyms)
    return ' '.join(new_words)

def augment_dataset(data: List[Dict[str, Any]], task_type: str, domain: str, task_id: int, augmentation_factor: int) -> List[Dict[str, Any]]:
    """Augment dataset by generating synthetic samples."""
    augmented_data = data.copy()
    for item in data:
        for _ in range(augmentation_factor - 1):
            new_item = item.copy()
            new_item['text'] = augment_text(item['text'])
            new_item['task_id'] = task_id
            new_item['domain'] = domain
            augmented_data.append(new_item)
    return augmented_data

def main():
    parser = argparse.ArgumentParser(description="Augment datasets for nlp-continual-learning")
    parser.add_argument('--data-dir', type=str, default='datasets/processed/', help='Input dataset directory')
    parser.add_argument('--output-dir', type=str, default='datasets/augmented/', help='Output directory for augmented datasets')
    parser.add_argument('--task-type', choices=['sentiment', 'classification', 'language_modeling', 'ner', 'intent'],
                        required=True, help='Task type to augment')
    parser.add_argument('--domains', nargs='+', default=['general'], help='Domains to augment')
    parser.add_argument('--augmentation-factor', type=int, default=2, help='Number of augmented samples per original sample')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load WordNet
    load_wordnet()
    
    # Setup directories
    data_dir = Path(args.data_dir) / args.task_type
    output_dir = Path(args.output_dir) / args.task_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            
            # Augment data
            logger.info(f"Augmenting {input_file}")
            augmented_data = augment_dataset(data, args.task_type, domain, args.domains.index(domain), args.augmentation_factor)
            
            # Save augmented data
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(augmented_data, f, indent=2)
            logger.info(f"Saved augmented data to {output_file}")

if __name__ == "__main__":
    main()