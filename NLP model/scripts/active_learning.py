"""
Script to implement active learning for selecting informative samples for annotation.
"""

import argparse
import yaml
from pathlib import Path
import logging
import json
import numpy as np
from typing import List, Dict, Any
from src.models.ewc_sentiment import EWCSentimentAnalyzer
from src.models.replay_classifier import ReplayClassifier
from src.models.progressive_lm import ProgressiveLMAnalyzer
from src.models.gem_ner import GEMNERAnalyzer
from src.models.online_intent import OnlineIntentAnalyzer
from src.utils.data_loader import load_sentiment_data, load_classification_data, load_language_modeling_data, load_ner_data, load_intent_data

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("ActiveLearning")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def uncertainty_sampling(model, data: List[Dict[str, Any]], task_id: int, task_type: str, n_samples: int) -> List[int]:
    """Select samples with highest uncertainty for annotation."""
    uncertainties = []
    for idx, item in enumerate(data):
        try:
            if task_type in ['sentiment', 'classification', 'intent']:
                pred = model.predict_single(item['text'], task_id, return_probability=True)
                prob = pred.probability
                entropy = -np.sum(prob * np.log2(prob + 1e-10))
                uncertainties.append((idx, entropy))
            elif task_type == 'ner':
                pred = model.predict_single(item['text'], task_id, return_probability=True)
                probs = [p for ent in pred for p in ent['probabilities']] if pred else []
                entropy = -np.mean([p * np.log2(p + 1e-10) for p in probs]) if probs else 0
                uncertainties.append((idx, entropy))
            else:
                uncertainties.append((idx, 0))
        except Exception as e:
            logger.warning(f"Uncertainty calculation failed for index {idx}: {str(e)}")
            uncertainties.append((idx, 0))
    
    uncertainties.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in uncertainties[:n_samples]]

def main():
    parser = argparse.ArgumentParser(description="Select informative samples using active learning")
    parser.add_argument('--model-name', choices=['ewc_sentiment', 'replay_classifier', 'progressive_lm', 'gem_ner', 'online_intent'],
                        required=True, help='Model to use for active learning')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='datasets/processed/', help='Directory containing unlabeled data')
    parser.add_argument('--save-dir', type=str, default='results/active_learning/', help='Directory to save selected samples')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the data')
    parser.add_argument('--task-id', type=int, required=True, help='Task ID for the domain')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of samples to select')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_classes = {
        'ewc_sentiment': EWCSentimentAnalyzer,
        'replay_classifier': ReplayClassifier,
        'progressive_lm': ProgressiveLMAnalyzer,
        'gem_ner': GEMNERAnalyzer,
        'online_intent': OnlineIntentAnalyzer
    }
    
    data_loaders = {
        'ewc_sentiment': load_sentiment_data,
        'replay_classifier': load_classification_data,
        'progressive_lm': load_language_modeling_data,
        'gem_ner': load_ner_data,
        'online_intent': load_intent_data
    }
    
    model_class = model_classes[args.model_name]
    data_loader = data_loaders[args.model_name]
    task_type = args.model_name.replace('_', '') if args.model_name != 'online_intent' else 'intent'
    
    ext = 'pt' if args.model_name in ['ewc_sentiment', 'progressive_lm', 'gem_ner'] else 'ckpt' if args.model_name == 'replay_classifier' else 'pkl'
    checkpoint_path = Path(args.save_dir) / f"{args.model_name}_{args.domain}_task_{args.task_id}.{ext}"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    model = model_class.load_from_checkpoint(str(checkpoint_path))
    model.verbose = args.verbose
    
    data_file = Path(args.data_dir) / task_type / args.domain / 'unlabeled.json'
    if not data_file.exists():
        logger.error(f"Unlabeled data not found: {data_file}")
        return
    
    data = data_loader(str(data_file), args.task_id, args.domain)
    
    logger.info(f"Selecting {args.n_samples} samples for {args.model_name} in {args.domain}")
    selected_indices = uncertainty_sampling(model, data, args.task_id, task_type, args.n_samples)
    selected_samples = [data[idx] for idx in selected_indices]
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"{args.model_name}_{args.domain}_selected.json"
    with open(output_file, 'w') as f:
        json.dump(selected_samples, f, indent=2)
    logger.info(f"Saved selected samples to {output_file}")

if __name__ == "__main__":
    main()