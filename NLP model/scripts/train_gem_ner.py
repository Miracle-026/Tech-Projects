"""
Training script for GEM-based Named Entity Recognition model.
"""

import argparse
import yaml
from pathlib import Path
import torch
from typing import List, Dict, Any
import logging
import json

from src.models.gem_ner import GEMNERAnalyzer, NERExample
from src.utils.data_loader import load_ner_data  # Assumes a utility function for loading NER data

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("GEMNERTraining")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def main():
    parser = argparse.ArgumentParser(description="Train GEM-based NER model")
    parser.add_argument('--config', type=str, default='configs/gem_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='datasets/ner/',
                        help='Directory containing NER datasets')
    parser.add_argument('--save-dir', type=str, default='results/models/',
                        help='Directory to save model checkpoints')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = GEMNERAnalyzer(
        model_name=config.get('model_name', 'bert-base-uncased'),
        memory_size=config.get('memory_size', 1000),
        learning_rate=config.get('learning_rate', 2e-5),
        batch_size=config.get('batch_size', 16),
        max_length=config.get('max_length', 512),
        device=config.get('device', 'auto'),
        save_dir=args.save_dir,
        verbose=args.verbose
    )
    
    # Load datasets
    data_dir = Path(args.data_dir)
    domains = config.get('domains', [])
    train_datasets: Dict[str, List[NERExample]] = {}
    test_datasets: Dict[str, List[NERExample]] = {}
    
    for domain in domains:
        train_file = data_dir / domain / 'train.json'
        test_file = data_dir / domain / 'test.json'
        if train_file.exists():
            train_data = load_ner_data(str(train_file))  # Assumes JSON format with text and entities
            train_datasets[domain] = [NERExample(**d, task_id=0, domain=domain) for d in train_data]
        if test_file.exists():
            test_data = load_ner_data(str(test_file))
            test_datasets[domain] = [NERExample(**d, task_id=0, domain=domain) for d in test_data]
    
    # Train sequentially on each domain
    for task_id, domain in enumerate(domains):
        if domain not in train_datasets:
            logger.warning(f"No training data for domain {domain}")
            continue
        
        logger.info(f"Training on domain: {domain}")
        metrics = model.train_task(
            train_data=train_datasets[domain],
            task_id=task_id,
            domain=domain,
            epochs=config.get('epochs', 3),
            validation_data=test_datasets.get(domain)
        )
        
        # Save checkpoint
        checkpoint_path = model.save_checkpoint(f"gem_ner_{domain}_task_{task_id}.pt")
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Evaluate on all previous tasks
        results = model.evaluate_all_tasks(test_datasets)
        for eval_task_id, eval_metrics in results.items():
            eval_domain = model.task_id_to_domain.get(eval_task_id, "unknown")
            logger.info(
                f"Evaluation on {eval_domain} (Task {eval_task_id}): "
                f"Accuracy: {eval_metrics.accuracy:.4f}, F1: {eval_metrics.f1_score:.4f}"
            )
        
        # Analyze forgetting
        forgetting_scores = model.analyze_forgetting(test_datasets)
        logger.info(f"Forgetting scores: {forgetting_scores}")
        
        # Save results
        results_path = Path(args.save_dir) / 'results' / f"{domain}_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump({
                'domain': domain,
                'task_id': task_id,
                'metrics': [vars(m) for m in metrics],
                'forgetting': forgetting_scores
            }, f, indent=2)
    
    # Generate and save summary
    summary = model.get_domain_summary()
    summary_path = Path(args.save_dir) / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")

if __name__ == "__main__":
    main()