"""
Script to fine-tune pre-trained models on new domains.
"""

import argparse
import yaml
from pathlib import Path
import logging
import json
from src.models.ewc_sentiment import EWCSentimentAnalyzer
from src.models.replay_classifier import ReplayClassifier
from src.models.progressive_lm import ProgressiveLMAnalyzer
from src.models.gem_ner import GEMNERAnalyzer
from src.models.online_intent import OnlineIntentAnalyzer
from src.utils.data_loader import load_sentiment_data, load_classification_data, load_language_modeling_data, load_ner_data, load_intent_data

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("TransferLearning")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained models on new domains")
    parser.add_argument('--model-name', choices=['ewc_sentiment', 'replay_classifier', 'progressive_lm', 'gem_ner', 'online_intent'],
                        required=True, help='Model to fine-tune')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='datasets/processed/', help='Directory containing new domain data')
    parser.add_argument('--checkpoint-dir', type=str, default='results/models/', help='Directory containing pre-trained checkpoints')
    parser.add_argument('--save-dir', type=str, default='results/models/', help='Directory to save fine-tuned checkpoints')
    parser.add_argument('--new-domain', type=str, required=True, help='New domain to fine-tune on')
    parser.add_argument('--task-id', type=int, required=True, help='Task ID for the new domain')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Model classes
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
    
    # Load pre-trained model
    ext = 'pt' if args.model_name in ['ewc_sentiment', 'progressive_lm', 'gem_ner'] else 'ckpt' if args.model_name == 'replay_classifier' else 'pkl'
    checkpoint_path = Path(args.checkpoint_dir) / f"{args.model_name}_{config['domains'][0]}_task_0.{ext}"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    model = model_class.load_from_checkpoint(str(checkpoint_path))
    model.verbose = args.verbose
    
    # Load new domain data
    train_file = Path(args.data_dir) / task_type / args.new_domain / 'train.json'
    test_file = Path(args.data_dir) / task_type / args.new_domain / 'test.json'
    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        return
    
    train_data = data_loader(str(train_file), args.task_id, args.new_domain)
    test_data = data_loader(str(test_file), args.task_id, args.new_domain) if test_file.exists() else None
    
    # Fine-tune model
    logger.info(f"Fine-tuning {args.model_name} on {args.new_domain}")
    metrics = model.train_task(
        train_data=train_data,
        task_id=args.task_id,
        domain=args.new_domain,
        epochs=config.get('epochs', 3),
        validation_data=test_data
    )
    
    # Save checkpoint
    checkpoint_path = Path(args.save_dir) / f"{args.model_name}_{args.new_domain}_task_{args.task_id}.{ext}"
    model.save_checkpoint(str(checkpoint_path))
    logger.info(f"Saved fine-tuned checkpoint: {checkpoint_path}")
    
    # Evaluate
    if test_data:
        eval_results = model.evaluate_all_tasks({args.new_domain: test_data})
        for task_id, eval_metrics in eval_results.items():
            logger.info(f"Evaluation on {args.new_domain} (Task {task_id}): {vars(eval_metrics)}")
    
    # Save results
    results_path = Path(args.save_dir) / 'results' / f"{args.model_name}_{args.new_domain}_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'model': args.model_name,
            'domain': args.new_domain,
            'task_id': args.task_id,
            'metrics': [vars(m) for m in metrics]
        }, f, indent=2)
    logger.info(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()