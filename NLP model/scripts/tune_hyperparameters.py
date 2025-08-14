"""
Script to perform hyperparameter tuning for all models using grid search.
"""

import argparse
import yaml
from pathlib import Path
import logging
import json
import itertools
from src.models.ewc_sentiment import EWCSentimentAnalyzer
from src.models.replay_classifier import ReplayClassifier
from src.models.progressive_lm import ProgressiveLMAnalyzer
from src.models.gem_ner import GEMNERAnalyzer
from src.models.online_intent import OnlineIntentAnalyzer
from src.utils.data_loader import load_sentiment_data, load_classification_data, load_language_modeling_data, load_ner_data, load_intent_data

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("HyperparameterTuner")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_search_grid(model_name: str) -> list:
    """Define hyperparameter search grid for each model."""
    if model_name == 'ewc_sentiment':
        return [
            {'learning_rate': lr, 'batch_size': bs, 'ewc_lambda': el}
            for lr, bs, el in itertools.product(
                [1e-5, 2e-5, 5e-5], [8, 16], [0.2, 0.4, 0.6]
            )
        ]
    elif model_name == 'replay_classifier':
        return [
            {'learning_rate': lr, 'batch_size': bs}
            for lr, bs in itertools.product(
                [1e-5, 2e-5, 5e-5], [8, 16]
            )
        ]
    elif model_name == 'progressive_lm':
        return [
            {'learning_rate': lr, 'batch_size': bs}
            for lr, bs in itertools.product(
                [1e-5, 2e-5, 5e-5], [8, 16]
            )
        ]
    elif model_name == 'gem_ner':
        return [
            {'learning_rate': lr, 'batch_size': bs}
            for lr, bs in itertools.product(
                [1e-5, 2e-5, 5e-5], [8, 16]
            )
        ]
    elif model_name == 'online_intent':
        return [
            {'learning_rate': lr, 'batch_size': bs}
            for lr, bs in itertools.product(
                [0.01, 0.05, 0.1], [8, 16]
            )
        ]
    return []

def tune_model(model_class, model_name: str, config: dict, train_data: list, test_data: list, task_id: int, domain: str, save_dir: str, verbose: bool) -> dict:
    """Tune a single model with grid search."""
    logger = logging.getLogger("HyperparameterTuner")
    search_grid = get_search_grid(model_name)
    best_metrics = None
    best_params = None
    best_score = float('-inf')
    
    for params in search_grid:
        logger.info(f"Testing {model_name} with params: {params}")
        model = model_class(
            model_name=config.get('model_name', 'bert-base-uncased' if model_name != 'online_intent' else None),
            memory_size=config.get('memory_size', 1000),
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            max_length=config.get('max_length', 128),
            ewc_lambda=params.get('ewc_lambda', 0.4) if model_name == 'ewc_sentiment' else None,
            device=config.get('device', 'cpu'),
            save_dir=save_dir,
            verbose=verbose
        )
        
        metrics = model.train_task(train_data, task_id, domain, epochs=1, validation_data=test_data)
        score = metrics[-1].f1_score if model_name != 'progressive_lm' else -metrics[-1].perplexity
        if score > best_score:
            best_score = score
            best_metrics = metrics[-1]
            best_params = params
    
    return {'params': best_params, 'metrics': vars(best_metrics), 'score': best_score}

def main():
    parser = argparse.ArgumentParser(description="Tune hyperparameters for all models")
    parser.add_argument('--data-dir', type=str, default='datasets/processed/', help='Directory containing processed datasets')
    parser.add_argument('--save-dir', type=str, default='results/tuning/', help='Directory to save tuning results')
    parser.add_argument('--configs-dir', type=str, default='configs/', help='Directory containing configuration files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load configurations
    configs = {
        'ewc_sentiment': Path(args.configs_dir) / 'ewc_config.yaml',
        'replay_classifier': Path(args.configs_dir) / 'replay_config.yaml',
        'progressive_lm': Path(args.configs_dir) / 'progressive_lm_config.yaml',
        'gem_ner': Path(args.configs_dir) / 'gem_config.yaml',
        'online_intent': Path(args.configs_dir) / 'online_config.yaml'
    }
    
    model_configs = {}
    domains = {}
    for model_name, config_path in configs.items():
        with open(config_path, 'r') as f:
            model_configs[model_name] = yaml.safe_load(f)
        task_type = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
        domains[task_type] = model_configs[model_name].get('domains', [])
    
    # Load datasets
    data_loaders = {
        'sentiment': load_sentiment_data,
        'classification': load_classification_data,
        'language_modeling': load_language_modeling_data,
        'ner': load_ner_data,
        'intent': load_intent_data
    }
    
    model_classes = {
        'ewc_sentiment': EWCSentimentAnalyzer,
        'replay_classifier': ReplayClassifier,
        'progressive_lm': ProgressiveLMAnalyzer,
        'gem_ner': GEMNERAnalyzer,
        'online_intent': OnlineIntentAnalyzer
    }
    
    # Tune each model
    results = {}
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model_class in model_classes.items():
        task_type = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
        for task_id, domain in enumerate(domains[task_type]):
            test_file = Path(args.data_dir) / task_type / domain / 'test.json'
            train_file = Path(args.data_dir) / task_type / domain / 'train.json'
            if not test_file.exists() or not train_file.exists():
                logger.warning(f"Skipping {model_name} for {domain}: Data not found")
                continue
            
            train_data = data_loaders[task_type](str(train_file), task_id, domain)
            test_data = data_loaders[task_type](str(test_file), task_id, domain)
            
            logger.info(f"Tuning {model_name} for {domain}")
            result = tune_model(
                model_class, model_name, model_configs[model_name], train_data[:100], test_data[:100], task_id, domain, str(save_dir), args.verbose
            )
            results[f"{model_name}_{domain}"] = result
            
            output_file = save_dir / f"{model_name}_{domain}_tuning.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved tuning results to {output_file}")
    
    # Save summary
    summary_file = save_dir / 'tuning_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved tuning summary to {summary_file}")

if __name__ == "__main__":
    main()