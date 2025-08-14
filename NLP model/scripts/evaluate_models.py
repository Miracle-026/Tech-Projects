"""
Script to evaluate and compare performance of all models on test datasets.
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
    logger = logging.getLogger("ModelEvaluator")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def load_models(configs: dict, save_dir: str, verbose: bool) -> dict:
    """Load all models from checkpoints."""
    models = {}
    model_configs = {
        'ewc_sentiment': (EWCSentimentAnalyzer, configs.get('ewc_config', 'configs/ewc_config.yaml'), 'pt'),
        'replay_classifier': (ReplayClassifier, configs.get('replay_config', 'configs/replay_config.yaml'), 'ckpt'),
        'progressive_lm': (ProgressiveLMAnalyzer, configs.get('progressive_lm_config', 'configs/progressive_lm_config.yaml'), 'pt'),
        'gem_ner': (GEMNERAnalyzer, configs.get('gem_config', 'configs/gem_config.yaml'), 'pt'),
        'online_intent': (OnlineIntentAnalyzer, configs.get('online_config', 'configs/online_config.yaml'), 'pkl')
    }
    
    for model_name, (model_class, config_path, ext) in model_configs.items():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        domains = config.get('domains', [])
        for task_id, domain in enumerate(domains):
            checkpoint_path = Path(save_dir) / f"{model_name}_{domain}_task_{task_id}.{ext}"
            if checkpoint_path.exists():
                models[f"{model_name}_{domain}"] = model_class.load_from_checkpoint(str(checkpoint_path))
                models[f"{model_name}_{domain}"].verbose = verbose
    return models

def load_test_datasets(data_dir: str, domains: dict) -> dict:
    """Load test datasets for all task types."""
    datasets = {}
    data_loaders = {
        'sentiment': load_sentiment_data,
        'classification': load_classification_data,
        'language_modeling': load_language_modeling_data,
        'ner': load_ner_data,
        'intent': load_intent_data
    }
    
    for task_type, domain_list in domains.items():
        datasets[task_type] = {}
        for task_id, domain in enumerate(domain_list):
            test_file = Path(data_dir) / task_type / domain / 'test.json'
            if test_file.exists():
                datasets[task_type][domain] = data_loaders[task_type](str(test_file), task_id, domain)
    return datasets

def evaluate_models(models: dict, datasets: dict) -> dict:
    """Evaluate all models on their respective test datasets."""
    results = {}
    for model_name, model in models.items():
        task_type = model_name.split('_')[0] if model_name != 'online_intent' else 'intent'
        domain = '_'.join(model_name.split('_')[1:-2] if model_name != 'online_intent' else model_name.split('_')[2:-2])
        if task_type in datasets and domain in datasets[task_type]:
            results[model_name] = model.evaluate_all_tasks({domain: datasets[task_type][domain]})
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on test datasets")
    parser.add_argument('--data-dir', type=str, default='datasets/processed/', help='Directory containing processed test datasets')
    parser.add_argument('--save-dir', type=str, default='results/models/', help='Directory containing model checkpoints')
    parser.add_argument('--output-dir', type=str, default='results/evaluation/', help='Directory to save evaluation results')
    parser.add_argument('--configs-dir', type=str, default='configs/', help='Directory containing configuration files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load configurations
    configs = {
        'ewc_config': Path(args.configs_dir) / 'ewc_config.yaml',
        'replay_config': Path(args.configs_dir) / 'replay_config.yaml',
        'progressive_lm_config': Path(args.configs_dir) / 'progressive_lm_config.yaml',
        'gem_config': Path(args.configs_dir) / 'gem_config.yaml',
        'online_config': Path(args.configs_dir) / 'online_config.yaml'
    }
    
    # Load domains from configs
    domains = {}
    for config_name, config_path in configs.items():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        task_type = config_name.replace('_config', '')
        if task_type == 'ewc': task_type = 'sentiment'
        elif task_type == 'replay': task_type = 'classification'
        elif task_type == 'progressive_lm': task_type = 'language_modeling'
        elif task_type == 'gem': task_type = 'ner'
        elif task_type == 'online': task_type = 'intent'
        domains[task_type] = config.get('domains', [])
    
    # Load models
    logger.info("Loading models...")
    models = load_models(configs, args.save_dir, args.verbose)
    
    # Load test datasets
    logger.info("Loading test datasets...")
    datasets = load_test_datasets(args.data_dir, domains)
    
    # Evaluate models
    logger.info("Evaluating models...")
    results = evaluate_models(models, datasets)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, eval_results in results.items():
        output_file = output_dir / f"{model_name}_eval.json"
        with open(output_file, 'w') as f:
            json.dump({str(task_id): vars(metrics) for task_id, metrics in eval_results.items()}, f, indent=2)
        logger.info(f"Saved evaluation results to {output_file}")
    
    # Summarize results
    summary = {}
    for model_name, eval_results in results.items():
        for task_id, metrics in eval_results.items():
            domain = models[model_name].task_id_to_domain.get(task_id, 'unknown')
            key = f"{model_name}_task_{task_id}"
            summary[key] = {
                'domain': domain,
                'metrics': vars(metrics)
            }
    
    summary_file = output_dir / 'evaluation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved evaluation summary to {summary_file}")

if __name__ == "__main__":
    main()