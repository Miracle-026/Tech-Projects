"""
Script to monitor model performance in production and detect data drift.
"""

import argparse
import yaml
from pathlib import Path
import logging
import json
import numpy as np
from scipy.stats import ks_2samp
from src.models.ewc_sentiment import EWCSentimentAnalyzer
from src.models.replay_classifier import ReplayClassifier
from src.models.progressive_lm import ProgressiveLMAnalyzer
from src.models.gem_ner import GEMNERAnalyzer
from src.models.online_intent import OnlineIntentAnalyzer
from src.utils.data_loader import load_sentiment_data, load_classification_data, load_language_modeling_data, load_ner_data, load_intent_data
from transformers import AutoTokenizer

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("ModelMonitor")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_text_features(texts: list, model_name: str = 'bert-base-uncased') -> np.ndarray:
    """Extract text features using BERT tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lengths = [len(tokenizer.tokenize(text)) for text in texts]
    return np.array(lengths).reshape(-1, 1)

def detect_drift(reference_data: list, new_data: list, model_name: str) -> dict:
    """Detect data drift using Kolmogorov-Smirnov test on text length distributions."""
    ref_features = get_text_features([item['text'] for item in reference_data], model_name)
    new_features = get_text_features([item['text'] for item in new_data], model_name)
    ks_stat, p_value = ks_2samp(ref_features.flatten(), new_features.flatten())
    return {'ks_stat': ks_stat, 'p_value': p_value, 'drift_detected': p_value < 0.05}

def monitor_model(model, model_name: str, test_data: list, new_data: list, task_type: str) -> dict:
    """Monitor model performance and drift."""
    logger = logging.getLogger("ModelMonitor")
    results = {'model': model_name, 'performance': {}, 'drift': {}}
    
    # Evaluate on test data
    eval_results = model.evaluate_all_tasks({model.task_id_to_domain.get(0, 'unknown'): test_data})
    results['performance'] = {str(k): vars(v) for k, v in eval_results.items()}
    
    # Detect drift
    if new_data:
        results['drift'] = detect_drift(test_data, new_data, 'bert-base-uncased' if model_name != 'online_intent' else 'distilbert-base-uncased')
        logger.info(f"Drift detection for {model_name}: KS Stat = {results['drift']['ks_stat']:.4f}, p-value = {results['drift']['p_value']:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Monitor model performance and detect data drift")
    parser.add_argument('--data-dir', type=str, default='datasets/processed/', help='Directory containing processed test datasets')
    parser.add_argument('--new-data-dir', type=str, default='datasets/new/', help='Directory containing new incoming data')
    parser.add_argument('--save-dir', type=str, default='results/monitoring/', help='Directory to save monitoring results')
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
    
    # Load models
    model_classes = {
        'ewc_sentiment': EWCSentimentAnalyzer,
        'replay_classifier': ReplayClassifier,
        'progressive_lm': ProgressiveLMAnalyzer,
        'gem_ner': GEMNERAnalyzer,
        'online_intent': OnlineIntentAnalyzer
    }
    
    models = {}
    for model_name, model_class in model_classes.items():
        for task_id, domain in enumerate(domains[model_name.replace('_', '') if model_name != 'online_intent' else 'intent']):
            ext = 'pt' if model_name in ['ewc_sentiment', 'progressive_lm', 'gem_ner'] else 'ckpt' if model_name == 'replay_classifier' else 'pkl'
            checkpoint_path = Path(args.save_dir) / f"{model_name}_{domain}_task_{task_id}.{ext}"
            if checkpoint_path.exists():
                models[f"{model_name}_{domain}"] = model_class.load_from_checkpoint(str(checkpoint_path))
                models[f"{model_name}_{domain}"].verbose = args.verbose
    
    # Load datasets
    data_loaders = {
        'sentiment': load_sentiment_data,
        'classification': load_classification_data,
        'language_modeling': load_language_modeling_data,
        'ner': load_ner_data,
        'intent': load_intent_data
    }
    
    results = {}
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in models.items():
        task_type = model_name.split('_')[0] if model_name != 'online_intent' else 'intent'
        domain = '_'.join(model_name.split('_')[1:-2] if model_name != 'online_intent' else model_name.split('_')[2:-2])
        test_file = Path(args.data_dir) / task_type / domain / 'test.json'
        new_file = Path(args.new_data_dir) / task_type / domain / 'new.json'
        
        if not test_file.exists():
            logger.warning(f"Skipping {model_name}: Test data not found")
            continue
        
        test_data = data_loaders[task_type](str(test_file), models[model_name].domain_to_task_id.get(domain, 0), domain)
        new_data = data_loaders[task_type](str(new_file), models[model_name].domain_to_task_id.get(domain, 0), domain) if new_file.exists() else []
        
        logger.info(f"Monitoring {model_name} for {domain}")
        result = monitor_model(model, model_name, test_data, new_data, task_type)
        results[model_name] = result
        
        output_file = save_dir / f"{model_name}_monitoring.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved monitoring results to {output_file}")
    
    # Save summary
    summary_file = save_dir / 'monitoring_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved monitoring summary to {summary_file}")

if __name__ == "__main__":
    main()