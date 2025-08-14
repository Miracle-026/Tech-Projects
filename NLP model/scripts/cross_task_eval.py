"""
Script to evaluate models on tasks outside their primary domain.
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
    logger = logging.getLogger("CrossTaskEvaluator")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on cross-task datasets")
    parser.add_argument('--data-dir', type=str, default='datasets/processed/', help='Directory containing processed test datasets')
    parser.add_argument('--save-dir', type=str, default='results/cross_task/', help='Directory to save evaluation results')
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
        task_type = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
        for task_id, domain in enumerate(domains[task_type]):
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
    
    # Cross-task evaluation
    for model_name, model in models.items():
        primary_task = model_name.split('_')[0] if model_name != 'online_intent' else 'intent'
        domain = '_'.join(model_name.split('_')[1:-2] if model_name != 'online_intent' else model_name.split('_')[2:-2])
        task_id = model.domain_to_task_id.get(domain, 0)
        
        for task_type in data_loaders.keys():
            if task_type == primary_task:
                continue  # Skip primary task
            for test_domain in domains.get(task_type, []):
                test_file = Path(args.data_dir) / task_type / test_domain / 'test.json'
                if not test_file.exists():
                    continue
                
                test_data = data_loaders[task_type](str(test_file), task_id, test_domain)
                try:
                    eval_results = model.evaluate_all_tasks({test_domain: test_data})
                    results[f"{model_name}_on_{task_type}_{test_domain}"] = {str(k): vars(v) for k, v in eval_results.items()}
                    logger.info(f"Evaluated {model_name} on {task_type}/{test_domain}: {results[f'{model_name}_on_{task_type}_{test_domain}']}")
                except Exception as e:
                    logger.warning(f"Evaluation failed for {model_name} on {task_type}/{test_domain}: {str(e)}")
                    continue
    
    # Save results
    for key, result in results.items():
        output_file = save_dir / f"{key}_eval.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved cross-task evaluation results to {output_file}")
    
    # Save summary
    summary_file = save_dir / 'cross_task_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved cross-task evaluation summary to {summary_file}")

if __name__ == "__main__":
    main()