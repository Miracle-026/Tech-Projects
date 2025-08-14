"""
Script to benchmark model performance on CPU and GPU.
"""

import argparse
import yaml
from pathlib import Path
import logging
import json
import time
import torch
from src.models.ewc_sentiment import EWCSentimentAnalyzer
from src.models.replay_classifier import ReplayClassifier
from src.models.progressive_lm import ProgressiveLMAnalyzer
from src.models.gem_ner import GEMNERAnalyzer
from src.models.online_intent import OnlineIntentAnalyzer
from src.models.multi_task import MultiTaskModel
from src.utils.data_loader import load_sentiment_data, load_classification_data, load_language_modeling_data, load_ner_data, load_intent_data

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("ModelBenchmarker")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def measure_performance(model, data: list, task_id: int, task_type: str, device: str, iterations: int = 100) -> dict:
    """Measure inference time and memory usage."""
    logger = logging.getLogger("ModelBenchmarker")
    model.to(device)
    model.eval()
    
    # Warm-up run
    for item in data[:5]:
        model.predict_single(item['text'], task_id, task_type=task_type)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(iterations):
        for item in data[:10]:  # Use small subset for speed
            model.predict_single(item['text'], task_id, task_type=task_type)
    avg_time = (time.time() - start_time) / (iterations * len(data[:10]))
    
    # Measure memory usage
    torch.cuda.empty_cache() if device == 'cuda' else None
    mem_before = torch.cuda.memory_allocated(device) if device == 'cuda' else 0
    for item in data[:5]:
        model.predict_single(item['text'], task_id, task_type=task_type)
    mem_after = torch.cuda.memory_allocated(device) if device == 'cuda' else 0
    memory_usage = (mem_after - mem_before) / 1e6 if device == 'cuda' else 0  # MB
    
    return {'avg_inference_time_ms': avg_time * 1000, 'memory_usage_mb': memory_usage}

def main():
    parser = argparse.ArgumentParser(description="Benchmark model performance on CPU and GPU")
    parser.add_argument('--data-dir', type=str, default='datasets/processed/', help='Directory containing test datasets')
    parser.add_argument('--checkpoint-dir', type=str, default='results/models/', help='Directory containing model checkpoints')
    parser.add_argument('--save-dir', type=str, default='results/benchmarks/', help='Directory to save benchmark results')
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
        'online_intent': Path(args.configs_dir) / 'online_config.yaml',
        'multi_task': Path(args.configs_dir) / 'ewc_config.yaml'  # Reuse for multi-task
    }
    
    model_configs = {}
    domains = {}
    for model_name, config_path in configs.items():
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_configs[model_name] = yaml.safe_load(f)
            task_type = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
            domains[task_type] = model_configs[model_name].get('domains', ['general'])
    
    # Model classes and data loaders
    model_classes = {
        'ewc_sentiment': EWCSentimentAnalyzer,
        'replay_classifier': ReplayClassifier,
        'progressive_lm': ProgressiveLMAnalyzer,
        'gem_ner': GEMNERAnalyzer,
        'online_intent': OnlineIntentAnalyzer,
        'multi_task': MultiTaskModel
    }
    
    data_loaders = {
        'ewc_sentiment': load_sentiment_data,
        'replay_classifier': load_classification_data,
        'progressive_lm': load_language_modeling_data,
        'gem_ner': load_ner_data,
        'online_intent': load_intent_data,
        'multi_task': load_sentiment_data
    }
    
    results = {}
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    for model_name, model_class in model_classes.items():
        task_type = model_name.replace('_', '') if model_name != 'online_intent' else 'intent'
        domain = domains[task_type][0]
        ext = 'pt' if model_name in ['ewc_sentiment', 'progressive_lm', 'gem_ner', 'multi_task'] else 'ckpt' if model_name == 'replay_classifier' else 'pkl'
        checkpoint_path = Path(args.checkpoint_dir) / f"{model_name}_{domain}_task_0.{ext}"
        
        if not checkpoint_path.exists():
            logger.warning(f"Skipping {model_name}: Checkpoint not found")
            continue
        
        model = model_class.load_from_checkpoint(str(checkpoint_path))
        model.verbose = args.verbose
        
        test_file = Path(args.data_dir) / task_type / domain / 'test.json'
        if not test_file.exists():
            logger.warning(f"Skipping {model_name}: Test data not found")
            continue
        
        test_data = data_loaders[model_name](str(test_file), 0, domain)
        
        results[model_name] = {}
        for device in devices:
            logger.info(f"Benchmarking {model_name} on {device}")
            perf = measure_performance(model, test_data, task_id=0, task_type=task_type, device=device)
            results[model_name][device] = perf
            logger.info(f"{model_name} on {device}: {perf}")
    
    # Save results
    results_file = save_dir / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved benchmark results to {results_file}")

if __name__ == "__main__":
    main()