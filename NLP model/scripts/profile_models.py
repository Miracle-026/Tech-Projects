"""
Script to profile model training performance and suggest optimizations.
"""

import cProfile
import pstats
import psutil
import time
import logging
from pathlib import Path
import argparse
from src.models.gem_ner import GEMNERAnalyzer
from src.models.online_intent import OnlineIntentAnalyzer
from src.models.ewc_sentiment import EWCSentimentAnalyzer
from src.utils.data_loader import load_ner_data, load_intent_data, load_sentiment_data

def setup_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("ModelProfiler")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

def profile_model(model, train_data, task_id: int, domain: str, epochs: int, profile_file: str):
    """Profile a model's training performance."""
    logger = logging.getLogger("ModelProfiler")
    
    # Measure memory usage
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024**2  # MB
    
    # Run profiling
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    metrics = model.train_task(train_data, task_id, domain, epochs=epochs)
    end_time = time.time()
    
    profiler.disable()
    end_memory = process.memory_info().rss / 1024**2  # MB
    
    # Save profile stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').dump_stats(profile_file)
    
    # Log results
    logger.info(f"Training Time: {end_time - start_time:.2f} seconds")
    logger.info(f"Memory Usage: Start={start_memory:.2f} MB, End={end_memory:.2f} MB")
    logger.info(f"Metrics: {metrics[-1].accuracy:.4f} (Accuracy), {metrics[-1].f1_score:.4f} (F1)")
    
    return stats

def suggest_optimizations(stats: pstats.Stats, model_name: str) -> list:
    """Suggest optimizations based on profiling stats."""
    optimizations = []
    
    # Analyze top time-consuming functions
    stats.sort_stats('cumulative')
    top_functions = stats.fstats[:5]  # Top 5 functions by cumulative time
    
    for func, (cc, nc, tt, ct, callers) in top_functions:
        if 'torch' in func[0] or 'transformers' in func[0]:
            optimizations.append("Optimize PyTorch/Transformers calls (e.g., use mixed precision training).")
        if 'nltk' in func[0]:
            optimizations.append("Cache NLTK preprocessing results to reduce overhead.")
        if 'sklearn' in func[0]:
            optimizations.append("Use sparse matrices for scikit-learn operations.")
    
    # Model-specific optimizations
    if model_name == 'gem_ner':
        optimizations.append("Reduce max_length for faster BERT processing.")
        optimizations.append("Batch gradient projection operations for GEM.")
    elif model_name == 'online_intent':
        optimizations.append("Increase batch_size for scikit-learn's SGDClassifier.")
        optimizations.append("Use incremental TF-IDF vectorization.")
    elif model_name == 'ewc_sentiment':
        optimizations.append("Lower ewc_lambda to reduce regularization overhead.")
        optimizations.append("Optimize Fisher matrix computation.")
    
    return list(set(optimizations))  # Remove duplicates

def main():
    parser = argparse.ArgumentParser(description="Profile model training performance")
    parser.add_argument('--data-dir', type=str, default='datasets/', help='Dataset directory')
    parser.add_argument('--save-dir', type=str, default='results/profiling/', help='Directory to save profiling results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize models
    models = {
        'gem_ner': GEMNERAnalyzer(model_name='bert-base-uncased', memory_size=100, batch_size=4, max_length=128, device='cpu', save_dir=args.save_dir, verbose=args.verbose),
        'online_intent': OnlineIntentAnalyzer(memory_size=100, learning_rate=0.01, batch_size=4, device='cpu', save_dir=args.save_dir, verbose=args.verbose),
        'ewc_sentiment': EWCSentimentAnalyzer(model_name='bert-base-uncased', memory_size=100, learning_rate=2e-5, batch_size=4, max_length=128, ewc_lambda=0.4, device='cpu', save_dir=args.save_dir, verbose=args.verbose)
    }
    
    # Load sample data
    data_dir = Path(args.data_dir)
    ner_data = load_ner_data(str(data_dir / 'ner/news/train.json'), task_id=0, domain='news')
    intent_data = load_intent_data(str(data_dir / 'intent/travel/train.json'), task_id=0, domain='travel')
    sentiment_data = load_sentiment_data(str(data_dir / 'sentiment/movies/train.json'), task_id=0, domain='movies')
    
    # Profile each model
    for model_name, model in models.items():
        logger.info(f"Profiling {model_name}")
        data = {
            'gem_ner': ner_data,
            'online_intent': intent_data,
            'ewc_sentiment': sentiment_data
        }[model_name]
        
        profile_file = str(Path(args.save_dir) / f'{model_name}_profile.prof')
        stats = profile_model(model, data[:10], task_id=0, domain=model_name.split('_')[1], epochs=1, profile_file=profile_file)
        
        # Suggest optimizations
        optimizations = suggest_optimizations(stats, model_name)
        logger.info(f"Optimizations for {model_name}:")
        for opt in optimizations:
            logger.info(f"  - {opt}")

if __name__ == '__main__':
    main()