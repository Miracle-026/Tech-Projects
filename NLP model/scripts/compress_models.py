"""
Script to compress models using pruning, quantization, and knowledge distillation.
"""

import argparse
import yaml
from pathlib import Path
import logging
import json
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from src.models.ewc_sentiment import EWCSentimentAnalyzer
from src.models.replay_classifier import ReplayClassifier
from src.models.progressive_lm import ProgressiveLMAnalyzer
from src.models.gem_ner import GEMNERAnalyzer
from src.models.online_intent import OnlineIntentAnalyzer
from src.utils.data_loader import load_sentiment_data, load_classification_data, load_language_modeling_data, load_ner_data, load_intent_data

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("ModelCompressor")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def prune_model(model, sparsity: float):
    """Apply L1 unstructured pruning to the model."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            prune.l1_unstructured(module, name='weight', amount=sparsity)
    return model

def quantize_model(model):
    """Apply dynamic quantization to the model."""
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

def distill_model(teacher_model, student_model, train_data: list, task_id: int, domain: str, epochs: int, learning_rate: float):
    """Distill knowledge from teacher to student model."""
    student_model.train()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    for epoch in range(epochs):
        for item in train_data:
            inputs = item['text']
            with torch.no_grad():
                teacher_outputs = teacher_model.predict_single(inputs, task_id, return_probability=True)
            student_outputs = student_model.predict_single(inputs, task_id, return_probability=True)
            teacher_probs = torch.tensor(teacher_outputs.probability, dtype=torch.float32).log()
            student_probs = torch.tensor(student_outputs.probability, dtype=torch.float32).log()
            loss = criterion(student_probs, teacher_probs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return student_model

def main():
    parser = argparse.ArgumentParser(description="Compress models using pruning, quantization, and distillation")
    parser.add_argument('--model-name', choices=['ewc_sentiment', 'replay_classifier', 'progressive_lm', 'gem_ner', 'online_intent'],
                        required=True, help='Model to compress')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='datasets/processed/', help='Directory containing training data')
    parser.add_argument('--checkpoint-dir', type=str, default='results/models/', help='Directory containing pre-trained checkpoints')
    parser.add_argument('--save-dir', type=str, default='results/compressed/', help='Directory to save compressed models')
    parser.add_argument('--domain', type=str, required=True, help='Domain of the model')
    parser.add_argument('--task-id', type=int, required=True, help='Task ID for the domain')
    parser.add_argument('--sparsity', type=float, default=0.3, help='Sparsity level for pruning')
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
    
    # Load teacher model
    ext = 'pt' if args.model_name in ['ewc_sentiment', 'progressive_lm', 'gem_ner'] else 'ckpt' if args.model_name == 'replay_classifier' else 'pkl'
    checkpoint_path = Path(args.checkpoint_dir) / f"{args.model_name}_{args.domain}_task_{args.task_id}.{ext}"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    teacher_model = model_class.load_from_checkpoint(str(checkpoint_path))
    teacher_model.verbose = args.verbose
    
    # Initialize student model (same architecture, smaller size)
    student_model = model_class(
        model_name=config.get('model_name', 'distilbert-base-uncased' if args.model_name != 'online_intent' else None),
        memory_size=config.get('memory_size', 1000),
        learning_rate=config.get('learning_rate', 2e-5),
        batch_size=config.get('batch_size', 16),
        max_length=config.get('max_length', 128),
        device=config.get('device', 'cpu'),
        save_dir=args.save_dir,
        verbose=args.verbose
    )
    
    # Load training data
    train_file = Path(args.data_dir) / task_type / args.domain / 'train.json'
    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        return
    train_data = data_loader(str(train_file), args.task_id, args.domain)
    
    # Apply compression
    logger.info(f"Compressing {args.model_name} for {args.domain}")
    
    # Step 1: Pruning
    logger.info("Applying pruning...")
    student_model = prune_model(student_model, args.sparsity)
    
    # Step 2: Distillation
    logger.info("Applying knowledge distillation...")
    student_model = distill_model(teacher_model, student_model, train_data[:100], args.task_id, args.domain, epochs=1, learning_rate=config.get('learning_rate', 2e-5))
    
    # Step 3: Quantization
    logger.info("Applying quantization...")
    student_model = quantize_model(student_model)
    
    # Save compressed model
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / f"{args.model_name}_{args.domain}_compressed_task_{args.task_id}.{ext}"
    student_model.save_checkpoint(str(checkpoint_path))
    logger.info(f"Saved compressed model to {checkpoint_path}")

if __name__ == "__main__":
    main()