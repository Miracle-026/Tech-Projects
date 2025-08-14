"""
Script to run end-to-end tests for all components in the repository.
"""

import argparse
import subprocess
import logging
from pathlib import Path
import json

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("EndToEndTester")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def run_command(command: str, logger: logging.Logger) -> bool:
    """Run a shell command and return success status."""
    logger.info(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Success: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run end-to-end tests for nlp-continual-learning")
    parser.add_argument('--data-dir', type=str, default='datasets/processed/', help='Directory containing test datasets')
    parser.add_argument('--save-dir', type=str, default='results/tests/', help='Directory to save test results')
    parser.add_argument('--configs-dir', type=str, default='configs/', help='Directory containing configuration files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Setup output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Test commands
    test_commands = [
        # Unit tests
        f"python -m unittest tests/test_models.py",
        # Model evaluation
        f"python scripts/evaluate_models.py --data-dir {args.data_dir} --save-dir results/models/ --configs-dir {args.configs_dir}",
        # Hyperparameter tuning (small subset)
        f"python scripts/tune_hyperparameters.py --data-dir {args.data_dir} --save-dir results/tuning/ --configs-dir {args.configs_dir}",
        # Data augmentation
        f"python scripts/augment_data.py --data-dir {args.data_dir} --output-dir datasets/augmented/ --task-type sentiment --domains movies",
        # Monitoring
        f"python scripts/monitor_models.py --data-dir {args.data_dir} --new-data-dir datasets/new/ --save-dir results/monitoring/",
        # Transfer learning
        f"python scripts/transfer_learning.py --model-name ewc_sentiment --config configs/ewc_config.yaml --data-dir {args.data_dir} --checkpoint-dir results/models/ --save-dir results/models/ --new-domain books --task-id 1",
        # Cross-task evaluation
        f"python scripts/cross_task_eval.py --data-dir {args.data_dir} --save-dir results/cross_task/ --configs-dir {args.configs_dir}",
        # Active learning
        f"python scripts/active_learning.py --model-name ewc_sentiment --config configs/ewc_config.yaml --data-dir {args.data_dir} --save-dir results/active_learning/ --domain movies --task-id 0 --n-samples 10",
        # Model compression
        f"python scripts/compress_models.py --model-name ewc_sentiment --config configs/ewc_config.yaml --data-dir {args.data_dir} --checkpoint-dir results/models/ --save-dir results/compressed/ --domain movies --task-id 0",
        # Retraining
        f"python scripts/retrain_models.py --model-name ewc_sentiment --config configs/ewc_config.yaml --data-dir {args.data_dir} --active-learning-dir results/active_learning/ --checkpoint-dir results/models/ --save-dir results/models/ --domain movies --task-id 0",
        # Data versioning
        f"python scripts/version_data.py --data-dir {args.data_dir} --version-dir datasets/versions/ --task-type sentiment --domains movies"
    ]
    
    # Run tests
    results = {}
    for cmd in test_commands:
        test_name = cmd.split()[1] if cmd.startswith('python') else cmd
        results[test_name] = {'success': run_command(cmd, logger)}
    
    # Save results
    results_file = save_dir / 'end_to_end_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved test results to {results_file}")
    
    # Check for failures
    failed_tests = [name for name, result in results.items() if not result['success']]
    if failed_tests:
        logger.error(f"Failed tests: {failed_tests}")
    else:
        logger.info("All end-to-end tests passed successfully")

if __name__ == "__main__":
    main()