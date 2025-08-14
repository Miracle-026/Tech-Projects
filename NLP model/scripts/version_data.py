"""
Script to manage dataset versions for reproducibility.
"""

import argparse
import yaml
from pathlib import Path
import logging
import json
import hashlib
from datetime import datetime

def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("DataVersioner")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def compute_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Manage dataset versions for reproducibility")
    parser.add_argument('--data-dir', type=str, default='datasets/processed/', help='Directory containing datasets')
    parser.add_argument('--version-dir', type=str, default='datasets/versions/', help='Directory to store version metadata')
    parser.add_argument('--task-type', choices=['sentiment', 'classification', 'language_modeling', 'ner', 'intent'],
                        required=True, help='Task type to version')
    parser.add_argument('--domains', nargs='+', default=['general'], help='Domains to version')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Setup directories
    data_dir = Path(args.data_dir) / args.task_type
    version_dir = Path(args.version_dir) / args.task_type
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Version datasets
    version_metadata = {}
    for domain in args.domains:
        version_metadata[domain] = {}
        for split in ['train', 'test', 'unlabeled']:
            data_file = data_dir / domain / f'{split}.json'
            if not data_file.exists():
                logger.warning(f"Skipping {data_file}: File not found")
                continue
            
            # Compute hash
            file_hash = compute_hash(data_file)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_id = f"{split}_{timestamp}_{file_hash[:8]}"
            
            # Copy file to version directory
            version_file = version_dir / domain / f"{version_id}.json"
            version_file.parent.mkdir(parents=True, exist_ok=True)
            with open(data_file, 'r') as src, open(version_file, 'w') as dst:
                dst.write(src.read())
            
            # Store metadata
            version_metadata[domain][split] = {
                'version_id': version_id,
                'file': str(version_file),
                'hash': file_hash,
                'timestamp': timestamp
            }
            logger.info(f"Versioned {data_file} as {version_id}")
    
    # Save metadata
    metadata_file = version_dir / 'version_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(version_metadata, f, indent=2)
    logger.info(f"Saved version metadata to {metadata_file}")

if __name__ == "__main__":
    main()