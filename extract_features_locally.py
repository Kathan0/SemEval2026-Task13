"""
Pre-compute features locally and save to cache files.

This script extracts handcrafted features for all tasks and saves them to cache files.
These cache files can then be uploaded to Google Drive and reused in Colab,
avoiding the need to re-extract features every time.

Usage:
    python extract_features_locally.py --task a                    # Extract Task A (100k samples)
    python extract_features_locally.py --task a --max-samples null # Extract all samples (~500k)
    python extract_features_locally.py --task all                  # Extract all tasks (100k each)
    python extract_features_locally.py --task a --skip-download    # Use existing data

Time estimate:
    - Task A: 20-40 minutes (100K training samples, default)
    - Task B: 20-40 minutes (100K training samples, default)
    - Task C: 20-40 minutes (100K training samples, default)
    - Full dataset: 2-4 hours per task (500K samples with --max-samples null)
"""

import sys
import argparse
from pathlib import Path
import time
from datetime import timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
from task_a.dataset import TaskADataset
from task_b.dataset import TaskBDataset
from task_c.dataset import TaskCDataset


def download_dataset(task: str, data_dir: Path):
    """
    Download dataset from HuggingFace and save as parquet.
    
    Args:
        task: Task letter ('a', 'b', 'c')
        data_dir: Directory to save parquet files
    """
    dataset_name = "DaniilOr/SemEval-2026-Task13"
    config_name = task.upper()  # 'A', 'B', or 'C'
    
    print(f"\n📥 Downloading dataset: {config_name}...")
    dataset = load_dataset(dataset_name, config_name)
    
    # Create directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet
    train_path = data_dir / 'train.parquet'
    val_path = data_dir / 'validation.parquet'
    
    print(f"  Saving train split ({len(dataset['train'])} samples)...")
    dataset['train'].to_pandas().to_parquet(train_path)
    
    print(f"  Saving validation split ({len(dataset['validation'])} samples)...")
    dataset['validation'].to_pandas().to_parquet(val_path)
    
    print(f"  ✅ Saved to {data_dir}")
    
    return train_path, val_path


def extract_task_features(task: str, skip_download: bool = False, max_samples: int = 100000):
    """
    Extract features for a specific task.
    
    Args:
        task: Task letter ('a', 'b', or 'c')
        skip_download: If True, assumes data already exists
        max_samples: Maximum number of samples to extract (default: 100000 for Colab)
    """
    print("\n" + "=" * 80)
    print(f"🚀 EXTRACTING FEATURES FOR TASK {task.upper()}")
    print("=" * 80)
    if max_samples:
        print(f"📊 Limiting to {max_samples:,} samples (training + validation)")
    else:
        print(f"📊 Processing all samples (full dataset)")
    print("=" * 80)
    
    start_time = time.time()
    
    # Setup paths
    data_dir = Path(f'data/task_{task}')
    cache_dir = data_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = data_dir / 'train.parquet'
    val_path = data_dir / 'validation.parquet'
    
    # Download data if needed
    if not skip_download or not train_path.exists():
        train_path, val_path = download_dataset(task, data_dir)
    else:
        print(f"\n📁 Using existing data in {data_dir}")
        if not train_path.exists() or not val_path.exists():
            print("  ⚠️  Data files not found! Downloading...")
            train_path, val_path = download_dataset(task, data_dir)
    
    # Load tokenizer
    print("\n🔧 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  ✅ Tokenizer loaded")
    
    # Select dataset class
    DatasetClass = {
        'a': TaskADataset,
        'b': TaskBDataset,
        'c': TaskCDataset
    }[task]
    
    # Extract TRAIN features
    print("\n" + "=" * 80)
    print("📊 EXTRACTING TRAINING SET FEATURES")
    print("=" * 80)
    
    train_start = time.time()
    train_ds = DatasetClass(
        data_path=str(train_path),
        tokenizer=tokenizer,
        max_length=512 if task in ['b', 'c'] else 256,
        cache_dir=str(cache_dir),
        use_cache=True,
        max_samples=max_samples if max_samples else None
    )
    train_time = time.time() - train_start
    
    print(f"\n✅ Train features extracted: {len(train_ds)} samples")
    print(f"   Time: {str(timedelta(seconds=int(train_time)))}")
    
    # Extract VALIDATION features
    print("\n" + "=" * 80)
    print("📊 EXTRACTING VALIDATION SET FEATURES")
    print("=" * 80)
    
    val_start = time.time()
    val_ds = DatasetClass(
        data_path=str(val_path),
        tokenizer=tokenizer,
        max_length=512 if task in ['b', 'c'] else 256,
        cache_dir=str(cache_dir),
        use_cache=True,
        max_samples=max_samples // 5 if max_samples else None  # 20% of training size
    )
    val_time = time.time() - val_start
    
    print(f"\n✅ Validation features extracted: {len(val_ds)} samples")
    print(f"   Time: {str(timedelta(seconds=int(val_time)))}")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ TASK {task.upper()} COMPLETE!")
    print("=" * 80)
    print(f"📁 Cache location: {cache_dir.absolute()}")
    print(f"⏱️  Total time: {str(timedelta(seconds=int(total_time)))}")
    print("\n📤 Next steps:")
    print(f"   1. Upload the entire '{cache_dir}' folder to Google Drive")
    print(f"   2. Place it at: MyDrive/colab_checkpoints/feature_cache/task_{task}/")
    print(f"   3. In Colab, download these files before training")
    print("=" * 80 + "\n")
    
    return cache_dir


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute features locally for faster Colab training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_features_locally.py --task a                    # Extract Task A (100k samples)
  python extract_features_locally.py --task all                  # Extract all tasks (100k each)
  python extract_features_locally.py --task a --max-samples null # Full dataset (~500k)
  python extract_features_locally.py --task a --skip-download    # Use existing data

After extraction, upload the cache folders to Google Drive:
  data/task_a/cache/ → MyDrive/colab_checkpoints/feature_cache/task_a/
  data/task_b/cache/ → MyDrive/colab_checkpoints/feature_cache/task_b/
  data/task_c/cache/ → MyDrive/colab_checkpoints/feature_cache/task_c/
        """
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['a', 'b', 'c', 'all'],
        default='a',
        help='Which task to extract features for (default: a)'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading data (assumes data already exists)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=lambda x: None if x.lower() == 'null' else int(x),
        default=100000,
        help='Maximum number of samples to extract (default: 100000 for Colab, use "null" for full dataset)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 LOCAL FEATURE EXTRACTION FOR SEMEVAL-2026 TASK 13")
    print("=" * 80)
    print(f"Task: {args.task.upper()}")
    print(f"Max samples: {f'{args.max_samples:,}' if args.max_samples else 'All (full dataset)'}")
    print(f"Skip download: {args.skip_download}")
    print("=" * 80)
    
    overall_start = time.time()
    
    if args.task == 'all':
        tasks = ['a', 'b', 'c']
        cache_dirs = []
        
        for task in tasks:
            try:
                cache_dir = extract_task_features(task, args.skip_download, args.max_samples)
                cache_dirs.append((task, cache_dir))
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user!")
                print(f"✅ Progress saved. You can resume by running the same command.")
                sys.exit(0)
            except Exception as e:
                print(f"\n❌ Error extracting Task {task.upper()}: {e}")
                print(f"   Continuing with next task...")
                continue
        
        # Final summary
        total_time = time.time() - overall_start
        print("\n" + "=" * 80)
        print("🎉 ALL TASKS COMPLETE!")
        print("=" * 80)
        print(f"⏱️  Total time: {str(timedelta(seconds=int(total_time)))}")
        print("\n📤 Upload these folders to Google Drive:")
        for task, cache_dir in cache_dirs:
            print(f"   {cache_dir.absolute()}")
            print(f"   → MyDrive/colab_checkpoints/feature_cache/task_{task}/")
        print("=" * 80)
        
    else:
        try:
            extract_task_features(args.task, args.skip_download, args.max_samples)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user!")
            print(f"✅ Progress saved. You can resume by running the same command.")
            sys.exit(0)


if __name__ == "__main__":
    main()
