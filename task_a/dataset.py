"""
Dataset loader for Task A: Binary Classification

Loads data, extracts features, and prepares batches for training/inference.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.features import UnifiedFeatureExtractor


class TaskADataset(Dataset):
    """
    Dataset for Task A (Binary Classification: Human vs AI).
    
    Features:
    - 110 handcrafted features (33 AST + 57 patterns + 8 perplexity + 12 stylometric)
    - StarCoder2 tokenization
    - Feature caching for efficiency
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        feature_extractor: Optional[UnifiedFeatureExtractor] = None,
        max_length: int = 512,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize Task A dataset.
        
        Args:
            data_path: Path to parquet file
            tokenizer: HuggingFace tokenizer
            feature_extractor: UnifiedFeatureExtractor instance
            max_length: Maximum sequence length
            use_cache: Whether to cache features
            cache_dir: Directory for feature cache
            max_samples: Limit number of samples (for faster testing)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_cache = use_cache
        self.max_samples = max_samples
        
        # Load data
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        
        # Limit samples if specified
        if max_samples is not None and len(self.df) > max_samples:
            print(f"  Limiting dataset from {len(self.df)} to {max_samples} samples")
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)
        
        print(f"[OK] Loaded {len(self.df)} samples")
        
        # Initialize feature extractor
        if feature_extractor is None:
            print("Initializing feature extractor...")
            self.feature_extractor = UnifiedFeatureExtractor(use_perplexity=False)
        else:
            self.feature_extractor = feature_extractor
        
        # Setup cache
        if use_cache:
            if cache_dir is None:
                cache_dir = self.data_path.parent / "cache"
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Include max_samples in cache filename to avoid conflicts
            cache_suffix = f"_{max_samples}" if max_samples else ""
            self.cache_file = self.cache_dir / f"{self.data_path.stem}_task_a_features{cache_suffix}.pt"
        
        # Extract features
        self._extract_features()
        
        print(f"[OK] Dataset ready: {len(self)} samples")
    
    def _extract_features(self):
        """Extract and cache handcrafted features with incremental saving."""
        # Check for complete cache
        if self.use_cache and self.cache_file.exists():
            print(f"Loading cached features from {self.cache_file}...")
            cache = torch.load(self.cache_file, weights_only=False)
            if cache.get('complete', False) and len(cache['features']) == len(self.df):
                self.features = cache['features']
                print(f"[OK] Loaded complete cached features")
                return
            else:
                print("  Partial cache found, will resume...")
        
        # Check for partial/temporary cache
        temp_cache_file = self.cache_file.with_suffix('.partial.pt') if self.use_cache else None
        start_idx = 0
        features = []
        
        if self.use_cache and temp_cache_file and temp_cache_file.exists():
            print(f"Resuming from partial cache: {temp_cache_file}")
            try:
                partial_cache = torch.load(temp_cache_file, weights_only=False)
                features = partial_cache['features'].tolist()
                start_idx = len(features)
                print(f"  Resuming from sample {start_idx}/{len(self.df)}")
            except Exception as e:
                print(f"  Warning: Could not load partial cache: {e}")
                features = []
                start_idx = 0
        else:
            print("Extracting features...")
        
        # Extract features with error handling and incremental saving
        save_interval = 10000  # Save every 10K samples
        
        for idx in range(start_idx, len(self.df)):
            try:
                row = self.df.iloc[idx]
                code = row['code']
                language = row.get('language', 'python')
                
                feat = self.feature_extractor.extract(code, language)
                features.append(feat)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(self.df)} samples")
                
                # Incremental save every N samples
                if self.use_cache and (idx + 1) % save_interval == 0:
                    print(f"  💾 Saving progress at {idx + 1} samples...")
                    torch.save({
                        'features': np.array(features),
                        'complete': False,
                        'processed': idx + 1
                    }, temp_cache_file)
                    
            except Exception as e:
                print(f"  ⚠️ Error processing sample {idx}: {e}")
                print(f"  Skipping sample and using zero features")
                # Use zero vector for failed samples
                feat_dim = self.feature_extractor.get_feature_count()
                features.append(np.zeros(feat_dim))
        
        self.features = np.array(features)
        
        # Check for NaN/Inf in features
        nan_mask = np.isnan(self.features) | np.isinf(self.features)
        if nan_mask.any():
            num_nan = nan_mask.sum()
            print(f"  ⚠️  Found {num_nan} NaN/Inf values in features - replacing with 0")
            self.features[nan_mask] = 0
        
        # Save final complete cache
        if self.use_cache:
            print(f"  💾 Saving final cache...")
            torch.save({
                'features': self.features,
                'complete': True,
                'processed': len(self.df)
            }, self.cache_file)
            # Remove temporary cache
            if temp_cache_file and temp_cache_file.exists():
                temp_cache_file.unlink()
            print(f"[OK] Cached features to {self.cache_file}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - input_ids: Tokenized code
                - attention_mask: Attention mask
                - features: Handcrafted features (110 dims)
                - label: Binary label (0=human, 1=AI)
        """
        row = self.df.iloc[idx]
        
        # Get code and label
        code = row['code']
        label = row['label']  # 0=human, 1=AI
        
        # Tokenize
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get features
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'features': features,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution."""
        labels = self.df['label'].values
        unique, counts = np.unique(labels, return_counts=True)
        
        return {
            'human': int(counts[unique == 0][0]) if 0 in unique else 0,
            'ai': int(counts[unique == 1][0]) if 1 in unique else 0
        }
    
    def print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "=" * 60)
        print("Task A Dataset Statistics")
        print("=" * 60)
        print(f"Total samples: {len(self)}")
        
        dist = self.get_class_distribution()
        print(f"\nClass distribution:")
        print(f"  Human: {dist['human']} ({dist['human']/len(self)*100:.1f}%)")
        print(f"  AI:    {dist['ai']} ({dist['ai']/len(self)*100:.1f}%)")
        
        print(f"\nFeature dimensions: {self.features.shape[1]}")
        print(f"Max sequence length: {self.max_length}")
        print("=" * 60 + "\n")


def create_task_a_dataloader(
    dataset: TaskADataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 2
) -> DataLoader:
    """
    Create DataLoader for Task A.
    
    Args:
        dataset: TaskADataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        persistent_workers: Keep workers alive between epochs (requires num_workers > 0)
        prefetch_factor: Number of batches to prefetch per worker
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )


if __name__ == "__main__":
    # Test dataset
    from transformers import AutoTokenizer
    
    print("Testing Task A Dataset")
    print("=" * 60)
    
    # Note: Replace with actual data path
    print("\nNote: This is a test script. Replace data_path with actual data.")
    print("Example usage:")
    print("""
    from transformers import AutoTokenizer
    from task_a.dataset import TaskADataset, create_task_a_dataloader
    
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
    dataset = TaskADataset(
        data_path="data/task_a/train.parquet",        tokenizer=tokenizer,
        max_length=512
    )
    
    dataset.print_statistics()
    
    dataloader = create_task_a_dataloader(
        dataset,
        batch_size=16,
        shuffle=True
    )
    
    # Iterate over batches
    for batch in dataloader:
        print(batch['input_ids'].shape)
        print(batch['features'].shape)
        print(batch['label'].shape)
        break
    """)
