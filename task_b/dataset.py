"""
Dataset loader for Task B: Authorship Attribution (Cascade)

Loads data, extracts features, and prepares batches for cascade training.
Supports both binary (Human vs AI) and family (11-class) classification.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Literal, Tuple

from src.features import UnifiedFeatureExtractor


class TaskBDataset(Dataset):
    """
    Dataset for Task B (Authorship Attribution with Cascade).
    
    Cascade strategy:
    1. Binary classifier: Human (0) vs AI (1-10)
    2. Family classifier: 10 AI families
    
    Features:
    - 110 handcrafted features
    - Meta-learning support (episode-based sampling)
    - Prototypical network support
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        feature_extractor: Optional[UnifiedFeatureExtractor] = None,
        max_length: int = 512,
        mode: Literal['binary', 'family'] = 'family',
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize Task B dataset.
        
        Args:
            data_path: Path to parquet file
            tokenizer: HuggingFace tokenizer
            feature_extractor: UnifiedFeatureExtractor instance
            max_length: Maximum sequence length
            mode: 'binary' (Human vs AI) or 'family' (11-class)
            use_cache: Whether to cache features
            cache_dir: Directory for feature cache
            max_samples: Limit number of samples (for Colab/testing)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.use_cache = use_cache
        self.max_samples = max_samples
        
        # Load data
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        print(f"✓ Loaded {len(self.df)} samples")
        
        # Limit samples if specified
        if max_samples is not None and len(self.df) > max_samples:
            print(f"  Limiting dataset from {len(self.df)} to {max_samples} samples")
            self.df = self.df.iloc[:max_samples].reset_index(drop=True)
        
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
            self.cache_dir.mkdir(exist_ok=True)
            # Include max_samples in cache filename to avoid conflicts
            cache_suffix = f"_{max_samples}" if max_samples else ""
            self.cache_file = self.cache_dir / f"{self.data_path.stem}_task_b_features{cache_suffix}.pt"
        
        # Extract features
        self._extract_features()
        
        # Create label mappings
        self._create_label_mappings()
        
        print(f"✓ Dataset ready: {len(self)} samples ({self.mode} mode)")
    
    def _extract_features(self):
        """Extract and cache handcrafted features."""
        if self.use_cache and self.cache_file.exists():
            print(f"Loading cached features from {self.cache_file}...")
            cache = torch.load(self.cache_file, weights_only=False)
            self.features = cache['features']
            print(f"✓ Loaded cached features")
        else:
            print("Extracting features...")
            features = []
            
            for idx, row in self.df.iterrows():
                code = row['code']
                language = row.get('language', 'python')
                
                feat = self.feature_extractor.extract(code, language)
                features.append(feat)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(self.df)} samples")
            
            self.features = np.array(features)
            
            # Cache features
            if self.use_cache:
                torch.save({'features': self.features}, self.cache_file)
                print(f"✓ Cached features to {self.cache_file}")
    
    def _create_label_mappings(self):
        """Create label mappings for binary and family modes."""
        # Original labels: 0=human, 1=deepseek, 2=qwen, 3=01-ai, 4=bigcode, 
        # 5=gemma, 6=phi, 7=meta-llama, 8=ibm-granite, 9=mistral, 10=openai
        # Binary: 0=human, 1=AI
        # Family: 0=human, 1-10=AI families (unchanged)
        
        self.label_to_name = {
            0: 'human',
            1: 'deepseek',
            2: 'qwen',
            3: '01-ai',
            4: 'bigcode',
            5: 'gemma',
            6: 'phi',
            7: 'meta-llama',
            8: 'ibm-granite',
            9: 'mistral',
            10: 'openai'
        }
        
        if self.mode == 'binary':
            # For binary mode, samples with label >= 1 are AI (label=1)
            self.binary_labels = (self.df['label'] > 0).astype(int).values
    
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
                - label: Binary (0/1) or Family (0-10)
                - original_label: Original label (0-10)
        """
        row = self.df.iloc[idx]
        
        # Get code and label
        code = row['code']
        original_label = row['label']
        
        if self.mode == 'binary':
            label = self.binary_labels[idx]
        else:
            label = original_label
        
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
            'label': torch.tensor(label, dtype=torch.long),
            'original_label': torch.tensor(original_label, dtype=torch.long)
        }
    
    def get_class_distribution(self) -> Dict:
        """Get class distribution."""
        if self.mode == 'binary':
            labels = self.binary_labels
            unique, counts = np.unique(labels, return_counts=True)
            return {
                'human': int(counts[unique == 0][0]) if 0 in unique else 0,
                'ai': int(counts[unique == 1][0]) if 1 in unique else 0
            }
        else:
            labels = self.df['label'].values
            unique, counts = np.unique(labels, return_counts=True)
            return {self.label_to_name[int(l)]: int(c) for l, c in zip(unique, counts)}
    
    def set_mode(self, mode: Literal['binary', 'family']):
        """Switch between binary and family mode."""
        self.mode = mode
        print(f"✓ Switched to {mode} mode")
    
    def get_support_query_split(
        self,
        k_shot: int = 5,
        n_query: int = 15
    ) -> Tuple[Dataset, Dataset]:
        """
        Create support/query split for meta-learning.
        
        Args:
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            
        Returns:
            (support_dataset, query_dataset)
        """
        # For prototypical networks
        support_indices = []
        query_indices = []
        
        for label in range(11):  # 0-10
            class_indices = np.where(self.df['label'] == label)[0]
            
            if len(class_indices) < k_shot + n_query:
                print(f"Warning: Class {label} has only {len(class_indices)} samples")
                continue
            
            np.random.shuffle(class_indices)
            support_indices.extend(class_indices[:k_shot])
            query_indices.extend(class_indices[k_shot:k_shot+n_query])
        
        # Create subset datasets
        support_dataset = torch.utils.data.Subset(self, support_indices)
        query_dataset = torch.utils.data.Subset(self, query_indices)
        
        return support_dataset, query_dataset
    
    def print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "=" * 60)
        print(f"Task B Dataset Statistics ({self.mode} mode)")
        print("=" * 60)
        print(f"Total samples: {len(self)}")
        
        dist = self.get_class_distribution()
        print(f"\nClass distribution:")
        for name, count in dist.items():
            print(f"  {name:12s}: {count:6d} ({count/len(self)*100:.1f}%)")
        
        print(f"\nFeature dimensions: {self.features.shape[1]}")
        print(f"Max sequence length: {self.max_length}")
        print("=" * 60 + "\n")


def create_task_b_dataloader(
    dataset: TaskBDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for Task B.
    
    Args:
        dataset: TaskBDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    # Test dataset
    print("Testing Task B Dataset")
    print("=" * 60)
    
    print("\nNote: This is a test script. Replace data_path with actual data.")
    print("Example usage:")
    print("""
    from transformers import AutoTokenizer
    from task_b.dataset import TaskBDataset, create_task_b_dataloader
    
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
    
    # For binary training
    dataset_binary = TaskBDataset(
        data_path="data/task_b/train.parquet",
        tokenizer=tokenizer,
        mode='binary'
    )
    dataset_binary.print_statistics()
    
    # For family training
    dataset_family = TaskBDataset(
        data_path="data/task_b/train.parquet",
        tokenizer=tokenizer,
        mode='family'
    )
    dataset_family.print_statistics()
    
    # Meta-learning support/query split
    support, query = dataset_family.get_support_query_split(k_shot=5, n_query=15)
    print(f"Support set: {len(support)} samples")
    print(f"Query set: {len(query)} samples")
    """)
