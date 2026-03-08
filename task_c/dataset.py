"""
Dataset loader for Task C: Hybrid Detection (Staged Learning)

Loads data, extracts features, and prepares batches for staged training.
Supports both binary (Human vs Hybrid) and fine-grained (4-class) classification.
Includes section-wise analysis for 8 sections.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Literal, List
import re

from src.features import UnifiedFeatureExtractor


class TaskCDataset(Dataset):
    """
    Dataset for Task C (Hybrid Detection with Staged Learning).
    
    Staged strategy:
    1. Binary classifier: Pure (Human/AI) vs Hybrid
    2. Fine-grained classifier: 4 classes (human, machine, machine_author, machine_humanized)
    
    Features:
    - 110 handcrafted features (global)
    - Section-wise features for 8 sections (880 dims)
    - Consistency features across sections
    """
    
    # Define 8 code sections for analysis
    SECTIONS = [
        'imports',
        'constants',
        'classes',
        'functions',
        'main_logic',
        'error_handling',
        'comments',
        'overall'
    ]
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        feature_extractor: Optional[UnifiedFeatureExtractor] = None,
        max_length: int = 512,
        mode: Literal['binary', 'fine'] = 'fine',
        use_sections: bool = True,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize Task C dataset.
        
        Args:
            data_path: Path to parquet file
            tokenizer: HuggingFace tokenizer
            feature_extractor: UnifiedFeatureExtractor instance
            max_length: Maximum sequence length
            mode: 'binary' (Pure vs Hybrid) or 'fine' (4-class)
            use_sections: Whether to extract section-wise features
            use_cache: Whether to cache features
            cache_dir: Directory for feature cache
            max_samples: Limit number of samples (for Colab/testing)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.use_sections = use_sections
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
            cache_name = f"{self.data_path.stem}_task_c_features"
            if use_sections:
                cache_name += "_sections"
            self.cache_file = self.cache_dir / f"{cache_name}{cache_suffix}.pt"
        
        # Extract features
        self._extract_features()
        
        # Create label mappings
        self._create_label_mappings()
        
        print(f"✓ Dataset ready: {len(self)} samples ({self.mode} mode)")
    
    def _split_into_sections(self, code: str) -> Dict[str, str]:
        """
        Split code into 8 sections for section-wise analysis.
        
        Returns:
            Dictionary mapping section name to code text
        """
        sections = {s: "" for s in self.SECTIONS}
        
        lines = code.split('\n')
        
        # Extract imports (top of file)
        import_lines = []
        for line in lines[:50]:  # Check first 50 lines
            if re.match(r'^\s*(import|from)\s+', line):
                import_lines.append(line)
        sections['imports'] = '\n'.join(import_lines)
        
        # Extract constants (UPPERCASE variables)
        const_pattern = r'^\s*[A-Z_][A-Z0-9_]*\s*='
        const_lines = [line for line in lines if re.match(const_pattern, line)]
        sections['constants'] = '\n'.join(const_lines)
        
        # Extract classes
        class_blocks = re.findall(r'class\s+\w+.*?(?=\nclass\s+|\ndef\s+|\Z)', code, re.DOTALL)
        sections['classes'] = '\n\n'.join(class_blocks)
        
        # Extract functions
        func_blocks = re.findall(r'def\s+\w+.*?(?=\ndef\s+|\nclass\s+|\Z)', code, re.DOTALL)
        sections['functions'] = '\n\n'.join(func_blocks[:5])  # First 5 functions
        
        # Extract main logic (if __name__ == '__main__')
        main_match = re.search(r'if\s+__name__\s*==\s*["\']__main__["\'].*', code, re.DOTALL)
        if main_match:
            sections['main_logic'] = main_match.group(0)
        
        # Extract error handling (try/except blocks)
        error_blocks = re.findall(r'try:.*?except.*?(?=\n\S|\Z)', code, re.DOTALL)
        sections['error_handling'] = '\n\n'.join(error_blocks)
        
        # Extract comments
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        docstring_blocks = re.findall(r'""".*?"""|\'\'\'.*?\'\'\'', code, re.DOTALL)
        sections['comments'] = '\n'.join(comment_lines) + '\n\n' + '\n\n'.join(docstring_blocks)
        
        # Overall (full code)
        sections['overall'] = code
        
        return sections
    
    def _extract_features(self):
        """Extract and cache handcrafted features."""
        cache_data = {}
        
        if self.use_cache and self.cache_file.exists():
            print(f"Loading cached features from {self.cache_file}...")
            cache_data = torch.load(self.cache_file, weights_only=False)
            self.features = cache_data['features']
            if self.use_sections:
                self.section_features = cache_data['section_features']
                self.consistency_features = cache_data['consistency_features']
            print(f"✓ Loaded cached features")
        else:
            print("Extracting features...")
            features = []
            section_features_list = []
            consistency_features_list = []
            
            for idx, row in self.df.iterrows():
                code = row['code']
                language = row.get('language', 'python')
                
                # Global features
                feat = self.feature_extractor.extract(code, language)
                features.append(feat)
                
                # Section-wise features
                if self.use_sections:
                    sections = self._split_into_sections(code)
                    section_feats = []
                    
                    for section_name in self.SECTIONS:
                        section_code = sections[section_name]
                        if section_code.strip():
                            s_feat = self.feature_extractor.extract(section_code, language)
                        else:
                            s_feat = np.zeros(110)  # Empty section
                        section_feats.append(s_feat)
                    
                    section_feats = np.array(section_feats)  # (8, 110)
                    section_features_list.append(section_feats.flatten())  # 880 dims
                    
                    # Consistency features (variance across sections)
                    consistency = np.var(section_feats, axis=0)  # (110,)
                    consistency_features_list.append(consistency)
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(self.df)} samples")
            
            self.features = np.array(features)
            
            if self.use_sections:
                self.section_features = np.array(section_features_list)
                self.consistency_features = np.array(consistency_features_list)
            
            # Cache features
            if self.use_cache:
                cache_data = {'features': self.features}
                if self.use_sections:
                    cache_data['section_features'] = self.section_features
                    cache_data['consistency_features'] = self.consistency_features
                torch.save(cache_data, self.cache_file)
                print(f"✓ Cached features to {self.cache_file}")
    
    def _create_label_mappings(self):
        """Create label mappings for binary and fine modes."""
        # Original labels: 0=human, 1=machine, 2=hybrid, 3=adversarial
        # Binary: 0=pure (human/machine), 1=hybrid (hybrid/adversarial)
        
        self.label_to_name = {
            0: 'human',
            1: 'machine',
            2: 'hybrid',
            3: 'adversarial'
        }
        
        if self.mode == 'binary':
            # For binary mode: 0,1 -> 0 (pure), 2,3 -> 1 (hybrid)
            self.binary_labels = (self.df['label'] >= 2).astype(int).values
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - input_ids: Tokenized code
                - attention_mask: Attention mask
                - features: Global handcrafted features (110 dims)
                - section_features: Section-wise features (880 dims) [if use_sections]
                - consistency_features: Consistency features (110 dims) [if use_sections]
                - label: Binary (0/1) or Fine-grained (0-3)
                - original_label: Original label (0-3)
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
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'features': features,
            'label': torch.tensor(label, dtype=torch.long),
            'original_label': torch.tensor(original_label, dtype=torch.long)
        }
        
        # Add section features
        if self.use_sections:
            result['section_features'] = torch.tensor(
                self.section_features[idx], dtype=torch.float32
            )
            result['consistency_features'] = torch.tensor(
                self.consistency_features[idx], dtype=torch.float32
            )
        
        return result
    
    def get_class_distribution(self) -> Dict:
        """Get class distribution."""
        if self.mode == 'binary':
            labels = self.binary_labels
            unique, counts = np.unique(labels, return_counts=True)
            return {
                'pure': int(counts[unique == 0][0]) if 0 in unique else 0,
                'hybrid': int(counts[unique == 1][0]) if 1 in unique else 0
            }
        else:
            labels = self.df['label'].values
            unique, counts = np.unique(labels, return_counts=True)
            return {self.label_to_name[int(l)]: int(c) for l, c in zip(unique, counts)}
    
    def set_mode(self, mode: Literal['binary', 'fine']):
        """Switch between binary and fine-grained mode."""
        self.mode = mode
        print(f"✓ Switched to {mode} mode")
    
    def print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "=" * 60)
        print(f"Task C Dataset Statistics ({self.mode} mode)")
        print("=" * 60)
        print(f"Total samples: {len(self)}")
        
        dist = self.get_class_distribution()
        print(f"\nClass distribution:")
        for name, count in dist.items():
            print(f"  {name:18s}: {count:6d} ({count/len(self)*100:.1f}%)")
        
        print(f"\nFeature dimensions:")
        print(f"  Global: {self.features.shape[1]}")
        if self.use_sections:
            print(f"  Section-wise: {self.section_features.shape[1]} (8 sections × 110)")
            print(f"  Consistency: {self.consistency_features.shape[1]}")
            print(f"  Total: {self.features.shape[1] + self.section_features.shape[1] + self.consistency_features.shape[1]}")
        
        print(f"\nMax sequence length: {self.max_length}")
        print("=" * 60 + "\n")


def create_task_c_dataloader(
    dataset: TaskCDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for Task C.
    
    Args:
        dataset: TaskCDataset instance
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
    print("Testing Task C Dataset")
    print("=" * 60)
    
    print("\nNote: This is a test script. Replace data_path with actual data.")
    print("Example usage:")
    print("""
    from transformers import AutoTokenizer
    from task_c.dataset import TaskCDataset, create_task_c_dataloader
    
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
    
    # For binary training (pure vs hybrid)
    dataset_binary = TaskCDataset(
        data_path="data/task_c/train.parquet",
        tokenizer=tokenizer,
        mode='binary',
        use_sections=True
    )
    dataset_binary.print_statistics()
    
    # For fine-grained training (4 classes)
    dataset_fine = TaskCDataset(
        data_path="data/task_c/train.parquet",
        tokenizer=tokenizer,
        mode='fine',
        use_sections=True
    )
    dataset_fine.print_statistics()
    
    # Create dataloader
    dataloader = create_task_c_dataloader(
        dataset_fine,
        batch_size=16,
        shuffle=True
    )
    
    # Iterate over batches
    for batch in dataloader:
        print(f"Input IDs: {batch['input_ids'].shape}")
        print(f"Features: {batch['features'].shape}")
        print(f"Section features: {batch['section_features'].shape}")
        print(f"Consistency features: {batch['consistency_features'].shape}")
        print(f"Labels: {batch['label'].shape}")
        break
    """)
