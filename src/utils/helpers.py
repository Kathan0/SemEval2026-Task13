"""
Common helper functions for all tasks.
"""

import os
import random
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[OK] Random seed set to {seed}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        
    Returns:
        torch.device object
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("[OK] Using CPU")
    
    return device


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"[OK] Loaded config from {config_path}")
    return config


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    save_dir: str,
    is_best: bool = False,
    checkpoint_dir: Optional[str] = None,
    filename: str = "checkpoint.pt"
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        metrics: Dictionary of metric values
        save_dir: Directory to save checkpoint (preferred)
        is_best: If True, also save as 'best_model.pt'
        checkpoint_dir: Alias for save_dir (for backward compatibility)
        filename: Checkpoint filename
    """
    # Handle both save_dir and checkpoint_dir parameters
    if checkpoint_dir is not None:
        save_dir = checkpoint_dir
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save latest checkpoint
    checkpoint_path = save_dir / filename
    torch.save(checkpoint, checkpoint_path)
    print(f"[OK] Saved checkpoint to {checkpoint_path}")
    
    # Save best model if this is the best so far
    if is_best:
        best_path = save_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"[OK] Saved best model to {best_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state (optional)
        scheduler: Scheduler to load state (optional)
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"[OK] Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Metrics: {checkpoint.get('metrics', {})}")
    
    return checkpoint


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with total and trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params
    }


def print_model_info(model: torch.nn.Module):
    """
    Print model information.
    
    Args:
        model: PyTorch model
    """
    params = count_parameters(model)
    
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters:    {params['frozen']:,}")
    print(f"Trainable %:          {params['trainable']/params['total']*100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test helpers
    print("Testing helper functions")
    print("=" * 60)
    
    # Test seed setting
    set_seed(42)
    
    # Test device selection
    device = get_device()
    
    # Test time formatting
    print(f"\n5 seconds: {format_time(5)}")
    print(f"90 seconds: {format_time(90)}")
    print(f"3700 seconds: {format_time(3700)}")
    
    print("\n[OK] All tests passed")
