"""
Task C: Hybrid Detection (4 classes: Human, Machine, Hybrid, Adversarial)

This module provides:
- TaskCModel: Staged classifier with hybrid and adversarial detection
- TaskCDataset: Dataset loader for Task C
- Training and inference scripts

Target: 83-86% macro F1
"""

from .model import TaskCModel

__all__ = ['TaskCModel']
