"""
Task B: Authorship Detection (11 classes: 1 Human + 10 LLM families)

This module provides:
- TaskBModel: Cascade classifier with meta-learning
- TaskBDataset: Dataset loader for Task B
- Training and inference scripts

Target: 75-78% macro F1
"""

from .model import TaskBModel

__all__ = ['TaskBModel']
