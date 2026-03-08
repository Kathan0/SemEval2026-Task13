"""
Task A: Binary Classification (Human vs AI-generated code)

This module provides:
- TaskAModel: Binary classifier with OOD detection
- TaskADataset: Dataset loader for Task A
- Training and inference scripts

Target: 87-90% macro F1
"""

from .model import TaskAModel, FocalLoss

__all__ = ['TaskAModel', 'FocalLoss']
