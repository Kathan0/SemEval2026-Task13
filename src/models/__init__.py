"""
Model architectures for AI-generated code detection.

This module provides:
- Base hybrid classifier combining StarCoder2-3B + handcrafted features
- Task-specific models for binary, authorship, and hybrid detection
- Multi-scale attention pooling
- Feature fusion networks
"""

from .base_model import (
    HybridCodeClassifier,
    MultiScaleAttentionPooling,
    FeatureFusionNetwork
)

from .task_a_model import TaskAModel, FocalLoss
from .task_b_model import TaskBModel
from .task_c_model import TaskCModel

__all__ = [
    'HybridCodeClassifier',
    'MultiScaleAttentionPooling',
    'FeatureFusionNetwork',
    'TaskAModel',
    'TaskBModel',
    'TaskCModel',
    'FocalLoss'
]
