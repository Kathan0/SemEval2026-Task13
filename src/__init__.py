"""
MyProject: Enhanced AI-Generated Code Detection

A comprehensive solution for SemEval-2026 Task 13 featuring:
- 110-145 handcrafted features (AST, LLM patterns, perplexity, stylometric)
- StarCoder2-3B backbone with multi-scale extraction
- Hybrid semantic-stylometric architecture
- Task-specific strategies (binary, cascade, staged learning)

Modules:
- features: Feature extraction (AST, patterns, perplexity, stylometric)
- models: Hybrid model architectures
- data: Dataset loading and augmentation (to be implemented)
- training: Training loops and losses (to be implemented)
- utils: Metrics and utilities (to be implemented)
"""

__version__ = "0.1.0"
__author__ = "Enhanced SemEval-2026 Task 13 Solution"

from . import features
from . import models

__all__ = ['features', 'models']
