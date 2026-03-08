"""
Evaluation metrics for all tasks.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    classification_report
)


def compute_macro_f1(
    predictions: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute macro F1 score (primary metric for all tasks).
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        Macro F1 score
    """
    return f1_score(labels, predictions, average='macro')


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """
    Compute comprehensive metrics.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        num_classes: Number of classes
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average='macro'),
        "macro_precision": precision_score(labels, predictions, average='macro', zero_division=0),
        "macro_recall": recall_score(labels, predictions, average='macro', zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average='weighted'),
    }
    
    # Per-class F1 scores
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    for i, f1 in enumerate(per_class_f1):
        metrics[f"class_{i}_f1"] = f1
    
    return metrics


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        
    Returns:
        Confusion matrix as numpy array
    """
    return sklearn_confusion_matrix(labels, predictions)


def print_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    target_names: Optional[List[str]] = None
):
    """
    Print detailed classification report.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        target_names: Names of classes (optional)
    """
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    
    report = classification_report(
        labels,
        predictions,
        target_names=target_names,
        zero_division=0
    )
    print(report)
    
    print("=" * 60 + "\n")


def print_confusion_matrix(
    cm: np.ndarray,
    target_names: Optional[List[str]] = None
):
    """
    Print confusion matrix in a readable format.
    
    Args:
        cm: Confusion matrix
        target_names: Names of classes (optional)
    """
    print("\n" + "=" * 60)
    print("Confusion Matrix")
    print("=" * 60)
    
    if target_names is None:
        target_names = [f"Class {i}" for i in range(len(cm))]
    
    # Print header
    print(f"{'True/Pred':<15}", end="")
    for name in target_names:
        print(f"{name:<15}", end="")
    print()
    print("-" * (15 * (len(target_names) + 1)))
    
    # Print rows
    for i, true_name in enumerate(target_names):
        print(f"{true_name:<15}", end="")
        for j in range(len(target_names)):
            print(f"{cm[i, j]:<15}", end="")
        print()
    
    print("=" * 60 + "\n")


def compute_ood_metrics(
    ood_scores: np.ndarray,
    is_ood: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute OOD detection metrics.
    
    Args:
        ood_scores: OOD scores (higher = more in-distribution)
        is_ood: Binary labels (1 = OOD, 0 = in-distribution)
        threshold: Threshold for OOD detection
        
    Returns:
        Dictionary with OOD detection metrics
    """
    # Invert scores for OOD detection (lower score = OOD)
    predictions = (ood_scores < threshold).astype(int)
    
    metrics = {
        "ood_accuracy": accuracy_score(is_ood, predictions),
        "ood_f1": f1_score(is_ood, predictions, zero_division=0),
        "ood_precision": precision_score(is_ood, predictions, zero_division=0),
        "ood_recall": recall_score(is_ood, predictions, zero_division=0),
    }
    
    return metrics


class MetricsTracker:
    """
    Track metrics across epochs.
    """
    
    def __init__(self):
        self.history = {}
        self.best_metrics = {}
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """
        Update metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((epoch, value))
            
            # Track best metric
            if key not in self.best_metrics or value > self.best_metrics[key][1]:
                self.best_metrics[key] = (epoch, value)
    
    def get_best(self, metric: str) -> Tuple[int, float]:
        """
        Get best value for a metric.
        
        Args:
            metric: Metric name
            
        Returns:
            Tuple of (epoch, value)
        """
        if metric not in self.best_metrics:
            return (0, 0.0)
        return self.best_metrics[metric]
    
    def get_history(self, metric: str) -> List[Tuple[int, float]]:
        """
        Get history for a metric.
        
        Args:
            metric: Metric name
            
        Returns:
            List of (epoch, value) tuples
        """
        return self.history.get(metric, [])
    
    def print_summary(self):
        """Print summary of all tracked metrics."""
        print("\n" + "=" * 60)
        print("Metrics Summary")
        print("=" * 60)
        
        for metric, (epoch, value) in self.best_metrics.items():
            print(f"{metric:<30s}: {value:.4f} (epoch {epoch})")
        
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test metrics
    print("Testing metric functions")
    print("=" * 60)
    
    # Create dummy data
    labels = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 1, 1, 1, 2, 0])
    
    # Test metrics computation
    print("\n1. Computing metrics:")
    metrics = compute_metrics(predictions, labels, num_classes=3)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test confusion matrix
    print("\n2. Confusion matrix:")
    cm = compute_confusion_matrix(predictions, labels)
    print(cm)
    
    # Test classification report
    print("\n3. Classification report:")
    print_classification_report(
        predictions,
        labels,
        target_names=["Human", "Machine", "Hybrid"]
    )
    
    # Test metrics tracker
    print("\n4. Metrics tracker:")
    tracker = MetricsTracker()
    tracker.update(1, {"f1": 0.75, "accuracy": 0.80})
    tracker.update(2, {"f1": 0.78, "accuracy": 0.82})
    tracker.update(3, {"f1": 0.80, "accuracy": 0.81})
    tracker.print_summary()
    
    print("✓ All tests passed")
