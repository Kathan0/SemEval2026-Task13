"""
Task A Model: Binary Classification (Human vs AI)

Simplest task: classify code as human-written or AI-generated.

Key features:
- Binary output (2 classes)
- OOD (Out-of-Distribution) detection head
- Focal loss for class imbalance
- Supervised contrastive learning
- 110 handcrafted features

Target performance: 87-90% macro F1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from src.models.base_model import HybridCodeClassifier


class TaskAModel(HybridCodeClassifier):
    """
    Binary classifier for Task A with OOD detection.
    
    Extends base hybrid classifier with:
    - Binary classification head
    - OOD detection scoring
    - Confidence calibration
    """
    
    def __init__(
        self,
        model_name: str = "bigcode/starcoder2-3b",
        handcrafted_dim: int = 110,
        layer_indices: list = [6, 9, 12],
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_8bit: bool = True,
        freeze_backbone: bool = False,
        freeze_layers: int = 0,
        ood_threshold: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Task A binary classifier.
        
        Args:
            model_name: Backbone model name
            handcrafted_dim: Dimension of handcrafted features (110)
            layer_indices: Layers to extract from backbone
            hidden_dim: Hidden dimension for fusion
            dropout: Dropout rate
            use_8bit: Use 8-bit quantization (GPU only)
            freeze_backbone: Whether to freeze backbone
            freeze_layers: Number of layers to freeze
            ood_threshold: Threshold for OOD detection
            device: Device for computation (cuda/cpu)
        """
        # Initialize base model with 2 classes
        super().__init__(
            model_name=model_name,
            num_classes=2,
            handcrafted_dim=handcrafted_dim,
            layer_indices=layer_indices,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_8bit=use_8bit and device == "cuda",
            freeze_backbone=freeze_backbone,
            freeze_layers=freeze_layers
        )
        
        self.device = device
        self.ood_threshold = ood_threshold
        
        # OOD detection head (predicts if sample is in-distribution)
        self.ood_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temperature scaling for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Move to device (backbone already on device if using 8-bit quantization)
        if device == "cuda" and torch.cuda.is_available():
            # When using 8-bit, backbone is already on device via device_map="auto"
            # This moves all other components (pooler, fusion, classifier, OOD head, etc.)
            if not use_8bit:
                self.to(device)
            else:
                # Move non-backbone components to device
                self.ood_head = self.ood_head.to(device)
                if hasattr(self, 'pooler'):
                    self.pooler = self.pooler.to(device)
                if hasattr(self, 'fusion'):
                    self.fusion = self.fusion.to(device)
                if hasattr(self, 'classifier'):
                    self.classifier = self.classifier.to(device)
                self.temperature = nn.Parameter(self.temperature.to(device))
        
        print(f"[OK] Task A Model initialized")
        print(f"  Classes: 2 (Human vs AI)")
        print(f"  Features: {handcrafted_dim}")
        print(f"  Device: {device}")
        print(f"  8-bit quantization: {use_8bit and device == 'cuda'}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with OOD detection.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            handcrafted_features: (batch, 110)
            labels: Optional binary labels {0: Human, 1: AI}
            return_embeddings: Whether to return feature embeddings
            
        Returns:
            Dictionary with:
                - logits: (batch, 2) classification logits
                - loss: Optional total loss
                - ood_scores: (batch,) OOD detection scores
                - probs: (batch, 2) calibrated probabilities
                - embeddings: Optional (batch, hidden_dim)
        """
        # Get base model outputs
        logits, base_loss, fused_features = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            handcrafted_features=handcrafted_features,
            labels=labels,
            return_hidden=True
        )
        
        # OOD detection score (higher = more in-distribution)
        ood_scores = self.ood_head(fused_features).squeeze(-1)
        
        # Temperature-scaled probabilities for calibration
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Prepare outputs
        outputs = {
            "logits": logits,
            "ood_scores": ood_scores,
            "probs": probs,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Classification loss
            cls_loss = base_loss
            
            # OOD loss (all training samples should be in-distribution)
            ood_target = torch.ones_like(ood_scores)
            ood_loss = F.binary_cross_entropy(ood_scores, ood_target)
            
            # Combined loss
            total_loss = cls_loss + 0.1 * ood_loss
            outputs["loss"] = total_loss
            outputs["cls_loss"] = cls_loss
            outputs["ood_loss"] = ood_loss
        
        if return_embeddings:
            outputs["embeddings"] = fused_features
        
        return outputs
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted_features: torch.Tensor,
        return_confidence: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            handcrafted_features: (batch, 110)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with:
                - predictions: (batch,) predicted labels {0: Human, 1: AI}
                - confidence: (batch,) prediction confidence [0-1]
                - ood_flags: (batch,) OOD detection flags (True = OOD)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                handcrafted_features=handcrafted_features
            )
            
            probs = outputs["probs"]
            ood_scores = outputs["ood_scores"]
            
            # Predictions
            predictions = torch.argmax(probs, dim=-1)
            
            # Confidence (max probability)
            confidence = torch.max(probs, dim=-1)[0]
            
            # Adjust confidence based on OOD scores
            adjusted_confidence = confidence * ood_scores
            
            # OOD flags (samples with low OOD scores)
            ood_flags = ood_scores < self.ood_threshold
            
            results = {
                "predictions": predictions,
                "ood_flags": ood_flags,
            }
            
            if return_confidence:
                results["confidence"] = adjusted_confidence
                results["raw_confidence"] = confidence
                results["ood_scores"] = ood_scores
            
            return results
    
    def get_prediction_labels(self, predictions: torch.Tensor) -> list:
        """Convert numeric predictions to labels."""
        label_map = {0: "human", 1: "machine"}
        return [label_map[p.item()] for p in predictions]


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focuses training on hard examples by down-weighting easy examples.
    """
    
    def __init__(self, alpha=0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        Args:
            alpha: Weighting factor. Can be:
                   - float: Single weight for binary classification
                   - list/tuple: Per-class weights [weight_class_0, weight_class_1, ...]
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, num_classes) logits
            targets: (batch,) class labels
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Per-class weights
                alpha_t = self.alpha.to(inputs.device)[targets]
                focal_loss = alpha_t * focal_loss
            elif self.alpha >= 0:
                # Binary weighting (original behavior)
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                focal_loss = alpha_t * focal_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    # Test Task A model
    print("Testing Task A Model")
    print("=" * 60)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create model
    print("\nInitializing model...")
    model = TaskAModel(
        model_name="bigcode/starcoder2-3b",
        handcrafted_dim=110,
        use_8bit=True,
        device=device
    )
    
    # Create dummy data
    batch_size = 2
    seq_len = 128
    
    print("\nCreating dummy inputs...")
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    handcrafted_features = torch.randn(batch_size, 110)
    labels = torch.tensor([0, 1])  # Human, AI
    
    # Move to device
    if device == "cuda" and not model.backbone.device_map:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        handcrafted_features = handcrafted_features.to(device)
        labels = labels.to(device)
    
    # Forward pass
    print("\nForward pass (training)...")
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        handcrafted_features=handcrafted_features,
        labels=labels,
        return_embeddings=True
    )
    
    print(f"\nOutputs:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Classification loss: {outputs['cls_loss'].item():.4f}")
    print(f"  OOD loss: {outputs['ood_loss'].item():.4f}")
    print(f"  OOD scores: {outputs['ood_scores'].detach().cpu().numpy()}")
    print(f"  Probabilities: {outputs['probs'].detach().cpu().numpy()}")
    
    # Prediction
    print("\nMaking predictions...")
    predictions = model.predict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        handcrafted_features=handcrafted_features
    )
    
    print(f"\nPredictions:")
    print(f"  Predicted labels: {predictions['predictions'].cpu().numpy()}")
    print(f"  True labels: {labels.cpu().numpy()}")
    print(f"  Confidence: {predictions['confidence'].cpu().numpy()}")
    print(f"  OOD flags: {predictions['ood_flags'].cpu().numpy()}")
    
    # Label conversion
    pred_labels = model.get_prediction_labels(predictions['predictions'])
    print(f"  Predicted as: {pred_labels}")
    
    print("\n" + "=" * 60)
    print("[OK] Task A model test complete")
    
    # Test focal loss
    print("\nTesting Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 1, 0])
    loss = focal_loss(logits, targets)
    print(f"  Focal loss: {loss.item():.4f}")
