"""
Task B Model: Authorship Detection (11 classes)

Most challenging task: classify code as human-written or identify which of 10 LLMs generated it.

Classes:
- 0: Human
- 1-10: GPT, Claude, Copilot, DeepSeek, Llama, StarCoder, Gemma, Mistral, Qwen, Other

Key features:
- Cascade strategy: Binary detection → Family classification
- Meta-learning for few-shot generalization to unseen generators
- OOD detection for completely new models
- 145 features (110 base + 35 generator-specific)

Target performance: 75-78% macro F1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from src.models.base_model import HybridCodeClassifier
import numpy as np


class BinaryStage(nn.Module):
    """
    Stage 1: Binary classification (Human vs AI).
    
    Filters out human-written code before fine-grained classification.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 2) logits for Human vs AI."""
        return self.classifier(features)


class FamilyStage(nn.Module):
    """
    Stage 2: LLM family classification (10 classes).
    
    Only applied to samples classified as AI-generated.
    """
    
    def __init__(self, hidden_dim: int, num_families: int = 10, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_families)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 10) logits for LLM families."""
        return self.classifier(features)


class MetaLearner(nn.Module):
    """
    Meta-learning module for few-shot generalization.
    
    Learns to adapt quickly to unseen generators using prototypical networks.
    """
    
    def __init__(self, hidden_dim: int, num_families: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_families = num_families
        
        # Prototype memory (one prototype per family)
        self.register_buffer(
            "prototypes",
            torch.zeros(num_families, hidden_dim)
        )
        self.register_buffer(
            "prototype_counts",
            torch.zeros(num_families)
        )
        
        # Adaptation network
        self.adaptation = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Update prototypes with new examples (during training).
        
        Args:
            features: (batch, hidden_dim) feature embeddings
            labels: (batch,) family labels (1-10)
        """
        for label in range(self.num_families):
            mask = labels == (label + 1)  # Labels are 1-indexed for families
            if mask.sum() > 0:
                family_features = features[mask]
                
                # Update running average
                old_count = self.prototype_counts[label]
                new_count = old_count + len(family_features)
                
                self.prototypes[label] = (
                    self.prototypes[label] * old_count +
                    family_features.mean(dim=0) * len(family_features)
                ) / new_count
                
                self.prototype_counts[label] = new_count
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict using prototypical network.
        
        Args:
            features: (batch, hidden_dim)
            
        Returns:
            (batch, num_families) similarity scores
        """
        # Compute distances to prototypes
        # Shape: (batch, num_families)
        distances = torch.cdist(features, self.prototypes)
        
        # Convert distances to similarities (negative distances)
        similarities = -distances
        
        return similarities
    
    def adapt(self, features: torch.Tensor) -> torch.Tensor:
        """
        Adapt features using prototype information.
        
        Args:
            features: (batch, hidden_dim)
            
        Returns:
            (batch, hidden_dim) adapted features
        """
        # Find nearest prototype for each sample
        distances = torch.cdist(features, self.prototypes)
        nearest_idx = distances.argmin(dim=1)
        nearest_prototypes = self.prototypes[nearest_idx]
        
        # Concatenate and adapt
        combined = torch.cat([features, nearest_prototypes], dim=1)
        adapted = self.adaptation(combined)
        
        return adapted


class TaskBModel(HybridCodeClassifier):
    """
    Cascade model for Task B with meta-learning.
    
    Architecture:
    1. Extract semantic + handcrafted features
    2. Stage 1: Binary (Human vs AI)
    3. Stage 2: Family classification (10 LLMs) with meta-learning
    4. OOD detection for unseen generators
    """
    
    def __init__(
        self,
        model_name: str = "bigcode/starcoder2-3b",
        handcrafted_dim: int = 145,  # 110 base + 35 generator-specific
        layer_indices: list = [6, 9, 12],
        hidden_dim: int = 512,
        num_families: int = 10,
        dropout: float = 0.1,
        use_8bit: bool = True,
        freeze_backbone: bool = False,
        freeze_layers: int = 0,
        use_meta_learning: bool = True,
        ood_threshold: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Task B cascade classifier.
        
        Args:
            model_name: Backbone model name
            handcrafted_dim: Dimension of handcrafted features (145)
            layer_indices: Layers to extract from backbone
            hidden_dim: Hidden dimension for fusion
            num_families: Number of LLM families (10)
            dropout: Dropout rate
            use_8bit: Use 8-bit quantization (GPU only)
            freeze_backbone: Whether to freeze backbone
            freeze_layers: Number of layers to freeze
            use_meta_learning: Enable meta-learning
            ood_threshold: Threshold for OOD detection
            device: Device for computation (cuda/cpu)
        """
        # Initialize base model (we'll override the classifier)
        super().__init__(
            model_name=model_name,
            num_classes=11,  # 1 human + 10 families
            handcrafted_dim=handcrafted_dim,
            layer_indices=layer_indices,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_8bit=use_8bit and device == "cuda",
            freeze_backbone=freeze_backbone,
            freeze_layers=freeze_layers
        )
        
        self.device = device
        self.num_families = num_families
        self.use_meta_learning = use_meta_learning
        self.ood_threshold = ood_threshold
        
        # Replace base classifier with cascade stages
        self.classifier = None  # Remove base classifier
        
        # Stage 1: Binary classification
        self.binary_stage = BinaryStage(hidden_dim, dropout)
        
        # Stage 2: Family classification
        self.family_stage = FamilyStage(hidden_dim, num_families, dropout)
        
        # Meta-learner for few-shot generalization
        if use_meta_learning:
            self.meta_learner = MetaLearner(hidden_dim, num_families)
        
        # OOD detection (Mahalanobis distance-based)
        self.register_buffer("class_means", torch.zeros(num_families, hidden_dim))
        self.register_buffer("class_covs", torch.eye(hidden_dim).unsqueeze(0).repeat(num_families, 1, 1))
        self.register_buffer("sample_counts", torch.zeros(num_families))
        
        # Temperature scaling
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Move to device (backbone already on device if using 8-bit quantization)
        if device == "cuda" and torch.cuda.is_available():
            # When using 8-bit, backbone is already on device via device_map="auto"
            # This moves all other components (pooler, fusion, classifier, etc.)
            if not use_8bit:
                self.to(device)
            else:
                # Move non-backbone components to device
                if hasattr(self, 'pooler'):
                    self.pooler = self.pooler.to(device)
                if hasattr(self, 'fusion'):
                    self.fusion = self.fusion.to(device)
                if hasattr(self, 'classifier'):
                    self.classifier = self.classifier.to(device)
                if hasattr(self, 'subgroup_heads'):
                    self.subgroup_heads = nn.ModuleDict({k: v.to(device) for k, v in self.subgroup_heads.items()})
                self.temperature = nn.Parameter(self.temperature.to(device))
        
        print(f"✓ Task B Model initialized")
        print(f"  Classes: 11 (1 Human + 10 LLM families)")
        print(f"  Features: {handcrafted_dim} (110 base + 35 generator-specific)")
        print(f"  Cascade: Binary → Family")
        print(f"  Meta-learning: {use_meta_learning}")
        print(f"  Device: {device}")
        print(f"  8-bit quantization: {use_8bit and device == 'cuda'}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        stage: str = "both",  # "binary", "family", or "both"
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through cascade model.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            handcrafted_features: (batch, 145)
            labels: Optional labels (0=human, 1-10=families)
            stage: Which stage to run ("binary", "family", or "both")
            return_embeddings: Whether to return feature embeddings
            
        Returns:
            Dictionary with stage-specific outputs
        """
        # Get fused features from base model
        outputs = super(HybridCodeClassifier, self).backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract and pool features
        all_hidden_states = outputs.hidden_states
        selected_hidden_states = [all_hidden_states[i] for i in self.layer_indices]
        semantic_features = self.pooler(selected_hidden_states, attention_mask)
        
        # Fuse with handcrafted features
        fused_features = self.fusion(semantic_features, handcrafted_features)
        
        outputs_dict = {}
        total_loss = 0.0
        
        # Stage 1: Binary classification
        if stage in ["binary", "both"]:
            binary_logits = self.binary_stage(fused_features)
            outputs_dict["binary_logits"] = binary_logits
            
            if labels is not None:
                # Convert labels to binary (0=human, 1=AI)
                binary_labels = (labels > 0).long()
                binary_loss = F.cross_entropy(binary_logits, binary_labels)
                outputs_dict["binary_loss"] = binary_loss
                total_loss += binary_loss
        
        # Stage 2: Family classification (only for AI samples)
        if stage in ["family", "both"]:
            # Adapt features with meta-learning
            if self.use_meta_learning and self.training:
                adapted_features = self.meta_learner.adapt(fused_features)
            else:
                adapted_features = fused_features
            
            family_logits = self.family_stage(adapted_features)
            outputs_dict["family_logits"] = family_logits
            
            # Meta-learning prototypical scores
            if self.use_meta_learning:
                proto_scores = self.meta_learner.predict(fused_features)
                outputs_dict["proto_scores"] = proto_scores
            
            if labels is not None:
                # Only compute family loss for AI samples (labels > 0)
                ai_mask = labels > 0
                if ai_mask.sum() > 0:
                    ai_labels = labels[ai_mask] - 1  # Convert to 0-9
                    ai_family_logits = family_logits[ai_mask]
                    family_loss = F.cross_entropy(ai_family_logits, ai_labels)
                    outputs_dict["family_loss"] = family_loss
                    total_loss += family_loss
                    
                    # Update meta-learner prototypes
                    if self.use_meta_learning and self.training:
                        self.meta_learner.update_prototypes(
                            fused_features[ai_mask],
                            labels[ai_mask]
                        )
        
        # Combine outputs for full 11-class prediction
        if stage == "both":
            # Combine binary and family predictions
            binary_probs = F.softmax(binary_logits / self.temperature, dim=-1)
            family_probs = F.softmax(family_logits / self.temperature, dim=-1)
            
            # Full probability: [P(human), P(AI) * P(family_0), ..., P(AI) * P(family_9)]
            p_human = binary_probs[:, 0:1]
            p_ai = binary_probs[:, 1:2]
            p_families = p_ai * family_probs
            
            full_probs = torch.cat([p_human, p_families], dim=1)
            outputs_dict["probs"] = full_probs
            
            # Logits (for compatibility)
            full_logits = torch.log(full_probs + 1e-10)
            outputs_dict["logits"] = full_logits
        
        if labels is not None and total_loss > 0:
            outputs_dict["loss"] = total_loss
        
        if return_embeddings:
            outputs_dict["embeddings"] = fused_features
        
        return outputs_dict
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted_features: torch.Tensor,
        return_confidence: bool = True,
        return_stage_predictions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Make cascade predictions.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            handcrafted_features: (batch, 145)
            return_confidence: Whether to return confidence scores
            return_stage_predictions: Whether to return stage-wise predictions
            
        Returns:
            Dictionary with predictions and metadata
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                handcrafted_features=handcrafted_features,
                stage="both",
                return_embeddings=True
            )
            
            probs = outputs["probs"]
            predictions = torch.argmax(probs, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
            
            results = {
                "predictions": predictions,
            }
            
            if return_confidence:
                results["confidence"] = confidence
            
            if return_stage_predictions:
                binary_probs = F.softmax(outputs["binary_logits"], dim=-1)
                binary_preds = torch.argmax(binary_probs, dim=-1)
                
                family_probs = F.softmax(outputs["family_logits"], dim=-1)
                family_preds = torch.argmax(family_probs, dim=-1)
                
                results["binary_predictions"] = binary_preds
                results["family_predictions"] = family_preds + 1  # Convert to 1-10
                results["binary_confidence"] = torch.max(binary_probs, dim=-1)[0]
                results["family_confidence"] = torch.max(family_probs, dim=-1)[0]
            
            return results
    
    def get_prediction_labels(self, predictions: torch.Tensor) -> List[str]:
        """Convert numeric predictions to labels."""
        label_map = {
            0: "human",
            1: "gpt",
            2: "claude", 
            3: "copilot",
            4: "deepseek",
            5: "llama",
            6: "starcoder",
            7: "gemma",
            8: "mistral",
            9: "qwen",
            10: "other"
        }
        return [label_map[p.item()] for p in predictions]


if __name__ == "__main__":
    # Test Task B model
    print("Testing Task B Model (Cascade + Meta-Learning)")
    print("=" * 60)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create model
    print("\nInitializing model...")
    model = TaskBModel(
        model_name="bigcode/starcoder2-3b",
        handcrafted_dim=145,
        use_8bit=True,
        use_meta_learning=True,
        device=device
    )
    
    # Create dummy data
    batch_size = 4
    seq_len = 128
    
    print("\nCreating dummy inputs...")
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    handcrafted_features = torch.randn(batch_size, 145)
    labels = torch.tensor([0, 1, 5, 8])  # Human, GPT, Llama, Mistral
    
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
        stage="both",
        return_embeddings=True
    )
    
    print(f"\nOutputs:")
    print(f"  Binary logits shape: {outputs['binary_logits'].shape}")
    print(f"  Family logits shape: {outputs['family_logits'].shape}")
    print(f"  Full probabilities shape: {outputs['probs'].shape}")
    print(f"  Total loss: {outputs['loss'].item():.4f}")
    print(f"  Binary loss: {outputs['binary_loss'].item():.4f}")
    if 'family_loss' in outputs:
        print(f"  Family loss: {outputs['family_loss'].item():.4f}")
    
    # Prediction
    print("\nMaking predictions...")
    predictions = model.predict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        handcrafted_features=handcrafted_features,
        return_stage_predictions=True
    )
    
    print(f"\nPredictions:")
    print(f"  Final predictions: {predictions['predictions'].cpu().numpy()}")
    print(f"  True labels: {labels.cpu().numpy()}")
    print(f"  Confidence: {predictions['confidence'].cpu().numpy()}")
    print(f"  Binary stage: {predictions['binary_predictions'].cpu().numpy()}")
    print(f"  Family stage: {predictions['family_predictions'].cpu().numpy()}")
    
    # Label conversion
    pred_labels = model.get_prediction_labels(predictions['predictions'])
    true_labels = model.get_prediction_labels(labels)
    print(f"  Predicted as: {pred_labels}")
    print(f"  True labels: {true_labels}")
    
    print("\n" + "=" * 60)
    print("✓ Task B model test complete")
