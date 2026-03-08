"""
Task C Model: Hybrid Detection (4 classes)

Detecting code with mixed human/AI authorship and adversarial examples.

Classes:
- 0: Human (fully human-written)
- 1: Machine (fully AI-generated)
- 2: Hybrid (mix of human and AI)
- 3: Adversarial (AI-generated then obfuscated)

Key features:
- Staged learning: Binary (Human vs Non-Human) → Fine-grained (4 classes)
- Section-wise attention for detecting hybrid boundaries
- Adversarial detection head for obfuscation patterns
- Windowed perplexity analysis
- 135 features (110 base + 25 adversarial-specific)

Target performance: 83-86% macro F1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from src.models.base_model import HybridCodeClassifier


class BinaryStage(nn.Module):
    """
    Stage 1: Binary classification (Human vs Non-Human).
    
    Separates fully human code from anything involving AI.
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
        """Returns (batch, 2) logits for Human vs Non-Human."""
        return self.classifier(features)


class FineGrainedStage(nn.Module):
    """
    Stage 2: Fine-grained classification (4 classes).
    
    Distinguishes between Machine, Hybrid, and Adversarial.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 4) logits for all classes."""
        return self.classifier(features)


class SectionWiseAttention(nn.Module):
    """
    Section-wise attention for detecting hybrid code boundaries.
    
    Splits code into sections and identifies which sections are AI-generated.
    """
    
    def __init__(self, hidden_dim: int, num_sections: int = 8):
        super().__init__()
        self.num_sections = num_sections
        
        # Section encoder
        self.section_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        
        # Section classifier (classify each section as human/AI)
        self.section_classifier = nn.Linear(hidden_dim, 2)
        
        # Aggregator
        self.aggregator = nn.Linear(num_sections * 2, hidden_dim)
    
    def forward(
        self,
        sequence_features: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence_features: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
            
        Returns:
            section_features: (batch, hidden_dim) aggregated section features
            section_scores: (batch, num_sections, 2) per-section classification
        """
        batch_size, seq_len, hidden_dim = sequence_features.shape
        
        # Split sequence into sections
        section_len = seq_len // self.num_sections
        sections = []
        
        for i in range(self.num_sections):
            start_idx = i * section_len
            end_idx = (i + 1) * section_len if i < self.num_sections - 1 else seq_len
            
            section = sequence_features[:, start_idx:end_idx, :]
            section_mask = attention_mask[:, start_idx:end_idx]
            
            # Pool section (mean pooling with mask)
            section_mask_expanded = section_mask.unsqueeze(-1)
            section_pooled = (section * section_mask_expanded).sum(dim=1) / section_mask_expanded.sum(dim=1).clamp(min=1)
            
            sections.append(section_pooled)
        
        # Stack sections: (batch, num_sections, hidden_dim)
        section_features = torch.stack(sections, dim=1)
        
        # Encode sections
        encoded_sections = self.section_encoder(section_features)
        
        # Classify each section
        section_scores = self.section_classifier(encoded_sections)
        
        # Aggregate section information
        section_probs = F.softmax(section_scores, dim=-1)
        section_flat = section_probs.reshape(batch_size, -1)
        aggregated = self.aggregator(section_flat)
        
        return aggregated, section_scores


class AdversarialDetectionHead(nn.Module):
    """
    Specialized head for detecting adversarial/obfuscated code.
    
    Looks for signs of:
    - Unusual variable renaming patterns
    - Comment removal/addition
    - Whitespace manipulation
    - Code structure changes while preserving semantics
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Adversarial pattern detector
        self.detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns adversarial score [0-1] where 1 = highly likely adversarial.
        """
        return self.detector(features).squeeze(-1)


class TaskCModel(HybridCodeClassifier):
    """
    Staged model for Task C with hybrid and adversarial detection.
    
    Architecture:
    1. Extract semantic + handcrafted features
    2. Section-wise attention for hybrid boundary detection
    3. Stage 1: Binary (Human vs Non-Human)
    4. Stage 2: Fine-grained (4 classes)
    5. Adversarial detection head
    """
    
    def __init__(
        self,
        model_name: str = "bigcode/starcoder2-3b",
        handcrafted_dim: int = 135,  # 110 base + 25 adversarial-specific
        layer_indices: list = [6, 9, 12],
        hidden_dim: int = 512,
        num_sections: int = 8,
        dropout: float = 0.1,
        use_8bit: bool = True,
        freeze_backbone: bool = False,
        freeze_layers: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Task C staged classifier.
        
        Args:
            model_name: Backbone model name
            handcrafted_dim: Dimension of handcrafted features (135)
            layer_indices: Layers to extract from backbone
            hidden_dim: Hidden dimension for fusion
            num_sections: Number of sections for hybrid detection
            dropout: Dropout rate
            use_8bit: Use 8-bit quantization (GPU only)
            freeze_backbone: Whether to freeze backbone
            freeze_layers: Number of layers to freeze
            device: Device for computation (cuda/cpu)
        """
        # Initialize base model (we'll override the classifier)
        super().__init__(
            model_name=model_name,
            num_classes=4,  # Human, Machine, Hybrid, Adversarial
            handcrafted_dim=handcrafted_dim,
            layer_indices=layer_indices,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_8bit=use_8bit and device == "cuda",
            freeze_backbone=freeze_backbone,
            freeze_layers=freeze_layers
        )
        
        self.device = device
        self.num_sections = num_sections
        
        # Replace base classifier with staged architecture
        self.classifier = None  # Remove base classifier
        
        # Section-wise attention for hybrid detection
        self.section_attention = SectionWiseAttention(self.hidden_size, num_sections)
        
        # Stage 1: Binary classification
        self.binary_stage = BinaryStage(hidden_dim + hidden_dim, dropout)  # Doubled for section features
        
        # Stage 2: Fine-grained classification
        self.fine_grained_stage = FineGrainedStage(hidden_dim + hidden_dim, dropout)
        
        # Adversarial detection head
        self.adversarial_head = AdversarialDetectionHead(hidden_dim, dropout)
        
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
                if hasattr(self, 'model_heads'):
                    self.model_heads = nn.ModuleDict({k: v.to(device) for k, v in self.model_heads.items()})
                if hasattr(self, 'section_attention'):
                    self.section_attention = self.section_attention.to(device)
                if hasattr(self, 'adversarial_head'):
                    self.adversarial_head = self.adversarial_head.to(device)
                self.temperature = nn.Parameter(self.temperature.to(device))
        
        print(f"✓ Task C Model initialized")
        print(f"  Classes: 4 (Human, Machine, Hybrid, Adversarial)")
        print(f"  Features: {handcrafted_dim} (110 base + 25 adversarial)")
        print(f"  Staged: Binary → Fine-grained")
        print(f"  Section-wise attention: {num_sections} sections")
        print(f"  Device: {device}")
        print(f"  8-bit quantization: {use_8bit and device == 'cuda'}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        stage: str = "both",  # "binary", "fine_grained", or "both"
        return_embeddings: bool = False,
        return_section_scores: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through staged model.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            handcrafted_features: (batch, 135)
            labels: Optional labels (0=human, 1=machine, 2=hybrid, 3=adversarial)
            stage: Which stage to run
            return_embeddings: Whether to return feature embeddings
            return_section_scores: Whether to return section-wise predictions
            
        Returns:
            Dictionary with stage-specific outputs
        """
        # Get multi-layer hidden states from backbone
        outputs = super(HybridCodeClassifier, self).backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract specified layers
        all_hidden_states = outputs.hidden_states
        selected_hidden_states = [all_hidden_states[i] for i in self.layer_indices]
        
        # Multi-scale pooling
        semantic_features = self.pooler(selected_hidden_states, attention_mask)
        
        # Section-wise attention (use last layer for section analysis)
        last_hidden_state = all_hidden_states[-1]
        section_features, section_scores = self.section_attention(
            last_hidden_state,
            attention_mask
        )
        
        # Fuse with handcrafted features
        fused_features = self.fusion(semantic_features, handcrafted_features)
        
        # Combine with section features
        combined_features = torch.cat([fused_features, section_features], dim=1)
        
        outputs_dict = {}
        total_loss = 0.0
        
        # Adversarial detection score
        adversarial_scores = self.adversarial_head(fused_features)
        outputs_dict["adversarial_scores"] = adversarial_scores
        
        # Stage 1: Binary classification
        if stage in ["binary", "both"]:
            binary_logits = self.binary_stage(combined_features)
            outputs_dict["binary_logits"] = binary_logits
            
            if labels is not None:
                # Convert labels to binary (0=human, 1-3=non-human)
                binary_labels = (labels > 0).long()
                binary_loss = F.cross_entropy(binary_logits, binary_labels)
                outputs_dict["binary_loss"] = binary_loss
                total_loss += binary_loss
        
        # Stage 2: Fine-grained classification
        if stage in ["fine_grained", "both"]:
            fine_grained_logits = self.fine_grained_stage(combined_features)
            outputs_dict["fine_grained_logits"] = fine_grained_logits
            
            if labels is not None:
                fine_grained_loss = F.cross_entropy(fine_grained_logits, labels)
                outputs_dict["fine_grained_loss"] = fine_grained_loss
                total_loss += fine_grained_loss
                
                # Adversarial detection loss
                adv_target = (labels == 3).float()
                adv_loss = F.binary_cross_entropy(adversarial_scores, adv_target)
                outputs_dict["adversarial_loss"] = adv_loss
                total_loss += 0.2 * adv_loss  # Weight adversarial loss
        
        # Combine outputs for full prediction
        if stage == "both":
            # Use fine-grained logits but boost adversarial class based on detection score
            logits = fine_grained_logits.clone()
            
            # Boost class 3 (adversarial) using adversarial detection scores
            logits[:, 3] += adversarial_scores * 2.0  # Scale factor
            
            outputs_dict["logits"] = logits
            
            # Calibrated probabilities
            probs = F.softmax(logits / self.temperature, dim=-1)
            outputs_dict["probs"] = probs
        
        if labels is not None and total_loss > 0:
            outputs_dict["loss"] = total_loss
        
        if return_embeddings:
            outputs_dict["embeddings"] = fused_features
        
        if return_section_scores:
            outputs_dict["section_scores"] = section_scores
        
        return outputs_dict
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted_features: torch.Tensor,
        return_confidence: bool = True,
        return_stage_predictions: bool = False,
        return_section_analysis: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Make staged predictions with hybrid analysis.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            handcrafted_features: (batch, 135)
            return_confidence: Whether to return confidence scores
            return_stage_predictions: Whether to return stage-wise predictions
            return_section_analysis: Whether to return section-wise analysis
            
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
                return_embeddings=True,
                return_section_scores=return_section_analysis
            )
            
            probs = outputs["probs"]
            predictions = torch.argmax(probs, dim=-1)
            confidence = torch.max(probs, dim=-1)[0]
            
            results = {
                "predictions": predictions,
            }
            
            if return_confidence:
                results["confidence"] = confidence
                results["adversarial_scores"] = outputs["adversarial_scores"]
            
            if return_stage_predictions:
                binary_probs = F.softmax(outputs["binary_logits"], dim=-1)
                binary_preds = torch.argmax(binary_probs, dim=-1)
                
                fine_grained_probs = F.softmax(outputs["fine_grained_logits"], dim=-1)
                fine_grained_preds = torch.argmax(fine_grained_probs, dim=-1)
                
                results["binary_predictions"] = binary_preds
                results["fine_grained_predictions"] = fine_grained_preds
                results["binary_confidence"] = torch.max(binary_probs, dim=-1)[0]
            
            if return_section_analysis and "section_scores" in outputs:
                section_scores = outputs["section_scores"]
                # (batch, num_sections, 2) -> (batch, num_sections)
                section_preds = torch.argmax(section_scores, dim=-1)
                results["section_predictions"] = section_preds
                results["section_scores"] = section_scores
            
            return results
    
    def get_prediction_labels(self, predictions: torch.Tensor) -> List[str]:
        """Convert numeric predictions to labels."""
        label_map = {
            0: "human",
            1: "machine",
            2: "hybrid",
            3: "adversarial"
        }
        return [label_map[p.item()] for p in predictions]
    
    def analyze_hybrid_sections(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted_features: torch.Tensor,
        tokenizer = None
    ) -> List[Dict]:
        """
        Detailed analysis of hybrid code sections.
        
        Returns list of section analyses with:
        - section_text: Decoded text of section
        - is_ai: Boolean indicating if section is AI-generated
        - confidence: Confidence score
        """
        predictions = self.predict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            handcrafted_features=handcrafted_features,
            return_section_analysis=True
        )
        
        if "section_predictions" not in predictions or tokenizer is None:
            return []
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        section_len = seq_len // self.num_sections
        
        analyses = []
        for b in range(batch_size):
            sample_analysis = []
            for s in range(self.num_sections):
                start_idx = s * section_len
                end_idx = (s + 1) * section_len if s < self.num_sections - 1 else seq_len
                
                section_tokens = input_ids[b, start_idx:end_idx]
                section_text = tokenizer.decode(section_tokens, skip_special_tokens=True)
                
                is_ai = predictions["section_predictions"][b, s].item() == 1
                section_probs = F.softmax(predictions["section_scores"][b, s], dim=-1)
                confidence = section_probs[1 if is_ai else 0].item()
                
                sample_analysis.append({
                    "section_id": s,
                    "section_text": section_text,
                    "is_ai": is_ai,
                    "confidence": confidence
                })
            
            analyses.append(sample_analysis)
        
        return analyses


if __name__ == "__main__":
    # Test Task C model
    print("Testing Task C Model (Staged + Hybrid Detection)")
    print("=" * 60)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create model
    print("\nInitializing model...")
    model = TaskCModel(
        model_name="bigcode/starcoder2-3b",
        handcrafted_dim=135,
        num_sections=8,
        use_8bit=True,
        device=device
    )
    
    # Create dummy data
    batch_size = 3
    seq_len = 128
    
    print("\nCreating dummy inputs...")
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    handcrafted_features = torch.randn(batch_size, 135)
    labels = torch.tensor([0, 1, 2])  # Human, Machine, Hybrid
    
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
        return_embeddings=True,
        return_section_scores=True
    )
    
    print(f"\nOutputs:")
    print(f"  Binary logits shape: {outputs['binary_logits'].shape}")
    print(f"  Fine-grained logits shape: {outputs['fine_grained_logits'].shape}")
    print(f"  Section scores shape: {outputs['section_scores'].shape}")
    print(f"  Adversarial scores: {outputs['adversarial_scores'].detach().cpu().numpy()}")
    print(f"  Total loss: {outputs['loss'].item():.4f}")
    print(f"  Binary loss: {outputs['binary_loss'].item():.4f}")
    print(f"  Fine-grained loss: {outputs['fine_grained_loss'].item():.4f}")
    print(f"  Adversarial loss: {outputs['adversarial_loss'].item():.4f}")
    
    # Prediction
    print("\nMaking predictions...")
    predictions = model.predict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        handcrafted_features=handcrafted_features,
        return_stage_predictions=True,
        return_section_analysis=True
    )
    
    print(f"\nPredictions:")
    print(f"  Final predictions: {predictions['predictions'].cpu().numpy()}")
    print(f"  True labels: {labels.cpu().numpy()}")
    print(f"  Confidence: {predictions['confidence'].cpu().numpy()}")
    print(f"  Binary stage: {predictions['binary_predictions'].cpu().numpy()}")
    print(f"  Section predictions shape: {predictions['section_predictions'].shape}")
    print(f"  Section predictions (sample 0): {predictions['section_predictions'][0].cpu().numpy()}")
    
    # Label conversion
    pred_labels = model.get_prediction_labels(predictions['predictions'])
    true_labels = model.get_prediction_labels(labels)
    print(f"  Predicted as: {pred_labels}")
    print(f"  True labels: {true_labels}")
    
    print("\n" + "=" * 60)
    print("✓ Task C model test complete")
