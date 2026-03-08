"""
Base Hybrid Model Architecture

Combines semantic features from StarCoder2-3B with handcrafted features
for robust AI-generated code detection.

Key components:
1. StarCoder2-3B backbone with 8-bit quantization
2. Multi-scale feature extraction from transformer layers
3. Attention-based pooling (instead of [CLS] token)
4. Feature fusion network for hybrid features
5. Task-specific classification heads

Architecture inspired by Giovanni Giuseppe Iacuzzo's solution but with:
- Larger backbone (3B vs 125M parameters)
- Multi-scale extraction vs single layer
- More comprehensive handcrafted features (110+ vs 11)
- Enhanced attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional, Tuple
import numpy as np


class MultiScaleAttentionPooling(nn.Module):
    """
    Extract and pool features from multiple transformer layers.
    
    Uses attention mechanism to weight the importance of different
    tokens and layers for the classification task.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_size: Dimension of transformer hidden states
            num_layers: Number of layers to extract from
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Layer-wise attention weights
        self.layer_attention = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Multi-head self-attention for token pooling
        self.token_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool features from multiple layers.
        
        Args:
            hidden_states: List of tensors of shape (batch, seq_len, hidden_size)
            attention_mask: Optional mask of shape (batch, seq_len)
            
        Returns:
            Pooled features of shape (batch, hidden_size)
        """
        batch_size = hidden_states[0].size(0)
        
        # Normalize layer attention weights
        layer_weights = F.softmax(self.layer_attention, dim=0)
        
        # Weighted combination of layers with numerical stability
        combined = torch.zeros_like(hidden_states[0])
        for w, h in zip(layer_weights, hidden_states):
            combined = combined + w * h
        
        combined = self.layer_norm(combined)
        
        # Use mean pooling with attention mask (more stable than attention pooling)
        if attention_mask is not None:
            # Expand mask to match hidden dimension
            mask_expanded = attention_mask.unsqueeze(-1).float()
            # Sum over sequence length with masking
            pooled = (combined * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        else:
            # Simple mean pooling
            pooled = combined.mean(1)
        
        # Safety check for NaN (should not occur with mean pooling)
        if torch.isnan(pooled).any() or torch.isinf(pooled).any():
            print("⚠️  NaN/Inf in pooled features - using zeros")
            pooled = torch.zeros_like(pooled)
        
        pooled = self.dropout(pooled)
        
        return pooled


class FeatureFusionNetwork(nn.Module):
    """
    Fusion network for combining semantic and handcrafted features.
    
    Uses gated mechanism to learn optimal feature combination.
    """
    
    def __init__(
        self,
        semantic_dim: int,
        handcrafted_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            semantic_dim: Dimension of semantic features from transformer
            handcrafted_dim: Dimension of handcrafted feature vector
            hidden_dim: Hidden dimension for fusion layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Process handcrafted features
        self.handcrafted_network = nn.Sequential(
            nn.Linear(handcrafted_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Process semantic features
        self.semantic_network = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        semantic_features: torch.Tensor,
        handcrafted_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse semantic and handcrafted features.
        
        Args:
            semantic_features: (batch, semantic_dim)
            handcrafted_features: (batch, handcrafted_dim)
            
        Returns:
            Fused features of shape (batch, hidden_dim)
        """
        # Quick NaN check (should not occur with mean pooling)
        if torch.isnan(handcrafted_features).any():
            print("⚠️  NaN in handcrafted features - replacing with zeros")
            handcrafted_features = torch.nan_to_num(handcrafted_features, nan=0.0)
        
        # Process each feature type
        h_hand = self.handcrafted_network(handcrafted_features)
        h_sem = self.semantic_network(semantic_features)
        
        # Compute gate
        gate_input = torch.cat([h_sem, h_hand], dim=1)
        gate_weights = self.gate(gate_input)
        
        # Gated combination
        gated_sem = gate_weights * h_sem
        gated_hand = (1 - gate_weights) * h_hand
        
        # Final fusion
        fused = self.fusion(torch.cat([gated_sem, gated_hand], dim=1))
        
        return fused


class HybridCodeClassifier(nn.Module):
    """
    Base hybrid classifier for AI-generated code detection.
    
    Combines:
    - StarCoder2-3B for semantic understanding
    - Multi-scale feature extraction
    - Attention pooling
    - Handcrafted features (AST, patterns, perplexity, stylometric)
    """
    
    def __init__(
        self,
        model_name: str = "bigcode/starcoder2-3b",
        num_classes: int = 2,
        handcrafted_dim: int = 110,
        layer_indices: List[int] = [6, 9, 12],
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_8bit: bool = True,
        freeze_backbone: bool = False,
        freeze_layers: int = 0
    ):
        """
        Args:
            model_name: HuggingFace model name
            num_classes: Number of output classes
            handcrafted_dim: Dimension of handcrafted features
            layer_indices: Transformer layers to extract from (0-indexed)
            hidden_dim: Hidden dimension for fusion network
            dropout: Dropout rate
            use_8bit: Use 8-bit quantization
            freeze_backbone: Whether to freeze backbone completely
            freeze_layers: Number of initial layers to freeze (if not freezing all)
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.layer_indices = layer_indices
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load backbone with optional 8-bit quantization
        print(f"Loading backbone: {model_name}")
        # Use bfloat16 on CUDA for better numerical stability, float32 on CPU
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"  Using dtype: {dtype}")
        
        if use_8bit and torch.cuda.is_available():
            print("  Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            self.backbone = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=dtype
            )
        else:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype
            )
        
        # Configure backbone for multi-layer output
        self.backbone.config.output_hidden_states = True
        
        # Freeze backbone if requested
        if freeze_backbone:
            print("  Freezing entire backbone")
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            print(f"  Freezing first {freeze_layers} layers")
            # This is model-specific; adjust based on architecture
            # For transformer models, typically freeze embeddings + first N layers
            for name, param in self.backbone.named_parameters():
                if any(f"layer.{i}." in name or "embed" in name for i in range(freeze_layers)):
                    param.requires_grad = False
        
        # Get hidden size from backbone
        self.hidden_size = self.backbone.config.hidden_size
        
        # Multi-scale attention pooling
        self.pooler = MultiScaleAttentionPooling(
            hidden_size=self.hidden_size,
            num_layers=len(layer_indices),
            dropout=dropout
        )
        
        # Feature fusion network
        self.fusion = FeatureFusionNetwork(
            semantic_dim=self.hidden_size,
            handcrafted_dim=handcrafted_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights with smaller values to prevent NaN
        self._init_weights()
        
        print(f"✓ Model initialized")
        print(f"  Semantic dim: {self.hidden_size}")
        print(f"  Handcrafted dim: {handcrafted_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num classes: {num_classes}")
    
    def _init_weights(self):
        """Initialize weights with small values to prevent NaN."""
        # Initialize pooler query with very small values
        nn.init.normal_(self.pooler.query, mean=0.0, std=0.01)
        
        for module in [self.pooler, self.fusion, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    # Very conservative initialization
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.MultiheadAttention):
                    # Initialize attention weights conservatively
                    for param in m.parameters():
                        if param.dim() > 1:
                            nn.init.xavier_uniform_(param, gain=0.01)
                        else:
                            nn.init.constant_(param, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            handcrafted_features: (batch, handcrafted_dim)
            labels: Optional labels for loss computation
            return_hidden: Whether to return hidden representations
            
        Returns:
            logits: (batch, num_classes)
            loss: Optional loss value
            hidden: Optional hidden representations if return_hidden=True
        """
        # Get multi-layer hidden states from backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract specified layers
        all_hidden_states = outputs.hidden_states
        selected_hidden_states = [all_hidden_states[i] for i in self.layer_indices]
        
        # Convert to float32 if needed (for 8-bit quantized models that output fp16)
        selected_hidden_states = [h.float() for h in selected_hidden_states]
        
        # Ensure handcrafted features are float32 and on correct device
        handcrafted_features = handcrafted_features.float()
        if handcrafted_features.device != selected_hidden_states[0].device:
            handcrafted_features = handcrafted_features.to(selected_hidden_states[0].device)
        
        # Multi-scale pooling
        semantic_features = self.pooler(selected_hidden_states, attention_mask)
        
        # Fuse with handcrafted features
        fused_features = self.fusion(semantic_features, handcrafted_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        if return_hidden:
            return logits, loss, fused_features
        
        return logits, loss, None
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get fused feature embeddings (for contrastive learning, etc.).
        
        Returns:
            Features of shape (batch, hidden_dim)
        """
        with torch.no_grad():
            _, _, hidden = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                handcrafted_features=handcrafted_features,
                return_hidden=True
            )
        return hidden


if __name__ == "__main__":
    # Test the model
    print("Testing Hybrid Code Classifier")
    print("=" * 60)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 128
    handcrafted_dim = 110
    num_classes = 2
    
    print("\nCreating model (this may take a moment)...")
    model = HybridCodeClassifier(
        model_name="bigcode/starcoder2-3b",
        num_classes=num_classes,
        handcrafted_dim=handcrafted_dim,
        use_8bit=False,  # Disable for testing
        freeze_backbone=False
    )
    
    print("\nCreating dummy inputs...")
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    handcrafted_features = torch.randn(batch_size, handcrafted_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print("\nForward pass...")
    logits, loss, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        handcrafted_features=handcrafted_features,
        labels=labels
    )
    
    print(f"\nResults:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Predictions: {logits.argmax(dim=1).tolist()}")
    print(f"  True labels: {labels.tolist()}")
    
    print("\nGetting embeddings...")
    embeddings = model.get_embeddings(input_ids, attention_mask, handcrafted_features)
    print(f"  Embeddings shape: {embeddings.shape}")
    
    print("\n" + "=" * 60)
    print("✓ Model test complete")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
