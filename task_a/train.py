import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path FIRST (critical for Colab)
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set environment variable for PyTorch memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml
from tqdm import tqdm
import numpy as np

# Now import project modules
try:
    from task_a.model import TaskAModel, FocalLoss
    from task_a.dataset import TaskADataset, create_task_a_dataloader
    from src.utils import (
        set_seed, get_device, load_config, save_checkpoint,
        compute_metrics, MetricsTracker, print_model_info
    )
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent.parent}")
    raise


class TaskATrainer:
    """Trainer for Task A: Binary Classification (Human vs AI)."""
    
    def __init__(self, config_path: str, resume_from: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to config.yaml
            resume_from: Path to checkpoint to resume from (optional)
        """
        # Load config
        self.config = load_config(config_path)
        self.output_dir = Path(self.config['paths']['checkpoint_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        set_seed(self.config['common']['seed'])
        
        # Get device
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Track checkpoint path for resuming
        self.resume_from = resume_from
        self.start_epoch = 0
        
        # Initialize tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['base_model']
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load datasets
        print("\nLoading datasets...")
        data_dir = Path(self.config['paths']['data_dir'])
        train_file = data_dir / self.config['data']['train_file']
        val_file = data_dir / self.config['data']['val_file']
        cache_dir = self.config['paths']['cache_dir']
        max_samples = self.config['data'].get('max_samples', None)
        
        self.train_dataset = TaskADataset(
            data_path=str(train_file),
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length'],
            cache_dir=cache_dir,
            max_samples=max_samples
        )
        self.val_dataset = TaskADataset(
            data_path=str(val_file),
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length'],
            cache_dir=cache_dir,
            max_samples=max_samples // 5 if max_samples else None  # Use ~20% of max for validation
        )
        
        # Print statistics
        self.train_dataset.print_statistics()
        self.val_dataset.print_statistics()
        
        # Create dataloaders
        self.train_loader = create_task_a_dataloader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data'].get('num_workers', 0),
            persistent_workers=self.config['data'].get('persistent_workers', False),
            prefetch_factor=self.config['data'].get('prefetch_factor', 2)
        )
        self.val_loader = create_task_a_dataloader(
            self.val_dataset,
            batch_size=self.config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 0),
            persistent_workers=self.config['data'].get('persistent_workers', False),
            prefetch_factor=self.config['data'].get('prefetch_factor', 2)
        )
        
        # Initialize model
        print("\nInitializing model...")
        
        # Calculate total handcrafted features
        num_features = (
            self.config['data'].get('num_ast_features', 33) +
            self.config['data'].get('num_pattern_features', 57) +
            self.config['data'].get('num_perplexity_features', 8) +
            self.config['data'].get('num_stylometric_features', 12)
        )
        
        self.model = TaskAModel(
            model_name=self.config['model']['base_model'],
            handcrafted_dim=num_features,
            layer_indices=self.config['model'].get('multi_scale_layers', [6, 9, 12]),
            hidden_dim=self.config['model'].get('hidden_size', 768),
            dropout=self.config['model'].get('hidden_dropout', 0.2),
            use_8bit=self.config['model'].get('use_8bit_quantization', True),
            freeze_backbone=self.config['model'].get('freeze_backbone', False),
            freeze_layers=self.config['model'].get('freeze_layers', 0),
            device=str(self.device)
        )
        self.model = self.model.to(self.device)
        print_model_info(self.model)
        
        # Initialize loss
        self.criterion = FocalLoss(
            alpha=self.config['training']['focal_alpha'],
            gamma=self.config['training']['focal_gamma']
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config['training']['num_epochs']
        warmup_steps = int(total_steps * self.config['training']['warmup_ratio'])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Mixed precision (disable if using 8-bit quantization due to compatibility issues)
        use_8bit = self.config['model'].get('use_8bit_quantization', True)
        self.use_amp = (
            self.config['common'].get('mixed_precision', False) and  # Changed default to False
            self.device.type == 'cuda' and 
            not use_8bit  # Disable AMP when using 8-bit quantization
        )
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        if use_8bit and self.config['common'].get('mixed_precision', False):
            print("⚠️  WARNING: Mixed precision disabled - incompatible with 8-bit quantization")
        
        # Resume from checkpoint if specified
        if self.resume_from:
            self._load_checkpoint(self.resume_from)
        
        print(f"\n[OK] Trainer initialized")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  8-bit quantization: {use_8bit}")
        print(f"  Mixed precision: {self.use_amp} {'(disabled due to 8-bit quant)' if use_8bit and not self.use_amp else ''}")
        if self.resume_from:
            print(f"  Resuming from epoch: {self.start_epoch}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training state."""
        from src.utils import load_checkpoint
        
        print(f"\n📥 Loading checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        # Update start epoch
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Will resume from epoch {self.start_epoch}")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Clear CUDA cache at start of epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get gradient accumulation steps
        grad_accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            # Move to device (non_blocking for async transfer)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            features = batch['features'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Forward pass with automatic mixed precision
            if self.use_amp:
                # Use bfloat16 for better numerical stability on CUDA
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    outputs = self.model(input_ids, attention_mask, features)
                    loss = self.criterion(outputs['logits'], labels)
                    # Scale loss by accumulation steps
                    loss = loss / grad_accum_steps
            else:
                outputs = self.model(input_ids, attention_mask, features)
                loss = self.criterion(outputs['logits'], labels)
                # Scale loss by accumulation steps
                loss = loss / grad_accum_steps
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  NaN/Inf detected at step {step}!")
                print(f"  Loss: {loss.item()}")
                print(f"  Logits range: [{outputs['logits'].min().item():.4f}, {outputs['logits'].max().item():.4f}]")
                print(f"  Features range: [{features.min().item():.4f}, {features.max().item():.4f}]")
                print("  Skipping this batch...")
                continue
            
            # Backward pass (accumulate gradients)
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights after accumulating gradients
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                if self.use_amp:
                    # Unscale gradients before clipping (only if not already unscaled)
                    self.scaler.unscale_(self.optimizer)
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
                    # Step optimizer
                    self.scaler.step(self.optimizer)
                    # Update scaler for next iteration
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            # Collect metrics
            total_loss += loss.item() * grad_accum_steps  # Unscale for logging
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item() * grad_accum_steps:.4f}"})
        
        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            num_classes=2
        )
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_logits = []
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            # Move to device (non_blocking for async transfer)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            features = batch['features'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, features)
            loss = self.criterion(outputs['logits'], labels)
            
            # Collect metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(outputs['logits'].cpu())
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            num_classes=2
        )
        
        # Add temperature-calibrated confidence scores
        all_logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(all_logits, dim=1)
        metrics['avg_confidence'] = probs.max(dim=1)[0].mean().item()
        
        # Clear CUDA cache after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return avg_loss, metrics
    
    def train(self):
        """Full training loop."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        best_f1 = 0
        
        # Adjust starting epoch if resuming from checkpoint
        start = self.start_epoch if self.start_epoch > 0 else 1
        for epoch in range(start, self.config['training']['num_epochs'] + 1):
            print(f"\n{'='*60}\nEpoch {epoch}/{self.config['training']['num_epochs']}\n{'='*60}")
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            val_loss, val_metrics = self.evaluate()
            
            # Update metrics tracker
            self.metrics_tracker.update(
                epoch=epoch,
                train_loss=train_loss,
                train_metrics=train_metrics,
                val_loss=val_loss,
                val_metrics=val_metrics
            )
            
            # Print metrics
            print(f"\nTrain - Loss: {train_loss:.4f}, F1: {train_metrics['macro_f1']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, F1: {val_metrics['macro_f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Confidence: {val_metrics['avg_confidence']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['macro_f1'] > best_f1
            if is_best:
                best_f1 = val_metrics['macro_f1']
                print(f"\n[OK] New best F1: {best_f1:.4f}")
            
            save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics={'val_f1': val_metrics['macro_f1'], 'val_loss': val_loss},
                save_dir=self.output_dir,
                is_best=is_best
            )
        
        # Save final metrics
        self.metrics_tracker.save(self.output_dir / 'metrics.json')
        self.metrics_tracker.plot(self.output_dir / 'training_curves.png')
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"[OK] Best validation F1: {best_f1:.4f}")
        print(f"[OK] Checkpoints saved to: {self.output_dir}")
        print("=" * 60)


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Task A model")
    parser.add_argument(
        '--config',
        type=str,
        default='task_a/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (e.g., checkpoints/task_a/best_model.pt)'
    )
    parser.add_argument(
        '--no-auto-resume',
        action='store_true',
        help='Disable automatic resume from last checkpoint'
    )
    args = parser.parse_args()
    
    # Auto-detect last checkpoint if not specified and not disabled
    resume_from = args.resume
    if resume_from is None and not args.no_auto_resume:
        # Load config to get checkpoint directory
        config = load_config(args.config)
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        
        if checkpoint_dir.exists():
            # Look for latest epoch checkpoint
            epoch_checkpoints = sorted(checkpoint_dir.glob('epoch_*.pt'), 
                                      key=lambda x: int(x.stem.split('_')[1]),
                                      reverse=True)
            if epoch_checkpoints:
                resume_from = str(epoch_checkpoints[0])
                print(f"\n🔄 Auto-detected checkpoint: {resume_from}")
                print("   (Use --no-auto-resume to start from scratch)")
    
    # Train
    trainer = TaskATrainer(args.config, resume_from=resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
