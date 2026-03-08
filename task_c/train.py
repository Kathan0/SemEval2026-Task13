"""
Training script for Task C: Hybrid Detection (Staged Learning)

Trains staged classifier with two stages:
1. Binary: Pure (Human/AI) vs Hybrid
2. Fine-grained: 4 classes (human, machine, machine_author, machine_humanized)
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path FIRST (critical for Colab)
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml
from tqdm import tqdm
import numpy as np

# Now import project modules
try:
    from task_c.model import TaskCModel
    from task_c.dataset import TaskCDataset, create_task_c_dataloader
    from src.utils import (
        set_seed, get_device, load_config, save_checkpoint,
        compute_metrics, MetricsTracker, print_model_info
    )
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    print(f"Python path: {sys.path}")
    raise


class TaskCTrainer:
    """Trainer for Task C: Hybrid Detection (Staged approach)."""
    
    def __init__(self, config_path: str, resume_from: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to config.yaml
            resume_from: Path to checkpoint to resume from (optional)
        """
        # Load config
        self.config = load_config(config_path)
        self.output_dir = Path(self.config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        set_seed(self.config['training']['seed'])
        
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
        
        # Train in stages: first binary, then fine-grained
        self.stage = 'binary'  # 'binary' or 'fine'
        
        # Load datasets
        print("\nLoading datasets...")
        data_dir = Path(self.config['paths']['data_dir'])
        train_file = data_dir / self.config['data']['train_file']
        val_file = data_dir / self.config['data']['val_file']
        max_samples = self.config['data'].get('max_samples', None)
        
        self.train_dataset = TaskCDataset(
            data_path=str(train_file),
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length'],
            mode=self.stage,
            use_sections=self.config['model']['use_sections'],
            max_samples=max_samples
        )
        self.val_dataset = TaskCDataset(
            data_path=str(val_file),
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length'],
            mode=self.stage,
            use_sections=self.config['model']['use_sections'],
            max_samples=max_samples // 5 if max_samples else None  # Use ~20% of max for validation
        )
        
        # Print statistics
        self.train_dataset.print_statistics()
        self.val_dataset.print_statistics()
        
        # Create dataloaders
        self.train_loader = create_task_c_dataloader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data'].get('num_workers', 0)
        )
        self.val_loader = create_task_c_dataloader(
            self.val_dataset,
            batch_size=self.config['training']['eval_batch_size'],
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 0)
        )
        
        # Initialize model
        print("\nInitializing model...")
        self.model = TaskCModel(
            base_model=self.config['model']['base_model'],
            num_handcrafted_features=self.config['model']['num_handcrafted_features'],
            num_sections=self.config['model']['num_sections'],
            pooling_layers=self.config['model']['pooling_layers'],
            dropout=self.config['model']['dropout'],
            use_sections=self.config['model']['use_sections']
        )
        self.model = self.model.to(self.device)
        print_model_info(self.model)
        
        # Initialize loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Mixed precision
        self.use_amp = self.config['training'].get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Resume from checkpoint if specified
        if self.resume_from:
            self._load_checkpoint(self.resume_from)
        
        print(f"\n✓ Trainer initialized (Stage: {self.stage})")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Mixed precision: {self.use_amp}")
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
        
        # Extract stage from metrics if available
        stage = checkpoint.get('metrics', {}).get('stage', self.stage)
        self.stage = stage
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Will resume from epoch {self.start_epoch} in {stage} stage")
    
    def switch_stage(self, stage: str):
        """Switch between binary and fine-grained stages."""
        self.stage = stage
        print(f"\n{'='*60}")
        print(f"Switching to {stage} stage")
        print(f"{'='*60}")
        
        # Update datasets
        self.train_dataset.set_mode(stage)
        self.val_dataset.set_mode(stage)
        
        # Rebuild dataloaders
        self.train_loader = create_task_c_dataloader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data'].get('num_workers', 0)
        )
        self.val_loader = create_task_c_dataloader(
            self.val_dataset,
            batch_size=self.config['training']['eval_batch_size'],
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 0)
        )
        
        # Reset optimizer for new stage
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Reset scheduler
        total_steps = len(self.train_loader) * self.config['training']['num_epochs']
        warmup_steps = int(total_steps * self.config['training']['warmup_ratio'])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.train_dataset.print_statistics()
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} ({self.stage})")
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Section features (if available)
            section_features = batch.get('section_features')
            consistency_features = batch.get('consistency_features')
            if section_features is not None:
                section_features = section_features.to(self.device)
                consistency_features = consistency_features.to(self.device)
            
            # Forward pass with automatic mixed precision
            if self.use_amp:
                # Use bfloat16 for better numerical stability on CUDA
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    outputs = self.model(
                        input_ids,
                        attention_mask,
                        features,
                        section_features,
                        consistency_features,
                        stage=self.stage
                    )
                    loss = self.criterion(outputs['logits'], labels)
                    
                    # Add adversarial loss for hybrid detection
                    if 'adv_logits' in outputs and self.config['training'].get('adv_weight', 0) > 0:
                        adv_loss = self.criterion(outputs['adv_logits'], labels)
                        loss = loss + self.config['training']['adv_weight'] * adv_loss
            else:
                outputs = self.model(
                    input_ids,
                    attention_mask,
                    features,
                    section_features,
                    consistency_features,
                    stage=self.stage
                )
                loss = self.criterion(outputs['logits'], labels)
                
                # Add adversarial loss for hybrid detection
                if 'adv_logits' in outputs and self.config['training'].get('adv_weight', 0) > 0:
                    adv_loss = self.criterion(outputs['adv_logits'], labels)
                    loss = loss + self.config['training']['adv_weight'] * adv_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Collect metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        num_classes = 2 if self.stage == 'binary' else 4
        metrics = compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            num_classes=num_classes
        )
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc=f"Evaluating ({self.stage})"):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Section features (if available)
            section_features = batch.get('section_features')
            consistency_features = batch.get('consistency_features')
            if section_features is not None:
                section_features = section_features.to(self.device)
                consistency_features = consistency_features.to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids,
                attention_mask,
                features,
                section_features,
                consistency_features,
                stage=self.stage
            )
            loss = self.criterion(outputs['logits'], labels)
            
            # Collect metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        num_classes = 2 if self.stage == 'binary' else 4
        metrics = compute_metrics(
            np.array(all_preds),
            np.array(all_labels),
            num_classes=num_classes
        )
        
        return avg_loss, metrics
    
    def train_stage(self, stage: str, num_epochs: int):
        """Train a single stage."""
        self.switch_stage(stage)
        
        best_f1 = 0
        
        # Adjust starting epoch if resuming from checkpoint
        start = self.start_epoch if self.start_epoch > 0 and stage == self.stage else 1
        for epoch in range(start, num_epochs + 1):
            print(f"\n{'='*60}\nEpoch {epoch}/{num_epochs} ({stage})\n{'='*60}")
            
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
            
            # Save checkpoint
            is_best = val_metrics['macro_f1'] > best_f1
            if is_best:
                best_f1 = val_metrics['macro_f1']
                print(f"\n✓ New best F1 ({stage}): {best_f1:.4f}")
            
            save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics={'val_f1': val_metrics['macro_f1'], 'val_loss': val_loss, 'stage': stage},
                save_dir=self.output_dir / stage,
                is_best=is_best
            )
        
        return best_f1
    
    def train(self):
        """Full staged training loop."""
        print("\n" + "=" * 60)
        print("Starting Staged Training")
        print("=" * 60)
        
        # Stage 1: Binary (Pure vs Hybrid)
        print("\n" + "=" * 60)
        print("Stage 1: Binary Classification (Pure vs Hybrid)")
        print("=" * 60)
        best_binary_f1 = self.train_stage('binary', self.config['training']['binary_epochs'])
        
        # Stage 2: Fine-grained (4 classes)
        print("\n" + "=" * 60)
        print("Stage 2: Fine-grained Classification (4 classes)")
        print("=" * 60)
        best_fine_f1 = self.train_stage('fine', self.config['training']['fine_epochs'])
        
        # Save final metrics
        self.metrics_tracker.save(self.output_dir / 'metrics.json')
        self.metrics_tracker.plot(self.output_dir / 'training_curves.png')
        
        print("\n" + "=" * 60)
        print("Staged Training Complete!")
        print(f"✓ Best binary F1: {best_binary_f1:.4f}")
        print(f"✓ Best fine-grained F1: {best_fine_f1:.4f}")
        print(f"✓ Checkpoints saved to: {self.output_dir}")
        print("=" * 60)


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Task C model (staged)")
    parser.add_argument(
        '--config',
        type=str,
        default='task_c/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from (e.g., checkpoints/task_c/binary/best_model.pt)'
    )
    args = parser.parse_args()
    
    # Train
    trainer = TaskCTrainer(args.config, resume_from=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
