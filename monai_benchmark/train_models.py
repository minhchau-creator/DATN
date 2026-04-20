#!/usr/bin/env python3
"""
Main Training Loop for 8-Model Benchmark
Unified training script for all architectures.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from tqdm import tqdm

# MONAI imports
from monai.data import NiftiBatch
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Reorient, ScaleIntensityRanged, EnsureTupleSize, ToTensor

# Custom imports
from models.builder import get_model
from models.losses import DiceCELoss
from models.eval_metrics import calculate_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class SegmentationDataset(Dataset):
    """
    3D Medical Image Segmentation Dataset
    """
    def __init__(self, image_paths, label_paths, transforms=None):
        self.image_paths = [Path(p) for p in image_paths]
        self.label_paths = [Path(p) for p in label_paths]
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = nib.load(str(self.image_paths[idx])).get_fdata()
        image = np.expand_dims(image, axis=0).astype(np.float32)  # Add channel
        
        # Load label
        label = nib.load(str(self.label_paths[idx])).get_fdata().astype(np.int64)
        
        # Ensure spatial dimensions match
        if image.shape != (1, *label.shape):
            logger.warning(f"Shape mismatch: {image.shape} vs {label.shape}")
            # Crop or pad as needed
            min_d = min(image.shape[1], label.shape[0])
            min_h = min(image.shape[2], label.shape[1])
            min_w = min(image.shape[3], label.shape[2])
            image = image[:, :min_d, :min_h, :min_w]
            label = label[:min_d, :min_h, :min_w]
        
        # Pad to target size (96, 96, 96)
        pad_size = (96, 96, 96)
        d, h, w = label.shape
        pd = max(0, pad_size[0] - d)
        ph = max(0, pad_size[1] - h)
        pw = max(0, pad_size[2] - w)
        
        if pd > 0 or ph > 0 or pw > 0:
            pad_before = (pd // 2, ph // 2, pw // 2)
            pad_after = (pd - pad_before[0], ph - pad_before[1], pw - pad_before[2])
            image = np.pad(image, ((0, 0), (pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])), mode='constant')
            label = np.pad(label, ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])), mode='constant')
        elif d > pad_size[0] or h > pad_size[1] or w > pad_size[2]:
            # Crop if too large
            image = image[:, :pad_size[0], :pad_size[1], :pad_size[2]]
            label = label[:pad_size[0], :pad_size[1], :pad_size[2]]
        
        # Preprocess
        image = self._preprocess(image)
        
        return {'image': torch.from_numpy(image).float(), 'label': torch.from_numpy(label).long()}
    
    def _preprocess(self, image):
        """Apply HU windowing"""
        hu_min, hu_max = -74, 140
        image = np.clip(image, hu_min, hu_max)
        image = (image - hu_min) / (hu_max - hu_min)
        return image


class Trainer:
    """Main training class"""
    def __init__(self, model_name, fold_idx, config, device='cuda:0'):
        self.model_name = model_name
        self.fold_idx = fold_idx
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Paths
        self.checkpoint_dir = Path(f"checkpoints/{model_name}/fold_{fold_idx}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / "best_model.pth"
        self.latest_model_path = self.checkpoint_dir / "latest_model.pth"
        self.metrics_file = Path(f"results/{model_name}_fold_{fold_idx}_metrics.json")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} - Fold {fold_idx}")
        logger.info(f"{'='*60}")
        
    def setup(self, train_paths, val_paths):
        """Setup model, data loaders, optimizer"""
        
        # Model
        logger.info(f"📦 Loading model: {self.model_name}")
        self.model = get_model(
            self.model_name,
            pretrained=(self.config.get('strategy') == 'fine_tune'),
            device=self.device
        )
        
        if self.model is None:
            raise ValueError(f"Could not load model {self.model_name}")
        
        # Data loaders
        logger.info(f"📂 Loading data for fold {self.fold_idx}")
        bs = self.config.get('batch_size', 8)
        self.train_loader = self._create_dataloader(train_paths, bs, shuffle=True)
        self.val_loader = self._create_dataloader(val_paths, 1, shuffle=False)
        
        # Loss and optimizer
        self.criterion = DiceCELoss(
            ce_weight=0.5,
            dice_weight=0.5,
            class_weights=self.config.get('class_weights', [0.1, 0.5, 0.4])
        )
        
        lr = self.config.get('lr', 1e-4)
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=self.config.get('weight_decay', 1e-5))
        
        # LR Scheduler
        epochs = self.config.get('epochs', 50)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        logger.info(f"✓ Setup complete: Model | Data Loaders | Loss | Optimizer")
        
    def _create_dataloader(self, file_paths, batch_size, shuffle=True):
        """Create dataloader from file paths"""
        image_paths = []
        label_paths = []
        
        for fname in file_paths:
            img_file = Path(f"dataset/Task_7/imagesTr/{fname.replace('.nii', '_0000.nii.gz')}")
            lbl_file = Path(f"dataset/Task_7/labelsTr/{fname.replace('.nii', '.nii.gz')}")
            
            if img_file.exists() and lbl_file.exists():
                image_paths.append(img_file)
                label_paths.append(lbl_file)
        
        dataset = SegmentationDataset(image_paths, label_paths)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch", leave=False)
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.update(1)
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        """Validate on val set, return metrics"""
        self.model.eval()
        all_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val Epoch", leave=False):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()  # (B, D, H, W)
                targets = labels.cpu().numpy()  # (B, D, H, W)
                
                # Per-scan metrics
                for b in range(preds.shape[0]):
                    metrics = calculate_metrics(preds[b], targets[b])
                    for key, val in metrics.items():
                        all_metrics[key].append(val)
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        return avg_metrics
    
    def train(self, train_paths, val_paths):
        """Main training loop"""
        self.setup(train_paths, val_paths)
        
        epochs = self.config.get('epochs', 50)
        best_dice = 0.0
        patience_counter = 0
        patience = self.config.get('patience', 10)
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Validate every N epochs
            val_freq = self.config.get('val_freq', 5)
            if (epoch + 1) % val_freq == 0 or epoch == 0:
                val_metrics = self.validate()
                avg_dice = val_metrics.get('avg_dice', 0.0)
                
                logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Dice: {avg_dice:.4f}")
                
                # Save if best
                if avg_dice > best_dice:
                    best_dice = avg_dice
                    self.save_checkpoint(val_metrics, is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            self.scheduler.step()
            
            # Save latest
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(None, is_best=False)
        
        logger.info(f"✓ Training complete. Best Dice: {best_dice:.4f}")
        return best_dice
    
    def save_checkpoint(self, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': metrics,
            'model_name': self.model_name,
            'fold': self.fold_idx
        }
        
        path = self.best_model_path if is_best else self.latest_model_path
        torch.save(checkpoint, path)
        
        if is_best:
            # Also save metrics
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_file, 'w') as f:
                # Convert numpy values to float for JSON serialization
                metrics_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in metrics.items()}
                json.dump(metrics_serializable, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', help='Model name')
    parser.add_argument('--folds', type=str, default='0', help='Comma-separated fold indices')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--config', type=str, default='monai_benchmark/configs/training.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    model_config = full_config['models'].get(args.model, {})
    common_config = full_config['common']
    model_config.update({k: v for k, v in common_config.items() if k not in model_config})
    
    # Load fold data
    splits_dir = Path('monai_benchmark/splits')
    folds = [int(f) for f in args.folds.split(',')]
    
    for fold_idx in folds:
        logger.info(f"\n{'*'*60}")
        logger.info(f"Processing Fold {fold_idx}")
        logger.info(f"{'*'*60}")
        
        fold_file = splits_dir / f"fold_{fold_idx}.json"
        with open(fold_file, 'r') as f:
            fold_data = json.load(f)
        
        train_paths = fold_data['train']
        val_paths = fold_data['val']
        
        # Train
        trainer = Trainer(args.model, fold_idx, model_config, device=args.device)
        trainer.train(train_paths, val_paths)
        
        torch.cuda.empty_cache()
    
    logger.info(f"\n✅ All folds complete for {args.model}")


if __name__ == "__main__":
    main()
