"""
Training pipeline for Urban Heat Island detection models - FIXED VERSION
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Optional
import json
import shutil
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import *
from models import UNet, LSTLoss, EarlyStopping, initialize_weights, count_parameters

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def check_disk_space(path: Path, required_mb: int = 1000) -> bool:
    """Check if sufficient disk space is available"""
    try:
        stat = shutil.disk_usage(path)
        available_mb = stat.free / (1024 * 1024)
        logger.info(f"Available disk space: {available_mb:.2f} MB")
        
        if available_mb < required_mb:
            logger.warning(f"Low disk space! Available: {available_mb:.2f} MB, Required: {required_mb} MB")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True


class UHIDataset(Dataset):
    """PyTorch Dataset for UHI data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 augment: bool = False, transform=None):
        """
        Args:
            X: Features array (N, H, W, C)
            y: Target array (N, H, W, 1)
            augment: Apply data augmentation
            transform: Additional transforms
        """
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)  # (N, C, H, W)
        self.y = torch.FloatTensor(y).permute(0, 3, 1, 2)  # (N, 1, H, W)
        self.augment = augment
        self.transform = transform
        
        # DIAGNOSTIC: Log data statistics
        logger.info(f"Dataset created - X range: [{self.X.min():.4f}, {self.X.max():.4f}]")
        logger.info(f"Dataset created - y range: [{self.y.min():.4f}, {self.y.max():.4f}]")
        logger.info(f"Dataset created - y mean: {self.y.mean():.4f}, y std: {self.y.std():.4f}")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            x, y = self._augment(x, y)
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def _augment(self, x, y):
        """Apply data augmentation"""
        # Random horizontal flip
        if torch.rand(1) > 0.5 and AUGMENTATION_CONFIG["flip_horizontal"]:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
        
        # Random vertical flip
        if torch.rand(1) > 0.5 and AUGMENTATION_CONFIG["flip_vertical"]:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        
        # Random rotation (90, 180, 270 degrees)
        if torch.rand(1) > 0.5:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[1, 2])
        
        # Add Gaussian noise
        if AUGMENTATION_CONFIG["noise_std"] > 0:
            noise = torch.randn_like(x) * AUGMENTATION_CONFIG["noise_std"]
            x = x + noise
        
        # Brightness adjustment
        if AUGMENTATION_CONFIG["brightness_range"] > 0:
            brightness = 1.0 + (torch.rand(1) - 0.5) * 2 * AUGMENTATION_CONFIG["brightness_range"]
            x = x * brightness
        
        return x, y


class Trainer:
    """Model trainer"""
    
    def __init__(self, model, device, config=TRAINING_CONFIG):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Loss function
        self.criterion = LSTLoss(
            alpha=config["loss_weights"]["mse"],
            beta=config["loss_weights"]["spatial"],
            gamma=config["loss_weights"]["physical"]
        )
        
        # Optimizer
        if config["optimizer"] == "adamw":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config["initial_lr"],
                weight_decay=config["weight_decay"]
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config["initial_lr"]
            )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config["patience"],
            mode="min"
        )
        
        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "lr": []
        }
        
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        warmup_epochs = self.config["warmup_epochs"]
        total_epochs = self.config["epochs"]
        
        # Warmup + Cosine annealing
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return epoch / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        loss_components = {"mse": 0, "spatial": 0, "physical": 0}
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # DIAGNOSTIC: Check for NaN/Inf in first batch
            if batch_idx == 0:
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logger.error(f"NaN/Inf detected in model output!")
                    logger.error(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
                logger.info(f"Batch 0 - Output range: [{output.min():.4f}, {output.max():.4f}]")
                logger.info(f"Batch 0 - Target range: [{target.min():.4f}, {target.max():.4f}]")
            
            loss, components = self.criterion(output, target, data)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at batch {batch_idx}!")
                logger.error(f"Loss components: {components}")
                continue  # Skip this batch
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += components[key]
            
            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "mse": components["mse"]
            })
        
        # Average losses
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss, _ = self.criterion(output, target, data)
                
                total_loss += loss.item()
                
                # Collect predictions for metrics
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        
        preds = np.concatenate(all_preds, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()
        
        # DIAGNOSTIC: Log prediction statistics BEFORE filtering
        logger.info(f"Raw predictions - range: [{preds.min():.4f}, {preds.max():.4f}]")
        logger.info(f"Raw predictions - mean: {preds.mean():.4f}, std: {preds.std():.4f}")
        logger.info(f"Raw targets - range: [{targets.min():.4f}, {targets.max():.4f}]")
        logger.info(f"Raw targets - mean: {targets.mean():.4f}, std: {targets.std():.4f}")
        logger.info(f"Unique prediction values: {len(np.unique(preds))}")
        
        # Check if predictions are constant
        if preds.std() < 1e-6:
            logger.error("⚠️ MODEL IS PREDICTING CONSTANT VALUES! ⚠️")
            logger.error(f"All predictions are approximately: {preds.mean():.4f}")
        
        # Remove NaN values
        mask = ~np.isnan(targets)
        preds = preds[mask]
        targets = targets[mask]
        
        logger.info(f"After NaN filtering - {len(preds)} valid samples")
        
        metrics = self._calculate_metrics(preds, targets)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, preds, targets):
        """Calculate evaluation metrics"""
        # DIAGNOSTIC: Check variance
        pred_var = np.var(preds)
        target_var = np.var(targets)
        
        logger.info(f"Prediction variance: {pred_var:.6f}")
        logger.info(f"Target variance: {target_var:.6f}")
        
        if pred_var < 1e-8:
            logger.warning("⚠️ Predictions have near-zero variance!")
        
        # Calculate R² with additional diagnostics
        try:
            r2 = r2_score(targets, preds)
            
            # Manual R² calculation for verification
            ss_res = np.sum((targets - preds) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2_manual = 1 - (ss_res / ss_tot)
            
            logger.info(f"R² (sklearn): {r2:.6f}")
            logger.info(f"R² (manual): {r2_manual:.6f}")
            logger.info(f"SS_res: {ss_res:.4f}, SS_tot: {ss_tot:.4f}")
            
        except Exception as e:
            logger.error(f"Error calculating R²: {e}")
            r2 = 0.0
        
        rmse = np.sqrt(mean_squared_error(targets, preds))
        mae = mean_absolute_error(targets, preds)
        mbe = np.mean(preds - targets)
        
        return {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "mbe": mbe
        }
    
    def train(self, train_loader, val_loader, save_dir: Path):
        """Full training loop"""
        logger.info(f"Starting training for {self.config['epochs']} epochs")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Check disk space before starting
        if not check_disk_space(save_dir, required_mb=2000):
            logger.warning("Proceeding with low disk space - checkpoints may fail!")
        
        # DIAGNOSTIC: Test model on first batch before training
        logger.info("\n" + "="*50)
        logger.info("PRE-TRAINING DIAGNOSTICS")
        logger.info("="*50)
        self.model.eval()
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            sample_x, sample_y = sample_batch[0][:1].to(self.device), sample_batch[1][:1].to(self.device)
            sample_output = self.model(sample_x)
            logger.info(f"Sample input shape: {sample_x.shape}")
            logger.info(f"Sample output shape: {sample_output.shape}")
            logger.info(f"Sample output range: [{sample_output.min():.4f}, {sample_output.max():.4f}]")
            logger.info(f"Sample target range: [{sample_y.min():.4f}, {sample_y.max():.4f}]")
        logger.info("="*50 + "\n")
        
        for epoch in range(self.config["epochs"]):
            logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss, train_components = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Metrics - R²: {val_metrics['r2']:.4f}, "
                       f"RMSE: {val_metrics['rmse']:.4f}°C, "
                       f"MAE: {val_metrics['mae']:.4f}°C")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)
            self.history["lr"].append(current_lr)
            
            # Save checkpoint (with error handling and cleanup)
            if (epoch + 1) % 5 == 0:
                try:
                    self._save_checkpoint(save_dir, epoch, val_loss, val_metrics)
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")
                    logger.info("Continuing training without checkpoint...")
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Save final model
        try:
            self._save_final_model(save_dir)
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
        
        # Save training history
        try:
            self._save_history(save_dir)
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")
        
        logger.info("Training complete!")
        
        return self.history
    
    def _save_checkpoint(self, save_dir, epoch, loss, metrics):
        """Save model checkpoint with error handling and cleanup"""
        # Clean up old checkpoints to save space (keep only last 2)
        existing_checkpoints = sorted(save_dir.glob("checkpoint_epoch_*.pth"))
        if len(existing_checkpoints) >= 2:
            # Remove oldest checkpoint
            oldest = existing_checkpoints[0]
            try:
                oldest.unlink()
                logger.info(f"Removed old checkpoint: {oldest.name}")
            except Exception as e:
                logger.warning(f"Could not remove old checkpoint: {e}")
        
        # Check disk space before saving
        if not check_disk_space(save_dir, required_mb=500):
            logger.warning("Skipping checkpoint save due to low disk space")
            return
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "metrics": metrics
        }
        
        path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        temp_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth.tmp"
        
        try:
            # Save to temporary file first
            torch.save(checkpoint, temp_path)
            # If successful, rename to final path
            temp_path.rename(path)
            logger.info(f"Saved checkpoint: {path}")
        except Exception as e:
            # Clean up temporary file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _save_final_model(self, save_dir):
        """Save final trained model"""
        # Check disk space
        if not check_disk_space(save_dir, required_mb=500):
            logger.error("Cannot save final model - insufficient disk space!")
            return
        
        # Save best model from early stopping
        if self.early_stopping.best_model is not None:
            try:
                torch.save(
                    self.early_stopping.best_model,
                    save_dir / "best_model.pth"
                )
                logger.info("Saved best model")
            except Exception as e:
                logger.error(f"Failed to save best model: {e}")
        
        # Save final model (state dict only - more compact)
        try:
            torch.save(
                self.model.state_dict(),
                save_dir / "final_model.pth"
            )
            logger.info("Saved final model")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
    
    def _save_history(self, save_dir):
        """Save training history"""
        # Convert numpy values to Python types for JSON serialization
        history_serializable = {}
        for key, values in self.history.items():
            if key in ["train_metrics", "val_metrics"]:
                history_serializable[key] = [
                    {k: float(v) for k, v in metrics.items()}
                    for metrics in values
                ]
            else:
                history_serializable[key] = [float(v) for v in values]
        
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history_serializable, f, indent=2)
        
        logger.info("Saved training history")


def validate_data_quality(X: np.ndarray, y: np.ndarray, split: str) -> bool:
    """
    Validate data quality before training
    Returns True if data is good, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DATA QUALITY CHECK: {split.upper()}")
    logger.info(f"{'='*60}")
    
    issues = []
    
    # Check for NaN/Inf
    if np.isnan(X).any() or np.isinf(X).any():
        nan_pct = np.isnan(X).sum() / X.size * 100
        inf_pct = np.isinf(X).sum() / X.size * 100
        issues.append(f"X contains NaN ({nan_pct:.2f}%) or Inf ({inf_pct:.2f}%)")
    
    if np.isnan(y).any() or np.isinf(y).any():
        nan_pct = np.isnan(y).sum() / y.size * 100
        inf_pct = np.isinf(y).sum() / y.size * 100
        issues.append(f"y contains NaN ({nan_pct:.2f}%) or Inf ({inf_pct:.2f}%)")
    
    # Check target variance
    y_std = y.std()
    y_mean = y.mean()
    y_min = y.min()
    y_max = y.max()
    
    logger.info(f"Target (LST) statistics:")
    logger.info(f"  Mean: {y_mean:.4f}°C")
    logger.info(f"  Std:  {y_std:.4f}°C")
    logger.info(f"  Min:  {y_min:.4f}°C")
    logger.info(f"  Max:  {y_max:.4f}°C")
    logger.info(f"  Unique values: {len(np.unique(y))}")
    
    if y_std < 0.5:
        issues.append(f"Target has very low variance (std={y_std:.4f}°C) - likely placeholder data!")
    
    if y_std < 1e-6:
        issues.append(f"Target has ZERO variance - all values are constant ({y_mean:.4f})")
    
    # # Check realistic temperature range for Jakarta
    # if y_min < 15 or y_max > 50:
    #     issues.append(f"Target values outside realistic range for Jakarta (15-50°C): [{y_min:.1f}, {y_max:.1f}]")
    
    # # Check feature variance
    # X_channel_stds = X.std(axis=(0, 1, 2))
    # low_variance_channels = np.where(X_channel_stds < 1e-6)[0]
    # if len(low_variance_channels) > 0:
    #     issues.append(f"Input channels with zero variance: {low_variance_channels.tolist()}")
    
    # Report results
    if issues:
        logger.error(f"❌ DATA QUALITY ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            logger.error(f"  {i}. {issue}")
        logger.error(f"\n{'='*60}")
        logger.error(f"CRITICAL: Cannot train model with problematic data!")
        logger.error(f"{'='*60}\n")
        return False
    else:
        logger.info(f"✓ Data quality checks passed")
        logger.info(f"{'='*60}\n")
        return True


def load_data(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed data with validation"""
    # Point to the cnn_dataset subdirectory
    data_dir = PROCESSED_DATA_DIR / "cnn_dataset" / split
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        logger.error(f"Please run preprocessing.py first to generate the dataset")
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    
    X_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"
    
    if not X_path.exists() or not y_path.exists():
        logger.error(f"Data files missing in {data_dir}")
        logger.error(f"Expected files: X.npy, y.npy")
        raise FileNotFoundError(f"Missing data files in {data_dir}")
    
    try:
        X = np.load(X_path)
        y = np.load(y_path)
    except Exception as e:
        logger.error(f"Failed to load data from {data_dir}: {e}")
        raise
    
    logger.info(f"Loaded {split} data: X shape {X.shape}, y shape {y.shape}")
    
    # Validate data quality
    if not validate_data_quality(X, y, split):
        logger.error(f"\n{'='*60}")
        logger.error(f"SOLUTION: Fix your preprocessing pipeline")
        logger.error(f"{'='*60}")
        logger.error(f"The issue is likely in preprocessing.py:")
        logger.error(f"  1. Check if LST is being calculated correctly from thermal band")
        logger.error(f"  2. Verify thermal band (ST_B10) is downloaded from Earth Engine")
        logger.error(f"  3. Remove any placeholder LST values")
        logger.error(f"  4. Ensure LST has reasonable temperature range (15-50°C for Jakarta)")
        logger.error(f"  5. Check that patches have sufficient LST variance")
        logger.error(f"\nRe-run preprocessing.py after fixes")
        logger.error(f"{'='*60}\n")
        raise ValueError(f"{split} data failed quality checks - cannot proceed with training")
    
    return X, y


def main():
    """Main training script"""
    logger.info("="*60)
    logger.info("URBAN HEAT ISLAND DETECTION - TRAINING PIPELINE")
    logger.info("="*60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and COMPUTE_CONFIG["use_gpu"] else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load data
    logger.info("\nLoading and validating data...")
    try:
        X_train, y_train = load_data("train")
        X_val, y_val = load_data("val")
    except Exception as e:
        logger.error(f"\n❌ Failed to load data: {e}")
        logger.error(f"\nPlease ensure:")
        logger.error(f"  1. preprocessing.py has been run successfully")
        logger.error(f"  2. Data exists in: {PROCESSED_DATA_DIR / 'cnn_dataset'}")
        logger.error(f"  3. LST data has real temperature variance (not placeholders)")
        return
    
    # Create datasets
    logger.info("\nCreating PyTorch datasets...")
    train_dataset = UHIDataset(X_train, y_train, augment=True)
    val_dataset = UHIDataset(X_val, y_val, augment=False)
    
    # Create dataloaders
    logger.info("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=COMPUTE_CONFIG["num_workers"],
        pin_memory=COMPUTE_CONFIG["pin_memory"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=COMPUTE_CONFIG["num_workers"],
        pin_memory=COMPUTE_CONFIG["pin_memory"]
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("\nInitializing model...")
    model = UNet(
        in_channels=CNN_CONFIG["input_channels"],
        out_channels=1
    )
    initialize_weights(model)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    logger.info(f"Model architecture: U-Net")
    logger.info(f"Input channels: {CNN_CONFIG['input_channels']}")
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(model, device)
    
    logger.info(f"Optimizer: {TRAINING_CONFIG['optimizer']}")
    logger.info(f"Initial learning rate: {TRAINING_CONFIG['initial_lr']}")
    logger.info(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    logger.info(f"Epochs: {TRAINING_CONFIG['epochs']}")
    
    # Train
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    try:
        history = trainer.train(train_loader, val_loader, MODEL_DIR)
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    # Print final metrics
    if history["val_metrics"]:
        final_metrics = history["val_metrics"][-1]
        logger.info("\nFINAL VALIDATION METRICS:")
        logger.info(f"  R² Score: {final_metrics['r2']:.4f} (target: ≥ {VALIDATION_CONFIG['targets']['r2']})")
        logger.info(f"  RMSE: {final_metrics['rmse']:.4f}°C (target: ≤ {VALIDATION_CONFIG['targets']['rmse']}°C)")
        logger.info(f"  MAE: {final_metrics['mae']:.4f}°C (target: ≤ {VALIDATION_CONFIG['targets']['mae']}°C)")
        logger.info(f"  MBE: {final_metrics['mbe']:.4f}°C")
        logger.info("="*50)

if __name__ == "__main__":
    main()