"""
Enhanced training pipeline with CNN + GBM ensemble - IMPROVED VERSION
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
import lightgbm as lgb
import pickle

from config import *
from models import UNet, LSTLoss, EarlyStopping, initialize_weights, count_parameters

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def load_normalization_stats() -> Optional[Dict]:
    """
    Load normalization statistics for denormalization during evaluation
    
    Returns:
        Normalization statistics dictionary or None if not found
    """
    stats_path = PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
    
    if not stats_path.exists():
        logger.warning("⚠️ Normalization stats not found - predictions will remain in normalized space")
        return None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    return stats

def denormalize_predictions(predictions: np.ndarray, norm_stats: Dict) -> np.ndarray:
    """
    Denormalize predictions back to Celsius
    
    Args:
        predictions: Normalized predictions
        norm_stats: Normalization statistics
        
    Returns:
        Denormalized predictions in Celsius
    """
    if norm_stats is None or 'target' not in norm_stats:
        logger.warning("⚠️ Cannot denormalize - no target stats available")
        return predictions
    
    target_mean = norm_stats['target']['mean']
    target_std = norm_stats['target']['std']
    
    denormalized = predictions * target_std + target_mean
    
    return denormalized

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
            X: NORMALIZED features array (N, H, W, C) - mean≈0, std≈1
            y: NORMALIZED target array (N, H, W, 1) - mean≈0, std≈1
            augment: Apply data augmentation
            transform: Additional transforms
        """
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)  # (N, C, H, W)
        self.y = torch.FloatTensor(y).permute(0, 3, 1, 2)  # (N, 1, H, W)
        self.augment = augment
        self.transform = transform
        
        logger.info(f"Dataset created - X range: [{self.X.min():.4f}, {self.X.max():.4f}]")
        logger.info(f"Dataset created - y range: [{self.y.min():.4f}, {self.y.max():.4f}]")
        logger.info(f"Dataset created - y mean: {self.y.mean():.4f}, y std: {self.y.std():.4f}")
        
        # Verify normalization
        if abs(self.y.mean()) > 0.5:
            logger.warning(f"⚠️ Target mean={self.y.mean():.4f} is far from 0 - data might not be normalized!")
        if not (0.5 < self.y.std() < 1.5):
            logger.warning(f"⚠️ Target std={self.y.std():.4f} is far from 1 - data might not be normalized!")
        
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
        if torch.rand(1) > 0.5 and AUGMENTATION_CONFIG["flip_horizontal"]:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
        
        if torch.rand(1) > 0.5 and AUGMENTATION_CONFIG["flip_vertical"]:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        
        if torch.rand(1) > 0.5:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[1, 2])
        
        if AUGMENTATION_CONFIG["noise_std"] > 0:
            noise = torch.randn_like(x) * AUGMENTATION_CONFIG["noise_std"]
            x = x + noise
        
        if AUGMENTATION_CONFIG["brightness_range"] > 0:
            brightness = 1.0 + (torch.rand(1) - 0.5) * 2 * AUGMENTATION_CONFIG["brightness_range"]
            x = x * brightness
        
        return x, y


def prepare_gbm_features(X: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare tabular features for GBM from spatial data
    
    Args:
        X: NORMALIZED image patches (N, H, W, C) - mean≈0, std≈1
        y: NORMALIZED target LST (N, H, W, 1) - mean≈0, std≈1
    
    Returns:
        features_df: DataFrame with aggregated features per patch (normalized)
        targets: Flattened targets (normalized)
    
    Note:
        Both inputs and outputs are in NORMALIZED space.
        GBM will learn in normalized space.
        Predictions must be denormalized during evaluation.
    """
    logger.info("Preparing GBM features from spatial data...")
    
    # ... rest of function stays the same
    
    n_samples, height, width, n_channels = X.shape
    
    # Extract per-patch statistics for each channel
    features_list = []
    
    for i in range(n_samples):
        patch_features = {}
        
        for ch in range(n_channels):
            channel_data = X[i, :, :, ch]
            
            # Statistical features
            patch_features[f'ch{ch}_mean'] = channel_data.mean()
            patch_features[f'ch{ch}_std'] = channel_data.std()
            patch_features[f'ch{ch}_min'] = channel_data.min()
            patch_features[f'ch{ch}_max'] = channel_data.max()
            patch_features[f'ch{ch}_median'] = np.median(channel_data)
            
            # Percentiles
            patch_features[f'ch{ch}_p25'] = np.percentile(channel_data, 25)
            patch_features[f'ch{ch}_p75'] = np.percentile(channel_data, 75)
        
        # Spatial features
        patch_features['height'] = height
        patch_features['width'] = width
        
        features_list.append(patch_features)
    
    features_df = pd.DataFrame(features_list)
    
    # Flatten targets (use mean LST per patch)
    targets = y.reshape(n_samples, -1).mean(axis=1)
    
    logger.info(f"GBM features shape: {features_df.shape}")
    logger.info(f"GBM targets shape: {targets.shape}")
    
    return features_df, targets


class GBMTrainer:
    """Trainer for Gradient Boosting Model"""
    
    def __init__(self, config=None):
        self.config = config or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 127,
            "max_depth": 12,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_child_samples": 100,
            "verbose": -1
        }
        self.model = None
        
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame, y_val: np.ndarray):
        """Train GBM model"""
        logger.info("Training GBM model...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        self.model = lgb.train(
            self.config,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        logger.info(f"GBM training complete. Best iteration: {self.model.best_iteration}")
        
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def save(self, path: Path):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved GBM model to {path}")
    
    def load(self, path: Path):
        """Load model"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded GBM model from {path}")

def create_temperature_stratified_sampler(y_train):
    """
    Create sampler that balances temperature ranges
    Ensures model sees equal amounts of hot and cold samples
    
    Args:
        y_train: Training targets (N, H, W, 1) - NORMALIZED
    
    Returns:
        WeightedRandomSampler
    """
    from torch.utils.data import WeightedRandomSampler
    
    # Get mean temperature per sample (in normalized space)
    sample_means = y_train.reshape(len(y_train), -1).mean(axis=1).flatten()
    
    # Define temperature bins
    # In normalized space: -1.5 = very cold, 0 = average, +1.5 = very hot
    bins = np.array([-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf])
    bin_indices = np.digitize(sample_means, bins)
    
    # Count samples per bin
    unique_bins, bin_counts = np.unique(bin_indices, return_counts=True)
    
    # Calculate weights (inverse frequency)
    # Rare bins get higher weight
    bin_weights = {bin_idx: len(sample_means) / count 
                   for bin_idx, count in zip(unique_bins, bin_counts)}
    
    # Assign weight to each sample
    sample_weights = np.array([bin_weights[bin_idx] for bin_idx in bin_indices])
    
    logger.info("Temperature-stratified sampling:")
    for bin_idx, count in zip(unique_bins, bin_counts):
        logger.info(f"  Bin {bin_idx}: {count} samples (weight: {bin_weights[bin_idx]:.2f})")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

class EnsembleTrainer:
    """Ensemble trainer combining CNN and GBM"""
    
    def __init__(self, cnn_model, device, config=TRAINING_CONFIG):
        self.cnn_model = cnn_model.to(device)
        self.device = device
        self.config = config
        
        # CNN components
        from models import VarianceAwareLoss
        self.criterion = VarianceAwareLoss(
            mse_weight=0.7,      # Main loss
            var_weight=0.2,      # Variance preservation
            spatial_weight=0.1   # Spatial smoothness
        )
        
        if config["optimizer"] == "adamw":
            self.optimizer = optim.AdamW(
                cnn_model.parameters(),
                lr=config["initial_lr"],
                weight_decay=config["weight_decay"]
            )
        else:
            self.optimizer = optim.Adam(
                cnn_model.parameters(),
                lr=config["initial_lr"]
            )
        
        self.scheduler = self._create_scheduler()
        self.early_stopping = EarlyStopping(patience=config["patience"], mode="min")
        
        # GBM trainer
        self.gbm_trainer = GBMTrainer()
        self.gbm_trained = False
        
        # Ensemble weights from config
        self.ensemble_weights = ENSEMBLE_WEIGHTS
        
        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "cnn_metrics": [],
            "gbm_metrics": [],
            "ensemble_metrics": [],
            "lr": []
        }
        
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        warmup_epochs = self.config["warmup_epochs"]
        total_epochs = self.config["epochs"]
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_cnn_epoch(self, train_loader):
        """Train CNN for one epoch"""
        self.cnn_model.train()
        total_loss = 0
        loss_components = {}
        
        pbar = tqdm(train_loader, desc="Training CNN")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.cnn_model(data)
            
            if batch_idx == 0:
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logger.error(f"NaN/Inf detected in CNN output!")
                logger.info(f"CNN Batch 0 - Output range: [{output.min():.4f}, {output.max():.4f}]")
            
            loss, components = self.criterion(output, target, data)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at batch {batch_idx}!")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value
            
            pbar.set_postfix({"loss": loss.item(), "mse": components["mse"]})
        
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def evaluate_cnn(self, val_loader):
        """Evaluate CNN model"""
        self.cnn_model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Evaluating CNN"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.cnn_model(data)
                loss, _ = self.criterion(output, target, data)
                
                total_loss += loss.item()
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        preds = np.concatenate(all_preds, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()
        
        # Remove NaN values
        mask = ~np.isnan(targets)
        preds = preds[mask]
        targets = targets[mask]
        
        metrics = self._calculate_metrics(preds, targets, "CNN")
        
        return avg_loss, metrics, all_preds
    
    def train_gbm(self, X_train, y_train, X_val, y_val):
        """Train GBM model"""
        logger.info("\n" + "="*60)
        logger.info("TRAINING GBM MODEL")
        logger.info("="*60)
        
        # Prepare features
        X_train_gbm, y_train_gbm = prepare_gbm_features(X_train, y_train)
        X_val_gbm, y_val_gbm = prepare_gbm_features(X_val, y_val)
        
        # Train
        self.gbm_trainer.train(X_train_gbm, y_train_gbm, X_val_gbm, y_val_gbm)
        self.gbm_trained = True
        
        # Evaluate
        train_preds = self.gbm_trainer.predict(X_train_gbm)
        val_preds = self.gbm_trainer.predict(X_val_gbm)
        
        train_metrics = self._calculate_metrics(train_preds, y_train_gbm, "GBM Train")
        val_metrics = self._calculate_metrics(val_preds, y_val_gbm, "GBM Val")
        
        logger.info(f"GBM Train Metrics - R²: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f}°C")
        logger.info(f"GBM Val Metrics - R²: {val_metrics['r2']:.4f}, RMSE: {val_metrics['rmse']:.4f}°C")
        
        return val_metrics
    
    def compute_optimal_weights(self, cnn_metrics, gbm_metrics):
        """
        Compute optimal weights based on model performance
        Uses inverse RMSE weighting
        """
        cnn_rmse = cnn_metrics['rmse']
        gbm_rmse = gbm_metrics['rmse']
        
        # If CNN is terrible (R² < 0.2), don't use it
        if cnn_metrics['r2'] < 0.2:
            logger.warning("⚠️ CNN R² < 0.2, using GBM only")
            return {"cnn": 0.0, "gbm": 1.0}
        
        # If GBM is terrible (shouldn't happen), don't use it
        if gbm_metrics['r2'] < 0.2:
            logger.warning("⚠️ GBM R² < 0.2, using CNN only")
            return {"cnn": 1.0, "gbm": 0.0}
        
        # Inverse RMSE weighting
        cnn_weight = (1 / cnn_rmse)
        gbm_weight = (1 / gbm_rmse)
        
        total = cnn_weight + gbm_weight
        cnn_weight = cnn_weight / total
        gbm_weight = gbm_weight / total
        
        logger.info(f"Optimal weights computed:")
        logger.info(f"  CNN: {cnn_weight:.4f} (RMSE: {cnn_rmse:.4f}°C, R²: {cnn_metrics['r2']:.4f})")
        logger.info(f"  GBM: {gbm_weight:.4f} (RMSE: {gbm_rmse:.4f}°C, R²: {gbm_metrics['r2']:.4f})")
        
        return {"cnn": cnn_weight, "gbm": gbm_weight}
    
    def diagnose_cnn_issues(self):
        """
        Diagnose CNN performance issues
        """
        logger.info("\n" + "="*60)
        logger.info("DIAGNOSING CNN ISSUES")
        logger.info("="*60)
        
        # Check if CNN is learning
        if len(self.history["train_loss"]) > 5:
            initial_loss = self.history["train_loss"][0]
            final_loss = self.history["train_loss"][-1]
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            logger.info(f"Training progress:")
            logger.info(f"  Initial loss: {initial_loss:.4f}")
            logger.info(f"  Final loss: {final_loss:.4f}")
            logger.info(f"  Improvement: {improvement:.2f}%")
            
            if improvement < 10:
                logger.warning("⚠️ CNN barely improved - possible issues:")
                logger.warning("  1. Learning rate too low")
                logger.warning("  2. Model stuck in local minimum")
                logger.warning("  3. Data preprocessing issues")
        
        # Check CNN validation performance trend
        if len(self.history["cnn_metrics"]) > 0:
            r2_scores = [m['r2'] for m in self.history["cnn_metrics"]]
            logger.info(f"\nCNN R² progression (last 5): {[f'{r:.4f}' for r in r2_scores[-5:]]}")
            
            best_r2 = max(r2_scores)
            logger.info(f"Best CNN R² achieved: {best_r2:.4f}")
            
            if best_r2 < 0.5:
                logger.error("❌ CNN R² never exceeded 0.5 - CRITICAL ISSUE")
                logger.error("Recommendations:")
                logger.error("  1. Check input data normalization")
                logger.error("  2. Reduce model complexity (fewer layers)")
                logger.error("  3. Increase learning rate (current: {:.6f})".format(self.config["initial_lr"]))
                logger.error("  4. Try training longer")
                logger.error("  5. Check for data leakage/preprocessing bugs")
                logger.error("  6. Verify target values are in reasonable range")
        
        logger.info("="*60)
    
    def evaluate_ensemble(self, val_loader, X_val, y_val):
        """
        Evaluate ensemble of CNN + GBM with improved strategy
        """
        logger.info("\n" + "="*60)
        logger.info("EVALUATING ENSEMBLE PREDICTIONS")
        logger.info("="*60)
        
        # Get CNN predictions
        _, cnn_metrics, cnn_preds_list = self.evaluate_cnn(val_loader)
        cnn_preds = np.concatenate(cnn_preds_list, axis=0)
        cnn_preds_patch = cnn_preds.reshape(cnn_preds.shape[0], -1).mean(axis=1)
        
        # Get GBM predictions
        if self.gbm_trained:
            X_val_gbm, y_val_gbm = prepare_gbm_features(X_val, y_val)
            gbm_preds = self.gbm_trainer.predict(X_val_gbm)
            gbm_metrics = self._calculate_metrics(gbm_preds, y_val_gbm, "GBM")
            
            # CRITICAL FIX 1: Analyze prediction scales
            cnn_mean, cnn_std = cnn_preds_patch.mean(), cnn_preds_patch.std()
            gbm_mean, gbm_std = gbm_preds.mean(), gbm_preds.std()
            target_mean, target_std = y_val_gbm.mean(), y_val_gbm.std()
            
            logger.info(f"\nPrediction scale analysis:")
            logger.info(f"  Target: mean={target_mean:.2f}°C, std={target_std:.2f}°C")
            logger.info(f"  CNN:    mean={cnn_mean:.2f}°C, std={cnn_std:.2f}°C")
            logger.info(f"  GBM:    mean={gbm_mean:.2f}°C, std={gbm_std:.2f}°C")
            
            # Check if we need normalization
            scale_diff_cnn = abs(cnn_std - target_std) / target_std
            scale_diff_gbm = abs(gbm_std - target_std) / target_std
            
            if scale_diff_cnn > 0.3 or scale_diff_gbm > 0.3:
                logger.info(f"\n⚠️ Large scale differences detected, applying normalization...")
                
                # Standardize predictions
                cnn_preds_normalized = (cnn_preds_patch - cnn_mean) / (cnn_std + 1e-8)
                gbm_preds_normalized = (gbm_preds - gbm_mean) / (gbm_std + 1e-8)
                
                # CRITICAL FIX 2: Use optimal weights
                optimal_weights = self.compute_optimal_weights(cnn_metrics, gbm_metrics)
                
                # Ensemble in normalized space
                ensemble_normalized = (
                    optimal_weights["cnn"] * cnn_preds_normalized +
                    optimal_weights["gbm"] * gbm_preds_normalized
                )
                
                # Transform back to target scale
                ensemble_preds_normalized = ensemble_normalized * target_std + target_mean
                ensemble_metrics_normalized = self._calculate_metrics(
                    ensemble_preds_normalized, y_val_gbm, "Ensemble (Normalized)"
                )
            else:
                logger.info("\n✓ Scales are similar, normalization not needed")
                optimal_weights = self.compute_optimal_weights(cnn_metrics, gbm_metrics)
                ensemble_preds_normalized = (
                    optimal_weights["cnn"] * cnn_preds_patch +
                    optimal_weights["gbm"] * gbm_preds
                )
                ensemble_metrics_normalized = self._calculate_metrics(
                    ensemble_preds_normalized, y_val_gbm, "Ensemble (Optimal)"
                )
            
            # CRITICAL FIX 3: Compare strategies
            # Strategy 1: Original fixed weights
            fixed_preds = (
                self.ensemble_weights["cnn"] * cnn_preds_patch +
                self.ensemble_weights["gbm"] * gbm_preds
            )
            fixed_metrics = self._calculate_metrics(fixed_preds, y_val_gbm, "Ensemble (Fixed)")
            
            # Print comparison
            logger.info("\n" + "="*60)
            logger.info("ENSEMBLE STRATEGY COMPARISON")
            logger.info("="*60)
            logger.info(f"{'Strategy':<25} {'R²':<10} {'RMSE (°C)':<12} {'MAE (°C)':<12}")
            logger.info("-"*60)
            logger.info(f"{'CNN Only':<25} {cnn_metrics['r2']:<10.4f} {cnn_metrics['rmse']:<12.4f} {cnn_metrics['mae']:<12.4f}")
            logger.info(f"{'GBM Only':<25} {gbm_metrics['r2']:<10.4f} {gbm_metrics['rmse']:<12.4f} {gbm_metrics['mae']:<12.4f}")
            logger.info("-"*60)
            logger.info(f"{'Fixed Weights':<25} {fixed_metrics['r2']:<10.4f} {fixed_metrics['rmse']:<12.4f} {fixed_metrics['mae']:<12.4f}")
            logger.info(f"{'Optimal + Normalized':<25} {ensemble_metrics_normalized['r2']:<10.4f} {ensemble_metrics_normalized['rmse']:<12.4f} {ensemble_metrics_normalized['mae']:<12.4f}")
            logger.info("="*60)
            
            # Determine best approach
            all_results = [
                ("CNN Only", cnn_metrics),
                ("GBM Only", gbm_metrics),
                ("Fixed Ensemble", fixed_metrics),
                ("Optimal Ensemble", ensemble_metrics_normalized)
            ]
            
            best_name, best_metrics = max(all_results, key=lambda x: x[1]['r2'])
            
            logger.info(f"\n🏆 BEST APPROACH: {best_name}")
            logger.info(f"   R²: {best_metrics['r2']:.4f}")
            logger.info(f"   RMSE: {best_metrics['rmse']:.4f}°C")
            logger.info(f"   MAE: {best_metrics['mae']:.4f}°C")
            
            if best_name == "Optimal Ensemble":
                logger.info(f"   Weights: CNN={optimal_weights['cnn']:.3f}, GBM={optimal_weights['gbm']:.3f}")
                self.ensemble_weights = optimal_weights
                final_metrics = ensemble_metrics_normalized
            elif best_name == "GBM Only":
                logger.warning("\n⚠️ GBM alone performs best - ensemble doesn't help")
                logger.warning("   Consider using GBM only or improving CNN performance")
                self.ensemble_weights = {"cnn": 0.0, "gbm": 1.0}
                final_metrics = gbm_metrics
            elif best_name == "CNN Only":
                logger.warning("\n⚠️ CNN alone performs best - GBM doesn't help")
                self.ensemble_weights = {"cnn": 1.0, "gbm": 0.0}
                final_metrics = cnn_metrics
            else:
                final_metrics = fixed_metrics
            
            # Calculate improvement
            baseline_r2 = max(cnn_metrics['r2'], gbm_metrics['r2'])
            ensemble_r2 = ensemble_metrics_normalized['r2']
            improvement = (ensemble_r2 - baseline_r2) / abs(baseline_r2) * 100 if baseline_r2 != 0 else 0
            
            if improvement > 1:
                logger.info(f"\n✅ Ensemble improves over best individual model by {improvement:.2f}%")
            elif improvement < -1:
                logger.warning(f"\n⚠️ Ensemble is {abs(improvement):.2f}% worse than best individual model")
            else:
                logger.info(f"\n➡️  Ensemble performance similar to best individual model")
            
            logger.info("="*60)
            
            return final_metrics
        else:
            logger.warning("GBM not trained, using CNN metrics only")
            return cnn_metrics
    
    def _calculate_metrics(self, preds, targets, name=""):
        """Calculate evaluation metrics"""
        
        # Load normalization stats for denormalization
        norm_stats = load_normalization_stats()
        
        # Denormalize predictions and targets to Celsius for metrics
        if norm_stats is not None:
            preds_denorm = denormalize_predictions(preds, norm_stats)
            targets_denorm = denormalize_predictions(targets, norm_stats)
            
            logger.debug(f"{name} - Denormalized: pred mean={preds_denorm.mean():.2f}°C, "
                        f"target mean={targets_denorm.mean():.2f}°C")
        else:
            logger.warning(f"{name} - Using normalized values (no stats found)")
            preds_denorm = preds
            targets_denorm = targets
        
        # Use denormalized values for all metrics
        pred_var = np.var(preds_denorm)
        target_var = np.var(targets_denorm)
        
        if name and pred_var < 1e-8:
            logger.warning(f"⚠️ {name} predictions have near-zero variance!")
        
        try:
            r2 = r2_score(targets_denorm, preds_denorm)
        except Exception as e:
            logger.error(f"Error calculating R² for {name}: {e}")
            r2 = 0.0
        
        rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
        mae = mean_absolute_error(targets_denorm, preds_denorm)
        mbe = np.mean(preds_denorm - targets_denorm)
        
        return {"r2": r2, "rmse": rmse, "mae": mae, "mbe": mbe}
    
    def train(self, train_loader, val_loader, X_train, y_train, X_val, y_val, save_dir: Path):
        """Full training loop for ensemble"""
        logger.info(f"Starting ensemble training for {self.config['epochs']} epochs")
        logger.info(f"CNN parameters: {count_parameters(self.cnn_model):,}")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Train GBM first (independent of CNN)
        gbm_metrics = self.train_gbm(X_train, y_train, X_val, y_val)
        
        # Train CNN
        logger.info("\n" + "="*60)
        logger.info("TRAINING CNN MODEL")
        logger.info("="*60)
        
        best_cnn_r2 = -float('inf')
        
        for epoch in range(self.config["epochs"]):
            logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            # Train CNN
            train_loss, train_components = self.train_cnn_epoch(train_loader)
            
            # Validate CNN
            val_loss, cnn_metrics, _ = self.evaluate_cnn(val_loader)
            
            # Track best CNN performance
            if cnn_metrics['r2'] > best_cnn_r2:
                best_cnn_r2 = cnn_metrics['r2']
                logger.info(f"✅ New best CNN R²: {best_cnn_r2:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Log metrics
            logger.info(f"CNN Train Loss: {train_loss:.4f}")
            logger.info(f"CNN Val Loss: {val_loss:.4f}")
            logger.info(f"CNN Val Metrics - R²: {cnn_metrics['r2']:.4f}, "
                       f"RMSE: {cnn_metrics['rmse']:.4f}°C, "
                       f"MAE: {cnn_metrics['mae']:.4f}°C")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["cnn_metrics"].append(cnn_metrics)
            self.history["lr"].append(current_lr)
            
            # Early stopping based on CNN validation loss
            if self.early_stopping(val_loss, self.cnn_model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Diagnose CNN issues
        self.diagnose_cnn_issues()
        
        # Final ensemble evaluation with improved method
        ensemble_metrics = self.evaluate_ensemble(val_loader, X_val, y_val)
        self.history["ensemble_metrics"] = ensemble_metrics
        
        # Save models
        self._save_final_models(save_dir)
        self._save_history(save_dir)
        
        logger.info("\nEnsemble training complete!")
        return self.history
    
    def _save_checkpoint(self, save_dir, epoch, loss, metrics):
        """Save checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "cnn_state_dict": self.cnn_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "metrics": metrics,
            "ensemble_weights": self.ensemble_weights
        }
        
        path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def _save_final_models(self, save_dir):
        """Save final models"""
        # Save CNN
        if self.early_stopping.best_model is not None:
            torch.save(self.early_stopping.best_model, save_dir / "best_cnn.pth")
            logger.info("Saved best CNN model")
        
        torch.save(self.cnn_model.state_dict(), save_dir / "final_cnn.pth")
        logger.info("Saved final CNN model")
        
        # Save GBM
        if self.gbm_trained:
            self.gbm_trainer.save(save_dir / "gbm_model.pkl")
        
        # Save ensemble config
        ensemble_config = {
            "weights": self.ensemble_weights,
            "cnn_path": "final_cnn.pth",
            "gbm_path": "gbm_model.pkl" if self.gbm_trained else None
        }
        
        with open(save_dir / "ensemble_config.json", "w") as f:
            json.dump(ensemble_config, f, indent=2)
        logger.info("Saved ensemble configuration")
    
    def _save_history(self, save_dir):
        """Save training history"""
        history_serializable = {}
        for key, values in self.history.items():
            if key in ["cnn_metrics", "gbm_metrics"]:
                history_serializable[key] = [
                    {k: float(v) for k, v in metrics.items()}
                    for metrics in values
                ]
            elif key == "ensemble_metrics":
                if isinstance(values, dict):
                    history_serializable[key] = {k: float(v) for k, v in values.items()}
            else:
                history_serializable[key] = [float(v) for v in values]
        
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history_serializable, f, indent=2)
        
        logger.info("Saved training history")


def validate_data_quality(X: np.ndarray, y: np.ndarray, split: str) -> bool:
    """Validate data quality before training"""
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
    
    # Calculate statistics
    X_mean = X.mean()
    X_std = X.std()
    y_mean = y.mean()
    y_std = y.std()
    y_min = y.min()
    y_max = y.max()
    
    # Log feature statistics
    logger.info(f"Features (X) statistics:")
    logger.info(f"  Mean: {X_mean:.4f}")
    logger.info(f"  Std:  {X_std:.4f}")
    logger.info(f"  Min:  {X.min():.4f}")
    logger.info(f"  Max:  {X.max():.4f}")
    
    # Log target statistics
    logger.info(f"Target (y) statistics:")
    logger.info(f"  Mean: {y_mean:.4f}")
    logger.info(f"  Std:  {y_std:.4f}")
    logger.info(f"  Min:  {y_min:.4f}")
    logger.info(f"  Max:  {y_max:.4f}")
    logger.info(f"  Unique values: {len(np.unique(y))}")
    
    # Check normalization for TRAINING split
    if split == "train":
        logger.info(f"\nNormalization checks (expecting mean≈0, std≈1):")
        
        if not (-0.2 < X_mean < 0.2):
            issues.append(f"X not properly normalized (mean={X_mean:.4f}, expected ≈0)")
            logger.warning(f"  ⚠️ X mean={X_mean:.4f} (should be ≈0)")
        else:
            logger.info(f"  ✅ X mean={X_mean:.4f}")
        
        if not (0.8 < X_std < 1.2):
            issues.append(f"X not properly normalized (std={X_std:.4f}, expected ≈1)")
            logger.warning(f"  ⚠️ X std={X_std:.4f} (should be ≈1)")
        else:
            logger.info(f"  ✅ X std={X_std:.4f}")
        
        if not (-0.2 < y_mean < 0.2):
            issues.append(f"y not properly normalized (mean={y_mean:.4f}, expected ≈0)")
            logger.warning(f"  ⚠️ y mean={y_mean:.4f} (should be ≈0)")
        else:
            logger.info(f"  ✅ y mean={y_mean:.4f}")
        
        if not (0.8 < y_std < 1.2):
            issues.append(f"y not properly normalized (std={y_std:.4f}, expected ≈1)")
            logger.warning(f"  ⚠️ y std={y_std:.4f} (should be ≈1)")
        else:
            logger.info(f"  ✅ y std={y_std:.4f}")
    else:
        # For val/test, just check they're in normalized space
        logger.info(f"\nValidation/Test data checks:")
        if abs(X_mean) > 5.0:
            logger.warning(f"  ⚠️ X mean={X_mean:.4f} seems too large for normalized data")
        if abs(y_mean) > 5.0:
            logger.warning(f"  ⚠️ y mean={y_mean:.4f} seems too large for normalized data")
    
    # Check variance
    if y_std < 0.1:
        issues.append(f"Target has very low variance (std={y_std:.4f})")
    
    if y_std < 1e-6:
        issues.append(f"Target has ZERO variance")
    
    # Report results
    if issues:
        logger.error(f"❌ DATA QUALITY ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            logger.error(f"  {i}. {issue}")
        return False
    else:
        logger.info(f"\n✅ Data quality checks passed")
        logger.info(f"{'='*60}\n")
        return True


def load_data(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed data"""
    data_dir = PROCESSED_DATA_DIR / "cnn_dataset" / split
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    
    logger.info(f"Loaded {split} data: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"  X range: [{X.min():.4f}, {X.max():.4f}]")
    logger.info(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
    
    if not validate_data_quality(X, y, split):
        raise ValueError(f"{split} data failed quality checks")
    
    return X, y


def main():
    """Main training script with ensemble"""
    logger.info("="*60)
    logger.info("URBAN HEAT ISLAND - ENSEMBLE TRAINING (IMPROVED)")
    logger.info("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() and COMPUTE_CONFIG["use_gpu"] else "cpu")
    logger.info(f"Using device: {device}")
    
    # Verify normalization stats exist
    logger.info("\n" + "="*60)
    logger.info("VERIFYING NORMALIZATION STATISTICS")
    logger.info("="*60)
    
    stats_path = PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
    if not stats_path.exists():
        logger.error("❌ Normalization stats not found!")
        logger.error(f"   Expected: {stats_path}")
        logger.error("   ")
        logger.error("   This file should be created during preprocessing.")
        logger.error("   Run preprocessing.py to create normalized data with statistics.")
        raise FileNotFoundError(f"Missing normalization stats: {stats_path}")
    
    # Load and display normalization stats
    import json
    with open(stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    logger.info(f"✅ Found normalization stats: {stats_path}")
    logger.info(f"   Features: {norm_stats.get('n_channels', 'N/A')} channels")
    
    if 'target' in norm_stats:
        target_mean = norm_stats['target']['mean']
        target_std = norm_stats['target']['std']
        logger.info(f"   Target LST: mean={target_mean:.2f}°C, std={target_std:.2f}°C")
        logger.info(f"   (These are the ORIGINAL values before normalization)")
    
    logger.info("="*60)
    
    # Load data
    logger.info("\n" + "="*60)
    logger.info("LOADING TRAINING DATA")
    logger.info("="*60)
    X_train, y_train = load_data("train")
    
    logger.info("\n" + "="*60)
    logger.info("LOADING VALIDATION DATA")
    logger.info("="*60)
    X_val, y_val = load_data("val")
    
    # Create datasets
    train_dataset = UHIDataset(X_train, y_train, augment=True)
    val_dataset = UHIDataset(X_val, y_val, augment=False)
    
    # Create stratified sampler for balanced temperature distribution
    sampler = create_temperature_stratified_sampler(y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        sampler=sampler,  # CHANGED: use sampler instead of shuffle
        # shuffle=True,   # REMOVED: don't use shuffle with sampler
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
    
    # Create CNN model
    logger.info("\nInitializing CNN model...")
    cnn_model = UNet(in_channels=CNN_CONFIG["input_channels"], out_channels=1)
    initialize_weights(cnn_model)
    logger.info(f"CNN parameters: {count_parameters(cnn_model):,}")
    
    # Create ensemble trainer
    logger.info("\nInitializing ensemble trainer...")
    ensemble_trainer = EnsembleTrainer(cnn_model, device)
    logger.info(f"Initial ensemble weights: CNN={ENSEMBLE_WEIGHTS['cnn']}, GBM={ENSEMBLE_WEIGHTS['gbm']}")
    
    # Train ensemble
    logger.info("\n" + "="*60)
    logger.info("STARTING ENSEMBLE TRAINING")
    logger.info("="*60)
    
    try:
        history = ensemble_trainer.train(
            train_loader, val_loader,
            X_train, y_train, X_val, y_val,
            MODEL_DIR
        )
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE TRAINING COMPLETE")
    logger.info("="*60)
    
    # Print final metrics
    if "ensemble_metrics" in history and history["ensemble_metrics"]:
        final_metrics = history["ensemble_metrics"]
        logger.info("\nFINAL ENSEMBLE METRICS (Denormalized to °C):")
        logger.info(f"  R² Score: {final_metrics['r2']:.4f} (target: ≥ {VALIDATION_CONFIG['targets']['r2']})")
        logger.info(f"  RMSE: {final_metrics['rmse']:.4f}°C (target: ≤ {VALIDATION_CONFIG['targets']['rmse']}°C)")
        logger.info(f"  MAE: {final_metrics['mae']:.4f}°C (target: ≤ {VALIDATION_CONFIG['targets']['mae']}°C)")
        logger.info(f"  MBE: {final_metrics['mbe']:.4f}°C")
        
        logger.info("\nNote: Metrics are calculated in DENORMALIZED space (actual °C)")
        logger.info("      Model trains on normalized data (mean≈0, std≈1)")
        logger.info("      Predictions are denormalized before computing metrics")
        
        # Check if targets met
        targets_met = (
            final_metrics['r2'] >= VALIDATION_CONFIG['targets']['r2'] and
            final_metrics['rmse'] <= VALIDATION_CONFIG['targets']['rmse'] and
            final_metrics['mae'] <= VALIDATION_CONFIG['targets']['mae']
        )
        
        if targets_met:
            logger.info("\n✅ ALL PERFORMANCE TARGETS MET!")
        else:
            logger.warning("\n⚠️ Some performance targets not met")
        
        logger.info("="*60)
    
    # Print comparison
    if history["cnn_metrics"]:
        cnn_final = history["cnn_metrics"][-1]
        logger.info("\nMODEL COMPARISON:")
        logger.info(f"  CNN Only - R²: {cnn_final['r2']:.4f}, RMSE: {cnn_final['rmse']:.4f}°C")
        if "ensemble_metrics" in history and history["ensemble_metrics"]:
            ens_final = history["ensemble_metrics"]
            logger.info(f"  Ensemble - R²: {ens_final['r2']:.4f}, RMSE: {ens_final['rmse']:.4f}°C")
            
            if cnn_final['r2'] > 0:
                improvement = (ens_final['r2'] - cnn_final['r2']) / abs(cnn_final['r2']) * 100
                logger.info(f"  Improvement: {improvement:+.2f}%")
        
        # Print final weights
        logger.info(f"\nFinal Ensemble Weights:")
        logger.info(f"  CNN: {ensemble_trainer.ensemble_weights['cnn']:.4f}")
        logger.info(f"  GBM: {ensemble_trainer.ensemble_weights['gbm']:.4f}")
        
        logger.info("="*60)


if __name__ == "__main__":
    main()