"""
Deep Learning models for Urban Heat Island detection with Ensemble support - FIXED VERSION
Key fixes:
1. Added bias penalty to LSTLoss
2. Added variance penalty to prevent constant predictions
3. Improved loss monitoring
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from config import CNN_CONFIG, ENSEMBLE_WEIGHTS

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 dropout: float = 0.1, use_batchnorm: bool = True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder block with downsampling"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv_block = ConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connections"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 
                                         kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels * 2, out_channels, dropout)
        
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', 
                            align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """U-Net architecture for LST prediction"""
    
    def __init__(self, in_channels: int = 15, out_channels: int = 1):
        super().__init__()
        
        filters = CNN_CONFIG["filters"]
        dropout_rates = CNN_CONFIG["dropout_rates"]
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, filters[0], dropout_rates[0])
        self.enc2 = EncoderBlock(filters[0], filters[1], dropout_rates[1])
        self.enc3 = EncoderBlock(filters[1], filters[2], dropout_rates[2])
        self.enc4 = EncoderBlock(filters[2], filters[3], dropout_rates[3])
        
        # Bottleneck
        self.bottleneck = ConvBlock(filters[3], filters[4], dropout_rates[4])
        
        # Decoder
        self.dec4 = DecoderBlock(filters[4], filters[3], dropout_rates[3])
        self.dec3 = DecoderBlock(filters[3], filters[2], dropout_rates[2])
        self.dec2 = DecoderBlock(filters[2], filters[1], dropout_rates[1])
        self.dec1 = DecoderBlock(filters[1], filters[0], dropout_rates[0])
        
        # Output
        self.output = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Output
        x = self.output(x)
        return x

class ProgressiveLSTLoss(nn.Module):
    """
    Loss function that adapts during training
    Early: Focus on MSE
    Middle: Add spatial smoothness
    Late: Add all constraints
    """
    
    def __init__(self):
        super().__init__()
        # Initial weights (MSE focus)
        self.mse_weight = 1.0
        self.variance_weight = 0.2
        self.bias_weight = 0.1
        self.spatial_weight = 0.0
        self.physical_weight = 0.0
        
        self.current_epoch = 0
        self.total_epochs = 200
        
        logger.info("ProgressiveLSTLoss initialized")
    
    def set_training_progress(self, epoch: int, total_epochs: int):
        """Update loss weights based on training progress"""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        progress = epoch / total_epochs
        
        if progress < 0.3:
            # Phase 1: MSE focus (0-30% of training)
            self.mse_weight = 1.0
            self.variance_weight = 0.2
            self.bias_weight = 0.1
            self.spatial_weight = 0.0
            self.physical_weight = 0.0
            phase = "WARMUP"
            
        elif progress < 0.7:
            # Phase 2: Add spatial (30-70% of training)
            self.mse_weight = 0.8
            self.variance_weight = 0.3
            self.bias_weight = 0.2
            self.spatial_weight = 0.1
            self.physical_weight = 0.05
            phase = "REFINEMENT"
            
        else:
            # Phase 3: Full complexity (70-100% of training)
            self.mse_weight = 0.7
            self.variance_weight = 0.3
            self.bias_weight = 0.2
            self.spatial_weight = 0.15
            self.physical_weight = 0.1
            phase = "FINE-TUNING"
        
        if epoch == 0 or (progress * 100) % 10 < (1.0 / total_epochs * 100):
            logger.info(f"Loss phase: {phase} (progress: {progress*100:.1f}%)")
    
    def forward(self, pred, target, features=None):
        """Calculate progressive loss"""
        components = {}
        
        # 1. MSE
        mse = F.mse_loss(pred, target)
        components['mse'] = mse.item()
        
        # 2. Variance preservation
        pred_var = torch.var(pred)
        target_var = torch.var(target)
        var_loss = (pred_var - target_var) ** 2
        components['variance'] = var_loss.item()
        
        # 3. Bias penalty
        bias_loss = (pred.mean() - target.mean()) ** 2
        components['bias'] = bias_loss.item()
        
        # 4. Spatial smoothness (only if weight > 0)
        spatial_loss = torch.tensor(0.0, device=pred.device)
        if self.spatial_weight > 0 and pred.shape[2] > 1 and pred.shape[3] > 1:
            dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            spatial_loss = torch.mean(dx**2) + torch.mean(dy**2)
        components['spatial'] = spatial_loss.item()
        
        # 5. Physical constraints (only if weight > 0)
        physical_loss = torch.tensor(0.0, device=pred.device)
        if self.physical_weight > 0:
            # Penalize extreme values (in normalized space: ±3 is ~3 std)
            min_norm, max_norm = -3.0, 3.0
            penalty = torch.relu(min_norm - pred) + torch.relu(pred - max_norm)
            physical_loss = penalty.mean()
        components['physical'] = physical_loss.item()
        
        # Total loss
        total = (
            self.mse_weight * mse +
            self.variance_weight * var_loss +
            self.bias_weight * bias_loss +
            self.spatial_weight * spatial_loss +
            self.physical_weight * physical_loss
        )
        
        return total, components

class LSTLoss(nn.Module):
    """
    Multi-task loss for LST prediction - FIXED VERSION
    
    New features:
    1. Bias penalty - forces mean prediction to match mean target
    2. Variance penalty - prevents constant predictions
    3. Better component tracking
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, 
                 gamma: float = 0.05, delta: float = 0.5, epsilon: float = 0.2):
        super().__init__()
        self.alpha = alpha      # MSE weight
        self.beta = beta        # Spatial weight
        self.gamma = gamma      # Physical weight
        self.delta = delta      # Bias weight (NEW)
        self.epsilon = epsilon  # Variance weight (NEW)
        
        logger.info(f"LSTLoss initialized with weights: MSE={alpha}, Spatial={beta}, "
                   f"Physical={gamma}, Bias={delta}, Variance={epsilon}")
        
    def mse_loss(self, pred, target):
        """Mean Squared Error"""
        mask = ~torch.isnan(target)
        return F.mse_loss(pred[mask], target[mask])
    
    def spatial_loss(self, pred, target):
        """Spatial smoothness loss"""
        # Calculate gradients
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # L1 loss on gradients
        loss_x = torch.abs(pred_grad_x - target_grad_x).mean()
        loss_y = torch.abs(pred_grad_y - target_grad_y).mean()
        
        return loss_x + loss_y
    
    def physical_loss(self, pred, features):
        """Physical consistency loss"""
        # Penalize unrealistic temperature values in NORMALIZED space
        # For normalized data: mean≈0, std≈1
        # Reasonable range: [-3, +3] (±3 standard deviations)
        min_norm, max_norm = -3.0, 3.0
        penalty = torch.relu(min_norm - pred) + torch.relu(pred - max_norm)
        return penalty.mean()
    
    def bias_loss(self, pred, target):
        """
        NEW: Bias penalty - forces mean prediction to match mean target
        This prevents systematic over/underprediction
        """
        mask = ~torch.isnan(target)
        pred_mean = pred[mask].mean()
        target_mean = target[mask].mean()
        
        # Squared difference of means
        bias = (pred_mean - target_mean) ** 2
        return bias
    
    def variance_loss(self, pred, target):
        """
        NEW: Variance penalty - prevents constant predictions
        Encourages prediction variance to match target variance
        """
        mask = ~torch.isnan(target)
        pred_var = pred[mask].var()
        target_var = target[mask].var()
        
        # Penalize if prediction variance is too low
        # Use relative difference
        if target_var > 1e-6:
            var_ratio = pred_var / (target_var + 1e-8)
            # Penalize if ratio < 0.5 (predictions too flat)
            variance_penalty = torch.relu(0.5 - var_ratio)
        else:
            variance_penalty = torch.tensor(0.0, device=pred.device)
        
        return variance_penalty
    
    def forward(self, pred, target, features=None):
        """Calculate total loss"""
        # Core losses
        mse = self.mse_loss(pred, target)
        spatial = self.spatial_loss(pred, target)
        
        # NEW: Bias and variance losses
        bias = self.bias_loss(pred, target)
        variance = self.variance_loss(pred, target)
        
        # Combine losses
        total_loss = (
            self.alpha * mse + 
            self.beta * spatial +
            self.delta * bias +
            self.epsilon * variance
        )
        
        # Physical loss (optional, based on features)
        physical = torch.tensor(0.0, device=pred.device)
        if features is not None:
            physical = self.physical_loss(pred, features)
            total_loss += self.gamma * physical
        
        # Return total loss and components for monitoring
        components = {
            "mse": mse.item(),
            "spatial": spatial.item(),
            "physical": physical.item(),
            "bias": bias.item(),           # NEW
            "variance": variance.item()    # NEW
        }
        
        return total_loss, components


class EarlyStopping:
    """Early stopping to prevent overfitting - IMPROVED"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta  # Minimum improvement required
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
        
        logger.info(f"EarlyStopping: patience={patience}, min_delta={min_delta}, mode={mode}")
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._is_improvement(score):
            improvement = abs(score - self.best_score)
            logger.info(f"  → Improvement: {improvement:.6f}")
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            logger.info(f"  → No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"  → Early stopping triggered!")
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}


class ModelEnsemble:
    """Ensemble of CNN and GBM models for production inference"""
    
    def __init__(self, cnn_model=None, gbm_model=None, weights=None, device='cpu'):
        """
        Initialize ensemble model
        
        Args:
            cnn_model: Trained CNN model (UNet)
            gbm_model: Trained GBM model (LightGBM)
            weights: Dictionary with 'cnn' and 'gbm' weights
            device: Device for CNN inference
        """
        self.cnn_model = cnn_model
        self.gbm_model = gbm_model
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.device = device
        
        if self.cnn_model is not None:
            self.cnn_model.to(device)
            self.cnn_model.eval()
        
        logger.info(f"Ensemble initialized with weights: {self.weights}")
    
    def _prepare_gbm_features(self, X: np.ndarray) -> pd.DataFrame:
        """Prepare GBM features from image patches"""
        n_samples, n_channels, height, width = X.shape
        
        features_list = []
        for i in range(n_samples):
            patch_features = {}
            
            for ch in range(n_channels):
                channel_data = X[i, ch, :, :]
                
                patch_features[f'ch{ch}_mean'] = channel_data.mean()
                patch_features[f'ch{ch}_std'] = channel_data.std()
                patch_features[f'ch{ch}_min'] = channel_data.min()
                patch_features[f'ch{ch}_max'] = channel_data.max()
                patch_features[f'ch{ch}_median'] = np.median(channel_data)
                patch_features[f'ch{ch}_p25'] = np.percentile(channel_data, 25)
                patch_features[f'ch{ch}_p75'] = np.percentile(channel_data, 75)
            
            patch_features['height'] = height
            patch_features['width'] = width
            
            features_list.append(patch_features)
        
        return pd.DataFrame(features_list)
    
    def predict(self, X, return_individual=False):
        """
        Make ensemble predictions
        
        Args:
            X: Input tensor/array (N, C, H, W) or (N, H, W, C)
            return_individual: If True, return individual model predictions
            
        Returns:
            Ensemble prediction, or (ensemble, cnn, gbm) if return_individual=True
        """
        # Convert to torch tensor if numpy
        if isinstance(X, np.ndarray):
            # Handle (N, H, W, C) -> (N, C, H, W)
            if X.ndim == 4 and X.shape[-1] < X.shape[1]:
                X = np.transpose(X, (0, 3, 1, 2))
            X_torch = torch.FloatTensor(X)
        else:
            X_torch = X
        
        # CNN prediction
        cnn_pred = None
        if self.cnn_model is not None:
            with torch.no_grad():
                X_device = X_torch.to(self.device)
                cnn_output = self.cnn_model(X_device)
                cnn_pred = cnn_output.cpu().numpy()
                
                # Average over spatial dimensions for patch-level prediction
                cnn_pred_patch = cnn_pred.reshape(cnn_pred.shape[0], -1).mean(axis=1)
        else:
            raise ValueError("CNN model not available")
        
        # GBM prediction
        gbm_pred = None
        if self.gbm_model is not None:
            X_numpy = X_torch.numpy()
            X_gbm = self._prepare_gbm_features(X_numpy)
            gbm_pred = self.gbm_model.predict(X_gbm)
        
        # Ensemble prediction
        if gbm_pred is not None:
            ensemble_pred = (
                self.weights["cnn"] * cnn_pred_patch +
                self.weights["gbm"] * gbm_pred
            )
        else:
            logger.warning("GBM not available, using CNN predictions only")
            ensemble_pred = cnn_pred_patch
        
        if return_individual:
            return ensemble_pred, cnn_pred, gbm_pred
        else:
            return ensemble_pred
    
    def predict_spatial(self, X):
        """
        Make spatial predictions (full resolution from CNN)
        
        Args:
            X: Input tensor/array (N, C, H, W)
            
        Returns:
            Spatial LST predictions (N, 1, H, W)
        """
        if self.cnn_model is None:
            raise ValueError("CNN model not available for spatial predictions")
        
        if isinstance(X, np.ndarray):
            if X.ndim == 4 and X.shape[-1] < X.shape[1]:
                X = np.transpose(X, (0, 3, 1, 2))
            X = torch.FloatTensor(X)
        
        with torch.no_grad():
            X = X.to(self.device)
            predictions = self.cnn_model(X)
        
        return predictions.cpu().numpy()
    
    @classmethod
    def load_from_directory(cls, model_dir: Path, device='cpu'):
        """
        Load ensemble from saved directory
        
        Args:
            model_dir: Directory containing saved models
            device: Device for CNN inference
            
        Returns:
            ModelEnsemble instance
        """
        model_dir = Path(model_dir)
        
        # Load ensemble config
        config_path = model_dir / "ensemble_config.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            weights = config.get('weights', ENSEMBLE_WEIGHTS)
        else:
            weights = ENSEMBLE_WEIGHTS
            logger.warning("No ensemble config found, using default weights")
        
        # Load CNN
        cnn_model = None
        cnn_path = model_dir / "final_cnn.pth"
        if not cnn_path.exists():
            cnn_path = model_dir / "best_cnn.pth"
        
        if cnn_path.exists():
            try:
                cnn_model = UNet(
                    in_channels=CNN_CONFIG["input_channels"],
                    out_channels=1
                )
                cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
                cnn_model.eval()
                logger.info(f"Loaded CNN model from {cnn_path}")
            except Exception as e:
                logger.error(f"Failed to load CNN: {e}")
        else:
            logger.warning("No CNN model found")
        
        # Load GBM
        gbm_model = None
        gbm_path = model_dir / "gbm_model.pkl"
        if gbm_path.exists():
            try:
                with open(gbm_path, 'rb') as f:
                    gbm_model = pickle.load(f)
                logger.info(f"Loaded GBM model from {gbm_path}")
            except Exception as e:
                logger.error(f"Failed to load GBM: {e}")
        else:
            logger.warning("No GBM model found")
        
        return cls(cnn_model, gbm_model, weights, device)
    
    def save(self, save_dir: Path):
        """Save ensemble models"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CNN
        if self.cnn_model is not None:
            torch.save(self.cnn_model.state_dict(), save_dir / "final_cnn.pth")
            logger.info("Saved CNN model")
        
        # Save GBM
        if self.gbm_model is not None:
            with open(save_dir / "gbm_model.pkl", 'wb') as f:
                pickle.dump(self.gbm_model, f)
            logger.info("Saved GBM model")
        
        # Save config
        import json
        config = {
            "weights": self.weights,
            "cnn_path": "final_cnn.pth",
            "gbm_path": "gbm_model.pkl" if self.gbm_model else None
        }
        with open(save_dir / "ensemble_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Saved ensemble configuration")


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model):
    """Initialize model weights"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


def test_model():
    """Test model architecture"""
    model = UNet(in_channels=15, out_channels=1)
    initialize_weights(model)
    
    # Test forward pass
    x = torch.randn(2, 15, 128, 128)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test loss
    criterion = LSTLoss()
    target = torch.randn(2, 1, 128, 128)
    loss, components = criterion(y, target, x)
    
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {components}")
    
    # Test ensemble
    print("\nTesting ensemble...")
    ensemble = ModelEnsemble(cnn_model=model, device='cpu')
    pred = ensemble.predict(x)
    print(f"Ensemble prediction shape: {pred.shape}")

class VarianceAwareLoss(nn.Module):
    """
    Loss function that preserves temperature variance
    Prevents the model from regressing to the mean
    """
    
    def __init__(self, mse_weight=0.7, var_weight=0.2, spatial_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.var_weight = var_weight
        self.spatial_weight = spatial_weight
        
    def forward(self, pred, target, features=None):
        """
        Args:
            pred: Predictions (N, 1, H, W) - NORMALIZED
            target: Ground truth (N, 1, H, W) - NORMALIZED
            features: Input features (optional, for physical constraints)
        
        Returns:
            loss, components dict
        """
        # 1. Standard MSE
        mse_loss = nn.functional.mse_loss(pred, target)
        
        # 2. Variance preservation loss
        # Penalize if prediction variance doesn't match target variance
        pred_var = torch.var(pred)
        target_var = torch.var(target)
        var_loss = (pred_var - target_var) ** 2
        
        # 3. Spatial smoothness (optional, from original LSTLoss)
        spatial_loss = 0.0
        if pred.shape[2] > 1 and pred.shape[3] > 1:
            # Gradient in x and y
            dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            spatial_loss = torch.mean(dx**2) + torch.mean(dy**2)
        
        # Total loss
        total_loss = (
            self.mse_weight * mse_loss +
            self.var_weight * var_loss +
            self.spatial_weight * spatial_loss
        )
        
        components = {
            "mse": mse_loss.item(),
            "variance": var_loss.item(),
            "spatial": spatial_loss.item() if isinstance(spatial_loss, torch.Tensor) else spatial_loss
        }
        
        return total_loss, components


if __name__ == "__main__":
    test_model()