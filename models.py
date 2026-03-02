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
    """Convolutional block with BatchNorm, activation, and residual connection"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 dropout: float = 0.1, use_batchnorm: bool = True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm)
        
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual projection: needed when in_channels != out_channels
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        
    def forward(self, x):
        identity = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity          # residual shortcut
        out = self.relu(out)
        out = self.dropout(out)
        return out


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


class ASPPBlock(nn.Module):
    """
    Atrous Spatial Pyramid Pooling — captures multi-scale context at the
    bottleneck, critical for urban scenes where heat sources span 1–10 pixels.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 dilations: tuple = (1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Projection: fuse (len(dilations)+1) branches → out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        outs = [b(x) for b in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=(h, w), mode='bilinear', align_corners=False)
        outs.append(gp)
        return self.proj(torch.cat(outs, dim=1))


class UNet(nn.Module):
    """U-Net architecture for LST prediction (Landsat-only: 10 input channels)
    
    Improvements:
    - Residual connections in ConvBlock (better gradient flow)
    - ASPP bottleneck (multi-scale urban context)
    """
    
    def __init__(self, in_channels: int = 10, out_channels: int = 1):
        super().__init__()
        
        filters = CNN_CONFIG["filters"]
        dropout_rates = CNN_CONFIG["dropout_rates"]
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, filters[0], dropout_rates[0])
        self.enc2 = EncoderBlock(filters[0], filters[1], dropout_rates[1])
        self.enc3 = EncoderBlock(filters[1], filters[2], dropout_rates[2])
        self.enc4 = EncoderBlock(filters[2], filters[3], dropout_rates[3])
        
        # Bottleneck: ASPP replaces plain ConvBlock for multi-scale context
        self.bottleneck = ASPPBlock(filters[3], filters[4],
                                    dilations=(1, 2, 4, 8))
        
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
        
        # Bottleneck (ASPP)
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
    Enhanced loss function preventing variance collapse and range compression.

    FIX 4: Spatial smoothness penalty REMOVED — it was blurring sharp land-cover
            boundaries (e.g. building/water edges) which are physically real.
            Replaced with a gradient-alignment loss that penalises *disagreement*
            with the target's spatial gradients rather than raw smoothness.

    FIX 5: Temperature-weighted MSE — weights each pixel by how far its true
            temperature is from the patch mean, so extreme-temperature pixels
            (rooftops, industrial zones) contribute more to training than the
            densely-sampled mid-temperature range.  This attacks the observed
            heteroscedastic fan in residuals.
    """

    def __init__(self):
        super().__init__()

        # Phase weights — updated each epoch via set_training_progress()
        self.mse_weight      = 0.6
        self.variance_weight = 0.1
        self.range_weight    = 0.1
        self.bias_weight     = 0.1
        self.gradient_weight = 0.1   # FIX 4: replaces spatial smoothness

        self.current_epoch = 0

        logger.info("ProgressiveLSTLoss v2 initialised "
                    "(gradient-alignment + temperature-weighted MSE, no spatial smoothness)")

    def set_training_progress(self, epoch: int, total_epochs: int):
        """Update loss component weights based on training phase."""
        self.current_epoch = epoch
        progress = epoch / max(total_epochs, 1)

        if progress < 0.3:
            # Phase 1 – WARMUP: lead with weighted MSE; light regularisation
            self.mse_weight      = 0.55
            self.variance_weight = 0.18   # stronger than before to resist collapse early
            self.range_weight    = 0.15
            self.bias_weight     = 0.05
            self.gradient_weight = 0.07
            phase = "WARMUP"

        elif progress < 0.6:
            # Phase 2 – REFINEMENT: increase variance/range pressure
            self.mse_weight      = 0.40
            self.variance_weight = 0.25   # was 0.18
            self.range_weight    = 0.22   # was 0.18
            self.bias_weight     = 0.07
            self.gradient_weight = 0.06
            phase = "REFINEMENT"

        else:
            # Phase 3 – FINE-TUNING: max variance / range pressure
            self.mse_weight      = 0.33
            self.variance_weight = 0.30   # was 0.22
            self.range_weight    = 0.25   # was 0.20
            self.bias_weight     = 0.07
            self.gradient_weight = 0.05
            phase = "FINE-TUNING"

        if epoch % 20 == 0:
            logger.info(f"Loss phase: {phase}  (epoch {epoch}/{total_epochs})  "
                        f"weights: mse={self.mse_weight}, var={self.variance_weight}, "
                        f"range={self.range_weight}, bias={self.bias_weight}, "
                        f"grad={self.gradient_weight}")

    def forward(self, pred, target, features=None):
        """
        Args:
            pred:     (N, 1, H, W) normalised predictions
            target:   (N, 1, H, W) normalised targets
            features: unused, kept for API compatibility
        Returns:
            total_loss (scalar), components (dict)
        """
        components = {}

        # ── 1. Temperature-weighted MSE (FIX 5) ──────────────────────────────
        # Weight each pixel by |target - patch_mean| + 1, so extreme-temp pixels
        # (hot rooftops, cool water) receive more gradient signal than the
        # densely-sampled mid-range pixels.  Weights are normalised so the
        # effective learning rate is unchanged on average.
        with torch.no_grad():
            patch_mean = target.mean(dim=[2, 3], keepdim=True)   # (N,1,1,1)
            weights    = (target - patch_mean).abs() + 1.0       # ≥ 1 everywhere
            weights    = weights / weights.mean()                 # mean = 1 → same LR scale

        mse = (weights * (pred - target) ** 2).mean()
        components['mse'] = mse.item()

        # ── 2. Variance preservation (asymmetric) ─────────────────────────────
        # Penalise under-variance 3× more than over-variance, since the observed
        # failure mode is pred_std < target_std (compression to mean).
        pred_std   = pred.std()   + 1e-8
        target_std = target.std() + 1e-8
        std_ratio  = pred_std / target_std
        # Asymmetric: under-variance (ratio < 1) costs 3× more
        sym_log     = torch.log(std_ratio) ** 2
        asym_boost  = torch.where(std_ratio < 1.0,
                                  torch.tensor(3.0, device=pred.device),
                                  torch.tensor(1.0, device=pred.device))
        variance_loss = sym_log * asym_boost
        components['variance']  = variance_loss.item()
        components['std_ratio'] = std_ratio.item()

        # ── 3. Slope penalty (range compression) — asymmetric ─────────────────
        pred_flat     = pred.flatten()
        target_flat   = target.flatten()
        pred_c        = pred_flat   - pred_flat.mean()
        target_c      = target_flat - target_flat.mean()
        cov           = (pred_c * target_c).mean()
        target_var    = (target_c ** 2).mean() + 1e-8
        slope         = cov / target_var
        # Asymmetric: penalise slope < 1 (under-spread) 3× harder than slope > 1
        raw_slope_loss = (slope - 1.0) ** 2
        asym_slope = torch.where(slope < 1.0,
                                 torch.tensor(3.0, device=pred.device),
                                 torch.tensor(1.0, device=pred.device))
        slope_loss    = raw_slope_loss * asym_slope
        components['slope'] = slope.item()
        components['range'] = slope_loss.item()

        # ── 4. Bias penalty ───────────────────────────────────────────────────
        bias      = (pred_flat.mean() - target_flat.mean()) / target_std
        bias_loss = bias ** 2
        components['bias'] = bias_loss.item()

        # ── 5. Gradient alignment (FIX 4 — replaces spatial smoothness) ──────
        # Penalise disagreement between predicted and target spatial gradients.
        # This preserves sharp edges instead of blurring them.
        pred_dx  = pred[:, :, :, 1:]  - pred[:, :, :, :-1]
        pred_dy  = pred[:, :, 1:, :]  - pred[:, :, :-1, :]
        tgt_dx   = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy   = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_loss = F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)
        components['gradient'] = grad_loss.item()

        # ── Distribution tracking ─────────────────────────────────────────────
        components['pred_std']   = pred_std.item()
        components['target_std'] = target_std.item()

        # ── Total ─────────────────────────────────────────────────────────────
        total = (
            self.mse_weight      * mse          +
            self.variance_weight * variance_loss +
            self.range_weight    * slope_loss    +
            self.bias_weight     * bias_loss     +
            self.gradient_weight * grad_loss
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
    """Test model architecture (Landsat-only: 10 channels)"""
    model = UNet(in_channels=10, out_channels=1)
    initialize_weights(model)
    
    # Test forward pass
    x = torch.randn(2, 10, 128, 128)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test loss
    criterion = ProgressiveLSTLoss()
    target = torch.randn(2, 1, 128, 128)
    loss, components = criterion(y, target)
    
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