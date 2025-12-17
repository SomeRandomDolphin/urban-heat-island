"""
Deep Learning models for Urban Heat Island detection
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

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


class LSTLoss(nn.Module):
    """Multi-task loss for LST prediction"""
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.2, gamma: float = 0.1):
        super().__init__()
        self.alpha = alpha  # MSE weight
        self.beta = beta    # Spatial weight
        self.gamma = gamma  # Physical weight
        
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
        """
        Physical consistency loss
        
        Args:
            pred: Predicted LST
            features: Input features (includes NDVI, etc.)
        """
        # For now, simple penalty if predictions are unrealistic
        # In practice, you'd extract NDVI from features and enforce relationships
        
        # Penalize unrealistic temperature values (20-50°C for Jakarta)
        min_temp, max_temp = 20.0, 50.0
        penalty = torch.relu(min_temp - pred) + torch.relu(pred - max_temp)
        return penalty.mean()
    
    def forward(self, pred, target, features=None):
        """
        Calculate total loss
        
        Args:
            pred: Predicted LST
            target: Ground truth LST
            features: Input features (optional, for physical loss)
        """
        mse = self.mse_loss(pred, target)
        spatial = self.spatial_loss(pred, target)
        
        total_loss = self.alpha * mse + self.beta * spatial
        
        if features is not None:
            physical = self.physical_loss(pred, features)
            total_loss += self.gamma * physical
        
        return total_loss, {
            "mse": mse.item(),
            "spatial": spatial.item(),
            "physical": physical.item() if features is not None else 0.0
        }


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_model = model.state_dict().copy()


class ModelEnsemble:
    """Ensemble of CNN and GBM models"""
    
    def __init__(self, cnn_model, gbm_model=None):
        self.cnn_model = cnn_model
        self.gbm_model = gbm_model
        self.weights = ENSEMBLE_WEIGHTS
        
    def predict(self, cnn_input, gbm_input=None):
        """
        Ensemble prediction
        
        Args:
            cnn_input: Input tensor for CNN
            gbm_input: Input dataframe for GBM (optional)
            
        Returns:
            Ensemble prediction
        """
        # CNN prediction
        self.cnn_model.eval()
        with torch.no_grad():
            cnn_pred = self.cnn_model(cnn_input)
        
        # If GBM available, combine predictions
        if self.gbm_model is not None and gbm_input is not None:
            gbm_pred = self.gbm_model.predict(gbm_input)
            
            # Convert to same shape and combine
            # This assumes appropriate reshaping logic
            final_pred = (self.weights["cnn"] * cnn_pred.cpu().numpy() + 
                         self.weights["gbm"] * gbm_pred)
        else:
            final_pred = cnn_pred
        
        return final_pred


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


if __name__ == "__main__":
    test_model()