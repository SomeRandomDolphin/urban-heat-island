"""
FIXED: UHI Inference Pipeline - Expects NORMALIZED data (same as training)
"""
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Optional, List
import json
import pickle

from config import *
from models import UNet

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class MCDropoutUNet(nn.Module):
    """UNet wrapper with properly implemented MC Dropout layers"""
    
    def __init__(self, base_model: UNet, dropout_rate: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self._add_dropout_layers()
        logger.info(f"✓ Added MC Dropout layers with rate={dropout_rate}")
        
    def _add_dropout_layers(self):
        """Recursively add dropout layers to the model"""
        def add_dropout_to_module(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                    setattr(module, name, nn.Sequential(
                        child,
                        nn.Dropout2d(p=self.dropout_rate)
                    ))
                else:
                    add_dropout_to_module(child)
        
        add_dropout_to_module(self.base_model)
    
    def forward(self, x):
        return self.base_model(x)
    
    def enable_mc_dropout(self):
        """Enable dropout for inference (MC Dropout)"""
        def enable_dropout(module):
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        
        self.eval()
        self.apply(enable_dropout)
    
    def disable_mc_dropout(self):
        """Disable dropout (standard inference)"""
        self.eval()


class DataNormalizer:
    """
    CRITICAL: Handles denormalization of predictions back to Celsius
    Inference now expects NORMALIZED inputs (same as training)
    """
    
    def __init__(self, stats_path: Path):
        """
        Load normalization statistics from training
        
        Args:
            stats_path: Path to normalization_stats.json
        """
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {stats_path}\n"
                f"These should be created during preprocessing."
            )
        
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        
        logger.info("✓ Loaded normalization statistics")
        logger.info(f"  Channels: {self.stats.get('n_channels', 'N/A')}")
        
        if 'target' in self.stats:
            logger.info(f"  Target LST: mean={self.stats['target']['mean']:.2f}°C, "
                       f"std={self.stats['target']['std']:.2f}°C")
    
    def denormalize_predictions(self, predictions: np.ndarray, 
                                prediction_type: str = "target") -> np.ndarray:
        """
        Denormalize predictions back to Celsius
        
        Args:
            predictions: Normalized predictions (mean≈0, std≈1)
            prediction_type: 'target' for CNN or 'gbm_target' for GBM
            
        Returns:
            Denormalized predictions in Celsius
        """
        if prediction_type not in self.stats:
            logger.warning(f"⚠️ '{prediction_type}' not in normalization stats, "
                          f"returning predictions as-is")
            return predictions
        
        stats = self.stats[prediction_type]
        mean = stats['mean']
        std = stats['std']
        
        denormalized = predictions * std + mean
        
        logger.debug(f"Denormalized {prediction_type}: "
                    f"mean={denormalized.mean():.2f}°C, "
                    f"std={denormalized.std():.2f}°C")
        
        return denormalized


class EnsemblePredictor:
    """
    FIXED: Ensemble predictor - Expects NORMALIZED input (same as training)
    Denormalization happens ONCE at the end, not multiple times
    """
    
    def __init__(self, 
                 cnn_model_path: Path, 
                 gbm_model_path: Path, 
                 ensemble_config_path: Path, 
                 normalization_stats_path: Path,
                 device: str = "cuda",
                 mc_dropout_rate: float = 0.1):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load normalization stats for denormalization
        self.normalizer = DataNormalizer(normalization_stats_path)
        
        # Load ensemble config
        with open(ensemble_config_path, 'r') as f:
            self.ensemble_config = json.load(f)
        
        self.weights = self.ensemble_config.get("weights", {"cnn": 0.5, "gbm": 0.5})
        
        logger.info(f"Ensemble weights - CNN: {self.weights['cnn']:.3f}, "
                   f"GBM: {self.weights['gbm']:.3f}")
        
        # Load base CNN model
        logger.info("Loading CNN model...")
        base_model = UNet(in_channels=CNN_CONFIG["input_channels"], out_channels=1)
        base_model.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
        
        # Wrap with MC Dropout
        self.cnn_model = MCDropoutUNet(base_model, dropout_rate=mc_dropout_rate)
        self.cnn_model.to(self.device)
        logger.info("✅ CNN model loaded with MC Dropout support")
        
        # Load GBM model
        logger.info("Loading GBM model...")
        with open(gbm_model_path, 'rb') as f:
            self.gbm_model = pickle.load(f)
        logger.info("✅ GBM model loaded")
    
    def validate_input(self, X: np.ndarray) -> bool:
        """Validate that input data is properly normalized"""
        logger.info("\n" + "="*70)
        logger.info("VALIDATING INPUT DATA")
        logger.info("="*70)
        
        X_mean = X.mean()
        X_std = X.std()
        
        logger.info(f"Input statistics:")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Mean: {X_mean:.4f}")
        logger.info(f"  Std:  {X_std:.4f}")
        logger.info(f"  Range: [{X.min():.4f}, {X.max():.4f}]")
        
        # Check if normalized
        is_normalized = (-0.5 < X_mean < 0.5) and (0.5 < X_std < 1.5)
        
        if is_normalized:
            logger.info("✅ Input data appears to be NORMALIZED (mean≈0, std≈1)")
            logger.info("  This is correct! Inference expects normalized data.")
            return True
        else:
            logger.warning("⚠️ Input data does NOT appear to be normalized!")
            logger.warning(f"  Mean={X_mean:.4f} (expected ≈0)")
            logger.warning(f"  Std={X_std:.4f} (expected ≈1)")
            return False
        
        logger.info("="*70 + "\n")
    
    def _predict_cnn_normalized(self, X: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """
        CNN predictions in NORMALIZED space
        
        Args:
            X: NORMALIZED features (N, H, W, C)
            
        Returns:
            Predictions in NORMALIZED space (DO NOT denormalize here)
        """
        logger.info("Running CNN predictions...")
        
        n_samples = X.shape[0]
        predictions = []
        
        self.cnn_model.disable_mc_dropout()
        
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), desc="CNN Inference"):
                batch = X[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).permute(0, 3, 1, 2).to(self.device)
                
                output = self.cnn_model(batch_tensor)
                predictions.append(output.cpu().numpy())
        
        # Concatenate predictions (KEEP NORMALIZED)
        predictions = np.concatenate(predictions, axis=0).squeeze(1)
        
        logger.info(f"  CNN raw predictions (normalized): "
                   f"mean={predictions.mean():.4f}, std={predictions.std():.4f}")
        
        return predictions  # Return NORMALIZED predictions
    
    def _predict_gbm_normalized(self, X: np.ndarray) -> np.ndarray:
        """
        GBM predictions in NORMALIZED space
        
        Args:
            X: NORMALIZED features (N, H, W, C)
            
        Returns:
            Predictions in NORMALIZED space (DO NOT denormalize here)
        """
        logger.info("Running GBM predictions...")
        
        features_list = []
        n_samples, height, width, n_channels = X.shape
        
        for i in tqdm(range(n_samples), desc="Extracting GBM features"):
            patch_features = {}
            
            for ch in range(n_channels):
                channel_data = X[i, :, :, ch]
                
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
        
        features_df = pd.DataFrame(features_list)
        
        # GBM prediction (KEEP NORMALIZED)
        predictions = self.gbm_model.predict(features_df, 
                                            num_iteration=self.gbm_model.best_iteration)
        
        logger.info(f"  GBM raw predictions (normalized): "
                   f"mean={predictions.mean():.4f}, std={predictions.std():.4f}")
        
        return predictions  # Return NORMALIZED predictions
    
    def predict_ensemble(self, X: np.ndarray, batch_size: int = 8, 
                        return_uncertainty: bool = False,
                        use_spatial_ensemble: bool = True,
                        skip_validation: bool = False) -> Dict[str, np.ndarray]:
        """
        Make ensemble predictions
        
        Args:
            X: NORMALIZED input features (N, H, W, C)
            batch_size: Batch size for inference
            return_uncertainty: Whether to compute MC Dropout uncertainty
            use_spatial_ensemble: Use spatial-level ensemble vs patch-level
            skip_validation: Skip input validation
            
        Returns:
            Dictionary with ensemble predictions in CELSIUS
        """
        logger.info("\n" + "="*70)
        logger.info("ENSEMBLE PREDICTION PIPELINE")
        logger.info("="*70)
        
        # Validate input
        if not skip_validation:
            if not self.validate_input(X):
                logger.error("❌ Input validation failed!")
        
        # Get predictions in NORMALIZED space
        cnn_preds_norm = self._predict_cnn_normalized(X, batch_size)
        gbm_preds_norm = self._predict_gbm_normalized(X)
        
        # Calculate patch-level averages (STILL NORMALIZED)
        cnn_preds_patch_norm = cnn_preds_norm.reshape(cnn_preds_norm.shape[0], -1).mean(axis=1)
        
        logger.info("\n📊 Prediction Statistics (NORMALIZED space):")
        logger.info(f"  CNN spatial: mean={cnn_preds_norm.mean():.4f}, std={cnn_preds_norm.std():.4f}")
        logger.info(f"  CNN patch avg: mean={cnn_preds_patch_norm.mean():.4f}, std={cnn_preds_patch_norm.std():.4f}")
        logger.info(f"  GBM patch: mean={gbm_preds_norm.mean():.4f}, std={gbm_preds_norm.std():.4f}")
        
        # Ensemble combination in NORMALIZED space
        if use_spatial_ensemble and self.weights["cnn"] > 0:
            logger.info("\n🔧 Using SPATIAL ensemble (normalized space)")
            
            # Broadcast GBM to spatial dimensions
            gbm_spatial_norm = np.zeros_like(cnn_preds_norm)
            for i in range(len(gbm_preds_norm)):
                gbm_spatial_norm[i] = gbm_preds_norm[i]
            
            # Weighted combination
            ensemble_preds_norm = (
                self.weights["cnn"] * cnn_preds_norm +
                self.weights["gbm"] * gbm_spatial_norm
            )
            
            ensemble_preds_patch_norm = ensemble_preds_norm.reshape(
                ensemble_preds_norm.shape[0], -1
            ).mean(axis=1)
            
        else:
            logger.info("\n🔧 Using PATCH-LEVEL ensemble (normalized space)")
            
            # Weighted combination at patch level
            ensemble_preds_patch_norm = (
                self.weights["cnn"] * cnn_preds_patch_norm +
                self.weights["gbm"] * gbm_preds_norm
            )
            
            # Broadcast to spatial
            ensemble_preds_norm = np.zeros_like(cnn_preds_norm)
            for i in range(len(ensemble_preds_patch_norm)):
                ensemble_preds_norm[i] = ensemble_preds_patch_norm[i]
        
        logger.info(f"\n📊 Ensemble (NORMALIZED): mean={ensemble_preds_norm.mean():.4f}, "
                   f"std={ensemble_preds_norm.std():.4f}")
        
        # DENORMALIZE ONCE - at the very end
        logger.info("\n🔄 Denormalizing predictions to Celsius...")
        
        cnn_preds = self.normalizer.denormalize_predictions(cnn_preds_norm, "target")
        gbm_preds = self.normalizer.denormalize_predictions(gbm_preds_norm, "target")
        cnn_preds_patch = self.normalizer.denormalize_predictions(cnn_preds_patch_norm, "target")
        ensemble_preds = self.normalizer.denormalize_predictions(ensemble_preds_norm, "target")
        ensemble_preds_patch = self.normalizer.denormalize_predictions(ensemble_preds_patch_norm, "target")
        
        logger.info(f"\n📈 Final Ensemble Statistics (CELSIUS):")
        logger.info(f"  Mean: {ensemble_preds.mean():.2f}°C")
        logger.info(f"  Std:  {ensemble_preds.std():.2f}°C")
        logger.info(f"  Range: [{ensemble_preds.min():.2f}, {ensemble_preds.max():.2f}]°C")
        
        results = {
            "ensemble": ensemble_preds,
            "cnn": cnn_preds,
            "gbm": gbm_preds,
            "ensemble_patch": ensemble_preds_patch,
            "cnn_patch": cnn_preds_patch
        }
        
        # Uncertainty estimation
        if return_uncertainty:
            logger.info("\n🎲 Computing MC Dropout uncertainty...")
            uncertainty = self._compute_mc_dropout_uncertainty(X, n_samples=50, batch_size=batch_size)
            results["uncertainty"] = uncertainty
        
        logger.info(f"\n✅ Ensemble predictions complete")
        logger.info("="*70 + "\n")
        
        return results
    
    def _compute_mc_dropout_uncertainty(self, X: np.ndarray, n_samples: int = 50, 
                                       batch_size: int = 8) -> np.ndarray:
        """Compute uncertainty using MC Dropout"""
        logger.info(f"  Running {n_samples} MC Dropout forward passes...")
        
        self.cnn_model.enable_mc_dropout()
        
        n_inputs = X.shape[0]
        mc_predictions = []
        
        for mc_iter in tqdm(range(n_samples), desc="MC Dropout samples"):
            predictions = []
            
            with torch.no_grad():
                for i in range(0, n_inputs, batch_size):
                    batch = X[i:i+batch_size]
                    batch_tensor = torch.FloatTensor(batch).permute(0, 3, 1, 2).to(self.device)
                    
                    output = self.cnn_model(batch_tensor)
                    predictions.append(output.cpu().numpy())
            
            preds = np.concatenate(predictions, axis=0).squeeze(1)
            
            # Denormalize each MC sample
            preds = self.normalizer.denormalize_predictions(preds, "target")
            mc_predictions.append(preds)
        
        self.cnn_model.disable_mc_dropout()
        
        # Calculate statistics
        mc_predictions = np.array(mc_predictions)
        mean_pred = np.mean(mc_predictions, axis=0)
        epistemic_uncertainty = np.std(mc_predictions, axis=0)
        
        logger.info(f"\n  MC Dropout Statistics:")
        logger.info(f"    Mean prediction: {mean_pred.mean():.2f}±{mean_pred.std():.2f}°C")
        logger.info(f"    Epistemic uncertainty: {epistemic_uncertainty.mean():.3f}°C")
        
        return epistemic_uncertainty


class PostProcessor:
    """Post-processing for LST predictions"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "bilateral_sigma_spatial": 5,
            "bilateral_sigma_range": 2.0,
            "temp_min": 20.0,
            "temp_max": 50.0
        }
    
    def apply_bilateral_filter(self, lst_map: np.ndarray) -> np.ndarray:
        """Apply bilateral filter to smooth while preserving edges"""
        import cv2
        
        logger.info("Applying bilateral filter...")
        
        if lst_map.std() < 0.5:
            logger.warning("⚠️ Input has low variance, skipping filtering")
            return lst_map
        
        lst_min, lst_max = lst_map.min(), lst_map.max()
        if lst_max - lst_min < 1.0:
            logger.warning("⚠️ Input range too small for filtering")
            return lst_map
        
        lst_normalized = ((lst_map - lst_min) / (lst_max - lst_min) * 255).astype(np.uint8)
        
        filtered = cv2.bilateralFilter(
            lst_normalized, 
            d=9,
            sigmaColor=self.config["bilateral_sigma_range"],
            sigmaSpace=self.config["bilateral_sigma_spatial"]
        )
        
        filtered = filtered.astype(np.float32) / 255.0 * (lst_max - lst_min) + lst_min
        
        logger.info(f"  Filtered: std={filtered.std():.2f}°C")
        return filtered
    
    def fill_nodata(self, lst_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Fill no-data pixels using spatial interpolation"""
        from scipy.interpolate import griddata
        
        if mask is None:
            mask = ~np.isnan(lst_map)
        
        if mask.all():
            return lst_map
        
        n_missing = (~mask).sum()
        logger.info(f"Filling {n_missing} no-data pixels ({n_missing/mask.size*100:.1f}%)...")
        
        y, x = np.indices(lst_map.shape)
        valid_coords = np.column_stack([x[mask], y[mask]])
        valid_values = lst_map[mask]
        invalid_coords = np.column_stack([x[~mask], y[~mask]])
        
        filled_values = griddata(
            valid_coords, valid_values, invalid_coords,
            method='cubic', fill_value=np.nanmean(lst_map)
        )
        
        filled_map = lst_map.copy()
        filled_map[~mask] = filled_values
        
        return filled_map
    
    def clip_values(self, lst_map: np.ndarray) -> np.ndarray:
        """Clip LST values to physically realistic range"""
        n_below = (lst_map < self.config["temp_min"]).sum()
        n_above = (lst_map > self.config["temp_max"]).sum()
        
        if n_below + n_above > 0:
            logger.info(f"Clipping {n_below + n_above} pixels to "
                       f"[{self.config['temp_min']}, {self.config['temp_max']}]°C")
        
        return np.clip(lst_map, self.config["temp_min"], self.config["temp_max"])
    
    def process(self, lst_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply full post-processing pipeline"""
        logger.info("Post-processing LST predictions...")
        logger.info(f"  Input: mean={lst_map.mean():.2f}°C, std={lst_map.std():.2f}°C")
        
        processed = self.fill_nodata(lst_map, mask)
        processed = self.apply_bilateral_filter(processed)
        processed = self.clip_values(processed)
        
        logger.info(f"  Output: mean={processed.mean():.2f}°C, std={processed.std():.2f}°C")
        logger.info("✓ Post-processing complete")
        
        return processed


def main():
    """Example usage of inference pipeline"""
    logger.info("="*70)
    logger.info("UHI INFERENCE PIPELINE - Expects NORMALIZED Data")
    logger.info("="*70)
    logger.info("Key features:")
    logger.info("  1. Expects NORMALIZED input (mean≈0, std≈1)")
    logger.info("  2. Same preprocessing as training")
    logger.info("  3. Automatic denormalization to Celsius")
    logger.info("  4. Input validation")
    logger.info("  5. MC Dropout uncertainty estimation")
    logger.info("="*70)
    
    # Define paths
    model_dir = MODEL_DIR
    cnn_model_path = model_dir / "final_cnn.pth"
    gbm_model_path = model_dir / "gbm_model.pkl"
    ensemble_config_path = model_dir / "ensemble_config.json"
    normalization_stats_path = PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
    
    # Check if all required files exist
    required_files = [
        cnn_model_path,
        gbm_model_path,
        ensemble_config_path,
        normalization_stats_path
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        logger.error("❌ Missing required files:")
        for f in missing_files:
            logger.error(f"  - {f}")
        logger.error("\nPlease run train_ensemble.py first!")
        return
    
    logger.info("✓ All required files found")
    
    # Initialize predictor
    predictor = EnsemblePredictor(
        cnn_model_path=cnn_model_path,
        gbm_model_path=gbm_model_path,
        ensemble_config_path=ensemble_config_path,
        normalization_stats_path=normalization_stats_path,
        device="cuda",
        mc_dropout_rate=0.05  # CHANGED: reduced from 0.1
    )
    
    logger.info("\n✓ Ensemble predictor initialized and ready for inference")
    logger.info("\nTo use this predictor:")
    logger.info("  # Load NORMALIZED test data")
    logger.info("  X_test = np.load('data/processed/cnn_dataset/test/X.npy')")
    logger.info("  ")
    logger.info("  # Make predictions (X_test should be normalized)")
    logger.info("  results = predictor.predict_ensemble(X_test, return_uncertainty=True)")
    logger.info("  ")
    logger.info("  # Results will be in Celsius (automatically denormalized)")
    logger.info("  ensemble_pred = results['ensemble']")


if __name__ == "__main__":
    main()