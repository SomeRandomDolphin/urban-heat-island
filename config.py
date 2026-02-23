"""
IMPROVED Configuration file for Urban Heat Island Detection System
Changes focus on: reducing overfitting, better generalization, stronger regularization
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STUDY AREA CONFIGURATION (Jakarta)
# ============================================================================
STUDY_AREA = {
    "name": "Jakarta Metropolitan Area",
    "bounds": {
        "min_lon": 106.6,
        "max_lon": 107.1,
        "min_lat": -6.4,
        "max_lat": -6.0
    },
    "epsg": 32748,  # WGS84 / UTM Zone 48S
    "buffer_km": 5
}

# ============================================================================
# SATELLITE DATA CONFIGURATION
# ============================================================================
LANDSAT_CONFIG = {
    "collection": "LANDSAT/LC08/C02/T1_L2",
    "bands": {
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
        "swir2": "SR_B7",
        "thermal": "ST_B10",
        "qa": "QA_PIXEL"
    },
    "cloud_threshold": 50,  # percentage
    "scale": 30,  # meters
    "thermal_scale": 100  # meters (native resolution)
}

SENTINEL2_CONFIG = {
    "collection": "COPERNICUS/S2_SR_HARMONIZED",
    "bands": {
        "blue": "B2",
        "green": "B3",
        "red": "B4",
        "nir": "B8",
        "swir1": "B11",
        "swir2": "B12",
        "scl": "SCL"
    },
    "cloud_classes": [3, 8, 9, 10],  # Cloud shadow, cloud medium/high probability, cirrus
    "scale": 10  # meters
}

# Landsat 8 thermal constants
THERMAL_CONSTANTS = {
    "K1": 774.8853,  # Band 10
    "K2": 1321.0789,  # Band 10
    "wavelength": 10.9e-6,  # meters
    "rho": 1.438e-2  # m·K
}

# ============================================================================
# TEMPORAL CONFIGURATION - IMPROVED
# ============================================================================
# OLD: Temporal split caused distribution shift
# NEW: Random stratified split maintaining seasonal distribution
DATE_RANGE = {
    # Use all available data
    "all_data_start": "2021-01-01",
    "all_data_end": "2024-12-31",
    
    # Split ratios
    "train_ratio": 0.65,
    "val_ratio": 0.15,
    "test_ratio": 0.20,
    
    # Ensure seasonal balance
    "stratify_by_season": True,
    "stratify_by_location": True,  # Also stratify spatially
}

# ============================================================================
# GRID CONFIGURATION
# ============================================================================
GRID_CONFIG = {
    "resolution": 50,  # meters
    "patch_size": 64,  # pixels for CNN input
    "overlap": 32  # pixels overlap between patches
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
SPECTRAL_INDICES = [
    "NDVI",   # Normalized Difference Vegetation Index
    "NDBI",   # Normalized Difference Built-up Index
    "MNDWI",  # Modified Normalized Difference Water Index
    "BSI",    # Bare Soil Index
    "UI",     # Urban Index
    "EBBI",   # Enhanced Built-Up and Bareness Index
    "albedo"  # Surface albedo
]

URBAN_FEATURES = [
    "building_density",
    "road_density",
    "green_space_ratio",
    "impervious_surface",
    "sky_view_factor",
    "avg_building_height"
]

METEOROLOGICAL_FEATURES = [
    "air_temp",
    "wind_speed",
    "solar_radiation",
    "humidity",
    "pressure"
]

TEMPORAL_FEATURES = [
    "hour",
    "DOY_sin",
    "DOY_cos",
    "season"
]

# ============================================================================
# MODEL CONFIGURATION - IMPROVED TO REDUCE OVERFITTING
# ============================================================================
CNN_CONFIG = {
    "architecture": "unet",
    "input_channels": 10,
    
    # REDUCED CAPACITY: Prevent overfitting
    "filters": [16, 32, 64, 128, 256],  # Was [32, 64, 128, 256, 512]
    
    # INCREASED DROPOUT: Better regularization
    "dropout_rates": [0.3, 0.4, 0.5, 0.5, 0.5],  # Was [0.2, 0.25, 0.3, 0.35, 0.4]
    
    "batch_norm": True,
    "activation": "relu",
    "use_attention": False,  # Keep simple
}

GBM_CONFIG = {
    "algorithm": "lightgbm",
    "params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        
        # REDUCED TREE COMPLEXITY: Prevent overfitting
        "num_leaves": 31,  # Was 127 - much simpler
        "max_depth": 6,    # Was 12 - much shallower
        
        # LEARNING: Let early stopping find the right number of trees
        "learning_rate": 0.05,
        "n_estimators": 2000,   # Was 500 — model was hitting the cap, not converging
        
        # SAMPLING: More aggressive feature sampling to reduce remaining gap
        "subsample": 0.7,         # Was 0.6 — slight increase to stabilize with more trees
        "subsample_freq": 1,
        "colsample_bytree": 0.4,  # Was 0.6 — more aggressive, reduces memorization
        
        # REGULARIZATION
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        # Reduced from 200 — was too restrictive, forcing mean-regression (slope=0.833)
        "min_child_samples": 50,
        # Controls leaf weight instead of count — more precise than min_child_samples alone
        "min_child_weight": 1e-3,
        # Prune splits that don't earn their keep — directly fights range compression
        "min_split_gain": 0.01,
        
        # Convergence
        "early_stopping_rounds": 50,
        "verbose": 100,
    }
}

# ADJUSTED ENSEMBLE WEIGHTS: Favor more robust GBM
ENSEMBLE_WEIGHTS = {
    "cnn": 0.35,   # Was 0.5 - reduced (CNN overfits more)
    "gbm": 0.55,   # Was 0.4 - increased (GBM more robust)
    "baseline": 0.1
}

# ============================================================================
# TRAINING CONFIGURATION - IMPROVED
# ============================================================================
TRAINING_CONFIG = {
    # Batch size
    "batch_size": 16,  # Was 12 - larger for better gradient estimates
    
    # Training duration
    "epochs": 500,  # Was 200 - rely more on early stopping
    
    # Learning rate
    "initial_lr": 0.001,  # Was 0.0015 - slightly more conservative
    "min_lr": 1e-6,       # Was 5e-7
    "warmup_epochs": 10,  # Was 20 - faster warmup
    
    # STRICTER EARLY STOPPING: Stop before overfitting
    "patience": 50,       # Was 50 - stop earlier
    "min_delta": 0.0005,   # Was 0.0005 - higher threshold
    
    # Optimizer
    "optimizer": "adamw",
    "weight_decay": 0.001,  # Was 0.0001 - MUCH stronger weight decay
    
    # Loss weights - FOCUS ON PRESERVING VARIANCE AND RANGE
    "loss_weights": {
        "mse": 0.5,        # Was 1.0 - share weight with other objectives
        "variance": 0.2,   # NEW: Preserve variance (prevent mean regression)
        "range": 0.15,     # NEW: Preserve range (prevent compression)
        "bias": 0.15,      # Prevent systematic bias
        "spatial": 0.0,    # Start at 0, increase during training
        "physical": 0.0,   # Start at 0, increase during training
    },
    
    # Gradient control
    "gradient_clip": 1.0,  # Was 0.5 - slightly looser
}

# STRONGER DATA AUGMENTATION: Better generalization
AUGMENTATION_CONFIG = {
    # Geometric augmentations
    "rotation_range": 30,      # Was 15 - more rotation
    "flip_horizontal": True,
    "flip_vertical": True,
    "elastic_deform": 0.1,     # NEW: Simulate terrain variations
    
    # Photometric augmentations
    "brightness_range": 0.15,  # Was 0.05 - simulate different solar angles
    "contrast_range": 0.15,    # NEW: Different atmospheric conditions
    "saturation_range": 0.10,  # NEW: Simulate sensor variations
    
    # Noise augmentations
    "noise_std": 0.10,         # Was 0.05 - more noise tolerance
    "gaussian_blur": 0.3,      # NEW: Simulate atmospheric blur
    
    # Advanced augmentations
    "mixup_alpha": 0.2,        # NEW: MixUp for better generalization
    "cutout_size": 8,          # NEW: Random masking
}

# ============================================================================
# VALIDATION CONFIGURATION - IMPROVED
# ============================================================================
VALIDATION_CONFIG = {
    # Metrics to track
    "metrics": ["r2", "rmse", "mae", "mbe", "slope", "std_ratio"],
    
    # Target performance (realistic)
    "targets": {
        "r2": 0.85,      # Was 0.80
        "rmse": 1.5,     # °C
        "mae": 1.0,      # °C
        "slope": 0.95,   # NEW: Check for range compression
        "std_ratio": 0.90,  # NEW: Check variance preservation
    },
    
    # Cross-validation
    "cv_folds": 5,
    "spatial_holdout": 0.15,
    "station_holdout": 0.20,
    
    # NEW: Spatial cross-validation
    "use_spatial_cv": True,
    "spatial_block_size": 5000,  # meters - blocks for spatial CV
}

# ============================================================================
# UHI ANALYSIS CONFIGURATION
# ============================================================================
UHI_CONFIG = {
    "classification": {
        "weak": (0, 2),      # °C
        "moderate": (2, 4),
        "strong": (4, 6),
        "very_strong": (6, 100)
    },
    "hotspot_threshold": 2.58,  # Gi* statistic (99% confidence)
    "spatial_weight_distance": 500,  # meters
    "rural_ndvi_threshold": 0.4,
    "urban_building_density_threshold": 0.7
}

# ============================================================================
# NORMALIZATION CONFIGURATION - CRITICAL
# ============================================================================
NORMALIZATION_CONFIG = {
    # Ensure consistent normalization between train/test
    "method": "zscore",  # Standardization
    "clip_outliers": True,
    "clip_std": 5.0,  # Clip values beyond ±5 std
    
    # Save/load normalization statistics
    "save_stats": True,
    "stats_file": "normalization_stats.json",
    
    # Validation
    "validate_normalized": True,
    "expected_mean_range": (-0.5, 0.5),
    "expected_std_range": (0.5, 1.5),
}

# ============================================================================
# POST-PROCESSING CONFIGURATION - NEW
# ============================================================================
POST_PROCESSING_CONFIG = {
    # Bias correction
    "apply_bias_correction": True,
    
    # Variance scaling
    "apply_variance_scaling": True,
    
    # Calibration
    "apply_calibration": True,
    "calibration_method": "isotonic",  # or "platt"
    
    # Spatial filtering
    "bilateral_filter": {
        "enabled": True,
        "sigma_spatial": 5,
        "sigma_range": 2.0,
    },
    
    # Physical constraints
    "temp_min": 20.0,  # °C
    "temp_max": 50.0,  # °C
}

# ============================================================================
# API & SENSOR CONFIGURATION
# ============================================================================
NAFAS_CONFIG = {
    "api_url": "https://api.nafas.id/v1",
    "timeout": 30,
    "retry_attempts": 3
}

BMKG_CONFIG = {
    "stations": [
        {"id": "96749", "name": "Kemayoran", "lat": -6.1667, "lon": 106.8500},
        {"id": "96745", "name": "Tanjung Priok", "lat": -6.1167, "lon": 106.8833},
        {"id": "96747", "name": "Halim", "lat": -6.2667, "lon": 106.8833}
    ]
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
OUTPUT_CONFIG = {
    "raster": {
        "format": "GTiff",
        "compress": "lzw",
        "tiled": True,
        "nodata": -9999
    },
    "color_scheme": {
        "lst": "RdYlBu_r",  # Red-Yellow-Blue reversed
        "uhi": "hot"
    },
    "export_formats": ["geotiff", "png", "json", "csv"]
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": BASE_DIR / "logs" / "uhi_pipeline.log",
    "encoding": "utf-8"
}

# Create logs directory
(BASE_DIR / "logs").mkdir(exist_ok=True)

# ============================================================================
# COMPUTATIONAL CONFIGURATION
# ============================================================================
COMPUTE_CONFIG = {
    "use_gpu": True,
    "gpu_memory_fraction": 0.9,
    "num_workers": 4,
    "prefetch_factor": 2,
    "pin_memory": True
}

# ============================================================================
# MONITORING CONFIGURATION - NEW
# ============================================================================
MONITORING_CONFIG = {
    # Track these metrics during training
    "track_metrics": [
        "r2", "rmse", "mae", "mbe",
        "slope", "intercept",
        "pred_mean", "pred_std",
        "target_mean", "target_std",
        "std_ratio", "range_ratio"
    ],
    
    # Warning thresholds
    "warnings": {
        "overfitting_threshold": 0.10,  # Train-val R² gap
        "variance_collapse_threshold": 0.80,  # pred_std / target_std
        "range_compression_threshold": 0.90,  # slope
    },
    
    # Log frequency
    "log_every_n_epochs": 5,
    "save_checkpoint_every_n_epochs": 10,
}