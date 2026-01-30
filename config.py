"""
Configuration file for Urban Heat Island Detection System - FIXED VERSION
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
# TEMPORAL CONFIGURATION
# ============================================================================
DATE_RANGE = {
    "train_start": "2021-01-01",
    "train_end": "2023-12-31",
    "val_start": "2024-01-01",
    "val_end": "2024-06-30",
    "test_start": "2024-07-01",
    "test_end": "2024-12-31"
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
# MODEL CONFIGURATION
# ============================================================================
CNN_CONFIG = {
    "architecture": "unet",
    "input_channels": 10,
    "filters": [32, 64, 128, 256, 512],  # REDUCED from [64, 128, 256, 512, 1024]
    "dropout_rates": [0.1, 0.15, 0.2, 0.25, 0.3],
    "batch_norm": True,
    "activation": "relu",
    "use_attention": False,  # Keep simple initially
}

GBM_CONFIG = {
    "algorithm": "lightgbm",
    "params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        
        # Tree structure - MORE COMPLEX
        "num_leaves": 255,  # INCREASED from 127
        "max_depth": 15,  # INCREASED from 12
        
        # Learning
        "learning_rate": 0.03,  # REDUCED from 0.05 - more trees
        "n_estimators": 2000,  # INCREASED from 1000
        
        # Sampling
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        
        # Regularization
        "reg_alpha": 0.05,  # REDUCED from 0.1
        "reg_lambda": 0.05,  # REDUCED from 0.1
        "min_child_samples": 50,  # REDUCED from 100 - allow more complex patterns
        
        # Convergence
        "early_stopping_rounds": 100,  # INCREASED from 50
        "verbose": 100,
    }
}

ENSEMBLE_WEIGHTS = {
    "cnn": 0.5,
    "gbm": 0.4,
    "baseline": 0.1
}

# ============================================================================
# TRAINING CONFIGURATION - FIXED
# ============================================================================
TRAINING_CONFIG = {
    # Batch size
    "batch_size": 12,  # CHANGED: Smaller batches for better gradients
    
    # Training duration
    "epochs": 200,  # CHANGED: More epochs (we have early stopping)
    
    # Learning rate
    "initial_lr": 0.0015,  # CHANGED: Slightly increased
    "min_lr": 5e-7,
    "warmup_epochs": 20,  # CHANGED: Longer warmup
    
    # Early stopping
    "patience": 40,  # CHANGED: More patience
    "min_delta": 0.0003,  # CHANGED: Smaller threshold
    
    # Optimizer
    "optimizer": "adamw",
    "weight_decay": 0.00001,
    
    # Loss weights (will be adjusted during training)
    "loss_weights": {
        "mse": 1.0,
        "spatial": 0.0,  # Start at 0, increase later
        "physical": 0.0,  # Start at 0, increase later
        "variance": 0.3,
        "bias": 0.2
    },
    
    # Gradient control
    "gradient_clip": 0.3,  # CHANGED: Tighter control
    "min_delta": 0.0003
}

# Data augmentation
AUGMENTATION_CONFIG = {
    "rotation_range": 15,  # degrees
    "flip_horizontal": True,
    "flip_vertical": True,
    "brightness_range": 0.05,
    "noise_std": 0.05
}

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================
VALIDATION_CONFIG = {
    "metrics": ["r2", "rmse", "mae", "mbe"],
    "targets": {
        "r2": 0.80,
        "rmse": 1.5,  # °C
        "mae": 1.0    # °C
    },
    "cv_folds": 5,
    "spatial_holdout": 0.15,
    "station_holdout": 0.20
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