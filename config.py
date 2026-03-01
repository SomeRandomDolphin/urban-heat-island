"""
Configuration file for Urban Heat Island Detection System
Study area  : Greater Jakarta (Jabodetabek)
Satellite   : Landsat 8 + Landsat 9 only
Date range  : January 2016 – December 2025
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
# Bounds cover the full DKI Jakarta administrative area:
# - Jakarta Utara, Jakarta Barat, Jakarta Pusat, Jakarta Selatan, Jakarta Timur
# - Kepulauan Seribu (Thousand Islands)
# Extended slightly beyond admin boundary to capture urban fringe / rural reference
# ============================================================================
STUDY_AREA = {
    "name": "Greater Jakarta (Jabodetabek)",
    "bounds": {
        "min_lon": 106.40,   # Western edge — covers Tangerang / Banten fringe
        "max_lon": 107.20,   # Eastern edge — covers Bekasi / Karawang border
        "min_lat":  -6.70,   # Southern edge — covers Depok / Bogor outskirts
        "max_lat":  -6.00,   # Northern edge — includes Jakarta Bay coastline
    },
    # Jabodetabek: Jakarta, Bogor, Depok, Tangerang, Bekasi
    # EPSG:32748 — WGS84 / UTM Zone 48S; standard projection for Java.
    "epsg": 32748,
    "buffer_km": 5
}

# ============================================================================
# SATELLITE DATA CONFIGURATION
# ============================================================================

# Landsat collection merges LC08 (launched 2013) and LC09 (launched 2021).
# Both carry OLI + TIRS instruments with identical band structure, so they can
# be treated as a single harmonised collection. Merging maximises revisit
# frequency and is especially important for 2021-onwards when both are active.
LANDSAT_CONFIG = {
    # Primary (LC08) and secondary (LC09) collections — merged at runtime
    "collection_l8": "LANDSAT/LC08/C02/T1_L2",
    "collection_l9": "LANDSAT/LC09/C02/T1_L2",
    "bands": {
        "coastal": "SR_B1",   # Coastal/Aerosol (not on TM/ETM+, but on OLI)
        "blue":    "SR_B2",
        "green":   "SR_B3",
        "red":     "SR_B4",
        "nir":     "SR_B5",
        "swir1":   "SR_B6",
        "swir2":   "SR_B7",
        "thermal": "ST_B10",
        "qa":      "QA_PIXEL"
    },
    # Landsat Collection-2 Level-2 scale factors (MUST apply before analysis)
    # Surface Reflectance: DN * 2.75e-5 - 0.2   → unitless [0, 1]
    # Surface Temperature: DN * 0.00341802 + 149.0 → Kelvin
    "sr_scale":    2.75e-5,
    "sr_offset":  -0.2,
    "st_scale":    0.00341802,
    "st_offset":  149.0,
    "cloud_threshold": 50,  # percentage
    "scale": 30,            # meters (OLI optical bands)
    "thermal_scale": 100    # meters (TIRS native, resampled to 30 m in C2)
}

# Landsat 8/9 TIRS thermal constants (Band 10)
THERMAL_CONSTANTS = {
    "K1": 774.8853,       # W/(m²·sr·μm)
    "K2": 1321.0789,      # K
    "wavelength": 10.9e-6,  # metres
    "rho": 1.438e-2         # m·K
}

# ============================================================================
# TEMPORAL CONFIGURATION
# ============================================================================
DATE_RANGE = {
    # Full download window — 10 years of data (2016-01-01 to 2025-12-31)
    # Landsat 8 operational from April 2013; Landsat 9 from September 2021.
    # Both are available for the full 2016–2025 range (L9 contributes from late 2021).
    "all_data_start": "2016-01-01",
    "all_data_end":   "2025-12-31",

    # Split ratios
    "train_ratio": 0.65,
    "val_ratio":   0.15,
    "test_ratio":  0.20,

    # Ensure seasonal balance in splits
    "stratify_by_season":   True,
    "stratify_by_location": True,
}

# ============================================================================
# GRID CONFIGURATION
# ============================================================================
GRID_CONFIG = {
    "resolution": 50,  # metres
    "patch_size": 64,  # pixels for CNN input
    "overlap": 32      # pixels overlap between patches
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
SPECTRAL_INDICES = [
    "NDVI",    # Normalized Difference Vegetation Index
    "NDBI",    # Normalized Difference Built-up Index
    "MNDWI",   # Modified Normalized Difference Water Index
    "BSI",     # Bare Soil Index
    "UI",      # Urban Index
    "EBBI",    # Enhanced Built-Up and Bareness Index
    "albedo"   # Surface albedo
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

    # Reduced capacity: prevents overfitting
    "filters": [16, 32, 64, 128, 256],  # Was [32, 64, 128, 256, 512]

    # Increased dropout: better regularisation
    "dropout_rates": [0.3, 0.4, 0.5, 0.5, 0.5],  # Was [0.2, 0.25, 0.3, 0.35, 0.4]

    "batch_norm": True,
    "activation": "relu",
    "use_attention": False,
}

GBM_CONFIG = {
    "algorithm": "lightgbm",
    "params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",

        "num_leaves": 31,   # Was 127
        "max_depth": 6,     # Was 12

        "learning_rate": 0.05,
        "n_estimators": 2000,

        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.4,

        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "min_child_samples": 50,
        "min_child_weight": 1e-3,
        "min_split_gain": 0.01,

        "early_stopping_rounds": 50,
        "verbose": 100,
    }
}

ENSEMBLE_WEIGHTS = {
    "cnn": 0.35,
    "gbm": 0.55,
    "baseline": 0.10
}

# ============================================================================
# TRAINING CONFIGURATION - IMPROVED
# ============================================================================
TRAINING_CONFIG = {
    "batch_size": 16,
    "epochs": 500,

    "initial_lr": 0.001,
    "min_lr": 1e-6,
    "warmup_epochs": 10,

    "patience": 50,
    "min_delta": 0.0005,

    "optimizer": "adamw",
    "weight_decay": 0.001,

    "loss_weights": {
        "mse":      0.50,
        "variance": 0.20,
        "range":    0.15,
        "bias":     0.15,
        "spatial":  0.00,
        "physical": 0.00,
    },

    "gradient_clip": 1.0,
}

AUGMENTATION_CONFIG = {
    "rotation_range":   30,
    "flip_horizontal":  True,
    "flip_vertical":    True,
    "elastic_deform":   0.10,

    "brightness_range": 0.15,
    "contrast_range":   0.15,
    "saturation_range": 0.10,

    "noise_std":        0.10,
    "gaussian_blur":    0.30,

    "mixup_alpha":      0.20,
    "cutout_size":      8,
}

# ============================================================================
# VALIDATION CONFIGURATION - IMPROVED
# ============================================================================
VALIDATION_CONFIG = {
    "metrics": ["r2", "rmse", "mae", "mbe", "slope", "std_ratio"],

    "targets": {
        "r2":       0.85,
        "rmse":     1.5,
        "mae":      1.0,
        "slope":    0.95,
        "std_ratio": 0.90,
    },

    "cv_folds": 5,
    "spatial_holdout": 0.15,
    "station_holdout": 0.20,

    "use_spatial_cv": True,
    "spatial_block_size": 5000,  # metres
}

# ============================================================================
# UHI ANALYSIS CONFIGURATION
# ============================================================================
UHI_CONFIG = {
    "classification": {
        "weak":      (0, 2),
        "moderate":  (2, 4),
        "strong":    (4, 6),
        "very_strong": (6, 100)
    },
    "hotspot_threshold": 2.58,       # Gi* statistic (99% confidence)
    "spatial_weight_distance": 500,  # metres
    "rural_ndvi_threshold": 0.4,
    "urban_building_density_threshold": 0.7
}

# ============================================================================
# NORMALIZATION CONFIGURATION
# ============================================================================
NORMALIZATION_CONFIG = {
    "method": "zscore",
    "clip_outliers": True,
    "clip_std": 5.0,

    "save_stats": True,
    "stats_file": "normalization_stats.json",

    "validate_normalized": True,
    "expected_mean_range": (-0.5, 0.5),
    "expected_std_range":  (0.5, 1.5),
}

# ============================================================================
# POST-PROCESSING CONFIGURATION
# ============================================================================
POST_PROCESSING_CONFIG = {
    "apply_bias_correction": True,
    "apply_variance_scaling": True,
    "apply_calibration": True,
    "calibration_method": "isotonic",

    "bilateral_filter": {
        "enabled": True,
        "sigma_spatial": 5,
        "sigma_range": 2.0,
    },

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
        {"id": "96749", "name": "Kemayoran (Jakarta Pusat)",   "lat": -6.1667, "lon": 106.8500},
        {"id": "96745", "name": "Tanjung Priok (Jakarta Utara)","lat": -6.1167, "lon": 106.8833},
        {"id": "96747", "name": "Halim (Jakarta Timur)",       "lat": -6.2667, "lon": 106.8833},
        {"id": "96743", "name": "Soekarno-Hatta (Tangerang)",  "lat": -6.1167, "lon": 106.6500},
        {"id": "96751", "name": "Pondok Betung (Tangerang Sel.)","lat": -6.2833, "lon": 106.7333},
        {"id": "96753", "name": "Citeko (Bogor)",              "lat": -6.7000, "lon": 107.0333},
        {"id": "96757", "name": "Depok",                       "lat": -6.4000, "lon": 106.8333},
        {"id": "96759", "name": "Bekasi",                      "lat": -6.2333, "lon": 107.0000},
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
        "lst": "RdYlBu_r",
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
# MONITORING CONFIGURATION
# ============================================================================
MONITORING_CONFIG = {
    "track_metrics": [
        "r2", "rmse", "mae", "mbe",
        "slope", "intercept",
        "pred_mean", "pred_std",
        "target_mean", "target_std",
        "std_ratio", "range_ratio"
    ],

    "warnings": {
        "overfitting_threshold": 0.10,
        "variance_collapse_threshold": 0.80,
        "range_compression_threshold": 0.90,
    },

    "log_every_n_epochs": 5,
    "save_checkpoint_every_n_epochs": 10,
}