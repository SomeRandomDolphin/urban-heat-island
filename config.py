"""
Configuration file for Urban Heat Island Detection System
All pipeline parameters are centralised here — preprocessing.py reads from this
file and contains no hardcoded magic numbers.
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
    "name": "Jakarta Metropolitan Area (DKI Jakarta)",
    "bounds": {
        "min_lon": 106.65,   # Western edge (Jakarta Barat coast)
        "max_lon": 107.00,   # Eastern edge (Jakarta Timur boundary)
        "min_lat": -6.40,    # Southern edge (Jakarta Selatan border)
        "max_lat": -6.00,    # Northern edge (includes Kepulauan Seribu waters)
    },
    # NOTE: Kepulauan Seribu extends far north (~-5.4°), but including the full
    # archipelago chain adds mostly open sea. Set max_lat=-5.85 to include the
    # nearest island cluster while keeping the AOI manageable.
    "epsg": 32748,           # WGS84 / UTM Zone 48S
    "buffer_km": 5,
}

# ============================================================================
# SATELLITE DATA CONFIGURATION
# ============================================================================

# Landsat collection merges LC08 (launched 2013) and LC09 (launched 2021).
# Both carry OLI + TIRS instruments with identical band structure.
LANDSAT_CONFIG = {
    "collection_l8": "LANDSAT/LC08/C02/T1_L2",
    "collection_l9": "LANDSAT/LC09/C02/T1_L2",
    "bands": {
        "coastal": "SR_B1",
        "blue":    "SR_B2",
        "green":   "SR_B3",
        "red":     "SR_B4",
        "nir":     "SR_B5",
        "swir1":   "SR_B6",
        "swir2":   "SR_B7",
        "thermal": "ST_B10",
        "qa":      "QA_PIXEL",
    },
    # Collection-2 Level-2 scale factors (applied before analysis)
    # Surface Reflectance: DN * 2.75e-5 - 0.2   → unitless [0, 1]
    # Surface Temperature: DN * 0.00341802 + 149.0 → Kelvin
    "sr_scale":         2.75e-5,
    "sr_offset":       -0.2,
    "st_scale":         0.00341802,
    "st_offset":        149.0,
    "cloud_threshold":  50,   # percent
    "scale":            30,   # meters (OLI optical bands)
    "thermal_scale":    100,  # meters (TIRS native; resampled to 30 m in C2)
}

SENTINEL2_CONFIG = {
    "collection": "COPERNICUS/S2_SR_HARMONIZED",
    "bands": {
        "blue":  "B2",
        "green": "B3",
        "red":   "B4",
        "nir":   "B8",
        "swir1": "B11",
        "swir2": "B12",
        "scl":   "SCL",
    },
    # SCL classes to mask out:
    # 1=Saturated/Defective, 3=Cloud Shadow, 8=Cloud Med, 9=Cloud High,
    # 10=Cirrus, 11=Snow/Ice
    "cloud_classes": [1, 3, 8, 9, 10, 11],
    "scale": 10,  # meters
    # Reflectance scale factor: S2_SR_HARMONIZED stores values × 10 000.
    # Values whose 75th percentile (positive pixels) exceeds this threshold
    # are treated as raw DN and divided by dn_scale_factor.
    "dn_detection_p75_threshold": 2.0,
    "dn_scale_factor":            10_000.0,
}

# Landsat 8/9 TIRS thermal constants (Band 10)
THERMAL_CONSTANTS = {
    "K1":         774.8853,   # W/(m²·sr·μm)
    "K2":        1321.0789,   # K
    "wavelength": 10.9e-6,    # metres
    "rho":        1.438e-2,   # m·K  (h·c / k_B, precomputed)
}

# ============================================================================
# TEMPORAL CONFIGURATION
# ============================================================================
DATE_RANGE = {
    "all_data_start":       "2016-01-01",
    "all_data_end":         "2025-12-31",
    "train_ratio":          0.65,
    "val_ratio":            0.15,
    "test_ratio":           0.20,
    "stratify_by_season":   True,
    "stratify_by_location": True,
}

# ============================================================================
# GRID CONFIGURATION
# ============================================================================
GRID_CONFIG = {
    "resolution": 50,  # metres
    "patch_size": 64,  # pixels for CNN input
    "overlap":    32,  # pixels overlap between patches
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
    "albedo",  # Surface albedo
]

URBAN_FEATURES = [
    "building_density",
    "road_density",
    "green_space_ratio",
    "impervious_surface",
    "sky_view_factor",
    "avg_building_height",
]

METEOROLOGICAL_FEATURES = [
    "air_temp",
    "wind_speed",
    "solar_radiation",
    "humidity",
    "pressure",
]

TEMPORAL_FEATURES = [
    "hour",
    "DOY_sin",
    "DOY_cos",
    "season",
]

# ============================================================================
# PREPROCESSING PIPELINE PARAMETERS
# All magic numbers from preprocessing.py are defined here.
# ============================================================================

PREPROCESSING_CONFIG = {
    # ── Memory / downsampling ──────────────────────────────────────────────
    # Maximum pixels (rows × cols) allowed per band before adaptive downsampling.
    # At float32 each pixel costs 4 bytes:
    #   10_000_000 px → ~38 MiB/band → ~343 MiB for 9 Landsat bands
    # Can also be overridden at runtime via: export UHI_MAX_PIXELS=<int>
    "max_pixels_default":       10_000_000,
    # Minimum allowed downsample factor (never go below 5% of native resolution)
    "min_downsample_factor":     0.05,
    # RAM budget fraction used when psutil is available (0.40 = 40% of free RAM)
    "ram_budget_fraction":       0.40,

    # ── Channel / feature order written into X ─────────────────────────────
    # This list determines the channel axis of the (N, H, W, C) arrays.
    # Must stay in sync with CNN_CONFIG["input_channels"].
    "channel_order": [
        "SR_B4", "SR_B5", "SR_B6", "SR_B7",
        "NDVI",  "NDBI",  "MNDWI", "BSI", "UI", "albedo",
    ],

    # ── LST temperature gates (three-tier system) ──────────────────────────
    # Tier 1 — pixel level (validate_lst):    nulls physically impossible pixels.
    # Tier 2 — patch mean (extract_patches):  rejects artifact-dominated patches.
    # Tier 3 — safety clip (create_training): final hard clip before saving.
    #
    # Rationale for bounds:
    #   min_pixel_temp=10°C  Sub-10°C is unambiguously cloud shadow / noise at
    #                        this latitude; Jakarta water bodies stay >20°C.
    #   max_pixel_temp=65°C  Dark asphalt / metal roofing can reach ~60-63°C at
    #                        midday. Values above 65°C are retrieval failures.
    #   min_patch_temp=15°C  Coastal/vegetated patches can reach 15–20°C in wet-
    #                        season mornings. The old 20°C floor truncated the cool
    #                        tail and hurt cool-end calibration.
    #   max_patch_temp=58°C  No legitimate 64×64 patch averages above 58°C over
    #                        Jakarta. Raised from 48°C to include genuinely hot
    #                        dense-urban industrial-zone patches (UHI signal).
    "lst_min_pixel_temp":       10.0,   # °C — Tier 1 & 3 lower bound
    "lst_max_pixel_temp":       65.0,   # °C — Tier 1 & 3 upper bound
    "lst_min_patch_temp":       15.0,   # °C — Tier 2 patch-mean lower bound
    "lst_max_patch_temp":       58.0,   # °C — Tier 2 patch-mean upper bound

    # LST Gaussian smoothing (applied after validate_lst)
    # A light blur suppresses 30m pixel-grid artefacts without smearing UHI
    # gradients (which span tens–hundreds of metres, i.e. >> 1 pixel).
    "lst_smooth_sigma":         0.8,    # pixels; set to 0.0 to disable

    # ── Scene-level QC thresholds ──────────────────────────────────────────
    # A scene is skipped if its LST std is below this value (flat/uniform scene)
    "scene_min_lst_std":        0.5,    # °C
    # A scene is skipped if fewer than this fraction of pixels are finite
    "scene_min_valid_ratio":    0.10,   # 10%

    # ── Patch extraction QC ────────────────────────────────────────────────
    "patch_size":               64,     # pixels
    # Stride used during extraction AND during mosaic reconstruction.
    # Keep these two values in sync — patch_stride is saved to metadata.json
    # and read back by the inference pipeline.
    "patch_stride":             24,     # pixels (was 32; 24 gives denser coverage)
    # Minimum fraction of finite LST pixels within a patch.
    # Raised from 0.80 to 0.95 to eliminate NaN-heavy edge/cloud patches that
    # caused the central spike at normalised LST ≈ 0.
    "patch_min_valid_ratio":    0.95,
    # Minimum LST standard deviation within a patch; rejects flat-temperature
    # (water, cloud-shadow, or ocean fill) patches.
    "patch_min_variance":       0.5,    # °C

    # ── NaN inpainting (iterative Gaussian) ───────────────────────────────
    # Sigma schedule for isotropic Gaussian inpainting passes.
    # Small holes fill first (tight local context), large holes fill later
    # (wider neighbourhood) → smooth gradients across hole boundaries.
    # Any pixels still NaN after all passes fall back to the patch median.
    "inpaint_sigmas":           [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0],

    # ── Sample weight binning ──────────────────────────────────────────────
    # Number of equal-width LST bins used to compute inverse-frequency
    # sample weights (addresses class imbalance across the temperature range).
    "sample_weight_n_bins":     10,

    # ── Scatter-plot subsampling ───────────────────────────────────────────
    # Maximum number of pixels used in diagnostic scatter/hexbin plots.
    # Keeps plot generation fast for large rasters.
    "plot_subsample_n":         50_000,

    # ── Temporal matching (multi-sensor fusion) ────────────────────────────
    # Maximum calendar days between a Landsat and Sentinel-2 acquisition
    # for them to be paired as a fused epoch.
    "max_time_diff_days":       16,
    # Fusion target resolution (metres).  Landsat LST is native 30 m;
    # Sentinel-2 spectral indices are resampled from 10 m to this value.
    "fusion_target_resolution": 30,

    # ── Stratified split random seed ──────────────────────────────────────
    "split_random_seed":        42,

    # ── Normalisation verification tolerances ────────────────────────────
    # After z-score normalisation, warn if the training split deviates beyond
    # these bounds (expected: mean ≈ 0, std ≈ 1).
    "norm_mean_tolerance":      0.1,
    "norm_std_low":             0.9,
    "norm_std_high":            1.1,

    # ── Leakage check tolerance ───────────────────────────────────────────
    # Maximum allowed absolute difference between the stored normalisation
    # stats and the actual training-data statistics.
    "leakage_tolerance":        0.01,
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
CNN_CONFIG = {
    "architecture":   "unet",
    "input_channels": 10,   # Must equal len(PREPROCESSING_CONFIG["channel_order"])

    # Reduced capacity: prevents overfitting (was [32, 64, 128, 256, 512])
    "filters":        [16, 32, 64, 128, 256],

    # Increased dropout: better regularisation (was [0.2, 0.25, 0.3, 0.35, 0.4])
    "dropout_rates":  [0.3, 0.4, 0.5, 0.5, 0.5],

    "batch_norm":     True,
    "activation":     "relu",
    "use_attention":  False,
}

GBM_CONFIG = {
    "algorithm":                "lightgbm",
    "bottleneck_pca_components": 32,
    "params": {
        "objective":             "regression",
        "metric":                "rmse",
        "boosting_type":         "gbdt",
        "num_leaves":            31,    # Was 127
        "max_depth":             6,     # Was 12
        "learning_rate":         0.03,  # Reduced from 0.05 — finer convergence
        "n_estimators":          3000,  # Increased from 2000 — more room with lower LR
        "subsample":             0.7,
        "subsample_freq":        1,
        "colsample_bytree":      0.4,
        "reg_alpha":             1.0,
        "reg_lambda":            1.0,
        "min_child_samples":     50,
        "min_child_weight":      1e-3,
        "min_split_gain":        0.01,
        "early_stopping_rounds": 100,   # Was 50 — more patience
        "verbose":               100,
    },
}

ENSEMBLE_WEIGHTS = {
    "cnn":      0.35,
    "gbm":      0.55,
    "baseline": 0.10,
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    "batch_size":    16,
    "epochs":        500,
    "initial_lr":    0.001,
    "min_lr":        1e-6,
    "warmup_epochs": 10,
    "patience":      50,
    "min_delta":     0.0005,
    "optimizer":     "adamw",
    "weight_decay":  0.01,   # Increased from 0.001 — weight norm was growing to 500+

    "loss_weights": {
        "mse":      0.30,   # Reduced — pure MSE encourages mean-regression (slope collapse)
        "variance": 0.40,   # Increased from 0.20 — combats slope=0.67 / std_ratio=0.75
        "range":    0.15,
        "bias":     0.15,   # Keeps MBE near zero
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
# VALIDATION CONFIGURATION
# ============================================================================
VALIDATION_CONFIG = {
    "metrics":            ["r2", "rmse", "mae", "mbe", "slope", "std_ratio"],
    "targets": {
        "r2":        0.85,
        "rmse":      1.5,
        "mae":       1.0,
        "slope":     0.95,
        "std_ratio": 0.90,
    },
    "cv_folds":           5,
    "spatial_holdout":    0.15,
    "station_holdout":    0.20,
    "use_spatial_cv":     True,
    "spatial_block_size": 5000,  # metres
}

# ============================================================================
# UHI ANALYSIS CONFIGURATION
# ============================================================================
UHI_CONFIG = {
    "classification": {
        "weak":       (0, 2),
        "moderate":   (2, 4),
        "strong":     (4, 6),
        "very_strong": (6, 100),
    },
    "hotspot_threshold":              2.58,   # Gi* statistic (99% confidence)
    "spatial_weight_distance":        500,    # metres
    "rural_ndvi_threshold":           0.4,
    "urban_building_density_threshold": 0.7,
}

# ============================================================================
# NORMALIZATION CONFIGURATION
# ============================================================================
NORMALIZATION_CONFIG = {
    "method":           "zscore",
    "clip_outliers":    True,
    "clip_std":         5.0,
    "save_stats":       True,
    "stats_file":       "normalization_stats.json",
    "validate_normalized": True,
    "expected_mean_range": (-0.5, 0.5),
    "expected_std_range":  (0.5, 1.5),
}

# ============================================================================
# POST-PROCESSING CONFIGURATION
# ============================================================================
POST_PROCESSING_CONFIG = {
    "apply_bias_correction":  True,
    "apply_variance_scaling": True,
    "apply_calibration":      True,
    "calibration_method":     "isotonic",
    "bilateral_filter": {
        "enabled":       True,
        "sigma_spatial": 5,
        "sigma_range":   2.0,
    },
    "temp_min": 20.0,   # °C
    "temp_max": 50.0,   # °C
}

# ============================================================================
# API & SENSOR CONFIGURATION
# ============================================================================
NAFAS_CONFIG = {
    "api_url":        "https://api.nafas.id/v1",
    "timeout":        30,
    "retry_attempts": 3,
}

BMKG_CONFIG = {
    "stations": [
        {"id": "96749", "name": "Kemayoran",     "lat": -6.1667, "lon": 106.8500},
        {"id": "96745", "name": "Tanjung Priok", "lat": -6.1167, "lon": 106.8833},
        {"id": "96747", "name": "Halim",         "lat": -6.2667, "lon": 106.8833},
    ],
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
OUTPUT_CONFIG = {
    "raster": {
        "format":   "GTiff",
        "compress": "lzw",
        "tiled":    True,
        "nodata":   -9999,
    },
    "color_scheme": {
        "lst": "RdYlBu_r",
        "uhi": "hot",
    },
    "export_formats": ["geotiff", "png", "json", "csv"],
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    "level":    "INFO",
    "format":   "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": BASE_DIR / "logs" / "uhi_pipeline.log",
    "encoding": "utf-8",
}

(BASE_DIR / "logs").mkdir(exist_ok=True)

# ============================================================================
# COMPUTATIONAL CONFIGURATION
# ============================================================================
COMPUTE_CONFIG = {
    "use_gpu":            True,
    "gpu_memory_fraction": 0.9,
    "num_workers":        4,
    "prefetch_factor":    2,
    "pin_memory":         True,
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
        "std_ratio", "range_ratio",
    ],
    "warnings": {
        "overfitting_threshold":       0.10,   # train-val R² gap; original: gap > 0.10
        "variance_collapse_threshold": 0.80,   # std_ratio floor; original: std_ratio < 0.80
        "range_compression_threshold": 0.85,   # slope floor; original: slope < 0.85 (was incorrectly 0.90)
    },
    "log_every_n_epochs":            5,
    "save_checkpoint_every_n_epochs": 10,
}

# ============================================================================
# SCHEDULER CONFIGURATION
# CosineAnnealingWarmRestarts (SGDR) — periodically resets LR to escape local
# minima. T_mult doubles the period after each restart.
# ============================================================================
SCHEDULER_CONFIG = {
    "T_0":     50,    # epochs before first restart
    "T_mult":  2,     # period multiplier after each restart (50 → 100 → 200)
    "eta_min": 1e-6,  # LR floor at the bottom of each cosine valley
}

# ============================================================================
# EARLY STOPPING CONFIGURATION
# ============================================================================
EARLY_STOPPING_CONFIG = {
    "patience":   50,     # mirrors TRAINING_CONFIG["patience"]; EnsembleTrainer used config.get("patience", 20) which resolved to 50
    "min_delta":  0.002,  # minimum R² gain to count as improvement (hardcoded in original EnsembleTrainer)
    "mode":       "max",  # 'max' for R², 'min' for loss/RMSE (hardcoded in original EnsembleTrainer)
}

# ============================================================================
# LAYER-WISE WEIGHT DECAY
# Deeper / output layers get stronger regularisation to prevent the monotonic
# weight-norm growth observed during training (75 → 320+ over 200 epochs).
# ============================================================================
LAYERWISE_WEIGHT_DECAY = {
    "enc1":       1e-3,
    "enc2":       2e-3,
    "enc3":       3e-3,
    "enc4":       4e-3,
    "bottleneck": 5e-3,
    "dec4":       4e-3,
    "dec3":       3e-3,
    "dec2":       2e-3,
    "dec1":       1e-3,
    "output":     5e-3,
}

# ============================================================================
# PROGRESSIVE LOSS PHASE WEIGHTS
# Three training phases that shift emphasis as training progresses.
# Weights must sum to 1.0 within each phase.
# ============================================================================
PROGRESSIVE_LOSS_CONFIG = {
    # Thresholds (fraction of total epochs) that divide phases
    "phase1_end": 0.30,   # 0%–30%  → WARMUP
    "phase2_end": 0.60,   # 30%–60% → REFINEMENT

    # Phase 1 — WARMUP: anchor on weighted MSE; light regularisation
    "phase1": {
        "mse":      0.65,
        "variance": 0.10,
        "range":    0.10,
        "bias":     0.05,
        "gradient": 0.10,
    },
    # Phase 2 — REFINEMENT: increase variance/range pressure
    "phase2": {
        "mse":      0.50,
        "variance": 0.18,
        "range":    0.18,
        "bias":     0.07,
        "gradient": 0.07,
    },
    # Phase 3 — FINE-TUNING: balance all terms
    "phase3": {
        "mse":      0.40,
        "variance": 0.22,
        "range":    0.20,
        "bias":     0.10,
        "gradient": 0.08,
    },
}

# ============================================================================
# AUGMENTATION PROBABILITY CONFIGURATION
# Controls per-transform apply probabilities and magnitude ranges.
# These override the coarser AUGMENTATION_CONFIG ranges for the live pipeline.
# ============================================================================
AUGMENTATION_PROB_CONFIG = {
    "geometric_prob":     0.80,  # probability of applying any geometric transform
    "flip_prob":          0.50,  # horizontal / vertical flip
    "rot90_prob":         0.50,  # 90-degree rotation
    "brightness_prob":    0.50,  # brightness jitter
    "brightness_range":   0.30,  # ± fraction (0.30 → ±15%)
    "contrast_prob":      0.40,  # contrast jitter
    "contrast_range":     0.30,  # ± fraction
    "noise_prob":         0.40,  # Gaussian noise
    "noise_std":          0.10,  # noise standard deviation (normalised units)
    "cutout_prob":        0.20,  # regional dropout (simulates cloud/missing data)
    "cutout_size_frac":   0.25,  # cutout patch = 25% of spatial dimension
    "mixup_prob":         0.20,  # MixUp
    "mixup_alpha":        0.20,  # Beta(alpha, alpha) mixing coefficient
}

# ============================================================================
# TEMPERATURE STRATIFIED SAMPLER
# Bins are in normalised LST space (mean≈0, std≈1).
# -1.0 ≈ cool tail, +1.0 ≈ hot tail. Inverse-frequency weighting balances bins.
# ============================================================================
STRATIFIED_SAMPLER_CONFIG = {
    "bins": [-float("inf"), -1.0, -0.5, 0.0, 0.5, 1.0, float("inf")],
}

# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================
CHECKPOINT_CONFIG = {
    "metrics":         ["r2", "rmse", "mae"],
    "primary_metric":  "r2",   # used to select best CNN for ensemble
}

# ============================================================================
# CONVRBLOCK RESIDUAL CONNECTION
# When True, ConvBlock adds a 1×1 shortcut projection so gradients can flow
# directly through the encoder/decoder without passing through both conv layers.
# ============================================================================
MODEL_ARCH_CONFIG = {
    "use_residual_connections": True,
    # Dropout mode: "channel" (Dropout2d, drops full channels) or
    # "pixel" (standard Dropout).  "pixel" is gentler in early layers.
    "dropout_mode": "pixel",
}

# ============================================================================
# DATA QUALITY CHECK TOLERANCES
# Thresholds used by validate_data_quality() in train_ensemble.py
# ============================================================================
DATA_QUALITY_CONFIG = {
    # Training split: check that data is properly z-score normalised
    "train_mean_tol":       0.20,   # |mean| must be below this
    "train_std_low":        0.80,   # std must be above this
    "train_std_high":       1.20,   # std must be below this
    # Val/test: looser check — just ensure values aren't wildly out of range
    "val_mean_max":         5.0,
    # Variance floor — targets below this std are considered degenerate
    "min_target_std":       0.10,
    "zero_variance_floor":  1e-6,
    # Minimum bin count for stratified error plots
    "stratified_min_bin_n": 5,
}

# ============================================================================
# GBM BOTTLENECK PCA
# Number of PCA dimensions used to compress CNN bottleneck features before
# passing them to LightGBM.  Without compression the high-dimensional
# bottleneck acts as a near-perfect memorisation key for training patches.
# ============================================================================
# NOTE: bottleneck_pca_components is also kept in GBM_CONFIG for backward compat.

# ============================================================================
# DISK SPACE CHECK
# ============================================================================
DISK_CONFIG = {
    "required_mb": 1000,  # minimum free MB before training starts
}

# ============================================================================
# DIAGNOSTICS / PLOTTING
# ============================================================================
DIAGNOSTICS_CONFIG = {
    "save_dpi":              150,
    "spatial_n_samples":     6,    # patches to show in spatial error map (plot 08)
    "stratified_n_bins":     10,   # temperature bins for stratified error (plot 09)
    "gbm_importance_top_n":  30,   # top-N features in GBM importance plot (plot 05)
    "r2_target_line":        0.85, # reference line on R² panel
    "slope_target":          1.0,  # reference line on slope panel
    "slope_danger":          0.85, # red warning line on slope panel
    "std_ratio_target":      1.0,
    "std_ratio_danger":      0.80,
}