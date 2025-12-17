"""
Data preprocessing pipeline for Urban Heat Island detection
Purpose: Process BOTH Landsat and Sentinel-2 data and fuse them
Does NOT download data - only transforms existing raw data files
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from typing import Dict, Tuple, List, Optional
import logging
from pathlib import Path
from scipy.ndimage import uniform_filter, generic_filter, zoom
import pyproj
from datetime import datetime, timedelta

from config import *

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class SatellitePreprocessor:
    """Transform raw satellite data into processed features"""
    
    def __init__(self, satellite_type: str = "landsat"):
        """
        Initialize preprocessor
        
        Args:
            satellite_type: 'landsat' or 'sentinel2'
        """
        self.satellite_type = satellite_type
        self.config = LANDSAT_CONFIG if satellite_type == "landsat" else SENTINEL2_CONFIG
        
    def resample_band(self, src_array: np.ndarray, 
                     src_resolution: int,
                     target_resolution: int,
                     method: str = 'cubic') -> np.ndarray:
        """
        Resample band to target resolution
        
        Args:
            src_array: Source array
            src_resolution: Source resolution in meters
            target_resolution: Target resolution in meters
            method: Resampling method ('cubic', 'bilinear', 'nearest')
            
        Returns:
            Resampled array
        """
        if src_resolution == target_resolution:
            return src_array
        
        zoom_factor = src_resolution / target_resolution
        
        order_map = {
            'nearest': 0,
            'bilinear': 1,
            'cubic': 3
        }
        order = order_map.get(method, 3)
        
        logger.debug(f"Resampling from {src_resolution}m to {target_resolution}m using {method}")
        resampled = zoom(src_array, zoom_factor, order=order, mode='nearest')
        
        return resampled
    
    def calculate_lst_from_thermal(self, thermal_band: np.ndarray, 
                                   ndvi: np.ndarray) -> np.ndarray:
        """
        Calculate Land Surface Temperature from thermal band
        Uses Single-Channel Algorithm with emissivity correction
        
        Args:
            thermal_band: Thermal infrared band (Landsat ST_B10)
            ndvi: Normalized Difference Vegetation Index
            
        Returns:
            LST in degrees Celsius
        """
        if self.satellite_type != "landsat":
            logger.warning("LST calculation only available for Landsat")
            return None
        
        # Convert DN to Kelvin using Landsat Collection 2 Level-2 scaling
        bt_kelvin = thermal_band * 0.00341802 + 149.0
        bt_celsius = bt_kelvin - 273.15
        
        # Calculate land surface emissivity from NDVI
        epsilon = np.where(
            ndvi < 0.2,
            0.973,  # Bare soil
            np.where(
                ndvi > 0.5,
                0.986,  # Full vegetation
                0.973 + 0.047 * ((ndvi - 0.2) / 0.3)  # Mixed pixels
            )
        )
        
        # Apply emissivity correction using Planck's law
        wavelength = 10.9e-6  # Band 10 wavelength (meters)
        h = 6.626e-34  # Planck's constant (J·s)
        c = 2.998e8    # Speed of light (m/s)
        sigma = 1.38e-23  # Boltzmann constant (J/K)
        rho = (h * c) / sigma  # ≈ 1.438e-2 m·K
        
        # LST with emissivity correction
        lst_celsius = bt_celsius / (1 + (wavelength * bt_kelvin / rho) * np.log(epsilon))
        
        return lst_celsius
    
    def validate_lst(self, lst: np.ndarray, 
                    min_temp: float = -50, 
                    max_temp: float = 60) -> Tuple[np.ndarray, Dict]:
        """
        Validate and clean LST data
        
        Args:
            lst: Land Surface Temperature array
            min_temp: Minimum realistic temperature (°C)
            max_temp: Maximum realistic temperature (°C)
            
        Returns:
            Tuple of (cleaned LST, validation stats)
        """
        lst_clean = lst.copy()
        
        # Count original valid pixels
        original_valid = np.isfinite(lst).sum()
        
        # Filter unrealistic values
        lst_clean[(lst < min_temp) | (lst > max_temp)] = np.nan
        
        # Calculate statistics
        valid_pixels = np.isfinite(lst_clean).sum()
        valid_ratio = valid_pixels / lst_clean.size
        
        stats = {
            "original_valid": int(original_valid),
            "filtered_valid": int(valid_pixels),
            "valid_ratio": float(valid_ratio),
            "mean": float(np.nanmean(lst_clean)) if valid_pixels > 0 else np.nan,
            "std": float(np.nanstd(lst_clean)) if valid_pixels > 0 else np.nan,
            "min": float(np.nanmin(lst_clean)) if valid_pixels > 0 else np.nan,
            "max": float(np.nanmax(lst_clean)) if valid_pixels > 0 else np.nan
        }
        
        logger.info(f"  LST validation: {valid_ratio*100:.1f}% valid pixels")
        logger.info(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]°C")
        logger.info(f"    Mean: {stats['mean']:.2f}°C, Std: {stats['std']:.2f}°C")
        
        return lst_clean, stats
    
    def calculate_spectral_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate spectral indices from raw bands
        
        Args:
            bands: Dictionary of band arrays
            
        Returns:
            Dictionary of calculated indices
        """
        indices = {}
        eps = 1e-8
        
        # Extract bands based on satellite type
        if self.satellite_type == "landsat":
            blue = bands.get("SR_B2", np.zeros_like(bands["SR_B4"])).astype(float)
            green = bands.get("SR_B3", np.zeros_like(bands["SR_B4"])).astype(float)
            red = bands["SR_B4"].astype(float)
            nir = bands["SR_B5"].astype(float)
            swir1 = bands["SR_B6"].astype(float)
            swir2 = bands["SR_B7"].astype(float)
        else:  # Sentinel-2
            blue = bands["B2"].astype(float)
            green = bands["B3"].astype(float)
            red = bands["B4"].astype(float)
            nir = bands["B8"].astype(float)
            swir1 = bands["B11"].astype(float)
            swir2 = bands["B12"].astype(float)
        
        # Calculate indices
        indices["NDVI"] = (nir - red) / (nir + red + eps)
        indices["NDBI"] = (swir1 - nir) / (swir1 + nir + eps)
        indices["MNDWI"] = (green - swir1) / (green + swir1 + eps)
        indices["BSI"] = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)
        indices["UI"] = (swir2 - nir) / (swir2 + nir + eps)
        
        # Albedo (simplified surface albedo)
        albedo = (0.356 * blue + 0.130 * red + 0.373 * nir + 
                  0.085 * swir1 + 0.072 * swir2 - 0.0018)
        indices["albedo"] = np.clip(albedo, 0, 1)
        
        logger.info(f"  Calculated {len(indices)} spectral indices")
        return indices
    
    def process_raw_file(self, raw_file: Path,
                        calculate_lst: bool = True) -> Optional[Dict[str, np.ndarray]]:
        """
        Process a single raw data file
        
        Args:
            raw_file: Path to raw .npz file
            calculate_lst: Whether to calculate LST from thermal band
            
        Returns:
            Dictionary of processed features or None if failed
        """
        logger.info(f"Processing: {raw_file.name}")
        
        try:
            # Load raw data
            raw_data = dict(np.load(raw_file))
            logger.info(f"  Loaded {len(raw_data)} raw bands")
            
            # Extract band data
            bands = {}
            for key, arr in raw_data.items():
                if (key.startswith('SR_B') or key.startswith('B')) and key not in ['BSI', 'B11', 'B12']:
                    bands[key] = arr
                elif key in ['B11', 'B12']:  # SWIR bands for Sentinel-2
                    bands[key] = arr
            
            if len(bands) == 0:
                logger.error(f"  No valid bands found in {raw_file}")
                return None
            
            # Calculate spectral indices
            indices = self.calculate_spectral_indices(bands)
            
            # Calculate LST if Landsat with thermal band
            lst_calculated = False
            if calculate_lst and self.satellite_type == "landsat":
                thermal = raw_data.get("ST_B10")
                if thermal is not None and "NDVI" in indices:
                    logger.info("  Calculating LST from thermal band...")
                    lst = self.calculate_lst_from_thermal(thermal, indices["NDVI"])
                    
                    if lst is not None:
                        # Validate and clean LST
                        lst_clean, lst_stats = self.validate_lst(lst)
                        
                        # Check if LST has sufficient variance
                        if lst_stats["std"] < 0.5:
                            logger.warning(f"  LST has low variance ({lst_stats['std']:.2f}°C)")
                        
                        if lst_stats["valid_ratio"] < 0.1:
                            logger.warning(f"  LST has low valid ratio ({lst_stats['valid_ratio']*100:.1f}%)")
                        
                        indices["LST"] = lst_clean
                        lst_calculated = True
                else:
                    logger.warning("  Cannot calculate LST: missing thermal band or NDVI")
            
            if not lst_calculated and self.satellite_type == "landsat":
                logger.warning("  No LST calculated")
            
            # Combine bands and indices
            processed = {**bands, **indices}
            
            logger.info(f"  ✓ Processed {len(processed)} features")
            return processed
            
        except Exception as e:
            logger.error(f"  ✗ Failed to process {raw_file}: {e}")
            import traceback
            traceback.print_exc()
            return None


class MultiSensorFusion:
    """Fuse data from Landsat and Sentinel-2"""
    
    def __init__(self):
        self.landsat_preprocessor = SatellitePreprocessor("landsat")
        self.sentinel2_preprocessor = SatellitePreprocessor("sentinel2")
    
    def temporal_match(self, landsat_dates: List[Tuple[datetime, Dict]],
                      sentinel2_dates: List[Tuple[datetime, Dict]],
                      max_time_diff_days: int = 16) -> List[Tuple[datetime, Dict, Dict, int]]:
        """
        Match Landsat and Sentinel-2 data by temporal proximity
        
        Args:
            landsat_dates: List of (date, data) tuples for Landsat
            sentinel2_dates: List of (date, data) tuples for Sentinel-2
            max_time_diff_days: Maximum allowed time difference
            
        Returns:
            List of (avg_date, landsat_data, sentinel2_data, time_diff) tuples
        """
        matches = []
        
        for ls_date, ls_data in landsat_dates:
            # Find closest Sentinel-2 image
            best_match = None
            min_diff = float('inf')
            
            for s2_date, s2_data in sentinel2_dates:
                time_diff = abs((ls_date - s2_date).days)
                if time_diff < min_diff:
                    min_diff = time_diff
                    best_match = (s2_date, s2_data)
            
            if best_match and min_diff <= max_time_diff_days:
                s2_date, s2_data = best_match
                avg_date = ls_date + (s2_date - ls_date) / 2
                matches.append((avg_date, ls_data, s2_data, min_diff))
                logger.info(f"  Matched {ls_date.date()} (Landsat) with {s2_date.date()} (Sentinel-2), diff={min_diff} days")
        
        logger.info(f"Created {len(matches)} temporal matches")
        return matches
    
    def fuse_data(self, landsat_data: Dict[str, np.ndarray],
                sentinel2_data: Dict[str, np.ndarray],
                time_diff: int,
                target_resolution: int = 30) -> Dict[str, np.ndarray]:
        """
        Fuse Landsat and Sentinel-2 data
        
        Strategy:
        - Use Landsat LST (only source with thermal data)
        - Use Sentinel-2 for higher resolution spectral indices (10m → 30m)
        - Weight by temporal proximity
        - Use common resolution (30m for consistency with Landsat)
        - Match spatial extents by cropping to minimum overlap
        
        Args:
            landsat_data: Landsat processed data
            sentinel2_data: Sentinel-2 processed data
            time_diff: Time difference in days
            target_resolution: Target resolution in meters
            
        Returns:
            Fused data dictionary
        """
        fused = {}
        
        # Get reference shape from Landsat LST (this defines our target extent)
        if "LST" not in landsat_data:
            logger.error("No LST in Landsat data - cannot fuse")
            return fused
        
        reference_shape = landsat_data["LST"].shape
        logger.debug(f"  Reference shape (Landsat LST): {reference_shape}")
        
        # Start with Landsat LST (only source with thermal data)
        fused["LST"] = landsat_data["LST"].copy()
        
        # Use Sentinel-2 for spectral indices (higher native resolution)
        spectral_indices = ["NDVI", "NDBI", "MNDWI", "BSI", "UI", "albedo"]
        
        # Calculate temporal weight (closer in time = higher weight for Sentinel-2)
        time_weight_s2 = 1.0 / (1.0 + time_diff / 16.0)
        time_weight_ls = 1.0 - time_weight_s2
        
        for idx in spectral_indices:
            has_s2 = idx in sentinel2_data
            has_ls = idx in landsat_data
            
            if has_s2 and has_ls:
                # Resample Sentinel-2 (10m) to target resolution (30m)
                s2_resampled = self.sentinel2_preprocessor.resample_band(
                    sentinel2_data[idx],
                    src_resolution=10,
                    target_resolution=target_resolution,
                    method='bilinear'
                )
                
                ls_data = landsat_data[idx]
                
                # Determine the minimum common extent
                min_height = min(s2_resampled.shape[0], ls_data.shape[0], reference_shape[0])
                min_width = min(s2_resampled.shape[1], ls_data.shape[1], reference_shape[1])
                
                # Crop to common extent
                s2_cropped = s2_resampled[:min_height, :min_width]
                ls_cropped = ls_data[:min_height, :min_width]
                
                # Weighted fusion
                fused[idx] = (time_weight_s2 * s2_cropped + 
                            time_weight_ls * ls_cropped)
                
                logger.debug(f"  {idx}: fused (S2 weight={time_weight_s2:.2f}, shape={fused[idx].shape})")
                
            elif has_s2:
                # Only Sentinel-2 available
                s2_resampled = self.sentinel2_preprocessor.resample_band(
                    sentinel2_data[idx],
                    src_resolution=10,
                    target_resolution=target_resolution,
                    method='bilinear'
                )
                
                # Crop to reference shape
                min_height = min(s2_resampled.shape[0], reference_shape[0])
                min_width = min(s2_resampled.shape[1], reference_shape[1])
                fused[idx] = s2_resampled[:min_height, :min_width]
                
                logger.debug(f"  {idx}: Sentinel-2 only (shape={fused[idx].shape})")
                
            elif has_ls:
                # Only Landsat available
                ls_data = landsat_data[idx]
                
                # Crop to reference shape
                min_height = min(ls_data.shape[0], reference_shape[0])
                min_width = min(ls_data.shape[1], reference_shape[1])
                fused[idx] = ls_data[:min_height, :min_width]
                
                logger.debug(f"  {idx}: Landsat only (shape={fused[idx].shape})")
        
        # Determine final consistent shape from fused spectral indices
        if len(fused) > 1:  # We have LST + at least one spectral index
            # Get the minimum dimensions across all fused features
            all_shapes = [arr.shape for arr in fused.values() if isinstance(arr, np.ndarray)]
            if all_shapes:
                final_height = min(shape[0] for shape in all_shapes)
                final_width = min(shape[1] for shape in all_shapes)
                final_shape = (final_height, final_width)
                
                # Crop all features to consistent final shape
                for key in list(fused.keys()):
                    fused[key] = fused[key][:final_height, :final_width]
                
                logger.debug(f"  Standardized all features to shape: {final_shape}")
            else:
                final_shape = reference_shape
        else:
            final_shape = reference_shape
        
        # Add raw bands from Landsat (cropped to final shape)
        for key in landsat_data.keys():
            if key.startswith('SR_B') and key not in fused:
                ls_band = landsat_data[key]
                fused[key] = ls_band[:final_shape[0], :final_shape[1]]
        
        # Verify all arrays have consistent shape
        shapes = {k: v.shape for k, v in fused.items() if isinstance(v, np.ndarray)}
        unique_shapes = set(shapes.values())
        
        if len(unique_shapes) > 1:
            logger.warning(f"  Inconsistent shapes after fusion: {shapes}")
            # Force consistency by cropping to minimum
            min_h = min(s[0] for s in unique_shapes)
            min_w = min(s[1] for s in unique_shapes)
            for key in fused.keys():
                if isinstance(fused[key], np.ndarray):
                    fused[key] = fused[key][:min_h, :min_w]
            final_shape = (min_h, min_w)
        
        logger.info(f"  Fused {len(fused)} features at {target_resolution}m resolution")
        logger.info(f"  Final shape: {final_shape}")
        
        # Validate minimum size
        if final_shape[0] < 64 or final_shape[1] < 64:
            logger.warning(f"  Warning: Fused data shape {final_shape} is too small for 64x64 patches")
        
        return fused

class FeatureEngineer:
    """Engineer additional features from processed satellite data"""
    
    def __init__(self):
        pass
        
    def calculate_impervious_surface(self, ndvi: np.ndarray, 
                                    ndbi: np.ndarray,
                                    mndwi: np.ndarray) -> np.ndarray:
        """
        Calculate impervious surface fraction
        ISF = (NDBI + (1 - NDVI) + (1 - MNDWI)) / 3
        
        Args:
            ndvi, ndbi, mndwi: Spectral indices
            
        Returns:
            Impervious surface fraction (0-1)
        """
        isf = (ndbi + (1 - ndvi) + (1 - mndwi)) / 3
        return np.clip(isf, 0, 1)
    
    def calculate_spatial_context(self, arr: np.ndarray, 
                                  window_sizes: List[int] = [3, 5, 7]) -> Dict[str, np.ndarray]:
        """
        Calculate spatial context features (neighborhood statistics)
        
        Args:
            arr: Input array
            window_sizes: List of window sizes
            
        Returns:
            Dictionary of spatial statistics
        """
        context = {}
        
        for ws in window_sizes:
            prefix = f"{ws}x{ws}"
            
            # Mean
            context[f"mean_{prefix}"] = uniform_filter(arr, size=ws, mode='reflect')
            
            # Standard deviation
            arr_sq = arr ** 2
            mean_sq = uniform_filter(arr_sq, size=ws, mode='reflect')
            context[f"std_{prefix}"] = np.sqrt(np.maximum(mean_sq - context[f"mean_{prefix}"] ** 2, 0))
        
        return context
    
    def encode_temporal_features(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """
        Encode temporal features
        
        Args:
            timestamp: Date/time of observation
            
        Returns:
            Dictionary of temporal features
        """
        doy = timestamp.dayofyear
        hour = timestamp.hour
        month = timestamp.month
        
        features = {
            "hour": hour / 24.0,
            "DOY_sin": np.sin(2 * np.pi * doy / 365),
            "DOY_cos": np.cos(2 * np.pi * doy / 365),
            "season": 1 if 4 <= month <= 10 else 0,  # Dry season in Indonesia
            "month": month / 12.0
        }
        
        return features


class DatasetCreator:
    """Create ML-ready training/validation/test datasets"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        
    def extract_patches(self, raster_data: Dict[str, np.ndarray],
                    patch_size: int = 64,
                    stride: int = 48,
                    min_valid_ratio: float = 0.7,
                    min_variance: float = 0.3) -> List[Dict]:
        """
        Extract patches from raster data with quality control
        
        Args:
            raster_data: Dictionary of raster arrays
            patch_size: Size of patches (pixels)
            stride: Stride between patches (pixels)
            min_valid_ratio: Minimum ratio of valid pixels required
            min_variance: Minimum LST variance (°C) required
            
        Returns:
            List of patch dictionaries
        """
        if "LST" not in raster_data:
            logger.error("Cannot extract patches: LST not found")
            return []
        
        # Verify all arrays have consistent shapes
        shapes = {k: v.shape for k, v in raster_data.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        unique_shapes = set(shapes.values())
        
        if len(unique_shapes) > 1:
            logger.error(f"Inconsistent array shapes in raster_data: {shapes}")
            return []
        
        height, width = raster_data["LST"].shape
        
        logger.info(f"  Extracting patches from {height}x{width} raster")
        
        if height < patch_size or width < patch_size:
            logger.error(f"Image too small: {height}x{width} < {patch_size}x{patch_size}")
            return []
        
        patches = []
        
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                patch = {
                    "position": (i, j),
                    "data": {}
                }
                
                # Extract patch for each feature
                valid_patch = True
                for name, arr in raster_data.items():
                    if isinstance(arr, np.ndarray) and arr.ndim == 2:
                        if arr.shape != (height, width):
                            logger.warning(f"  Skipping feature {name}: shape mismatch {arr.shape} vs ({height}, {width})")
                            valid_patch = False
                            break
                        
                        patch_data = arr[i:i+patch_size, j:j+patch_size]
                        
                        # Verify patch dimensions
                        if patch_data.shape != (patch_size, patch_size):
                            logger.warning(f"  Invalid patch shape for {name}: {patch_data.shape}")
                            valid_patch = False
                            break
                        
                        patch["data"][name] = patch_data
                
                if not valid_patch:
                    continue
                
                # Quality control on LST
                if "LST" not in patch["data"]:
                    continue
                
                patch_lst = patch["data"]["LST"]
                
                # Verify patch shape one more time
                if patch_lst.shape != (patch_size, patch_size):
                    logger.warning(f"  LST patch has wrong shape: {patch_lst.shape}")
                    continue
                
                valid_pixels = np.isfinite(patch_lst).sum()
                valid_ratio = valid_pixels / (patch_size * patch_size)
                
                if valid_ratio >= min_valid_ratio:
                    # Check variance
                    lst_std = np.nanstd(patch_lst)
                    if lst_std >= min_variance:
                        patches.append(patch)
        
        logger.info(f"  Extracted {len(patches)} valid patches")
        return patches
    
    def create_training_samples(self, patches: List[Dict],
                                temporal_features: Dict,
                                channel_order: List[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create training samples from patches
        
        Args:
            patches: List of patch dictionaries
            temporal_features: Temporal feature dictionary
            channel_order: Order of input channels
            
        Returns:
            Tuple of (X, y, metadata)
        """
        if len(patches) == 0:
            raise ValueError("No patches provided")
        
        if channel_order is None:
            # Default channel order (works for both Landsat and fused data)
            channel_order = [
                "SR_B4", "SR_B5", "SR_B6", "SR_B7",  # Red, NIR, SWIR1, SWIR2
                "NDVI", "NDBI", "MNDWI", "BSI", "UI",
                "albedo"
            ]
        
        n_samples = len(patches)
        patch_size = patches[0]["data"]["LST"].shape[0]
        n_channels = len(channel_order)
        
        X = np.zeros((n_samples, patch_size, patch_size, n_channels), dtype=np.float32)
        y = np.zeros((n_samples, patch_size, patch_size, 1), dtype=np.float32)
        
        for idx, patch in enumerate(patches):
            # Stack channels
            for channel_idx, feature in enumerate(channel_order):
                if feature in patch["data"]:
                    X[idx, :, :, channel_idx] = patch["data"][feature]
                else:
                    logger.warning(f"Feature {feature} not found in patch {idx}")
            
            # Target (LST)
            y[idx, :, :, 0] = patch["data"]["LST"]
        
        # Replace NaN with zeros
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        metadata = {
            "n_samples": n_samples,
            "patch_size": patch_size,
            "n_channels": n_channels,
            "channel_order": channel_order,
            "temporal_features": temporal_features
        }
        
        logger.info(f"Created training samples: X={X.shape}, y={y.shape}")
        return X, y, metadata
    
    def create_train_val_test_split(self, X: np.ndarray, y: np.ndarray,
                                   dates: np.ndarray,
                                   split_method: str = "temporal",
                                   split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Dict:
        """
        Create train/validation/test splits
        
        Args:
            X: Features array
            y: Target array
            dates: Array of dates
            split_method: 'temporal' or 'random'
            split_ratios: (train, val, test) ratios
            
        Returns:
            Dictionary with train/val/test splits
        """
        if split_method == "temporal":
            # Sort by date
            sort_idx = np.argsort(dates)
            X = X[sort_idx]
            y = y[sort_idx]
            dates = dates[sort_idx]
            
            # Calculate split indices
            n_train = int(split_ratios[0] * len(X))
            n_val = int(split_ratios[1] * len(X))
            
            splits = {
                "X_train": X[:n_train],
                "y_train": y[:n_train],
                "dates_train": dates[:n_train],
                "X_val": X[n_train:n_train+n_val],
                "y_val": y[n_train:n_train+n_val],
                "dates_val": dates[n_train:n_train+n_val],
                "X_test": X[n_train+n_val:],
                "y_test": y[n_train+n_val:],
                "dates_test": dates[n_train+n_val:]
            }
            
        elif split_method == "random":
            from sklearn.model_selection import train_test_split
            
            test_size = split_ratios[2]
            val_ratio = split_ratios[1] / (1 - split_ratios[0])
            
            X_train, X_temp, y_train, y_temp, dates_train, dates_temp = train_test_split(
                X, y, dates, test_size=(1-split_ratios[0]), random_state=42
            )
            X_val, X_test, y_val, y_test, dates_val, dates_test = train_test_split(
                X_temp, y_temp, dates_temp, test_size=val_ratio, random_state=42
            )
            
            splits = {
                "X_train": X_train,
                "y_train": y_train,
                "dates_train": dates_train,
                "X_val": X_val,
                "y_val": y_val,
                "dates_val": dates_val,
                "X_test": X_test,
                "y_test": y_test,
                "dates_test": dates_test
            }
        else:
            raise ValueError(f"Unknown split method: {split_method}")
        
        logger.info(f"Created {split_method} split ({split_ratios}):")
        logger.info(f"  Train: {len(splits['X_train'])} samples")
        logger.info(f"  Val: {len(splits['X_val'])} samples")
        logger.info(f"  Test: {len(splits['X_test'])} samples")
        
        return splits
    
    def save_dataset(self, splits: Dict, output_dir: Path, metadata: Dict):
        """
        Save dataset to disk
        
        Args:
            splits: Dictionary with train/val/test splits
            output_dir: Output directory
            metadata: Metadata dictionary
        """
        output_dir = Path(output_dir)
        
        # Save each split
        for split_name in ['train', 'val', 'test']:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(split_dir / "X.npy", splits[f"X_{split_name}"])
            np.save(split_dir / "y.npy", splits[f"y_{split_name}"])
            
            if f"dates_{split_name}" in splits:
                np.save(split_dir / "dates.npy", splits[f"dates_{split_name}"])
        
        # Save metadata
        import json
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            # Convert numpy types to native Python types
            metadata_clean = {}
            for k, v in metadata.items():
                if isinstance(v, (np.integer, np.floating)):
                    metadata_clean[k] = float(v)
                elif isinstance(v, np.ndarray):
                    metadata_clean[k] = v.tolist()
                else:
                    metadata_clean[k] = v
            json.dump(metadata_clean, f, indent=2)
        
        logger.info(f"✓ Dataset saved to {output_dir}")


def main():
    """Main preprocessing pipeline - processes and fuses multi-sensor data"""
    logger.info("="*70)
    logger.info("MULTI-SENSOR PREPROCESSING PIPELINE")
    logger.info("="*70)
    
    # Define input directories
    raw_data_dir = RAW_DATA_DIR
    landsat_dir = raw_data_dir / "landsat"
    sentinel2_dir = raw_data_dir / "sentinel2"
    
    # Check what data is available
    has_landsat = landsat_dir.exists()
    has_sentinel2 = sentinel2_dir.exists()
    
    if not has_landsat and not has_sentinel2:
        logger.error(f"No raw data found in {raw_data_dir}")
        logger.error("Please run earth_engine_loader.py first to download data")
        return
    
    logger.info(f"Data availability:")
    logger.info(f"  Landsat: {'✓' if has_landsat else '✗'}")
    logger.info(f"  Sentinel-2: {'✓' if has_sentinel2 else '✗'}")
    
    # Initialize components
    landsat_preprocessor = SatellitePreprocessor(satellite_type="landsat")
    sentinel2_preprocessor = SatellitePreprocessor(satellite_type="sentinel2")
    fusion = MultiSensorFusion()
    dataset_creator = DatasetCreator()
    feature_engineer = FeatureEngineer()
    
    # Step 1: Process Landsat data
    landsat_processed = []
    if has_landsat:
        logger.info("\n" + "="*70)
        logger.info("STEP 1A: Process Landsat data")
        logger.info("="*70)
        
        landsat_files = sorted(landsat_dir.glob("landsat_*.npz"))
        logger.info(f"Found {len(landsat_files)} Landsat files")
        
        for raw_file in landsat_files:
            # Extract date from filename
            parts = raw_file.stem.split('_')
            year = int(parts[1])
            month = int(parts[2])
            timestamp = pd.Timestamp(year=year, month=month, day=15)
            
            # Process the file
            processed = landsat_preprocessor.process_raw_file(raw_file, calculate_lst=True)
            
            if processed is None:
                continue
            
            # Validate LST
            if "LST" not in processed:
                logger.warning(f"  No LST in {raw_file.name}, skipping")
                continue
            
            lst_std = np.nanstd(processed["LST"])
            lst_valid_ratio = np.sum(np.isfinite(processed["LST"])) / processed["LST"].size
            
            if lst_std < 0.5:
                logger.warning(f"  LST variance too low ({lst_std:.2f}°C), skipping")
                continue
            
            if lst_valid_ratio < 0.1:
                logger.warning(f"  LST valid ratio too low ({lst_valid_ratio*100:.1f}%), skipping")
                continue
            
            landsat_processed.append((timestamp, processed))
            logger.info(f"  ✓ {raw_file.name}")
        
        logger.info(f"\n✓ Processed {len(landsat_processed)} Landsat files")
    
    # Step 2: Process Sentinel-2 data
    sentinel2_processed = []
    if has_sentinel2:
        logger.info("\n" + "="*70)
        logger.info("STEP 1B: Process Sentinel-2 data")
        logger.info("="*70)
        
        sentinel2_files = sorted(sentinel2_dir.glob("sentinel2_*.npz"))
        logger.info(f"Found {len(sentinel2_files)} Sentinel-2 files")
        
        for raw_file in sentinel2_files:
            # Extract date from filename
            parts = raw_file.stem.split('_')
            year = int(parts[1])
            month = int(parts[2])
            timestamp = pd.Timestamp(year=year, month=month, day=15)
            
            # Process the file
            processed = sentinel2_preprocessor.process_raw_file(raw_file, calculate_lst=False)
            
            if processed is None:
                continue
            
            # Check if we have essential indices
            required = ["NDVI", "NDBI", "MNDWI"]
            if not all(idx in processed for idx in required):
                logger.warning(f"  Missing required indices in {raw_file.name}, skipping")
                continue
            
            sentinel2_processed.append((timestamp, processed))
            logger.info(f"  ✓ {raw_file.name}")
        
        logger.info(f"\n✓ Processed {len(sentinel2_processed)} Sentinel-2 files")
    
    # Step 3: Fuse data or use single sensor
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Fuse multi-sensor data")
    logger.info("="*70)
    
    all_fused_data = []
    all_dates = []
    
    if has_landsat and has_sentinel2 and len(landsat_processed) > 0 and len(sentinel2_processed) > 0:
        # Perform temporal matching and fusion
        logger.info("Performing multi-sensor fusion...")
        
        matches = fusion.temporal_match(landsat_processed, sentinel2_processed, max_time_diff_days=16)
        
        for avg_date, ls_data, s2_data, time_diff in matches:
            logger.info(f"\nFusing data (time_diff={time_diff} days):")
            fused_data = fusion.fuse_data(ls_data, s2_data, time_diff, target_resolution=30)
            
            # Add impervious surface
            if "NDVI" in fused_data and "NDBI" in fused_data and "MNDWI" in fused_data:
                isf = feature_engineer.calculate_impervious_surface(
                    fused_data["NDVI"],
                    fused_data["NDBI"],
                    fused_data["MNDWI"]
                )
                fused_data["impervious_surface"] = isf
            
            all_fused_data.append(fused_data)
            all_dates.append(avg_date)
        
        logger.info(f"\n✓ Created {len(all_fused_data)} fused datasets")
        
    elif has_landsat and len(landsat_processed) > 0:
        # Use Landsat only
        logger.info("Using Landsat data only (no Sentinel-2 available)")
        
        for timestamp, ls_data in landsat_processed:
            # Add impervious surface
            if "NDVI" in ls_data and "NDBI" in ls_data and "MNDWI" in ls_data:
                isf = feature_engineer.calculate_impervious_surface(
                    ls_data["NDVI"],
                    ls_data["NDBI"],
                    ls_data["MNDWI"]
                )
                ls_data["impervious_surface"] = isf
            
            all_fused_data.append(ls_data)
            all_dates.append(timestamp)
        
        logger.info(f"✓ Using {len(all_fused_data)} Landsat datasets")
        
    else:
        logger.error("No valid data available for training")
        return
    
    if len(all_fused_data) == 0:
        logger.error("No data available after processing")
        return
    
    # Step 4: Extract patches
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Extract patches")
    logger.info("="*70)
    
    all_patches = []
    for idx, fused_data in enumerate(all_fused_data):
        patches = dataset_creator.extract_patches(
            fused_data,
            patch_size=64,
            stride=48,
            min_valid_ratio=0.7,
            min_variance=0.3
        )
        
        for patch in patches:
            patch["date"] = all_dates[idx]
        
        all_patches.extend(patches)
        logger.info(f"  Dataset {idx+1}/{len(all_fused_data)}: {len(patches)} patches")
    
    logger.info(f"\n✓ Total patches extracted: {len(all_patches)}")
    
    if len(all_patches) == 0:
        logger.error("No patches extracted")
        return
    
    # Step 5: Create training samples
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Create training samples")
    logger.info("="*70)
    
    temporal_features = feature_engineer.encode_temporal_features(all_dates[0])
    X, y, metadata = dataset_creator.create_training_samples(all_patches, temporal_features)
    
    dates = np.array([patch["date"] for patch in all_patches])
    
    # Step 6: Create splits
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Create train/val/test splits")
    logger.info("="*70)
    
    splits = dataset_creator.create_train_val_test_split(
        X, y, dates,
        split_method="temporal",
        split_ratios=(0.7, 0.15, 0.15)
    )
    
    # Step 7: Save dataset
    logger.info("\n" + "="*70)
    logger.info("STEP 6: Save dataset")
    logger.info("="*70)
    
    # Add fusion info to metadata
    metadata["fusion_info"] = {
        "has_landsat": has_landsat,
        "has_sentinel2": has_sentinel2,
        "n_landsat": len(landsat_processed) if has_landsat else 0,
        "n_sentinel2": len(sentinel2_processed) if has_sentinel2 else 0,
        "n_fused": len(all_fused_data),
        "fusion_strategy": "temporal_weighted" if has_landsat and has_sentinel2 else "single_sensor"
    }
    
    output_dataset_dir = PROCESSED_DATA_DIR / "cnn_dataset"
    dataset_creator.save_dataset(splits, output_dataset_dir, metadata)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("✓ MULTI-SENSOR PREPROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(f"Data sources:")
    logger.info(f"  Landsat files: {len(landsat_processed) if has_landsat else 0}")
    logger.info(f"  Sentinel-2 files: {len(sentinel2_processed) if has_sentinel2 else 0}")
    logger.info(f"  Fused datasets: {len(all_fused_data)}")
    logger.info(f"Patches:")
    logger.info(f"  Total patches: {len(all_patches)}")
    logger.info(f"Training data:")
    logger.info(f"  Training samples: {len(splits['X_train'])}")
    logger.info(f"  Validation samples: {len(splits['X_val'])}")
    logger.info(f"  Test samples: {len(splits['X_test'])}")
    logger.info(f"Output:")
    logger.info(f"  Dataset saved to: {output_dataset_dir}")
    logger.info(f"  Fusion strategy: {metadata['fusion_info']['fusion_strategy']}")
    logger.info("="*70)
    logger.info("\nNext step: Run model training")


if __name__ == "__main__":
    main()