"""
Utility functions for Urban Heat Island project
"""
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
import rasterio
from rasterio.transform import from_bounds
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

from config import *

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path = None):
    """Setup logging configuration"""
    if log_file is None:
        log_file = LOGGING_CONFIG["file"]
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=LOGGING_CONFIG["level"],
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    metrics = {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mbe": np.mean(y_pred - y_true),
        "correlation": pearsonr(y_true, y_pred)[0],
        "n_samples": len(y_true)
    }
    
    return metrics


def plot_training_history(history: Dict, save_path: Path = None):
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R² score
    ax = axes[0, 1]
    r2_scores = [m["r2"] for m in history["val_metrics"]]
    ax.plot(r2_scores, color="green")
    ax.axhline(y=VALIDATION_CONFIG["targets"]["r2"], 
               color='r', linestyle='--', label='Target')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("R² Score")
    ax.set_title("Validation R² Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[1, 0]
    rmse_scores = [m["rmse"] for m in history["val_metrics"]]
    ax.plot(rmse_scores, color="orange")
    ax.axhline(y=VALIDATION_CONFIG["targets"]["rmse"], 
               color='r', linestyle='--', label='Target')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Validation RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 1]
    ax.plot(history["lr"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, 
                           save_path: Path = None):
    """
    Plot prediction scatter plot
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_path: Path to save plot
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    
    # 1:1 line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
    
    # Add metrics text
    textstr = '\n'.join([
        f'R² = {metrics["r2"]:.4f}',
        f'RMSE = {metrics["rmse"]:.2f}°C',
        f'MAE = {metrics["mae"]:.2f}°C',
        f'MBE = {metrics["mbe"]:.2f}°C',
        f'n = {metrics["n_samples"]:,}'
    ])
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Observed LST (°C)', fontsize=12)
    ax.set_ylabel('Predicted LST (°C)', fontsize=12)
    ax.set_title('Predicted vs. Observed Land Surface Temperature', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scatter plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_uhi_intensity(lst: np.ndarray, ndvi: np.ndarray, 
                           building_density: np.ndarray = None) -> Dict:
    """
    Calculate UHI intensity
    
    Args:
        lst: Land Surface Temperature array
        ndvi: NDVI array
        building_density: Building density array (optional)
        
    Returns:
        Dictionary with UHI statistics
    """
    # Define urban and rural areas
    if building_density is not None:
        urban_mask = building_density > UHI_CONFIG["urban_building_density_threshold"]
    else:
        # Use NDVI as proxy
        urban_mask = ndvi < 0.3
    
    rural_mask = ndvi > UHI_CONFIG["rural_ndvi_threshold"]
    
    # Calculate reference temperatures
    t_urban = np.nanmean(lst[urban_mask])
    t_rural = np.nanmean(lst[rural_mask])
    
    # UHI intensity per pixel
    uhi_intensity = lst - t_rural
    
    # Classify intensity
    weak = (uhi_intensity >= 0) & (uhi_intensity < 2)
    moderate = (uhi_intensity >= 2) & (uhi_intensity < 4)
    strong = (uhi_intensity >= 4) & (uhi_intensity < 6)
    very_strong = uhi_intensity >= 6
    
    results = {
        "mean_uhi_intensity": t_urban - t_rural,
        "max_uhi_intensity": np.nanmax(uhi_intensity),
        "urban_temp_mean": t_urban,
        "rural_temp_mean": t_rural,
        "uhi_map": uhi_intensity,
        "classification": {
            "weak": np.sum(weak),
            "moderate": np.sum(moderate),
            "strong": np.sum(strong),
            "very_strong": np.sum(very_strong)
        }
    }
    
    return results


def calculate_hotspots(lst: np.ndarray, coords: np.ndarray = None,
                      distance_threshold: float = 500) -> np.ndarray:
    """
    Calculate hot spots using Getis-Ord Gi* statistic
    
    Args:
        lst: Land Surface Temperature array
        coords: Coordinates array (N, 2) with (x, y)
        distance_threshold: Distance threshold for spatial weights (meters)
        
    Returns:
        Gi* statistic array
    """
    # Flatten arrays
    lst_flat = lst.flatten()
    valid_mask = ~np.isnan(lst_flat)
    lst_valid = lst_flat[valid_mask]
    
    if coords is None:
        # Create grid coordinates
        h, w = lst.shape
        y, x = np.mgrid[0:h, 0:w]
        coords = np.column_stack([x.flatten(), y.flatten()])
    
    coords_valid = coords[valid_mask]
    
    # Calculate distances
    distances = cdist(coords_valid, coords_valid)
    
    # Calculate spatial weights (inverse distance)
    weights = np.zeros_like(distances)
    mask = (distances > 0) & (distances <= distance_threshold)
    weights[mask] = 1.0 / distances[mask]
    
    # Normalize weights
    row_sums = weights.sum(axis=1)
    weights = weights / row_sums[:, np.newaxis]
    
    # Calculate Gi* statistic
    n = len(lst_valid)
    mean = np.mean(lst_valid)
    std = np.std(lst_valid)
    
    weighted_sum = weights @ lst_valid
    weight_sum = weights.sum(axis=1)
    weight_sq_sum = (weights ** 2).sum(axis=1)
    
    numerator = weighted_sum - mean * weight_sum
    denominator = std * np.sqrt((n * weight_sq_sum - weight_sum ** 2) / (n - 1))
    
    gi_star = numerator / (denominator + 1e-8)
    
    # Map back to full array
    gi_star_full = np.full(lst.size, np.nan)
    gi_star_full[valid_mask] = gi_star
    gi_star_map = gi_star_full.reshape(lst.shape)
    
    return gi_star_map


def save_geotiff(array: np.ndarray, output_path: Path, 
                bounds: Dict, crs: str = None):
    """
    Save array as GeoTIFF
    
    Args:
        array: 2D numpy array
        output_path: Output file path
        bounds: Dictionary with min_lon, max_lon, min_lat, max_lat
        crs: Coordinate reference system (default: WGS84)
    """
    if crs is None:
        crs = f"EPSG:{STUDY_AREA['epsg']}"
    
    # Calculate transform
    height, width = array.shape
    transform = from_bounds(
        bounds["min_lon"], bounds["min_lat"],
        bounds["max_lon"], bounds["max_lat"],
        width, height
    )
    
    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        compress='lzw',
        tiled=True,
        nodata=-9999
    ) as dst:
        dst.write(array, 1)
    
    logger.info(f"Saved GeoTIFF to {output_path}")


def plot_lst_map(lst: np.ndarray, save_path: Path = None, 
                title: str = "Land Surface Temperature"):
    """
    Plot LST map
    
    Args:
        lst: LST array
        save_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot
    im = ax.imshow(lst, cmap='RdYlBu_r', interpolation='bilinear')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Temperature (°C)')
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved LST map to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_report(metrics: Dict, uhi_stats: Dict, 
                   output_path: Path):
    """
    Generate analysis report
    
    Args:
        metrics: Model performance metrics
        uhi_stats: UHI statistics
        output_path: Output file path
    """
    report = {
        "metadata": {
            "date_generated": datetime.now().isoformat(),
            "study_area": STUDY_AREA["name"],
            "model_type": "U-Net + LightGBM Ensemble"
        },
        "model_performance": metrics,
        "uhi_analysis": uhi_stats,
        "targets_met": {
            "r2": metrics.get("r2", 0) >= VALIDATION_CONFIG["targets"]["r2"],
            "rmse": metrics.get("rmse", 999) <= VALIDATION_CONFIG["targets"]["rmse"],
            "mae": metrics.get("mae", 999) <= VALIDATION_CONFIG["targets"]["mae"]
        }
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved report to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("URBAN HEAT ISLAND ANALYSIS REPORT")
    print("="*60)
    print(f"\nModel Performance:")
    print(f"  R² Score: {metrics.get('r2', 0):.4f}")
    print(f"  RMSE: {metrics.get('rmse', 0):.2f}°C")
    print(f"  MAE: {metrics.get('mae', 0):.2f}°C")
    print(f"\nUHI Intensity:")
    print(f"  Mean: {uhi_stats.get('mean_uhi_intensity', 0):.2f}°C")
    print(f"  Maximum: {uhi_stats.get('max_uhi_intensity', 0):.2f}°C")
    print(f"  Urban Temperature: {uhi_stats.get('urban_temp_mean', 0):.2f}°C")
    print(f"  Rural Temperature: {uhi_stats.get('rural_temp_mean', 0):.2f}°C")
    print("="*60)


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Generate dummy data
    y_true = np.random.randn(1000) * 5 + 30
    y_pred = y_true + np.random.randn(1000) * 1.5
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print(f"\nMetrics: {metrics}")
    
    # Plot
    plot_prediction_scatter(y_true, y_pred)
    
    print("\nUtility functions working correctly!")