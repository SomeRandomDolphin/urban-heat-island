"""
UHI Analysis - Calculate UHI intensity, detect hotspots, and generate reports
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm

from config import *

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class UHIAnalyzer:
    """Analyze UHI intensity and patterns"""
    
    def __init__(self, lst_map: np.ndarray, coords: Optional[np.ndarray] = None):
        """
        Initialize UHI analyzer
        
        Args:
            lst_map: LST map (H, W) in Celsius
            coords: Optional coordinate array (H, W, 2) with (lon, lat)
        """
        self.lst_map = lst_map
        self.coords = coords
        self.uhi_map = None
        self.reference_temps = {}
        
        logger.info(f"Initialized UHI Analyzer for {lst_map.shape} map")
        logger.info(f"LST range: [{lst_map.min():.2f}, {lst_map.max():.2f}]°C")
    
    def define_reference_areas(self, urban_mask: np.ndarray, 
                              rural_mask: np.ndarray) -> Dict[str, float]:
        """
        Define urban and rural reference temperatures
        
        Args:
            urban_mask: Boolean mask for urban areas (CBD, high density)
            rural_mask: Boolean mask for rural areas (low density, high vegetation)
            
        Returns:
            Dictionary with reference temperatures
        """
        logger.info("Defining reference areas...")
        
        # Calculate reference temperatures
        T_urban = self.lst_map[urban_mask].mean()
        T_rural = self.lst_map[rural_mask].mean()
        
        logger.info(f"Urban reference temperature: {T_urban:.2f}°C ({urban_mask.sum()} pixels)")
        logger.info(f"Rural reference temperature: {T_rural:.2f}°C ({rural_mask.sum()} pixels)")
        logger.info(f"Temperature difference: {T_urban - T_rural:.2f}°C")
        
        self.reference_temps = {
            "T_urban": T_urban,
            "T_rural": T_rural,
            "T_diff": T_urban - T_rural,
            "urban_std": self.lst_map[urban_mask].std(),
            "rural_std": self.lst_map[rural_mask].std()
        }
        
        return self.reference_temps
    
    def calculate_uhi_intensity(self, rural_reference: Optional[float] = None) -> np.ndarray:
        """
        Calculate per-pixel UHI intensity
        
        Args:
            rural_reference: Rural reference temperature (uses calculated if None)
            
        Returns:
            UHI intensity map (H, W)
        """
        logger.info("Calculating UHI intensity...")
        
        if rural_reference is None:
            if "T_rural" not in self.reference_temps:
                raise ValueError("Rural reference not defined. Call define_reference_areas() first.")
            rural_reference = self.reference_temps["T_rural"]
        
        self.uhi_map = self.lst_map - rural_reference
        
        logger.info(f"UHI intensity range: [{self.uhi_map.min():.2f}, {self.uhi_map.max():.2f}]°C")
        
        return self.uhi_map
    
    def classify_uhi_intensity(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Classify UHI intensity into categories
        
        Categories:
            0: No UHI or cooling (<0°C)
            1: Weak UHI (0-1°C)
            2: Moderate UHI (1-2°C)
            3: Strong UHI (2-3°C)
            4: Very Strong UHI (>3°C)
            
        Returns:
            Classification map and category counts
        """
        if self.uhi_map is None:
            raise ValueError("UHI intensity not calculated. Call calculate_uhi_intensity() first.")
        
        logger.info("Classifying UHI intensity...")
        
        classified = np.zeros_like(self.uhi_map, dtype=np.int8)
        classified[self.uhi_map < 0] = 0
        classified[(self.uhi_map >= 0) & (self.uhi_map < 1)] = 1
        classified[(self.uhi_map >= 1) & (self.uhi_map < 2)] = 2
        classified[(self.uhi_map >= 2) & (self.uhi_map < 3)] = 3
        classified[self.uhi_map >= 3] = 4
        
        categories = {
            "No UHI/Cooling": (classified == 0).sum(),
            "Weak UHI": (classified == 1).sum(),
            "Moderate UHI": (classified == 2).sum(),
            "Strong UHI": (classified == 3).sum(),
            "Very Strong UHI": (classified == 4).sum()
        }
        
        logger.info("UHI Classification:")
        for category, count in categories.items():
            pct = count / classified.size * 100
            logger.info(f"  {category}: {count} pixels ({pct:.2f}%)")
        
        return classified, categories
    
    def calculate_statistics(self) -> Dict[str, float]:
        """
        Calculate UHI statistical metrics
        
        Returns:
            Dictionary of statistics
        """
        if self.uhi_map is None:
            raise ValueError("UHI intensity not calculated.")
        
        logger.info("Calculating UHI statistics...")
        
        # Only consider positive UHI (areas warmer than rural reference)
        uhi_positive = self.uhi_map[self.uhi_map > 0]
        
        stats = {
            "max_intensity": float(self.uhi_map.max()),
            "mean_intensity": float(self.uhi_map.mean()),
            "mean_positive_intensity": float(uhi_positive.mean()) if len(uhi_positive) > 0 else 0.0,
            "median_intensity": float(np.median(self.uhi_map)),
            "std_intensity": float(self.uhi_map.std()),
            "spatial_extent_km2": float((self.uhi_map > 2).sum() * 0.0025),  # 50m resolution = 0.0025 km²
            "magnitude": float(self.uhi_map[self.uhi_map > 0].sum())  # Integrated intensity
        }
        
        logger.info("UHI Statistics:")
        logger.info(f"  Maximum intensity: {stats['max_intensity']:.2f}°C")
        logger.info(f"  Mean intensity: {stats['mean_intensity']:.2f}°C")
        logger.info(f"  Spatial extent (>2°C): {stats['spatial_extent_km2']:.2f} km²")
        logger.info(f"  UHI magnitude: {stats['magnitude']:.2f}°C·pixels")
        
        return stats


class HotspotDetector:
    """Detect and analyze UHI hotspots using Getis-Ord Gi* statistic"""
    
    def __init__(self, lst_map: np.ndarray, resolution: float = 50.0):
        """
        Initialize hotspot detector
        
        Args:
            lst_map: LST map (H, W)
            resolution: Spatial resolution in meters
        """
        self.lst_map = lst_map
        self.resolution = resolution
        self.hotspot_map = None
        
    def calculate_gi_star(self, search_radius: float = 500.0) -> np.ndarray:
        """
        Calculate Getis-Ord Gi* statistic
        
        Args:
            search_radius: Search radius in meters for spatial weights
            
        Returns:
            Gi* statistic map (H, W)
        """
        logger.info(f"Calculating Gi* statistic (radius={search_radius}m)...")
        
        height, width = self.lst_map.shape
        gi_star = np.zeros_like(self.lst_map)
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:height, 0:width] * self.resolution
        
        # Global statistics
        X_bar = np.mean(self.lst_map)
        S = np.std(self.lst_map)
        n = self.lst_map.size
        
        # Calculate for each pixel (this is slow for large images)
        logger.info("Computing Gi* for all pixels...")
        
        # Subsample for efficiency (can be adjusted)
        step = max(1, int(search_radius / self.resolution / 2))
        
        for i in tqdm(range(0, height, step), desc="Gi* calculation"):
            for j in range(0, width, step):
                # Calculate distances to all other pixels
                dist_y = (y_coords - y_coords[i, j]) ** 2
                dist_x = (x_coords - x_coords[i, j]) ** 2
                distances = np.sqrt(dist_y + dist_x)
                
                # Spatial weights (inverse distance within radius)
                weights = np.zeros_like(distances)
                in_radius = distances <= search_radius
                weights[in_radius] = 1.0 / (distances[in_radius] + 1e-6)
                weights = weights / weights.sum() if weights.sum() > 0 else weights
                
                # Weighted sum
                weighted_sum = np.sum(weights * self.lst_map)
                sum_weights = np.sum(weights)
                
                # Gi* statistic
                if sum_weights > 0:
                    numerator = weighted_sum - X_bar * sum_weights
                    sum_weights_sq = np.sum(weights ** 2)
                    denominator = S * np.sqrt((n * sum_weights_sq - sum_weights ** 2) / (n - 1))
                    
                    if denominator > 1e-6:
                        gi_star[i, j] = numerator / denominator
        
        # Interpolate to full resolution
        if step > 1:
            from scipy.interpolate import RegularGridInterpolator
            
            y_sub = np.arange(0, height, step)
            x_sub = np.arange(0, width, step)
            gi_star_sub = gi_star[::step, ::step]
            
            interp = RegularGridInterpolator(
                (y_sub, x_sub), gi_star_sub,
                method='linear', bounds_error=False, fill_value=0
            )
            
            y_full, x_full = np.mgrid[0:height, 0:width]
            points = np.column_stack([y_full.ravel(), x_full.ravel()])
            gi_star = interp(points).reshape(height, width)
        
        self.hotspot_map = gi_star
        
        logger.info(f"Gi* range: [{gi_star.min():.3f}, {gi_star.max():.3f}]")
        logger.info(f"Significant hotspots (Gi* > 1.96): {(gi_star > 1.96).sum()} pixels")
        logger.info(f"Highly significant hotspots (Gi* > 2.58): {(gi_star > 2.58).sum()} pixels")
        
        return gi_star
    
    def identify_hotspots(self, confidence_level: float = 0.95) -> Tuple[np.ndarray, List[Dict]]:
        """
        Identify significant hotspots
        
        Args:
            confidence_level: Statistical confidence (0.95 or 0.99)
            
        Returns:
            Binary hotspot map and list of hotspot regions
        """
        if self.hotspot_map is None:
            raise ValueError("Gi* not calculated. Call calculate_gi_star() first.")
        
        # Thresholds
        threshold = 1.96 if confidence_level == 0.95 else 2.58
        
        logger.info(f"Identifying hotspots at {confidence_level*100:.0f}% confidence...")
        
        # Binary hotspot map
        hotspots = self.hotspot_map > threshold
        
        # Connected component analysis to find hotspot regions
        from scipy.ndimage import label
        labeled_hotspots, n_hotspots = label(hotspots)
        
        logger.info(f"Found {n_hotspots} hotspot regions")
        
        # Extract hotspot properties
        hotspot_list = []
        for region_id in range(1, n_hotspots + 1):
            region_mask = labeled_hotspots == region_id
            region_pixels = region_mask.sum()
            
            if region_pixels < 4:  # Skip tiny regions
                continue
            
            # Get region properties
            y_indices, x_indices = np.where(region_mask)
            
            hotspot_info = {
                "id": region_id,
                "n_pixels": int(region_pixels),
                "area_km2": float(region_pixels * (self.resolution / 1000) ** 2),
                "centroid_y": float(y_indices.mean()),
                "centroid_x": float(x_indices.mean()),
                "mean_lst": float(self.lst_map[region_mask].mean()),
                "max_lst": float(self.lst_map[region_mask].max()),
                "mean_gi_star": float(self.hotspot_map[region_mask].mean()),
                "max_gi_star": float(self.hotspot_map[region_mask].max())
            }
            
            hotspot_list.append(hotspot_info)
        
        # Sort by area (largest first)
        hotspot_list.sort(key=lambda x: x["area_km2"], reverse=True)
        
        logger.info(f"Identified {len(hotspot_list)} significant hotspot regions")
        
        return hotspots, hotspot_list
    
    def prioritize_hotspots(self, hotspot_list: List[Dict], 
                           population_density: Optional[np.ndarray] = None,
                           vulnerability_index: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Prioritize hotspots for intervention
        
        Args:
            hotspot_list: List of hotspot dictionaries
            population_density: Optional population density map
            vulnerability_index: Optional vulnerability index map
            
        Returns:
            DataFrame with prioritized hotspots
        """
        logger.info("Prioritizing hotspots for intervention...")
        
        df = pd.DataFrame(hotspot_list)
        
        # Scoring criteria
        # 1. UHI intensity (40%)
        df["intensity_score"] = (df["mean_lst"] - df["mean_lst"].min()) / (df["mean_lst"].max() - df["mean_lst"].min())
        
        # 2. Spatial extent (30%)
        df["extent_score"] = (df["area_km2"] - df["area_km2"].min()) / (df["area_km2"].max() - df["area_km2"].min())
        
        # 3. Statistical significance (30%)
        df["significance_score"] = (df["max_gi_star"] - 1.96) / (df["max_gi_star"].max() - 1.96)
        df["significance_score"] = df["significance_score"].clip(0, 1)
        
        # Combined priority score
        df["priority_score"] = (
            0.4 * df["intensity_score"] +
            0.3 * df["extent_score"] +
            0.3 * df["significance_score"]
        )
        
        # Add population/vulnerability if available
        if population_density is not None:
            logger.info("Incorporating population density...")
            # Extract population for each hotspot
            # (Implementation would depend on your population data structure)
        
        if vulnerability_index is not None:
            logger.info("Incorporating vulnerability index...")
            # Similar to population density
        
        # Sort by priority
        df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        
        logger.info(f"Top 5 priority hotspots:")
        for idx, row in df.head(5).iterrows():
            logger.info(f"  {idx+1}. Area: {row['area_km2']:.3f} km², "
                       f"LST: {row['mean_lst']:.1f}°C, "
                       f"Priority: {row['priority_score']:.3f}")
        
        return df


class ValidationAnalyzer:
    """Validate predictions against ground truth"""
    
    def __init__(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Initialize validator
        
        Args:
            predictions: Predicted LST map
            ground_truth: Ground truth LST map
        """
        self.predictions = predictions.flatten()
        self.ground_truth = ground_truth.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(self.predictions) | np.isnan(self.ground_truth))
        self.predictions = self.predictions[mask]
        self.ground_truth = self.ground_truth[mask]
        
        logger.info(f"Initialized validator with {len(self.predictions)} valid pixels")
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate validation metrics"""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        r2 = r2_score(self.ground_truth, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.ground_truth, self.predictions))
        mae = mean_absolute_error(self.ground_truth, self.predictions)
        mbe = np.mean(self.predictions - self.ground_truth)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            self.ground_truth, self.predictions
        )
        
        metrics = {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "mbe": mbe,
            "slope": slope,
            "intercept": intercept,
            "correlation": r_value,
            "p_value": p_value
        }
        
        logger.info("Validation Metrics:")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}°C")
        logger.info(f"  MAE: {mae:.4f}°C")
        logger.info(f"  MBE: {mbe:.4f}°C")
        logger.info(f"  Regression: y = {slope:.4f}x + {intercept:.4f}")
        
        return metrics
    
    def plot_validation(self, output_path: Path):
        """Create validation plots"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        ax = axes[0]
        ax.scatter(self.ground_truth, self.predictions, alpha=0.5, s=10)
        
        # 1:1 line
        min_val = min(self.ground_truth.min(), self.predictions.min())
        max_val = max(self.ground_truth.max(), self.predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
        
        # Regression line
        slope, intercept, _, _, _ = stats.linregress(self.ground_truth, self.predictions)
        x_line = np.array([min_val, max_val])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'b-', label=f'Fit: y={slope:.3f}x+{intercept:.3f}')
        
        ax.set_xlabel('Observed LST (°C)')
        ax.set_ylabel('Predicted LST (°C)')
        ax.set_title('Predicted vs Observed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Residual plot
        ax = axes[1]
        residuals = self.predictions - self.ground_truth
        ax.scatter(self.ground_truth, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Observed LST (°C)')
        ax.set_ylabel('Residual (°C)')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved validation plot: {output_path}")
        plt.close()


def generate_report(uhi_stats: Dict, hotspots_df: pd.DataFrame, 
                   validation_metrics: Optional[Dict], output_path: Path):
    """Generate comprehensive UHI analysis report"""
    
    logger.info("Generating UHI analysis report...")

    if hotspots_df is not None and not hotspots_df.empty and "area_km2" in hotspots_df.columns:
        total_hotspot_area = f"{hotspots_df['area_km2'].sum():.2f} km²"
    else:
        total_hotspot_area = "0.00 km²"

    report = {
        "title": "Urban Heat Island Analysis Report",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uhi_characterization": {
            "maximum_intensity": f"{uhi_stats['max_intensity']:.2f}°C",
            "mean_intensity": f"{uhi_stats['mean_intensity']:.2f}°C",
            "spatial_extent": f"{uhi_stats['spatial_extent_km2']:.2f} km²",
            "magnitude": f"{uhi_stats['magnitude']:.2f}°C·pixels"
        },
        "hotspot_summary": {
            "total_hotspots": len(hotspots_df),
            "total_hotspot_area": total_hotspot_area,
            "largest_hotspot": f"{hotspots_df.iloc[0]['area_km2']:.3f} km²" if len(hotspots_df) > 0 else "N/A",
            "highest_temperature": f"{hotspots_df['max_lst'].max():.2f}°C" if len(hotspots_df) > 0 else "N/A"
        },
        "top_priority_hotspots": hotspots_df.head(20).to_dict('records') if len(hotspots_df) > 0 else []
    }
    
    if validation_metrics:
        report["validation"] = {
            "r2_score": f"{validation_metrics['r2']:.4f}",
            "rmse": f"{validation_metrics['rmse']:.4f}°C",
            "mae": f"{validation_metrics['mae']:.4f}°C",
            "bias": f"{validation_metrics['mbe']:.4f}°C"
        }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to: {output_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("UHI ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"Maximum UHI intensity: {uhi_stats['max_intensity']:.2f}°C")
    logger.info(f"UHI spatial extent: {uhi_stats['spatial_extent_km2']:.2f} km²")
    logger.info(f"Number of hotspots: {len(hotspots_df)}")
    if validation_metrics:
        logger.info(f"Validation R²: {validation_metrics['r2']:.4f}")
        logger.info(f"Validation RMSE: {validation_metrics['rmse']:.4f}°C")
    logger.info("="*60)


if __name__ == "__main__":
    # Example usage
    logger.info("UHI Analysis module loaded")
    logger.info("Use: from uhi_analysis import UHIAnalyzer, HotspotDetector, ValidationAnalyzer")