"""
UHI Visualization - Create maps, plots, and output products
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image
import json

from config import *

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class UHIVisualizer:
    """Create visualizations for UHI analysis"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10), dpi: int = 300):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['figure.titlesize'] = 16
    
    def create_lst_map(self, lst_data: np.ndarray, output_path: Path,
                      title: str = "Land Surface Temperature Map",
                      vmin: Optional[float] = None, vmax: Optional[float] = None,
                      add_colorbar: bool = True):
        """
        Create LST map visualization
        
        Args:
            lst_data: LST data array
            output_path: Path to save figure
            title: Plot title
            vmin, vmax: Value range for colormap
            add_colorbar: Whether to add colorbar
        """
        logger.info(f"Creating LST map: {title}")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create custom colormap (blue to red)
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                 '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('thermal', colors, N=n_bins)
        
        # Plot
        im = ax.imshow(lst_data, cmap=cmap, vmin=vmin, vmax=vmax, 
                      interpolation='nearest', aspect='auto')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        if add_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Temperature (°C)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved LST map: {output_path}")
        plt.close()
    
    def create_uhi_intensity_map(self, uhi_data: np.ndarray, output_path: Path,
                                title: str = "UHI Intensity Map"):
        """
        Create UHI intensity map with classification
        
        Args:
            uhi_data: UHI intensity data
            output_path: Path to save figure
            title: Plot title
        """
        logger.info(f"Creating UHI intensity map: {title}")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Classify UHI intensity
        classified = np.zeros_like(uhi_data)
        classified[uhi_data < 0] = 0        # Cooling
        classified[(uhi_data >= 0) & (uhi_data < 1)] = 1   # Weak
        classified[(uhi_data >= 1) & (uhi_data < 2)] = 2   # Moderate
        classified[(uhi_data >= 2) & (uhi_data < 3)] = 3   # Strong
        classified[uhi_data >= 3] = 4                        # Very Strong
        
        # Custom colormap
        colors = ['#3288bd', '#99d594', '#fee08b', '#fc8d59', '#d53e4f']
        cmap = mcolors.ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Plot
        im = ax.imshow(classified, cmap=cmap, norm=norm, 
                      interpolation='nearest', aspect='auto')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        # Colorbar with labels
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, 
                           boundaries=bounds, ticks=[0, 1, 2, 3, 4])
        cbar.set_label('UHI Category', fontsize=12)
        cbar.ax.set_yticklabels(['No UHI/Cooling', 'Weak\n(0-1°C)', 
                                 'Moderate\n(1-2°C)', 'Strong\n(2-3°C)', 
                                 'Very Strong\n(>3°C)'])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved UHI intensity map: {output_path}")
        plt.close()
    
    def create_hotspot_map(self, lst_data: np.ndarray, hotspot_mask: np.ndarray,
                          gi_star: np.ndarray, output_path: Path,
                          title: str = "UHI Hotspot Map"):
        """
        Create hotspot map with Gi* statistics
        
        Args:
            lst_data: LST data
            hotspot_mask: Binary hotspot mask
            gi_star: Gi* statistic map
            output_path: Path to save figure
            title: Plot title
        """
        logger.info(f"Creating hotspot map: {title}")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: LST with hotspot overlay
        ax = axes[0]
        
        # Base LST map
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                 '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        cmap = LinearSegmentedColormap.from_list('thermal', colors, N=256)
        
        im1 = ax.imshow(lst_data, cmap=cmap, alpha=0.8, 
                       interpolation='nearest', aspect='auto')
        
        # Overlay hotspots
        hotspot_overlay = np.ma.masked_where(~hotspot_mask, hotspot_mask)
        ax.contour(hotspot_overlay, levels=[0.5], colors='red', 
                  linewidths=2, linestyles='solid')
        
        ax.set_title('LST with Hotspot Boundaries', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label('LST (°C)', fontsize=11)
        
        # Right: Gi* statistic
        ax = axes[1]
        
        # Gi* colormap (diverging)
        im2 = ax.imshow(gi_star, cmap='RdYlBu_r', vmin=-3, vmax=3,
                       interpolation='nearest', aspect='auto')
        
        # Mark significant areas
        ax.contour(gi_star, levels=[1.96], colors='black', 
                  linewidths=1.5, linestyles='dashed', alpha=0.7)
        ax.contour(gi_star, levels=[2.58], colors='black', 
                  linewidths=2, linestyles='solid')
        
        ax.set_title('Gi* Statistic (Hotspot Analysis)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        cbar2.set_label('Gi* Statistic', fontsize=11)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved hotspot map: {output_path}")
        plt.close()
    
    def create_uncertainty_map(self, lst_data: np.ndarray, uncertainty: np.ndarray,
                              output_path: Path, title: str = "Prediction Uncertainty Map"):
        """
        Create uncertainty visualization
        
        Args:
            lst_data: LST predictions
            uncertainty: Uncertainty (standard deviation)
            output_path: Path to save figure
            title: Plot title
        """
        logger.info(f"Creating uncertainty map: {title}")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: LST predictions
        ax = axes[0]
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                 '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        cmap = LinearSegmentedColormap.from_list('thermal', colors, N=256)
        
        im1 = ax.imshow(lst_data, cmap=cmap, interpolation='nearest', aspect='auto')
        ax.set_title('LST Predictions', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label('Temperature (°C)', fontsize=11)
        
        # Right: Uncertainty
        ax = axes[1]
        im2 = ax.imshow(uncertainty, cmap='YlOrRd', interpolation='nearest', aspect='auto')
        ax.set_title('Prediction Uncertainty (Std Dev)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        cbar2.set_label('Uncertainty (°C)', fontsize=11)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved uncertainty map: {output_path}")
        plt.close()
    
    def create_statistics_dashboard(self, uhi_stats: Dict, classification: Dict,
                                hotspots_df: pd.DataFrame, output_path: Path):
        """
        Create comprehensive statistics dashboard
        
        Args:
            uhi_stats: UHI statistics dictionary
            classification: UHI classification counts
            hotspots_df: Hotspot dataframe
            output_path: Path to save figure
        """
        logger.info("Creating statistics dashboard...")
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # 1. UHI Statistics Summary (top-left)
        ax = fig.add_subplot(gs[0, :2])
        ax.axis('off')
        
        stats_text = f"""UHI CHARACTERIZATION SUMMARY

    Maximum Intensity: {uhi_stats['max_intensity']:.2f}°C
    Mean Intensity: {uhi_stats['mean_intensity']:.2f}°C
    Mean Positive Intensity: {uhi_stats['mean_positive_intensity']:.2f}°C
    Spatial Extent (>2°C): {uhi_stats['spatial_extent_km2']:.2f} km²
    UHI Magnitude: {uhi_stats['magnitude']:.2f}°C·pixels"""
        
        ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 2. UHI Classification Pie Chart (top-right)
        ax = fig.add_subplot(gs[0, 2])
        
        # Filter out zero values for cleaner pie chart
        labels = []
        sizes = []
        for label, size in classification.items():
            if size > 0:
                labels.append(label)
                sizes.append(size)
        
        if len(sizes) > 0:
            colors_pie = ['#3288bd', '#99d594', '#fee08b', '#fc8d59', '#d53e4f']
            # Use only as many colors as we have categories
            colors_pie = colors_pie[:len(labels)]
            
            ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 9})
            ax.set_title('UHI Classification', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No classification data', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        # 3. Top Hotspots Bar Chart (middle row, full width)
        ax = fig.add_subplot(gs[1, :])
        
        if len(hotspots_df) > 0:
            top_n = min(15, len(hotspots_df))
            top_hotspots = hotspots_df.head(top_n)
            
            x = np.arange(top_n)
            width = 0.35
            
            ax.bar(x - width/2, top_hotspots['area_km2'], width, 
                label='Area (km²)', color='steelblue', alpha=0.8)
            ax.bar(x + width/2, top_hotspots['mean_lst'] / 10, width,
                label='Mean LST / 10 (°C)', color='coral', alpha=0.8)
            
            ax.set_xlabel('Hotspot Rank', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(f'Top {top_n} Priority Hotspots', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'#{i+1}' for i in range(top_n)])
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No hotspot data available', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Top Priority Hotspots', fontsize=12, fontweight='bold')
        
        # 4. Hotspot Size Distribution (bottom-left)
        ax = fig.add_subplot(gs[2, 0])
        
        if len(hotspots_df) > 0 and 'area_km2' in hotspots_df.columns:
            ax.hist(hotspots_df['area_km2'], bins=min(20, len(hotspots_df)), 
                color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Hotspot Area (km²)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title('Hotspot Size Distribution', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No area data', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Hotspot Size Distribution', fontsize=11, fontweight='bold')
        
        # 5. Hotspot Temperature Distribution (bottom-middle)
        ax = fig.add_subplot(gs[2, 1])
        
        if len(hotspots_df) > 0 and 'mean_lst' in hotspots_df.columns:
            ax.hist(hotspots_df['mean_lst'], bins=min(20, len(hotspots_df)), 
                color='coral', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Mean LST (°C)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title('Hotspot Temperature Distribution', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No temperature data', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Hotspot Temperature Distribution', fontsize=11, fontweight='bold')
        
        # 6. Priority Score Distribution (bottom-right)
        ax = fig.add_subplot(gs[2, 2])
        
        if len(hotspots_df) > 0 and 'priority_score' in hotspots_df.columns:
            ax.hist(hotspots_df['priority_score'], bins=min(20, len(hotspots_df)), 
                color='green', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Priority Score', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title('Priority Score Distribution', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No priority data', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Priority Score Distribution', fontsize=11, fontweight='bold')
        
        plt.suptitle('UHI Analysis Dashboard', fontsize=18, fontweight='bold')
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved statistics dashboard: {output_path}")
        plt.close()


class OutputGenerator:
    """Generate final output products"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize output generator
        
        Args:
            output_dir: Directory for output products
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
    
    def export_geotiff(self, data: np.ndarray, output_name: str,
                      bounds: Tuple[float, float, float, float],
                      crs: str = "EPSG:32748", metadata: Optional[Dict] = None):
        """
        Export data as GeoTIFF
        
        Args:
            data: 2D array to export
            output_name: Output filename
            bounds: Geographic bounds (west, south, east, north)
            crs: Coordinate reference system
            metadata: Additional metadata
        """
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        
        output_path = self.output_dir / output_name
        logger.info(f"Exporting GeoTIFF: {output_name}")
        
        height, width = data.shape
        transform = from_bounds(*bounds, width, height)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs=CRS.from_string(crs),
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(data, 1)
            
            if metadata:
                dst.update_tags(**metadata)
        
        logger.info(f"Saved: {output_path}")
    
    def export_shapefile(self, hotspots_df: pd.DataFrame, 
                        geometry_col: str = 'geometry',
                        output_name: str = "hotspots.shp",
                        crs: str = "EPSG:32748"):
        """
        Export hotspots as shapefile
        
        Args:
            hotspots_df: Hotspot dataframe with geometry
            geometry_col: Name of geometry column
            output_name: Output filename
            crs: Coordinate reference system
        """
        import geopandas as gpd
        
        output_path = self.output_dir / output_name
        logger.info(f"Exporting shapefile: {output_name}")
        
        gdf = gpd.GeoDataFrame(hotspots_df, geometry=geometry_col, crs=crs)
        gdf.to_file(output_path)
        
        logger.info(f"Saved: {output_path}")
    
    def export_csv(self, data: pd.DataFrame, output_name: str):
        """
        Export data as CSV
        
        Args:
            data: DataFrame to export
            output_name: Output filename
        """
        output_path = self.output_dir / output_name
        logger.info(f"Exporting CSV: {output_name}")
        
        data.to_csv(output_path, index=False)
        
        logger.info(f"Saved: {output_path}")
    
    def create_metadata_file(self, metadata: Dict, output_name: str = "metadata.json"):
        """
        Create metadata file
        
        Args:
            metadata: Metadata dictionary
            output_name: Output filename
        """
        output_path = self.output_dir / output_name
        logger.info(f"Creating metadata file: {output_name}")
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved: {output_path}")


def create_time_series_animation(lst_maps: List[np.ndarray], dates: List[str],
                                 output_path: Path, fps: int = 2):
    """
    Create time-series animation
    
    Args:
        lst_maps: List of LST maps
        dates: List of date strings
        output_path: Output video path
        fps: Frames per second
    """
    logger.info(f"Creating time-series animation with {len(lst_maps)} frames...")
    
    import imageio
    
    # Create colormap
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
             '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    cmap = LinearSegmentedColormap.from_list('thermal', colors, N=256)
    
    # Find global min/max
    vmin = min(m.min() for m in lst_maps)
    vmax = max(m.max() for m in lst_maps)
    
    frames = []
    
    for i, (lst_map, date) in enumerate(zip(lst_maps, dates)):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(lst_map, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation='nearest', aspect='auto')
        
        ax.set_title(f'Land Surface Temperature - {date}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Temperature (°C)', fontsize=12)
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        
        plt.close()
    
    # Save as video
    imageio.mimsave(output_path, frames, fps=fps)
    logger.info(f"Saved animation: {output_path}")


if __name__ == "__main__":
    logger.info("UHI Visualization module loaded")
    logger.info("Use: from uhi_visualization import UHIVisualizer, OutputGenerator")