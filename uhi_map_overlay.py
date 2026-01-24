"""
UHI Map Overlay - Display UHI data on real maps using Folium and Contextily
Creates interactive and static maps with real geographic context
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patches as mpatches
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import folium
from folium.plugins import HeatMap, MarkerCluster
from branca.colormap import LinearColormap
import contextily as ctx

from config import *

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class RealMapOverlay:
    """Create UHI visualizations overlaid on real maps"""
    
    def __init__(self, bounds: Tuple[float, float, float, float], 
                 crs: str = "EPSG:32748"):
        """
        Initialize map overlay
        
        Args:
            bounds: Geographic bounds (west, south, east, north) in degrees
            crs: Coordinate reference system (default: UTM Zone 48S for Jakarta)
        """
        self.bounds = bounds  # (minx, miny, maxx, maxy)
        self.crs = crs
        
        # Calculate center - convert to native Python float
        self.center_lon = float((bounds[0] + bounds[2]) / 2)
        self.center_lat = float((bounds[1] + bounds[3]) / 2)
        
        logger.info(f"Map bounds: {bounds}")
        logger.info(f"Center: ({self.center_lat:.4f}, {self.center_lon:.4f})")
    
    def create_interactive_map(self, lst_map: np.ndarray, 
                              uhi_map: Optional[np.ndarray] = None,
                              hotspots_df: Optional[pd.DataFrame] = None,
                              output_path: Path = None,
                              add_heatmap: bool = True) -> folium.Map:
        """
        Create interactive Folium map with UHI overlay
        
        Args:
            lst_map: LST data array
            uhi_map: UHI intensity data array (optional)
            hotspots_df: Hotspot dataframe with coordinates
            output_path: Path to save HTML
            add_heatmap: Add heatmap layer
            
        Returns:
            Folium map object
        """
        logger.info("Creating interactive Folium map...")
        
        # Create base map
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=12,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        # Add different tile layers
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False
        ).add_to(m)
        
        # Create LST overlay as ImageOverlay with opacity
        if lst_map is not None:
            self._add_lst_overlay(m, lst_map, "Land Surface Temperature")
        
        # Create UHI overlay
        if uhi_map is not None:
            self._add_uhi_overlay(m, uhi_map, "UHI Intensity")
        
        # Add heatmap layer
        if add_heatmap and lst_map is not None:
            self._add_heatmap_layer(m, lst_map)
        
        # Add hotspot markers
        if hotspots_df is not None and len(hotspots_df) > 0:
            self._add_hotspot_markers(m, hotspots_df)
        
        # Add legend
        self._add_legend(m)
        
        # Add layer control
        folium.LayerControl(position='topright').add_to(m)
        
        # Add title
        title_html = '''
            <div style="position: fixed; 
                        top: 10px; left: 50px; width: 400px; height: 50px; 
                        background-color: white; border:2px solid grey; z-index:9999; 
                        font-size:16px; padding: 10px; border-radius: 5px;
                        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
                <b>Urban Heat Island Analysis - Jakarta</b><br>
                <small>Land Surface Temperature & Hotspots</small>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save if path provided
        if output_path:
            m.save(str(output_path))
            logger.info(f"✓ Saved interactive map: {output_path}")
        
        return m
    
    def _add_lst_overlay(self, m: folium.Map, lst_map: np.ndarray, name: str):
        """Add LST as semi-transparent overlay"""
        from PIL import Image
        from io import BytesIO
        import base64
        
        # Normalize LST to 0-255
        lst_norm = ((lst_map - lst_map.min()) / (lst_map.max() - lst_map.min()) * 255).astype(np.uint8)
        
        # Create RGBA image with colormap
        from matplotlib import cm
        colormap = cm.get_cmap('RdYlBu_r')
        lst_rgba = (colormap(lst_norm) * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(lst_rgba)
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Add as ImageOverlay
        folium.raster_layers.ImageOverlay(
            image=f'data:image/png;base64,{img_str}',
            bounds=[[self.bounds[1], self.bounds[0]], [self.bounds[3], self.bounds[2]]],
            opacity=0.6,
            name=name,
            overlay=True,
            control=True
        ).add_to(m)
    
    def _add_uhi_overlay(self, m: folium.Map, uhi_map: np.ndarray, name: str):
        """Add UHI intensity as overlay"""
        from PIL import Image
        from io import BytesIO
        import base64
        from matplotlib import cm
        
        # Classify UHI
        classified = np.zeros_like(uhi_map)
        classified[uhi_map < 0] = 0
        classified[(uhi_map >= 0) & (uhi_map < 1)] = 1
        classified[(uhi_map >= 1) & (uhi_map < 2)] = 2
        classified[(uhi_map >= 2) & (uhi_map < 3)] = 3
        classified[uhi_map >= 3] = 4
        
        # Create custom colormap
        colors = np.array([
            [50, 136, 189, 255],    # Blue (cooling)
            [153, 213, 148, 255],   # Green (weak)
            [254, 224, 139, 255],   # Yellow (moderate)
            [252, 141, 89, 255],    # Orange (strong)
            [213, 62, 79, 255]      # Red (very strong)
        ])
        
        # Map classifications to colors
        uhi_rgba = colors[classified.astype(int)]
        
        # Create PIL image
        img = Image.fromarray(uhi_rgba.astype(np.uint8))
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Add as ImageOverlay
        folium.raster_layers.ImageOverlay(
            image=f'data:image/png;base64,{img_str}',
            bounds=[[self.bounds[1], self.bounds[0]], [self.bounds[3], self.bounds[2]]],
            opacity=0.7,
            name=name,
            overlay=True,
            control=True
        ).add_to(m)
    
    def _add_heatmap_layer(self, m: folium.Map, lst_map: np.ndarray):
        """Add heatmap layer from LST data"""
        # Sample points from LST map
        height, width = lst_map.shape
        step = max(1, height // 50)  # Sample ~50x50 points
        
        heat_data = []
        lst_min = float(lst_map.min())
        lst_max = float(lst_map.max())
        
        for i in range(0, height, step):
            for j in range(0, width, step):
                # Calculate lat/lon for this pixel
                lat = float(self.bounds[1] + (i / height) * (self.bounds[3] - self.bounds[1]))
                lon = float(self.bounds[0] + (j / width) * (self.bounds[2] - self.bounds[0]))
                
                # Normalize temperature to 0-1 for weight
                weight = float((lst_map[i, j] - lst_min) / (lst_max - lst_min))
                
                heat_data.append([lat, lon, weight])
        
        # Add heatmap
        HeatMap(
            heat_data,
            name='Temperature Heatmap',
            min_opacity=0.3,
            max_zoom=15,
            radius=15,
            blur=20,
            gradient={0.0: 'blue', 0.5: 'yellow', 1.0: 'red'},
            overlay=True,
            control=True,
            show=False
        ).add_to(m)
    
    def _add_hotspot_markers(self, m: folium.Map, hotspots_df: pd.DataFrame):
        """Add hotspot markers to map"""
        # Create marker cluster
        marker_cluster = MarkerCluster(name='Hotspot Locations').add_to(m)
        
        for idx, row in hotspots_df.head(20).iterrows():  # Top 20 hotspots
            # Convert pixel coordinates to lat/lon
            if 'centroid_y' in row and 'centroid_x' in row:
                # Assuming coordinates are in pixels, need to convert
                # You may need to adjust this based on your coordinate system
                lat = self.center_lat  # Placeholder
                lon = self.center_lon  # Placeholder
            else:
                continue
            
            # Convert all numeric values to native Python types
            mean_lst = float(row.get('mean_lst', 0)) if pd.notna(row.get('mean_lst')) else 0.0
            max_lst = float(row.get('max_lst', 0)) if pd.notna(row.get('max_lst')) else 0.0
            area_km2 = float(row.get('area_km2', 0)) if pd.notna(row.get('area_km2')) else 0.0
            priority_score = float(row.get('priority_score', 0)) if pd.notna(row.get('priority_score')) else 0.0
            rank = int(row.get('rank', idx+1)) if pd.notna(row.get('rank')) else idx+1
            
            # Create popup content
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4 style="margin: 0;">Hotspot #{rank}</h4>
                <hr style="margin: 5px 0;">
                <b>Mean LST:</b> {mean_lst:.1f}°C<br>
                <b>Max LST:</b> {max_lst:.1f}°C<br>
                <b>Area:</b> {area_km2:.3f} km²<br>
                <b>Priority:</b> {priority_score:.3f}
            </div>
            """
            
            # Color based on priority
            if priority_score > 0.75:
                color = 'red'
                icon = 'fire'
            elif priority_score > 0.5:
                color = 'orange'
                icon = 'warning-sign'
            else:
                color = 'yellow'
                icon = 'info-sign'
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color=color, icon=icon, prefix='glyphicon'),
                tooltip=f"Hotspot #{rank}"
            ).add_to(marker_cluster)
    
    def _add_legend(self, m: folium.Map):
        """Add legend to map"""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px; border-radius: 5px;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
            <p style="margin: 0; font-weight: bold;">UHI Intensity</p>
            <hr style="margin: 5px 0;">
            <p style="margin: 3px 0;"><span style="background-color: #3288bd; width: 20px; height: 15px; display: inline-block;"></span> No UHI/Cooling</p>
            <p style="margin: 3px 0;"><span style="background-color: #99d594; width: 20px; height: 15px; display: inline-block;"></span> Weak (0-2°C)</p>
            <p style="margin: 3px 0;"><span style="background-color: #fee08b; width: 20px; height: 15px; display: inline-block;"></span> Moderate (2-4°C)</p>
            <p style="margin: 3px 0;"><span style="background-color: #fc8d59; width: 20px; height: 15px; display: inline-block;"></span> Strong (4-6°C)</p>
            <p style="margin: 3px 0;"><span style="background-color: #d53e4f; width: 20px; height: 15px; display: inline-block;"></span> Very Strong (>6°C)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def create_static_basemap_overlay(self, lst_map: np.ndarray,
                                     output_path: Path,
                                     title: str = "UHI Map with Basemap",
                                     basemap_source: str = 'OpenStreetMap',
                                     alpha: float = 0.6):
        """
        Create static map with basemap using contextily
        
        Args:
            lst_map: LST data array
            output_path: Path to save figure
            title: Map title
            basemap_source: Basemap provider (see contextily.providers)
            alpha: Transparency of LST overlay
        """
        logger.info(f"Creating static map with {basemap_source} basemap...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Convert bounds to Web Mercator for contextily
        from pyproj import Transformer
        
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        west_merc, south_merc = transformer.transform(self.bounds[0], self.bounds[1])
        east_merc, north_merc = transformer.transform(self.bounds[2], self.bounds[3])
        
        # Create colormap
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                 '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        cmap = LinearSegmentedColormap.from_list('thermal', colors, N=256)
        
        # Plot LST data
        extent = [west_merc, east_merc, south_merc, north_merc]
        im = ax.imshow(lst_map, extent=extent, cmap=cmap, 
                      alpha=alpha, interpolation='bilinear', zorder=2)
        
        # Add basemap
        try:
            if basemap_source == 'OpenStreetMap':
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=13)
            elif basemap_source == 'Satellite':
                ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=13)
            elif basemap_source == 'CartoDB':
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=13)
            else:
                ctx.add_basemap(ax, zoom=13)
        except Exception as e:
            logger.warning(f"Failed to add basemap: {e}")
            logger.warning("Continuing without basemap...")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Temperature (°C)', fontsize=14, fontweight='bold')
        
        # Add title
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        
        # Add scale bar
        self._add_scale_bar(ax, extent)
        
        # Add north arrow
        self._add_north_arrow(ax, extent)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved static map: {output_path}")
        plt.close()
    
    def create_side_by_side_comparison(self, lst_map: np.ndarray,
                                      uhi_map: np.ndarray,
                                      output_path: Path,
                                      basemap_source: str = 'OpenStreetMap'):
        """
        Create side-by-side comparison with and without basemap
        
        Args:
            lst_map: LST data
            uhi_map: UHI intensity data
            output_path: Output path
            basemap_source: Basemap provider
        """
        logger.info("Creating side-by-side comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Convert bounds to Web Mercator
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        west_merc, south_merc = transformer.transform(self.bounds[0], self.bounds[1])
        east_merc, north_merc = transformer.transform(self.bounds[2], self.bounds[3])
        extent = [west_merc, east_merc, south_merc, north_merc]
        
        # Left: LST with basemap
        ax = axes[0]
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                 '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        cmap = LinearSegmentedColormap.from_list('thermal', colors, N=256)
        
        im1 = ax.imshow(lst_map, extent=extent, cmap=cmap, 
                       alpha=0.6, interpolation='bilinear', zorder=2)
        
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=13)
        except:
            pass
        
        ax.set_title('Land Surface Temperature', fontsize=16, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label('LST (°C)', fontsize=12)
        
        # Right: UHI intensity with basemap
        ax = axes[1]
        
        # Classify UHI
        classified = np.zeros_like(uhi_map)
        classified[uhi_map < 0] = 0
        classified[(uhi_map >= 0) & (uhi_map < 1)] = 1
        classified[(uhi_map >= 1) & (uhi_map < 2)] = 2
        classified[(uhi_map >= 2) & (uhi_map < 3)] = 3
        classified[uhi_map >= 3] = 4
        
        from matplotlib.colors import ListedColormap, BoundaryNorm
        colors_uhi = ['#3288bd', '#99d594', '#fee08b', '#fc8d59', '#d53e4f']
        cmap_uhi = ListedColormap(colors_uhi)
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = BoundaryNorm(bounds, cmap_uhi.N)
        
        im2 = ax.imshow(classified, extent=extent, cmap=cmap_uhi, norm=norm,
                       alpha=0.7, interpolation='nearest', zorder=2)
        
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=13)
        except:
            pass
        
        ax.set_title('UHI Intensity Classification', fontsize=16, fontweight='bold')
        cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04,
                            boundaries=bounds, ticks=[0, 1, 2, 3, 4])
        cbar2.ax.set_yticklabels(['No UHI', 'Weak', 'Moderate', 'Strong', 'Very Strong'])
        
        plt.suptitle('Urban Heat Island Analysis - Jakarta', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved comparison map: {output_path}")
        plt.close()
    
    def _add_scale_bar(self, ax, extent):
        """Add scale bar to map"""
        from matplotlib_scalebar.scalebar import ScaleBar
        
        try:
            scalebar = ScaleBar(1, location='lower left', 
                              box_alpha=0.7, color='black')
            ax.add_artist(scalebar)
        except:
            # Fallback if matplotlib_scalebar not available
            pass
    
    def _add_north_arrow(self, ax, extent):
        """Add north arrow to map"""
        # Simple north arrow
        x = extent[1] - (extent[1] - extent[0]) * 0.1
        y = extent[3] - (extent[3] - extent[2]) * 0.1
        
        ax.annotate('N', xy=(x, y), xytext=(x, y - (extent[3] - extent[2]) * 0.05),
                   fontsize=16, fontweight='bold', ha='center',
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                   bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black'))
    
    def export_to_webmap(self, lst_map: np.ndarray, uhi_map: np.ndarray,
                        hotspots_df: pd.DataFrame, output_dir: Path):
        """
        Export complete web map package
        
        Args:
            lst_map: LST data
            uhi_map: UHI intensity data
            hotspots_df: Hotspot dataframe
            output_dir: Output directory for web files
        """
        logger.info("Exporting web map package...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create interactive map
        web_map = self.create_interactive_map(
            lst_map, uhi_map, hotspots_df,
            output_path=output_dir / "index.html"
        )
        
        # Create static maps
        self.create_static_basemap_overlay(
            lst_map,
            output_dir / "lst_basemap.png",
            basemap_source='OpenStreetMap'
        )
        
        self.create_side_by_side_comparison(
            lst_map, uhi_map,
            output_dir / "comparison_basemap.png"
        )
        
        # Create README
        readme_content = f"""
# Urban Heat Island Web Map

## Files
- `index.html` - Interactive web map (open in browser)
- `lst_basemap.png` - Static LST map with OpenStreetMap
- `comparison_basemap.png` - Side-by-side LST and UHI comparison

## Usage
1. Open `index.html` in a web browser
2. Use layer controls (top right) to toggle different layers
3. Click on hotspot markers for detailed information
4. Switch between different basemap styles

## Map Bounds
- West: {self.bounds[0]:.4f}°
- South: {self.bounds[1]:.4f}°
- East: {self.bounds[2]:.4f}°
- North: {self.bounds[3]:.4f}°

## Layers
- **OpenStreetMap** - Default street map
- **Light Map** - Minimalist CartoDB basemap
- **Dark Map** - Dark theme CartoDB basemap
- **Satellite** - Esri World Imagery
- **LST Overlay** - Land Surface Temperature
- **UHI Overlay** - UHI Intensity Classification
- **Temperature Heatmap** - Smooth temperature gradient
- **Hotspot Markers** - Top priority hotspots

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        logger.info(f"✓ Web map package saved to: {output_dir}")
        logger.info(f"  Open {output_dir / 'index.html'} in your browser!")


def demo_usage():
    """Demonstrate usage"""
    logger.info("="*60)
    logger.info("REAL MAP OVERLAY DEMO")
    logger.info("="*60)
    
    # Example: Jakarta bounds (adjust to your actual data)
    bounds = (106.6, -6.4, 107.1, -6.0)  # (west, south, east, north)
    
    # Load example data (replace with your actual data)
    data_dir = OUTPUT_DIR / "predictions" / "data"
    
    if not data_dir.exists():
        logger.error("No prediction data found. Run inference first!")
        return
    
    lst_map = np.load(data_dir / "lst_processed.npy")
    uhi_map = np.load(data_dir / "uhi_intensity.npy")
    
    hotspots_df = pd.DataFrame()
    if (data_dir / "hotspots.csv").exists():
        hotspots_df = pd.read_csv(data_dir / "hotspots.csv")
    
    # Create map overlay
    overlay = RealMapOverlay(bounds)
    
    # Export web map package
    webmap_dir = OUTPUT_DIR / "webmap"
    overlay.export_to_webmap(lst_map, uhi_map, hotspots_df, webmap_dir)
    
    logger.info("\n" + "="*60)
    logger.info("DEMO COMPLETE")
    logger.info("="*60)
    logger.info(f"Open: {webmap_dir / 'index.html'}")


if __name__ == "__main__":
    demo_usage()