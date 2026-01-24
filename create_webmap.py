"""
Standalone script to create web map from existing predictions
Run this after pipeline to create interactive maps
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse

from config import *
from uhi_map_overlay import RealMapOverlay

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Create interactive web map from UHI analysis results"
    )
    parser.add_argument("--data-dir", type=str, 
                       default="outputs/data",
                       help="Directory containing analysis results")
    parser.add_argument("--output-dir", type=str,
                       default="webmap",
                       help="Output directory for web map")
    parser.add_argument("--lst-file", type=str, default="lst_processed.npy",
                       help="LST data file name")
    parser.add_argument("--uhi-file", type=str, default="uhi_intensity.npy",
                       help="UHI intensity file name")
    parser.add_argument("--hotspots-file", type=str, default="hotspots.csv",
                       help="Hotspots CSV file name")
    
    args = parser.parse_args()
    
    # Extract bounds from config
    bounds = (
        STUDY_AREA["bounds"]["min_lon"],
        STUDY_AREA["bounds"]["min_lat"],
        STUDY_AREA["bounds"]["max_lon"],
        STUDY_AREA["bounds"]["max_lat"]
    )
    
    logger.info("="*70)
    logger.info("INTERACTIVE WEB MAP CREATOR")
    logger.info("="*70)
    logger.info(f"Study Area: {STUDY_AREA['name']}")
    logger.info(f"Bounds: {bounds}")
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Check data directory
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Run the pipeline first to generate analysis results")
        return
    
    # Load data
    logger.info("\nLoading data...")
    
    lst_path = data_dir / args.lst_file
    if not lst_path.exists():
        logger.error(f"LST file not found: {lst_path}")
        return
    lst_map = np.load(lst_path)
    logger.info(f"✓ Loaded LST: {lst_map.shape}")
    
    uhi_path = data_dir / args.uhi_file
    if not uhi_path.exists():
        logger.warning(f"UHI file not found: {uhi_path}")
        uhi_map = None
    else:
        uhi_map = np.load(uhi_path)
        logger.info(f"✓ Loaded UHI: {uhi_map.shape}")
    
    hotspots_path = data_dir / args.hotspots_file
    if not hotspots_path.exists():
        logger.warning(f"Hotspots file not found: {hotspots_path}")
        hotspots_df = pd.DataFrame()
    else:
        hotspots_df = pd.read_csv(hotspots_path)
        logger.info(f"✓ Loaded {len(hotspots_df)} hotspots")
    
    # Create map overlay
    logger.info("\nCreating web map...")
    
    overlay = RealMapOverlay(bounds)
    
    # Generate all map products
    logger.info("\n" + "="*70)
    logger.info("GENERATING MAP PRODUCTS")
    logger.info("="*70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Interactive web map
    logger.info("\n1. Creating interactive Folium map...")
    web_map = overlay.create_interactive_map(
        lst_map,
        uhi_map,
        hotspots_df,
        output_path=output_dir / "index.html",
        add_heatmap=True
    )
    
    # 2. Static map with basemap
    logger.info("\n2. Creating static OpenStreetMap overlay...")
    try:
        overlay.create_static_basemap_overlay(
            lst_map,
            output_path=output_dir / "lst_openstreetmap.png",
            title=f"Land Surface Temperature - {STUDY_AREA['name']}",
            basemap_source='OpenStreetMap',
            alpha=0.6
        )
    except Exception as e:
        logger.error(f"Failed to create OSM overlay: {e}")
        logger.error("Install contextily: pip install contextily")
    
    # 3. Satellite basemap
    logger.info("\n3. Creating static satellite overlay...")
    try:
        overlay.create_static_basemap_overlay(
            lst_map,
            output_path=output_dir / "lst_satellite.png",
            title="Land Surface Temperature - Satellite View",
            basemap_source='Satellite',
            alpha=0.5
        )
    except Exception as e:
        logger.error(f"Failed to create satellite overlay: {e}")
    
    # 4. Side-by-side comparison
    if uhi_map is not None:
        logger.info("\n4. Creating side-by-side comparison...")
        try:
            overlay.create_side_by_side_comparison(
                lst_map,
                uhi_map,
                output_path=output_dir / "comparison_map.png",
                basemap_source='OpenStreetMap'
            )
        except Exception as e:
            logger.error(f"Failed to create comparison: {e}")
    
    # Create README
    logger.info("\n5. Creating README...")
    readme_content = f"""
# Urban Heat Island Interactive Web Map
## {STUDY_AREA['name']}

## Quick Start
**Open `index.html` in your web browser!**

## Files Generated

### Interactive Maps
- **`index.html`** - Main interactive web map
  - Multiple basemap layers (street, satellite, light, dark)
  - Toggle LST and UHI overlays
  - Temperature heatmap
  - Clickable hotspot markers
  - Layer controls in top-right corner

### Static Maps
- **`lst_openstreetmap.png`** - LST overlay on street map
- **`lst_satellite.png`** - LST overlay on satellite imagery
- **`comparison_map.png`** - Side-by-side LST and UHI comparison

## Map Features

### Basemap Layers (toggle in top-right)
- **OpenStreetMap** - Default street map with labels
- **Light Map** - Minimalist CartoDB design
- **Dark Map** - Dark theme for contrast
- **Satellite** - Esri World Imagery

### Data Layers
- **Land Surface Temperature** - Semi-transparent thermal overlay
- **UHI Intensity** - Classified into 5 categories
- **Temperature Heatmap** - Smooth gradient visualization
- **Hotspot Markers** - Top 20 priority locations

### Hotspot Information
Click any hotspot marker to see:
- Temperature statistics
- Area coverage
- Priority score

## Map Coverage
- **Area:** {STUDY_AREA['name']}
- **West:** {bounds[0]:.4f}°
- **South:** {bounds[1]:.4f}°
- **East:** {bounds[2]:.4f}°
- **North:** {bounds[3]:.4f}°
- **CRS:** EPSG:{STUDY_AREA['epsg']}

## Legend

### UHI Intensity Categories
- 🔵 **No UHI/Cooling** - Temperature below rural reference
- 🟢 **Weak** - 0-2°C above rural reference
- 🟡 **Moderate** - 2-4°C above rural reference
- 🟠 **Strong** - 4-6°C above rural reference
- 🔴 **Very Strong** - >6°C above rural reference

## Usage Tips

1. **Zoom and Pan** - Use mouse wheel or +/- buttons
2. **Switch Layers** - Use layer control (top-right)
3. **Adjust Opacity** - Multiple layers can be visible
4. **View Details** - Click hotspot markers for info
5. **Share** - Send index.html to others (it's standalone!)

## Technical Details
- Coordinate System: WGS84 (EPSG:4326)
- Resolution: 50m per pixel
- Data Source: Landsat 8/9 satellite imagery
- Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Requirements (for viewing)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection (for basemap tiles)
- No additional software needed!

## Data Statistics
- LST Range: [{lst_map.min():.1f}, {lst_map.max():.1f}]°C
- LST Mean: {lst_map.mean():.1f}°C
- LST Std: {lst_map.std():.1f}°C
"""
    
    if uhi_map is not None:
        readme_content += f"""
- UHI Range: [{uhi_map.min():.1f}, {uhi_map.max():.1f}]°C
- UHI Mean: {uhi_map.mean():.1f}°C
"""
    
    if len(hotspots_df) > 0:
        readme_content += f"""
- Hotspots Detected: {len(hotspots_df)}
- Highest Priority: #{hotspots_df.iloc[0]['rank'] if 'rank' in hotspots_df else 1}
"""
    
    readme_content += """

---
Generated by Urban Heat Island Analysis Pipeline
"""
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("WEB MAP CREATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"\n📁 Output directory: {output_dir.absolute()}")
    logger.info(f"\n🌐 Main Files:")
    logger.info(f"   - {output_dir / 'index.html'} (open this in browser!)")
    logger.info(f"   - {output_dir / 'README.md'}")
    
    if (output_dir / "lst_openstreetmap.png").exists():
        logger.info(f"   - {output_dir / 'lst_openstreetmap.png'}")
    if (output_dir / "comparison_map.png").exists():
        logger.info(f"   - {output_dir / 'comparison_map.png'}")
    
    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS")
    logger.info("="*70)
    logger.info("\n1. Open index.html in your web browser")
    logger.info("2. Use layer controls (top-right) to toggle layers")
    logger.info("3. Click hotspot markers for details")
    logger.info("4. Try different basemap styles")
    logger.info("5. Share the HTML file with collaborators!")
    logger.info("\n" + "="*70)


if __name__ == "__main__":
    main()