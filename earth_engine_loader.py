"""
Google Earth Engine data downloader
Purpose: Download RAW satellite bands from Earth Engine for later processing
Does NOT perform any processing - just downloads raw data
"""
import sys
import ee
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from pathlib import Path
import urllib.request
import tempfile
import rasterio

from config import *

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class EarthEngineLoader:
    """Download RAW satellite bands from Earth Engine"""
    
    def __init__(self, sensors: List[str] = ["landsat", "sentinel2"]):
        """
        Initialize Earth Engine loader
        
        Args:
            sensors: List of sensors to download ('landsat', 'sentinel2')
        """
        try:
            ee.Initialize(project='nukobot-366809')
            logger.info("Earth Engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Earth Engine: {e}")
            logger.info("Please authenticate with: earthengine authenticate")
            raise
        
        self.sensors = sensors
        
    def get_study_area(self) -> ee.Geometry:
        """Get study area geometry from config"""
        bounds = STUDY_AREA["bounds"]
        return ee.Geometry.Rectangle([
            bounds["min_lon"],
            bounds["min_lat"],
            bounds["max_lon"],
            bounds["max_lat"]
        ])
    
    def load_landsat_collection(self, start_date: str, end_date: str,
                               cloud_threshold: int = 50) -> ee.ImageCollection:
        """
        Load Landsat 8/9 image collection
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cloud_threshold: Maximum cloud cover percentage
            
        Returns:
            Filtered image collection
        """
        study_area = self.get_study_area()
        
        collection = (ee.ImageCollection(LANDSAT_CONFIG["collection"])
                     .filterBounds(study_area)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUD_COVER', cloud_threshold)))
        
        size = collection.size().getInfo()
        logger.info(f"Found {size} Landsat images")
        return collection
    
    def load_sentinel2_collection(self, start_date: str, end_date: str,
                                 cloud_threshold: int = 50) -> ee.ImageCollection:
        """
        Load Sentinel-2 image collection
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            cloud_threshold: Maximum cloud cover percentage
            
        Returns:
            Filtered image collection
        """
        study_area = self.get_study_area()
        
        collection = (ee.ImageCollection(SENTINEL2_CONFIG["collection"])
                     .filterBounds(study_area)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)))
        
        size = collection.size().getInfo()
        logger.info(f"Found {size} Sentinel-2 images")
        return collection
    
    def apply_cloud_mask_landsat(self, image: ee.Image) -> ee.Image:
        """
        Apply cloud mask to Landsat image using QA_PIXEL band
        
        Args:
            image: Landsat image
            
        Returns:
            Cloud-masked image
        """
        qa = image.select('QA_PIXEL')
        
        # Bit 3: Cloud, Bit 4: Cloud Shadow
        cloud = qa.bitwiseAnd(1 << 3).eq(0)
        shadow = qa.bitwiseAnd(1 << 4).eq(0)
        
        mask = cloud.And(shadow)
        return image.updateMask(mask)
    
    def apply_cloud_mask_sentinel2(self, image: ee.Image) -> ee.Image:
        """
        Apply cloud mask to Sentinel-2 image using SCL band
        
        Args:
            image: Sentinel-2 image
            
        Returns:
            Cloud-masked image
        """
        scl = image.select('SCL')
        
        # Exclude: saturated (1), cloud shadows (3), clouds (8,9), cirrus (10), snow/ice (11)
        mask = (scl.neq(1).And(scl.neq(3))
                .And(scl.neq(8)).And(scl.neq(9))
                .And(scl.neq(10)).And(scl.neq(11)))
        
        return image.updateMask(mask)
    
    def create_composite(self, collection: ee.ImageCollection,
                        sensor_type: str,
                        method: str = "median") -> ee.Image:
        """
        Create temporal composite from image collection
        
        Args:
            collection: Image collection
            sensor_type: 'landsat' or 'sentinel2'
            method: Compositing method ('median' or 'mean')
            
        Returns:
            Composite image
        """
        # Apply cloud masking
        if sensor_type == "landsat":
            masked = collection.map(self.apply_cloud_mask_landsat)
        else:
            masked = collection.map(self.apply_cloud_mask_sentinel2)
        
        # Create composite
        if method == "median":
            composite = masked.median()
        elif method == "mean":
            composite = masked.mean()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Created {method} composite from {collection.size().getInfo()} images")
        return composite
    
    def download_band(self, image: ee.Image, band_name: str,
                     region: ee.Geometry, scale: int,
                     dimensions: List[int]) -> Optional[np.ndarray]:
        """
        Download a single band as numpy array
        
        Args:
            image: Earth Engine image
            band_name: Band name to download
            region: Region geometry
            scale: Resolution in meters
            dimensions: [width, height] in pixels
            
        Returns:
            Numpy array or None if failed
        """
        try:
            logger.info(f"  Downloading {band_name}...")
            
            band_image = image.select(band_name)
            
            # Get value range for this band
            stats = band_image.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=region,
                scale=scale * 10,  # Use coarser scale for stats
                maxPixels=1e8
            ).getInfo()
            
            band_min = stats.get(f'{band_name}_min')
            band_max = stats.get(f'{band_name}_max')
            
            if band_min is None or band_max is None:
                logger.warning(f"    No valid data for {band_name}")
                return None
            
            if band_min == band_max:
                band_max = band_min + 1
            
            logger.info(f"    Range: [{band_min:.4f}, {band_max:.4f}]")
            
            # Download as GeoTIFF
            url = band_image.getThumbURL({
                'region': region,
                'dimensions': dimensions,
                'format': 'GEO_TIFF'
            })
            
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                urllib.request.urlretrieve(url, tmp_path)
                
                with rasterio.open(tmp_path) as src:
                    arr = src.read(1).astype(np.float32)
                    
                    # Handle nodata
                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan
                    
                    # Denormalize if needed (Earth Engine returns normalized values)
                    if arr.max() <= 255 and arr.min() >= 0:
                        arr = arr / 255.0 * (band_max - band_min) + band_min
                    
                    logger.info(f"    ✓ Shape: {arr.shape}, "
                              f"Range: [{np.nanmin(arr):.2f}, {np.nanmax(arr):.2f}]")
                    return arr
                    
            finally:
                Path(tmp_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"    ✗ Failed to download {band_name}: {e}")
            return None
    
    def download_landsat_data(self, composite: ee.Image,
                             output_file: Path,
                             scale: int = 30,
                             max_dimension: int = 2000) -> bool:
        """
        Download all Landsat bands needed for analysis
        
        Args:
            composite: Landsat composite image
            output_file: Output file path
            scale: Resolution in meters
            max_dimension: Maximum dimension to avoid size limits
            
        Returns:
            True if successful
        """
        region = self.get_study_area()
        
        # Calculate dimensions
        bounds = region.bounds().getInfo()['coordinates'][0]
        min_lon, min_lat = bounds[0]
        max_lon, max_lat = bounds[2]
        
        width_km = (max_lon - min_lon) * 111
        height_km = (max_lat - min_lat) * 111
        width_px = int(width_km * 1000 / scale)
        height_px = int(height_km * 1000 / scale)
        
        # Limit dimensions
        if width_px > max_dimension or height_px > max_dimension:
            aspect_ratio = width_px / height_px
            if aspect_ratio > 1:
                width_px = max_dimension
                height_px = int(max_dimension / aspect_ratio)
            else:
                height_px = max_dimension
                width_px = int(max_dimension * aspect_ratio)
            logger.warning(f"  Limiting to {height_px}x{width_px} pixels")
        
        logger.info(f"  Resolution: {height_px}x{width_px} at {scale}m")
        
        # Bands to download
        bands_to_download = [
            'SR_B1',   # Coastal/Aerosol
            'SR_B2',   # Blue
            'SR_B3',   # Green
            'SR_B4',   # Red
            'SR_B5',   # NIR
            'SR_B6',   # SWIR1
            'SR_B7',   # SWIR2
            'ST_B10',  # Thermal
            'QA_PIXEL' # Quality band
        ]
        
        arrays = {}
        for band_name in bands_to_download:
            arr = self.download_band(
                composite, band_name, region, scale, [width_px, height_px]
            )
            if arr is not None:
                arrays[band_name] = arr
        
        # Save to disk
        if arrays:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_file, **arrays)
            logger.info(f"  ✓ Saved {len(arrays)} bands to {output_file}")
            return True
        else:
            logger.error(f"  ✗ No bands downloaded")
            return False
    
    def download_sentinel2_data(self, composite: ee.Image,
                               output_file: Path,
                               scale: int = 10,
                               max_dimension: int = 1500) -> bool:
        """
        Download all Sentinel-2 bands needed for analysis
        
        Args:
            composite: Sentinel-2 composite image
            output_file: Output file path
            scale: Resolution in meters
            max_dimension: Maximum dimension to avoid size limits
            
        Returns:
            True if successful
        """
        region = self.get_study_area()
        
        # Calculate dimensions
        bounds = region.bounds().getInfo()['coordinates'][0]
        min_lon, min_lat = bounds[0]
        max_lon, max_lat = bounds[2]
        
        width_km = (max_lon - min_lon) * 111
        height_km = (max_lat - min_lat) * 111
        width_px = int(width_km * 1000 / scale)
        height_px = int(height_km * 1000 / scale)
        
        # Limit dimensions
        if width_px > max_dimension or height_px > max_dimension:
            aspect_ratio = width_px / height_px
            if aspect_ratio > 1:
                width_px = max_dimension
                height_px = int(max_dimension / aspect_ratio)
            else:
                height_px = max_dimension
                width_px = int(max_dimension * aspect_ratio)
            logger.warning(f"  Limiting to {height_px}x{width_px} pixels")
        
        logger.info(f"  Resolution: {height_px}x{width_px} at {scale}m")
        
        # Bands to download
        bands_to_download = [
            'B2',   # Blue (10m)
            'B3',   # Green (10m)
            'B4',   # Red (10m)
            'B8',   # NIR (10m)
            'B11',  # SWIR1 (20m)
            'B12',  # SWIR2 (20m)
            'SCL'   # Scene Classification (20m)
        ]
        
        arrays = {}
        for band_name in bands_to_download:
            arr = self.download_band(
                composite, band_name, region, scale, [width_px, height_px]
            )
            if arr is not None:
                arrays[band_name] = arr
        
        # Save to disk
        if arrays:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_file, **arrays)
            logger.info(f"  ✓ Saved {len(arrays)} bands to {output_file}")
            return True
        else:
            logger.error(f"  ✗ No bands downloaded")
            return False
    
    def download_monthly_data(self, year: int, month: int,
                             output_dir: Path) -> Dict[str, Optional[Path]]:
        """
        Download monthly data for all configured sensors
        
        Args:
            year: Year
            month: Month (1-12)
            output_dir: Output directory
            
        Returns:
            Dictionary mapping sensor -> output file path (or None if failed)
        """
        # Define date range
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"DOWNLOADING DATA: {year}-{month:02d}")
        logger.info(f"Date range: {start_str} to {end_str}")
        logger.info(f"{'='*70}")
        
        results = {}
        
        # Download Landsat
        if "landsat" in self.sensors:
            logger.info("\n--- LANDSAT 8/9 ---")
            try:
                collection = self.load_landsat_collection(start_str, end_str)
                
                if collection.size().getInfo() == 0:
                    logger.warning("No Landsat images available")
                    results["landsat"] = None
                else:
                    composite = self.create_composite(collection, "landsat")
                    output_file = output_dir / "landsat" / f"landsat_{year}_{month:02d}.npz"
                    
                    success = self.download_landsat_data(composite, output_file)
                    results["landsat"] = output_file if success else None
                    
            except Exception as e:
                logger.error(f"Landsat download failed: {e}")
                import traceback
                traceback.print_exc()
                results["landsat"] = None
        
        # Download Sentinel-2
        if "sentinel2" in self.sensors:
            logger.info("\n--- SENTINEL-2 ---")
            try:
                collection = self.load_sentinel2_collection(start_str, end_str)
                
                if collection.size().getInfo() == 0:
                    logger.warning("No Sentinel-2 images available")
                    results["sentinel2"] = None
                else:
                    composite = self.create_composite(collection, "sentinel2")
                    output_file = output_dir / "sentinel2" / f"sentinel2_{year}_{month:02d}.npz"
                    
                    success = self.download_sentinel2_data(composite, output_file)
                    results["sentinel2"] = output_file if success else None
                    
            except Exception as e:
                logger.error(f"Sentinel-2 download failed: {e}")
                import traceback
                traceback.print_exc()
                results["sentinel2"] = None
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("DOWNLOAD SUMMARY")
        logger.info(f"{'='*70}")
        for sensor, filepath in results.items():
            if filepath and filepath.exists():
                logger.info(f"{sensor.upper()}: ✓ {filepath}")
            else:
                logger.info(f"{sensor.upper()}: ✗ Failed")
        
        return results
    
    def download_date_range(self, start_year: int, start_month: int,
                           end_year: int, end_month: int,
                           output_dir: Path) -> List[Dict[str, Optional[Path]]]:
        """
        Download data for a range of months
        
        Args:
            start_year: Start year
            start_month: Start month
            end_year: End year
            end_month: End month
            output_dir: Output directory
            
        Returns:
            List of results for each month
        """
        all_results = []
        
        current_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)
        
        while current_date <= end_date:
            results = self.download_monthly_data(
                current_date.year,
                current_date.month,
                output_dir
            )
            all_results.append(results)
            
            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        logger.info(f"\n{'='*70}")
        logger.info("ALL DOWNLOADS COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total months processed: {len(all_results)}")
        
        return all_results


def main():
    """Download satellite data for training"""
    logger.info("="*70)
    logger.info("SATELLITE DATA DOWNLOAD")
    logger.info("="*70)
    
    # Initialize loader
    loader = EarthEngineLoader(sensors=["landsat", "sentinel2"])
    
    # Define output directory
    output_dir = RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download data for training period
    # Example: January 2024 to June 2024
    results = loader.download_date_range(
        start_year=2024,
        start_month=1,
        end_year=2024,
        end_month=6,
        output_dir=output_dir
    )
    
    # Summary
    total_success = sum(1 for r in results if any(r.values()))
    logger.info(f"\n{'='*70}")
    logger.info(f"Successfully downloaded: {total_success}/{len(results)} months")
    logger.info(f"Data saved to: {output_dir}")
    logger.info(f"{'='*70}")
    logger.info("\nNext step: Run preprocessing.py to process the downloaded data")


if __name__ == "__main__":
    main()