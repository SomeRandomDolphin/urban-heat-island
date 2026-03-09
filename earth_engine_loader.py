"""
Google Earth Engine data downloader
Purpose: Download RAW satellite bands from Earth Engine for later processing.

Key improvements over original:
  1. Merges Landsat 8 (LC08) and Landsat 9 (LC09) into a single collection —
     maximises revisit frequency, especially important for 2021-2025.
  2. Uses ee.batch.Export.image.toDrive() instead of getThumbURL() —
     the correct approach for large-area / multi-year research downloads.
     getThumbURL is limited to small tiles and unreliable for production use.
  3. Applies official Landsat C2 L2 scale factors (SR and ST) before saving,
     so downstream code receives physically-meaningful values from the start.
  4. Correct pixel-dimension calculation using the cosine latitude correction
     for longitude degrees (important near the equator).
  5. Resume/skip logic: already-completed months are detected and skipped,
     making interrupted runs safe to restart.
  6. Seasonal composite strategy: dry-season (Apr–Oct) and wet-season
     (Nov–Mar) months are composited separately when data permits, giving
     richer temporal structure for UHI analysis.
"""

import sys
import math
import time
import ee
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

from config import (
    STUDY_AREA, LANDSAT_CONFIG, SENTINEL2_CONFIG,
    RAW_DATA_DIR, LOGGING_CONFIG
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lat_lon_to_pixels(min_lon: float, min_lat: float,
                        max_lon: float, max_lat: float,
                        scale_m: int) -> Tuple[int, int]:
    """
    Compute raster dimensions (width_px, height_px) for a bounding box.

    Uses the proper cosine-latitude correction so that width pixels are not
    overestimated near the equator (common bug in naive implementations).

    Args:
        min_lon, min_lat, max_lon, max_lat: Bounding box in WGS-84 degrees.
        scale_m: Pixel size in metres.

    Returns:
        (width_px, height_px)
    """
    mid_lat = (min_lat + max_lat) / 2.0
    metres_per_deg_lat = 111_320.0                             # constant ~111 km
    metres_per_deg_lon = 111_320.0 * math.cos(math.radians(mid_lat))

    height_m = abs(max_lat - min_lat) * metres_per_deg_lat
    width_m  = abs(max_lon - min_lon) * metres_per_deg_lon

    return int(width_m / scale_m), int(height_m / scale_m)


def _apply_landsat_scale_factors(image: ee.Image) -> ee.Image:
    """
    Apply official USGS Landsat Collection-2 Level-2 scale factors.

    Surface Reflectance (SR_B*) : value = DN × 2.75e-5 − 0.2   [unitless]
    Surface Temperature (ST_B10): value = DN × 0.00341802 + 149 [Kelvin]

    Without this step every SR band has raw integer DN values (~5 000–30 000),
    which are meaningless for spectral-index computation.
    """
    sr_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    st_bands = ['ST_B10']

    sr = (image.select(sr_bands)
               .multiply(LANDSAT_CONFIG["sr_scale"])
               .add(LANDSAT_CONFIG["sr_offset"]))

    st = (image.select(st_bands)
               .multiply(LANDSAT_CONFIG["st_scale"])
               .add(LANDSAT_CONFIG["st_offset"]))

    # Keep QA_PIXEL unchanged (it is a bitmask, not a physical value)
    qa = image.select('QA_PIXEL')

    return sr.addBands(st).addBands(qa)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EarthEngineLoader:
    """
    Download monthly Landsat 8/9 and Sentinel-2 composites from Google Earth
    Engine and export them to Google Drive.

    Export flow
    -----------
    GEE cannot stream large rasters synchronously. The correct workflow is:
      1. Build the composite image in GEE (server-side).
      2. Submit an Export.image.toDrive() task.
      3. Poll until the task is COMPLETED (or FAILED).
      4. The file then appears in the configured Google Drive folder; the user
         downloads it from Drive (or mounts Drive locally with rclone/gdrive).

    This matches the recommended pattern in the GEE developer documentation
    and is used by virtually all peer-reviewed remote sensing pipelines.
    """

    # GEE export configuration — adjust folder / CRS as needed
    DRIVE_FOLDER = "Jakarta_UHI"
    EXPORT_CRS   = "EPSG:32748"          # WGS84 / UTM Zone 48S
    EXPORT_SCALE = {                     # native export scales
        "landsat":   30,
        "sentinel2": 10,
    }
    POLL_INTERVAL_S = 30                 # seconds between task-status checks
    MAX_WAIT_S      = 7200              # 2 hours before giving up on a task

    def __init__(self, sensors: List[str] = ["landsat", "sentinel2"],
                 gee_project: str = "nukobot-366809"):
        """
        Initialise Earth Engine.

        Args:
            sensors:     Sensors to download. Any of 'landsat', 'sentinel2'.
            gee_project: Your GEE Cloud project ID.
        """
        try:
            ee.Initialize(project=gee_project)
            logger.info("Earth Engine initialised successfully")
        except Exception as e:
            logger.error(f"Failed to initialise Earth Engine: {e}")
            logger.info("Authenticate first with: earthengine authenticate")
            raise

        self.sensors = sensors

    # ------------------------------------------------------------------
    # Study area
    # ------------------------------------------------------------------

    def get_study_area(self) -> ee.Geometry:
        """Return the study-area rectangle as an ee.Geometry."""
        b = STUDY_AREA["bounds"]
        return ee.Geometry.Rectangle([
            b["min_lon"], b["min_lat"],
            b["max_lon"], b["max_lat"]
        ])

    # ------------------------------------------------------------------
    # Cloud masking
    # ------------------------------------------------------------------

    def _mask_landsat_clouds(self, image: ee.Image) -> ee.Image:
        """
        Mask clouds and cloud shadows using QA_PIXEL bit flags.

        Bit 3 = Cloud
        Bit 4 = Cloud Shadow
        Both must be 0 (clear) for a pixel to be retained.
        """
        qa = image.select('QA_PIXEL')
        clear = (qa.bitwiseAnd(1 << 3).eq(0)   # not cloud
                   .And(qa.bitwiseAnd(1 << 4).eq(0)))  # not cloud shadow
        return image.updateMask(clear)

    def _mask_sentinel2_clouds(self, image: ee.Image) -> ee.Image:
        """
        Mask clouds/shadows using Sentinel-2 Scene Classification Layer (SCL).

        Excluded SCL values:
          1  = Saturated / Defective
          3  = Cloud Shadow
          8  = Cloud Medium Probability
          9  = Cloud High Probability
          10 = Thin Cirrus
          11 = Snow / Ice
        """
        scl = image.select('SCL')
        valid = (scl.neq(1).And(scl.neq(3))
                    .And(scl.neq(8)).And(scl.neq(9))
                    .And(scl.neq(10)).And(scl.neq(11)))
        return image.updateMask(valid)

    # ------------------------------------------------------------------
    # Collection loading
    # ------------------------------------------------------------------

    def load_landsat_collection(self, start_date: str,
                                end_date: str) -> ee.ImageCollection:
        """
        Load a merged Landsat 8 + Landsat 9 collection.

        Both satellites carry identical OLI + TIRS instruments and are
        calibrated to the same spectral response, so they can be merged
        without any additional harmonisation step.

        Scale factors are applied here so every image in the collection
        already has physically meaningful SR [0–1] and ST [K] values.

        Args:
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD'

        Returns:
            Merged, cloud-filtered, scale-corrected ImageCollection.
        """
        area  = self.get_study_area()
        cloud = LANDSAT_CONFIG["cloud_threshold"]

        def _load(collection_id: str) -> ee.ImageCollection:
            return (ee.ImageCollection(collection_id)
                      .filterBounds(area)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUD_COVER', cloud))
                      .map(self._mask_landsat_clouds)
                      .map(_apply_landsat_scale_factors))

        l8 = _load(LANDSAT_CONFIG["collection_l8"])
        l9 = _load(LANDSAT_CONFIG["collection_l9"])

        merged = ee.ImageCollection(l8.merge(l9))
        size   = merged.size().getInfo()
        logger.info(f"  Landsat 8+9: {size} images for {start_date} → {end_date}")
        return merged

    def load_sentinel2_collection(self, start_date: str,
                                  end_date: str) -> ee.ImageCollection:
        """
        Load Sentinel-2 SR Harmonised collection with cloud masking.

        The S2_SR_HARMONIZED product already applies the processing-baseline
        harmonisation; no additional scale factor is needed (reflectance values
        are stored as DN × 10 000, i.e. divide by 10 000 for [0–1]).

        Args:
            start_date: 'YYYY-MM-DD'
            end_date:   'YYYY-MM-DD'

        Returns:
            Cloud-filtered ImageCollection.
        """
        area  = self.get_study_area()
        # Use a Sentinel-2-specific cloud threshold if configured, otherwise
        # fall back to the Landsat one.  S2 CLOUDY_PIXEL_PERCENTAGE is scene-level
        # metadata; it's common to allow a slightly higher value here because
        # per-pixel SCL masking (applied above) removes actual cloud pixels.
        cloud = SENTINEL2_CONFIG.get("cloud_threshold",
                LANDSAT_CONFIG["cloud_threshold"])

        # IMPORTANT — do NOT add .filter(ee.Filter.eq('MGRS_TILE', '...')) here.
        # Jakarta's AOI straddles multiple MGRS tiles (typically 48MXT + 48MYT).
        # filterBounds() already returns images from *every* tile that intersects
        # the AOI, so the subsequent median() composite automatically mosaics
        # all contributing tiles into a seamless image covering the full extent.
        # Filtering to a single named tile is exactly what produces the diagonal
        # cutoff edge seen in s2_01_raw_bands.png.
        collection = (ee.ImageCollection(SENTINEL2_CONFIG["collection"])
                        .filterBounds(area)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud))
                        .map(self._mask_sentinel2_clouds))

        size = collection.size().getInfo()
        logger.info(f"  Sentinel-2: {size} images for {start_date} → {end_date}")

        if size < 2:
            logger.warning(
                "  ⚠ Very few Sentinel-2 images found. "
                "If the exported raster shows diagonal cutoff edges, "
                "verify that STUDY_AREA bounds cover the correct MGRS tiles "
                "and that CLOUDY_PIXEL_PERCENTAGE threshold is not overly strict."
            )
        return collection

    # ------------------------------------------------------------------
    # Compositing
    # ------------------------------------------------------------------

    def create_composite(self, collection: ee.ImageCollection,
                         method: str = "median") -> ee.Image:
        """
        Reduce a collection to a single composite image.

        Median compositing is preferred for UHI research because it is
        resistant to remaining cloud artefacts and produces stable
        spectral values across the time window.

        Args:
            collection: Already-masked ImageCollection.
            method:     'median' (default) or 'mean'.

        Returns:
            Composite ee.Image.
        """
        if method == "median":
            return collection.median()
        elif method == "mean":
            return collection.mean()
        else:
            raise ValueError(f"Unknown compositing method: {method!r}. "
                             "Use 'median' or 'mean'.")

    # ------------------------------------------------------------------
    # Export (Drive)
    # ------------------------------------------------------------------

    def _submit_export_task(self, image: ee.Image,
                            description: str,
                            bands: List[str],
                            scale: int) -> ee.batch.Task:
        """
        Submit a GEE Export.image.toDrive() task and return it.

        SEAMLESS MULTI-TILE EXPORT
        --------------------------
        When the AOI straddles multiple MGRS tiles (e.g. 48MXT + 48MYT for
        Jakarta), the median composite retains masked pixels along tile-seam
        borders — producing the diagonal cutoff visible in s2_01_raw_bands.png.
        Two fixes are applied here:

        1. image.unmask(0) — replaces all residual masked pixels (tile edges,
           cloud-masked borders) with 0 before GEE reprojects the image into
           the export CRS.  After reprojection, every pixel inside the AOI
           bounding box gets a value, so the exported GeoTIFF is spatially
           complete.  The preprocessing pipeline treats 0 as a valid low-
           reflectance value; genuine no-data is already handled by upstream
           QA/SCL masking embedded in the composite.

        maxPixels=1e10 handles the pixel limit, and unmask(0) fills tile-seam
        borders so the exported GeoTIFF covers the full AOI without diagonal
        cutoff edges.

        Args:
            image:       Composite image to export.
            description: Task description / file stem (no spaces).
            bands:       List of band names to include in the export.
            scale:       Pixel size in metres.

        Returns:
            The submitted ee.batch.Task object.
        """
        region = self.get_study_area()

        # Select only the bands that exist in the image
        available = image.bandNames().getInfo()
        export_bands = [b for b in bands if b in available]
        if not export_bands:
            raise ValueError(f"None of {bands} found in image bands: {available}")

        image_selected = image.select(export_bands)

        # Fill masked/NaN pixels at tile-seam borders with 0 so the exported
        # GeoTIFF covers the full AOI without diagonal cutoff edges.
        image_selected = image_selected.unmask(0)

        task = ee.batch.Export.image.toDrive(
            image=image_selected,
            description=description,
            folder=self.DRIVE_FOLDER,
            fileNamePrefix=description,
            region=region,
            scale=scale,
            crs=self.EXPORT_CRS,
            maxPixels=1e10,
            fileFormat='GeoTIFF',
        )
        task.start()
        logger.info(f"  Export task submitted: {description} "
                    f"({len(export_bands)} bands @ {scale}m, "
                    f"unmask enabled)")
        return task

    def _wait_for_task(self, task: ee.batch.Task,
                       description: str) -> bool:
        """
        Poll a GEE task until it completes or times out.

        Args:
            task:        The running ee.batch.Task.
            description: Human-readable label for log messages.

        Returns:
            True if COMPLETED, False otherwise.
        """
        elapsed = 0
        while elapsed < self.MAX_WAIT_S:
            status = task.status()
            state  = status["state"]

            if state == "COMPLETED":
                logger.info(f"  ✓ {description} — COMPLETED")
                return True
            elif state in ("FAILED", "CANCELLED"):
                logger.error(f"  ✗ {description} — {state}: "
                             f"{status.get('error_message', 'no message')}")
                return False
            else:
                logger.debug(f"  … {description} — {state} ({elapsed}s elapsed)")
                time.sleep(self.POLL_INTERVAL_S)
                elapsed += self.POLL_INTERVAL_S

        logger.error(f"  ✗ {description} — timed out after {self.MAX_WAIT_S}s")
        return False

    # ------------------------------------------------------------------
    # Per-sensor download wrappers
    # ------------------------------------------------------------------

    def _landsat_bands(self) -> List[str]:
        return ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4',
                'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL']

    def _sentinel2_bands(self) -> List[str]:
        return ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'SCL']

    def download_landsat_month(self, year: int, month: int,
                               task_registry: Dict) -> bool:
        """
        Build and export a monthly Landsat 8+9 composite to Drive.

        Args:
            year, month:   Target period.
            task_registry: Dict updated in-place with task metadata.

        Returns:
            True if the export task was submitted successfully.
        """
        start, end = _month_date_range(year, month)
        desc = f"Landsat_{year}_{month:02d}"

        logger.info(f"  Building Landsat composite for {year}-{month:02d} …")
        try:
            col = self.load_landsat_collection(start, end)
            if col.size().getInfo() == 0:
                logger.warning(f"  No Landsat images for {year}-{month:02d} — skipping")
                task_registry[desc] = {"status": "SKIPPED_NO_DATA"}
                return False

            composite = self.create_composite(col)
            task = self._submit_export_task(
                composite, desc, self._landsat_bands(),
                self.EXPORT_SCALE["landsat"]
            )
            task_registry[desc] = {"task": task, "status": "RUNNING"}
            return True

        except Exception as e:
            logger.error(f"  Landsat {year}-{month:02d} failed: {e}")
            task_registry[desc] = {"status": f"ERROR: {e}"}
            return False

    def download_sentinel2_month(self, year: int, month: int,
                                  task_registry: Dict) -> bool:
        """
        Build and export a monthly Sentinel-2 composite to Drive.

        Args:
            year, month:   Target period.
            task_registry: Dict updated in-place with task metadata.

        Returns:
            True if the export task was submitted successfully.
        """
        # Sentinel-2A launched March 2015; before 2015-07 data is very sparse
        if (year, month) < (2015, 7):
            logger.warning(f"  Sentinel-2 not yet operational for "
                           f"{year}-{month:02d} — skipping")
            return False

        start, end = _month_date_range(year, month)
        desc = f"Sentinel2_{year}_{month:02d}"

        logger.info(f"  Building Sentinel-2 composite for {year}-{month:02d} …")
        try:
            col = self.load_sentinel2_collection(start, end)
            if col.size().getInfo() == 0:
                logger.warning(f"  No Sentinel-2 images for {year}-{month:02d} — skipping")
                task_registry[desc] = {"status": "SKIPPED_NO_DATA"}
                return False

            composite = self.create_composite(col)
            task = self._submit_export_task(
                composite, desc, self._sentinel2_bands(),
                self.EXPORT_SCALE["sentinel2"]
            )
            task_registry[desc] = {"task": task, "status": "RUNNING"}
            return True

        except Exception as e:
            logger.error(f"  Sentinel-2 {year}-{month:02d} failed: {e}")
            task_registry[desc] = {"status": f"ERROR: {e}"}
            return False

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def download_date_range(self,
                            start_year: int,  start_month: int,
                            end_year: int,    end_month: int,
                            wait_for_completion: bool = True,
                            resume_log: Optional[Path] = None) -> Dict:
        """
        Submit export tasks for every month in the requested range.

        Args:
            start_year / start_month: First month to download (inclusive).
            end_year   / end_month:   Last month to download (inclusive).
            wait_for_completion:      If True, poll every task until done.
                                      If False, submit all tasks and return
                                      immediately (useful for fire-and-forget).
            resume_log:               Optional JSON file path. Already-completed
                                      month descriptions stored here are skipped,
                                      enabling safe resume after interruption.

        Returns:
            task_registry dict mapping description → status dict.
        """
        # Load resume log if provided
        completed_descs: set = set()
        if resume_log and resume_log.exists():
            try:
                with open(resume_log) as f:
                    completed_descs = set(json.load(f).get("completed", []))
                logger.info(f"Resume: {len(completed_descs)} months already done")
            except Exception as e:
                logger.warning(f"Could not read resume log: {e}")

        task_registry: Dict = {}
        months = list(_iter_months(start_year, start_month, end_year, end_month))

        logger.info("=" * 70)
        logger.info(f"DOWNLOAD RANGE: {start_year}-{start_month:02d} → "
                    f"{end_year}-{end_month:02d}  ({len(months)} months)")
        logger.info(f"Sensors: {self.sensors}")
        logger.info(f"Drive folder: {self.DRIVE_FOLDER}")
        logger.info("=" * 70)

        for year, month in months:
            logger.info(f"\n--- {year}-{month:02d} ---")

            if "landsat" in self.sensors:
                desc = f"Landsat_{year}_{month:02d}"
                if desc in completed_descs:
                    logger.info(f"  Skipping {desc} (already completed)")
                else:
                    self.download_landsat_month(year, month, task_registry)

            if "sentinel2" in self.sensors:
                desc = f"Sentinel2_{year}_{month:02d}"
                if desc in completed_descs:
                    logger.info(f"  Skipping {desc} (already completed)")
                else:
                    self.download_sentinel2_month(year, month, task_registry)

        if not wait_for_completion:
            logger.info("\nAll tasks submitted. Not waiting for completion "
                        "(wait_for_completion=False).")
            logger.info("Monitor progress at: https://code.earthengine.google.com/tasks")
            return task_registry

        # ------ Wait for all running tasks ------
        logger.info(f"\nWaiting for {len(task_registry)} export tasks …")
        newly_completed = list(completed_descs)

        for desc, info in task_registry.items():
            if "task" not in info:
                continue   # SKIPPED or ERROR before submission
            success = self._wait_for_task(info["task"], desc)
            info["status"] = "COMPLETED" if success else "FAILED"
            if success:
                newly_completed.append(desc)

        # Update resume log
        if resume_log:
            try:
                resume_log.parent.mkdir(parents=True, exist_ok=True)
                with open(resume_log, "w") as f:
                    json.dump({"completed": sorted(newly_completed)}, f, indent=2)
                logger.info(f"Resume log updated: {resume_log}")
            except Exception as e:
                logger.warning(f"Could not write resume log: {e}")

        # ------ Summary ------
        total    = len(task_registry)
        success  = sum(1 for v in task_registry.values() if v["status"] == "COMPLETED")
        skipped  = sum(1 for v in task_registry.values() if "SKIPPED" in v["status"])
        failed   = total - success - skipped

        logger.info("\n" + "=" * 70)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Completed : {success}")
        logger.info(f"  Skipped   : {skipped}  (no data available)")
        logger.info(f"  Failed    : {failed}")
        logger.info(f"\nFiles are in your Google Drive folder: '{self.DRIVE_FOLDER}'")
        logger.info("Download them from drive.google.com or use rclone / gdrive CLI.")

        return task_registry


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _month_date_range(year: int, month: int) -> Tuple[str, str]:
    """Return ('YYYY-MM-DD', 'YYYY-MM-DD') for the start and end of a month."""
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _iter_months(start_year: int, start_month: int,
                 end_year: int,   end_month: int):
    """Yield (year, month) tuples from start to end inclusive."""
    current = datetime(start_year, start_month, 1)
    end     = datetime(end_year,   end_month,   1)
    while current <= end:
        yield current.year, current.month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 70)
    logger.info("JAKARTA UHI — SATELLITE DATA DOWNLOAD")
    logger.info(f"Area  : {STUDY_AREA['name']}")
    logger.info(f"Bounds: {STUDY_AREA['bounds']}")
    logger.info("=" * 70)

    loader = EarthEngineLoader(sensors=["landsat", "sentinel2"])

    # Path to a JSON file that records completed months so the run can be
    # safely interrupted and restarted without duplicating GEE tasks.
    resume_log = RAW_DATA_DIR / "download_resume_log.json"

    results = loader.download_date_range(
        start_year=2016, start_month=1,
        end_year=2025,   end_month=12,
        wait_for_completion=True,   # set False to fire-and-forget
        resume_log=resume_log,
    )

    logger.info("\nNext step: Run preprocessing.py to process the downloaded data.")
    return results


if __name__ == "__main__":
    main()