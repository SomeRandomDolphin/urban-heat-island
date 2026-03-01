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
    STUDY_AREA, LANDSAT_CONFIG,
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
    Download monthly Landsat 8/9 composites from Google Earth Engine
    and export them to Google Drive for Urban Heat Island (UHI) research
    over Greater Jakarta (Jabodetabek), Indonesia.
    """

    DRIVE_FOLDER    = "Jabodetabek_UHI"
    EXPORT_CRS      = "EPSG:32748"       # WGS84 / UTM Zone 48S
    EXPORT_SCALE    = {"landsat": 30}    # native Landsat resolution (metres)
    POLL_INTERVAL_S = 30                 # seconds between task-status checks
    MAX_WAIT_S      = 7200               # 2 hours before giving up on a task

    def __init__(self, sensors: List[str] = ["landsat"],
                 gee_project: str = "nukobot-366809"):
        """
        Initialise Earth Engine.

        Args:
            sensors:     Sensors to download. Only 'landsat' is supported.
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
        """
        Return the study-area rectangle as an ee.Geometry.

        Covers Greater Jakarta (Jabodetabek):
          Longitude: 106.40°E – 107.20°E
          Latitude :   6.70°S –   6.00°S
        Includes: Jakarta, Bogor, Depok, Tangerang, Bekasi.
        """
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

    # ------------------------------------------------------------------
    # Collection loading
    # ------------------------------------------------------------------

    def load_landsat_collection(self, start_date: str,
                                end_date: str) -> ee.ImageCollection:
        """
        Load a merged Landsat 8 + Landsat 9 collection.

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

    # ------------------------------------------------------------------
    # Compositing
    # ------------------------------------------------------------------

    def create_composite(self, collection: ee.ImageCollection,
                         method: str = "median") -> ee.Image:
        """
        Reduce a collection to a single composite image.

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

        task = ee.batch.Export.image.toDrive(
            image=image.select(export_bands),
            description=description,
            folder=self.DRIVE_FOLDER,
            fileNamePrefix=description,
            region=region,
            scale=scale,
            crs=self.EXPORT_CRS,
            maxPixels=1e10,   # Jabodetabek bbox ~7.7M pixels @ 30 m; well within limit
            fileFormat='GeoTIFF',
        )
        task.start()
        logger.info(f"  Export task submitted: {description} "
                    f"({len(export_bands)} bands @ {scale}m)")
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
    logger.info("GREATER JAKARTA (JABODETABEK) UHI — LANDSAT 8/9 DATA DOWNLOAD")
    logger.info(f"Area  : {STUDY_AREA['name']}")
    logger.info(f"Bounds: {STUDY_AREA['bounds']}")
    logger.info("=" * 70)

    loader = EarthEngineLoader(sensors=["landsat"])

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