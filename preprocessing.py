"""
Data preprocessing pipeline for Urban Heat Island detection
Purpose: Process BOTH Landsat and Sentinel-2 data and fuse them
Does NOT download data - only transforms existing raw data files
"""
import sys
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from typing import Dict, Tuple, List, Optional
import logging
from pathlib import Path
from scipy.ndimage import uniform_filter, zoom
import pyproj
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, StratifiedKFold

import rasterio
from rasterio.enums import Resampling as RasterioResampling

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    LANDSAT_CONFIG, SENTINEL2_CONFIG,
    LOGGING_CONFIG, STUDY_AREA,
)

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS & VISUALISATION MODULE
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend – safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ---------------------------------------------------------------------------
# GeoTIFF band loader
# ---------------------------------------------------------------------------

_LANDSAT_BAND_ORDER = [
    "SR_B1", "SR_B2", "SR_B3", "SR_B4",
    "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL",
]
_SENTINEL2_BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12", "SCL"]

# ---------------------------------------------------------------------------
# Memory-budget configuration
# ---------------------------------------------------------------------------
# Maximum number of pixels (rows × cols) allowed per band before adaptive
# downsampling kicks in.  At float32 each pixel costs 4 bytes, so:
#   12_000_000 px  →  ~46 MiB per band  →  ~410 MiB for 9 Landsat bands
# Raise this if your machine has more headroom; lower it if you still OOM.
# The value can also be overridden via an environment variable:
#   export UHI_MAX_PIXELS=8000000
#
# FIX: Raised default from 1_921_100 to 12_000_000.
# The previous budget (~1.9 M px) caused aggressive downsampling that varied
# epoch-to-epoch depending on each scene's bounding box.  Different downsample
# factors → different grid_rows/grid_cols per epoch → wrong patch-column mode →
# patches placed in incorrect canvas columns → blocky/misaligned mosaic.
# At 12 M px each 1152×1024 scene (~1.18 M px) loads at full native resolution,
# giving consistent grid dimensions across all epochs.
import os as _os
_DEFAULT_MAX_PIXELS = 10_000_000
MAX_PIXELS: int = int(_os.environ.get("UHI_MAX_PIXELS", _DEFAULT_MAX_PIXELS))


def _compute_downsample_factor(rows: int, cols: int,
                                n_bands: int,
                                max_pixels: int = MAX_PIXELS) -> float:
    """
    Return the linear scale factor (≤ 1.0) needed so that rows*cols ≤ max_pixels.

    We also peek at available system RAM (via psutil if installed, otherwise we
    use a conservative 1 GiB budget) and tighten the factor if even the
    downsampled load would exceed half the free memory.

    Args:
        rows, cols : native raster dimensions
        n_bands    : number of bands that will be loaded (affects RAM estimate)
        max_pixels : pixel budget (default MAX_PIXELS)

    Returns:
        factor ∈ (0, 1] — 1.0 means no downsampling needed
    """
    native_px = rows * cols

    # Factor driven by pixel budget
    if native_px <= max_pixels:
        pixel_factor = 1.0
    else:
        pixel_factor = (max_pixels / native_px) ** 0.5  # sqrt because both dims scale

    # Factor driven by available RAM (optional, requires psutil)
    ram_factor = 1.0
    try:
        import psutil
        free_bytes = psutil.virtual_memory().available
        # Budget: use at most 40 % of free RAM for the bands (conservative)
        budget_bytes = free_bytes * 0.40
        bytes_per_band_native = native_px * 4  # float32
        bytes_total_native = bytes_per_band_native * n_bands
        if bytes_total_native > budget_bytes:
            ram_factor = (budget_bytes / bytes_total_native) ** 0.5
    except ImportError:
        pass  # psutil not installed — fall back to pixel budget only

    factor = min(pixel_factor, ram_factor)
    return max(factor, 0.05)   # never go below 5 % of native resolution


def load_tif_as_bands(tif_path: Path,
                       max_pixels: int = MAX_PIXELS) -> Optional[Dict[str, np.ndarray]]:
    """
    Load a GeoTIFF exported by earth_engine_loader.py and return a dict of
    {band_name: 2-D float32 array}.

    Band identification priority:
      1. Rasterio band description stored in the file (set by GEE export).
      2. Positional fallback using the known export order (inferred from
         the number of bands in the file).
      3. Generic ``band_N`` keys as a last resort.

    The earth_engine_loader applies Landsat C2 L2 scale factors before
    exporting, so TIF values are already physically meaningful:
      - SR bands  : surface reflectance [0.0 – 1.0]
      - ST_B10    : brightness temperature in Kelvin (~280–330 K)
      - S2 bands  : surface reflectance × 10 000
    QA_PIXEL is kept as integer-valued float for bitmask operations.

    Adaptive downsampling
    ─────────────────────
    If the native raster exceeds *max_pixels* (or would exceed 40 % of free
    RAM when psutil is installed), the bands are read at a reduced resolution
    using rasterio's decimated-read feature with bilinear resampling (nearest-
    neighbour for the QA/SCL mask bands).  The downsample factor and resulting
    shape are logged so you can audit exactly what happened.
    """
    try:
        with rasterio.open(tif_path) as src:
            n_bands      = src.count
            native_rows  = src.height
            native_cols  = src.width
            descriptions = [src.descriptions[i] for i in range(n_bands)]
            has_desc     = any(d is not None and str(d).strip() for d in descriptions)

            # Detect south-up rasters (positive y scale in affine transform).
            # Standard north-up GeoTIFFs have a negative y scale (affine[4] < 0),
            # meaning row 0 is the northernmost row.  A positive y scale means row 0
            # is the SOUTHERNMOST row, which would flip the mosaic north-south.
            # We normalise to north-up by flipping after reading.
            _needs_vflip = (src.transform.e > 0)   # transform.e == pixel height (row scale)

            if not has_desc:
                if n_bands == len(_LANDSAT_BAND_ORDER):
                    descriptions = _LANDSAT_BAND_ORDER
                elif n_bands == len(_SENTINEL2_BAND_ORDER):
                    descriptions = _SENTINEL2_BAND_ORDER
                else:
                    descriptions = [f"band_{i+1}" for i in range(n_bands)]

            # ── Adaptive downsampling ──────────────────────────────────────
            factor = _compute_downsample_factor(
                native_rows, native_cols, n_bands, max_pixels
            )
            if factor < 1.0:
                out_rows = max(1, int(round(native_rows * factor)))
                out_cols = max(1, int(round(native_cols * factor)))
                logger.warning(
                    f"  [Downsample] {tif_path.name}: "
                    f"native {native_rows}×{native_cols} "
                    f"→ {out_rows}×{out_cols}  "
                    f"(factor={factor:.3f}, max_pixels={max_pixels:,}). "
                    f"Set UHI_MAX_PIXELS env-var to change this threshold."
                )
            else:
                out_rows, out_cols = native_rows, native_cols

            # Bands whose values are bitmasks / class labels — use NN resampling
            _MASK_BAND_NAMES = {"QA_PIXEL", "SCL", "QA60"}

            nodata = src.nodata
            bands: Dict[str, np.ndarray] = {}
            for i, name in enumerate(descriptions, start=1):
                is_mask = (name in _MASK_BAND_NAMES)
                resample_method = (
                    RasterioResampling.nearest if is_mask
                    else RasterioResampling.bilinear
                )
                arr = src.read(
                    i,
                    out_shape=(out_rows, out_cols),
                    resampling=resample_method,
                ).astype(np.float32)
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                if _needs_vflip:
                    arr = np.flipud(arr)
                bands[name] = arr

        actual_rows, actual_cols = next(iter(bands.values())).shape
        if _needs_vflip:
            logger.warning(
                f"  [Orientation] {tif_path.name}: south-up raster detected "
                f"(affine y-scale > 0) — flipped vertically to north-up. "
                f"This normalises patch grid_row=0 to the northernmost row."
            )

        # ── Replace GEE unmask(0) fill pixels with NaN ─────────────────
        # earth_engine_loader calls image.unmask(0) before exporting so that
        # the GeoTIFF bounding-box is fully populated (no masked-pixel gaps at
        # tile seams or ocean borders).  That means large ocean / out-of-scene
        # regions are stored as exactly 0 in EVERY spectral band.
        #
        # Problem: the all-zero pixels are valid float values, so rasterio
        # reads them as real data.  When the ocean region covers >50 % of the
        # raster (common for coastal Jakarta tiles), the nanmedian of the red
        # band is pulled toward 0, causing the DN-detection heuristic in
        # calculate_spectral_indices() to mis-classify already-scaled
        # GeoTIFFs as "raw DN" (or vice-versa), and collapsing every index
        # to a flat, featureless map.
        #
        # Fix: identify pixels where ALL spectral bands (excluding QA/SCL
        # bitmask bands) are exactly 0.0, and replace them with NaN.
        # This preserves genuine dark-water pixels (which have non-zero
        # values in at least one band) while eliminating the fill region.
        spectral_band_names = [
            k for k in bands
            if k not in {"QA_PIXEL", "QA60", "SCL"}
        ]
        if spectral_band_names:
            fill_mask = np.ones(
                bands[spectral_band_names[0]].shape, dtype=bool
            )
            for bn in spectral_band_names:
                fill_mask &= (bands[bn] == 0.0)

            n_fill = int(fill_mask.sum())
            if n_fill > 0:
                fill_pct = n_fill / fill_mask.size * 100
                logger.info(
                    f"  [Fill pixels] {tif_path.name}: replacing {n_fill:,} "
                    f"all-zero fill pixels ({fill_pct:.1f}% of image) with NaN. "
                    f"These are GEE unmask(0) tile-edge / ocean fill values."
                )
                for bn in spectral_band_names:
                    bands[bn][fill_mask] = np.nan
            del fill_mask
        # ───────────────────────────────────────────────────────────────

        logger.info(
            f"  Loaded {n_bands} bands from {tif_path.name} "
            f"[{actual_rows}×{actual_cols}]: {list(bands.keys())}"
        )
        return bands
    except Exception as exc:
        logger.error(f"  Failed to load {tif_path}: {exc}")
        return None


class PreprocessingDiagnostics:
    """
    Centralised diagnostics for the Urban Heat Island preprocessing pipeline.

    Call the individual `plot_*` methods at the appropriate pipeline stage.
    All figures are written to `output_dir / 'diagnostics'`.
    """

    def __init__(self, output_dir: Path):
        self.out = Path(output_dir) / "diagnostics"
        self.out.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Diagnostics] Plots will be saved to: {self.out}")

    # ------------------------------------------------------------------
    # 1. Raw band overview
    # ------------------------------------------------------------------
    def plot_raw_bands(self, raw_data: dict, title: str = "Raw Bands Overview",
                       filename: str = "01_raw_bands.png"):
        """Grid of all 2-D arrays in raw_data (bands + any existing indices)."""
        keys = [k for k, v in raw_data.items()
                if isinstance(v, np.ndarray) and v.ndim == 2]
        if not keys:
            logger.warning("[Diagnostics] plot_raw_bands: no 2-D arrays found.")
            return

        n = len(keys)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
        axes = np.array(axes).flatten()

        for ax, key in zip(axes, keys):
            arr = raw_data[key]
            valid = arr[np.isfinite(arr)]
            if valid.size >= 2:
                vmin, vmax = np.nanpercentile(valid, [2, 98])
                if vmax - vmin < 1e-6:
                    vmin, vmax = valid.min(), valid.max()
                if vmax - vmin < 1e-6:
                    vmin, vmax = vmin - 0.05, vmax + 0.05
            else:
                vmin, vmax = 0.0, 1.0
            im = ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(key, fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # 2. Spectral indices overview
    # ------------------------------------------------------------------
    def plot_spectral_indices(self, indices: dict,
                               filename: str = "02_spectral_indices.png"):
        """Side-by-side spatial maps of all computed spectral indices."""
        index_cmaps = {
            "NDVI":   "RdYlGn",
            "NDBI":   "RdBu_r",
            "MNDWI":  "Blues",
            "BSI":    "YlOrBr",
            "UI":     "hot",
            "albedo": "gray",
            "LST":    "inferno",
        }

        keys = [k for k in index_cmaps if k in indices]
        if not keys:
            logger.warning("[Diagnostics] plot_spectral_indices: no recognised indices.")
            return

        ncols = 3
        nrows = (len(keys) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
        axes = np.array(axes).flatten()

        for ax, key in zip(axes, keys):
            arr = indices[key]
            valid = arr[np.isfinite(arr)]
            if valid.size >= 2:
                vmin, vmax = np.nanpercentile(valid, [2, 98])
                if vmax - vmin < 1e-6:
                    vmin, vmax = valid.min(), valid.max()
            else:
                vmin, vmax = 0.0, 1.0
            cmap = index_cmaps.get(key, "viridis")
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            suffix = "°C" if key == "LST" else ""
            ax.set_title(f"{key}  [{vmin:.2f}{suffix} – {vmax:.2f}{suffix}]", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes[len(keys):]:
            ax.set_visible(False)

        fig.suptitle("Spectral Indices", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # 3. LST validation
    # ------------------------------------------------------------------
    def plot_lst_validation(self, lst_raw: np.ndarray, lst_clean: np.ndarray,
                             stats: dict, filename: str = "03_lst_validation.png"):
        """Before/after LST cleaning + distribution + spatial map."""
        fig = plt.figure(figsize=(16, 10), layout="constrained")
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

        # Spatial: raw LST
        ax0 = fig.add_subplot(gs[0, :2])
        vmin, vmax = np.nanpercentile(lst_raw[np.isfinite(lst_raw)], [2, 98])
        im0 = ax0.imshow(lst_raw, cmap="inferno", vmin=vmin, vmax=vmax)
        ax0.set_title("LST – Raw (°C)", fontweight="bold")
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0)

        # Spatial: clean LST
        ax1 = fig.add_subplot(gs[0, 2:])
        vmin2, vmax2 = np.nanpercentile(lst_clean[np.isfinite(lst_clean)], [2, 98]) \
            if np.any(np.isfinite(lst_clean)) else (0, 1)
        im1 = ax1.imshow(lst_clean, cmap="inferno", vmin=vmin2, vmax=vmax2)
        ax1.set_title("LST – Cleaned (°C)", fontweight="bold")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1)

        # Histogram comparison
        ax2 = fig.add_subplot(gs[1, :2])
        raw_vals = lst_raw[np.isfinite(lst_raw)].ravel()
        clean_vals = lst_clean[np.isfinite(lst_clean)].ravel()
        ax2.hist(raw_vals, bins=80, alpha=0.5, color="steelblue", label="Raw", density=True)
        ax2.hist(clean_vals, bins=80, alpha=0.7, color="tomato", label="Cleaned", density=True)
        ax2.set_xlabel("Temperature (°C)")
        ax2.set_ylabel("Density")
        ax2.set_title("LST Distribution: Raw vs Cleaned")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Stats table
        ax3 = fig.add_subplot(gs[1, 2:])
        ax3.axis("off")
        stat_rows = [
            ["Metric", "Value"],
            ["Original valid px", f"{stats.get('original_valid', 'N/A'):,}"],
            ["Filtered valid px", f"{stats.get('filtered_valid', 'N/A'):,}"],
            ["Valid ratio", f"{stats.get('valid_ratio', 0)*100:.1f} %"],
            ["Mean (°C)", f"{stats.get('mean', float('nan')):.2f}"],
            ["Std  (°C)", f"{stats.get('std', float('nan')):.2f}"],
            ["Min  (°C)", f"{stats.get('min', float('nan')):.2f}"],
            ["Max  (°C)", f"{stats.get('max', float('nan')):.2f}"],
        ]
        tbl = ax3.table(cellText=stat_rows[1:], colLabels=stat_rows[0],
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.6)
        ax3.set_title("LST Validation Statistics", fontweight="bold")

        fig.suptitle("Land Surface Temperature – Validation Diagnostics",
                     fontsize=13, fontweight="bold")
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # 4. LST vs spectral index scatter plots
    # ------------------------------------------------------------------
    def plot_lst_vs_indices(self, indices: dict,
                             filename: str = "04_lst_vs_indices.png"):
        """Scatter plots of LST against each spectral index."""
        if "LST" not in indices:
            logger.warning("[Diagnostics] plot_lst_vs_indices: LST not in indices.")
            return

        lst = indices["LST"].ravel()
        valid_mask = np.isfinite(lst)
        compare = ["NDVI", "NDBI", "MNDWI", "BSI", "UI", "albedo"]
        available = [k for k in compare if k in indices]

        if not available:
            return

        ncols = 3
        nrows = (len(available) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
        axes = np.array(axes).flatten()

        for ax, key in zip(axes, available):
            idx_vals = indices[key].ravel()
            mask = valid_mask & np.isfinite(idx_vals)
            x, y = idx_vals[mask], lst[mask]

            # Subsample for speed if very large
            if len(x) > 50_000:
                rng = np.random.default_rng(0)
                sel = rng.choice(len(x), 50_000, replace=False)
                x, y = x[sel], y[sel]

            ax.hexbin(x, y, gridsize=50, cmap="YlOrRd", mincnt=1)
            # Simple linear fit
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                xr = np.linspace(x.min(), x.max(), 200)
                ax.plot(xr, p(xr), "b--", lw=1.5, label=f"slope={z[0]:.2f}")
                ax.legend(fontsize=8)
            except Exception:
                pass

            corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else float("nan")
            ax.set_xlabel(key, fontsize=9)
            ax.set_ylabel("LST (°C)", fontsize=9)
            ax.set_title(f"LST vs {key}  (r={corr:.2f})", fontsize=9)
            ax.grid(True, alpha=0.3)

        for ax in axes[len(available):]:
            ax.set_visible(False)

        fig.suptitle("LST vs Spectral Index Correlations", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # 5. Patch quality diagnostics
    # ------------------------------------------------------------------
    def plot_patch_diagnostics(self, patches: list,
                                raster_data: dict = None,
                                n_preview: int = 12,
                                filename: str = "05_patch_diagnostics.png"):
        """
        Distributions of per-patch LST statistics + optional visual patch previews.

        Patches produced by DatasetCreator.extract_patches() store only
        position and lightweight statistics  (no data copy):
            {"position": (row, col), "_lst_mean": float, "_lst_std": float}

        Args:
            patches      : list of patch dicts from extract_patches()
            raster_data  : full raster dict (optional).  When provided, a fourth
                           row of visual patch thumbnails is added — showing the
                           raw LST slice for a sample of patches spanning the
                           temperature distribution.  This is useful to diagnose
                           NaN-fill artefacts (flat blobs) and fusion edge effects.
            n_preview    : number of patch thumbnails to show (default 12)
            filename     : output filename
        """
        if not patches:
            logger.warning("[Diagnostics] plot_patch_diagnostics: no patches.")
            return

        # Extract per-patch stats — fall back gracefully if keys are missing
        means = [p.get("_lst_mean", np.nan) for p in patches]
        stds  = [p.get("_lst_std",  np.nan) for p in patches]
        means = np.array(means, dtype=float)
        stds  = np.array(stds,  dtype=float)

        # Filter out NaN entries (patches that stored no stats)
        valid = np.isfinite(means) & np.isfinite(stds)
        means_v = means[valid]
        stds_v  = stds[valid]

        has_previews = (raster_data is not None and "LST" in raster_data and
                        len(patches) > 0)
        nrows_stats = 1
        nrows_total = nrows_stats + (1 if has_previews else 0)

        fig = plt.figure(figsize=(16, 7 + (5 if has_previews else 0)), layout="constrained")
        gs = gridspec.GridSpec(nrows_total, 3, figure=fig, hspace=0.55, wspace=0.35)

        # ── Row 0: stat histograms + scatter ──────────────────────────────────
        # Mean LST distribution
        ax0 = fig.add_subplot(gs[0, 0])
        if means_v.size > 0:
            ax0.hist(means_v, bins=min(40, max(1, means_v.size // 5)),
                     color="steelblue", edgecolor="white", alpha=0.85)
            ax0.axvline(means_v.mean(), color="black", linestyle="--", lw=1.5,
                        label=f"μ={means_v.mean():.1f}°C")
            ax0.axvline(15.0, color="green",  linestyle=":",  lw=1.2, label="min=15°C")
            ax0.axvline(58.0, color="crimson", linestyle=":", lw=1.2, label="max=58°C")
            ax0.legend(fontsize=7)
        ax0.set_xlabel("Patch LST Mean (°C)", fontsize=8)
        ax0.set_ylabel("Count")
        ax0.set_title("Patch LST Mean Distribution")
        ax0.grid(True, alpha=0.3)

        # Std LST distribution
        ax1 = fig.add_subplot(gs[0, 1])
        if stds_v.size > 0:
            ax1.hist(stds_v, bins=min(40, max(1, stds_v.size // 5)),
                     color="darkorange", edgecolor="white", alpha=0.85)
            ax1.axvline(stds_v.mean(), color="black", linestyle="--", lw=1.5,
                        label=f"μ={stds_v.mean():.1f}°C")
            ax1.legend(fontsize=8)
        ax1.set_xlabel("Patch LST Std (°C)", fontsize=8)
        ax1.set_ylabel("Count")
        ax1.set_title("Patch LST Std Distribution")
        ax1.grid(True, alpha=0.3)

        # 2-D scatter: mean vs std (quality overview)
        ax2 = fig.add_subplot(gs[0, 2])
        if means_v.size > 0 and stds_v.size > 0:
            sc = ax2.scatter(means_v, stds_v, c=stds_v, cmap="YlOrRd",
                             alpha=0.4, s=6, edgecolors="none")
            plt.colorbar(sc, ax=ax2, label="Patch LST Std (°C)")
        ax2.set_xlabel("Patch LST Mean (°C)", fontsize=8)
        ax2.set_ylabel("Patch LST Std (°C)", fontsize=8)
        ax2.set_title("Patch Quality: Mean vs Std")
        ax2.grid(True, alpha=0.3)

        # ── Row 1: visual patch thumbnails ────────────────────────────────────
        if has_previews:
            lst_raster = raster_data["LST"]
            patch_size = 64  # default; infer from first patch if possible
            if patches:
                r0, c0 = patches[0]["position"]
                ps_infer = min(64, lst_raster.shape[0] - r0, lst_raster.shape[1] - c0)
                if ps_infer > 0:
                    patch_size = ps_infer

            # Select n_preview patches evenly spread across the mean-temperature range
            valid_patches = [p for p in patches
                             if np.isfinite(p.get("_lst_mean", np.nan))
                             and np.isfinite(p.get("_lst_std",  np.nan))]
            if len(valid_patches) > n_preview:
                # Evenly spaced by sorted mean temperature
                sorted_by_mean = sorted(valid_patches, key=lambda p: p["_lst_mean"])
                step = max(1, len(sorted_by_mean) // n_preview)
                preview_patches = sorted_by_mean[::step][:n_preview]
            else:
                preview_patches = valid_patches[:n_preview]

            n_cols_prev = min(n_preview, 6)
            n_rows_prev = (len(preview_patches) + n_cols_prev - 1) // n_cols_prev

            # Compute a shared colour scale for all preview tiles
            sample_slices = []
            for pp in preview_patches:
                r, c = pp["position"]
                sl = lst_raster[r:r+patch_size, c:c+patch_size]
                finite = sl[np.isfinite(sl)]
                if finite.size > 0:
                    sample_slices.append(finite)
            if sample_slices:
                all_finite = np.concatenate(sample_slices)
                vmin_p, vmax_p = np.nanpercentile(all_finite, [2, 98])
            else:
                vmin_p, vmax_p = 20.0, 55.0

            inner_gs = gridspec.GridSpecFromSubplotSpec(
                n_rows_prev, n_cols_prev, subplot_spec=gs[1, :],
                hspace=0.15, wspace=0.05
            )

            for tile_idx, pp in enumerate(preview_patches):
                r, c = pp["position"]
                sl = lst_raster[r:r+patch_size, c:c+patch_size].copy()
                nan_frac = float((~np.isfinite(sl)).mean()) if sl.size > 0 else 0.0
                # Mark NaN pixels red via masked array so they're visible
                sl_masked = np.ma.masked_invalid(sl)

                ax_t = fig.add_subplot(inner_gs[tile_idx // n_cols_prev,
                                                tile_idx %  n_cols_prev])
                cmap_preview = plt.cm.inferno.copy()
                cmap_preview.set_bad("cyan", alpha=1.0)   # NaN pixels → cyan
                ax_t.imshow(sl_masked, cmap=cmap_preview,
                            vmin=vmin_p, vmax=vmax_p, interpolation="nearest")

                # Flag if NaN fraction is high (potential fill artifact)
                nan_warn = f" ⚠NaN:{nan_frac*100:.0f}%" if nan_frac > 0.02 else ""
                ax_t.set_title(f"μ={pp['_lst_mean']:.1f}°C\nσ={pp['_lst_std']:.1f}°C"
                               f"{nan_warn}", fontsize=6, pad=2)
                ax_t.axis("off")

            # Patch-row caption
            fig.text(
                0.5, 0.01,
                "Visual patch previews (LST °C, inferno scale — cyan = NaN/missing pixels)."
                "  Thumbnails span cool→hot; flat cyan blobs indicate NaN-fill artefacts.",
                ha="center", fontsize=7, style="italic", color="dimgray"
            )

        fig.suptitle(
            f"Patch Quality Diagnostics  "
            f"(n={len(patches):,} total, {valid.sum():,} with stats)",
            fontsize=13, fontweight="bold"
        )
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # 6. Data split distributions
    # ------------------------------------------------------------------
    def plot_split_distributions(self, splits: dict,
                                  filename: str = "06_split_distributions.png"):
        """Compare LST distributions across train / val / test splits."""
        split_keys = [("X_train", "y_train", "Train"),
                      ("X_val",   "y_val",   "Val"),
                      ("X_test",  "y_test",  "Test")]
        colors = ["steelblue", "darkorange", "seagreen"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. LST distribution per split
        ax = axes[0]
        for (_, yk, label), col in zip(split_keys, colors):
            if yk in splits:
                vals = splits[yk].ravel()
                ax.hist(vals, bins=60, alpha=0.6, color=col, label=label, density=True)
        ax.set_xlabel("LST (°C or normalised)")
        ax.set_ylabel("Density")
        ax.set_title("LST Distribution per Split")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Mean ± std bar chart
        ax = axes[1]
        split_labels = []
        split_means, split_stds, split_ns = [], [], []
        for _, yk, label in split_keys:
            if yk in splits:
                v = splits[yk].ravel()
                split_labels.append(label)
                split_means.append(v.mean())
                split_stds.append(v.std())
                split_ns.append(len(splits[yk]))

        bars = ax.bar(split_labels, split_means, yerr=split_stds,
                      color=colors[:len(split_labels)], capsize=6, alpha=0.8)
        for bar, n in zip(bars, split_ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                    f"n={n:,}", ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")
        ax.set_ylabel("Mean LST ± Std")
        ax.set_title("Split Statistics")
        ax.grid(True, alpha=0.3, axis="y")

        # 3. Sample count pie chart
        ax = axes[2]
        if split_ns:
            ax.pie(split_ns, labels=split_labels, colors=colors[:len(split_ns)],
                   autopct="%1.1f%%", startangle=90)
            ax.set_title("Sample Count per Split")

        fig.suptitle("Train / Validation / Test Split Diagnostics",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # 7. Normalisation diagnostics
    # ------------------------------------------------------------------
    def plot_normalization_diagnostics(self, X_raw: np.ndarray, X_norm: np.ndarray,
                                        y_raw: np.ndarray, y_norm: np.ndarray,
                                        channel_names: list = None,
                                        filename: str = "07_normalization.png"):
        """Before/after normalisation distributions for features and targets."""
        n_channels = X_raw.shape[-1]
        channel_names = channel_names or [f"Ch{i}" for i in range(n_channels)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        def _plot_channel_stats(ax, X, label_prefix, color):
            means = [X[:, :, :, c].mean() for c in range(n_channels)]
            stds  = [X[:, :, :, c].std()  for c in range(n_channels)]
            x_pos = np.arange(n_channels)
            ax.bar(x_pos, means, yerr=stds, color=color, alpha=0.7, capsize=3,
                   label="mean ± std")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=7)
            ax.axhline(0, color="black", lw=0.8, linestyle="--")
            ax.set_title(f"{label_prefix} Feature Statistics (per channel)")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        _plot_channel_stats(axes[0, 0], X_raw,  "Raw",        "steelblue")
        _plot_channel_stats(axes[0, 1], X_norm, "Normalised", "darkorange")

        # Target distributions
        for ax, vals, label, color in [
            (axes[1, 0], y_raw.ravel(),  "Target LST – Raw",        "steelblue"),
            (axes[1, 1], y_norm.ravel(), "Target LST – Normalised", "darkorange"),
        ]:
            ax.hist(vals, bins=80, color=color, alpha=0.8, edgecolor="white", density=True)
            ax.axvline(vals.mean(), color="black", linestyle="--", lw=1.5,
                       label=f"μ={vals.mean():.3f}")
            ax.axvline(vals.mean() + vals.std(), color="gray", linestyle=":", lw=1,
                       label=f"σ={vals.std():.3f}")
            ax.axvline(vals.mean() - vals.std(), color="gray", linestyle=":", lw=1)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.set_title(label)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Normalisation Diagnostics", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # 8. Channel correlation heatmap
    # ------------------------------------------------------------------
    def plot_channel_correlation(self, X: np.ndarray, channel_names: list = None,
                                  filename: str = "08_channel_correlation.png"):
        """Pearson correlation matrix across all input channels."""
        n_channels = X.shape[-1]
        channel_names = channel_names or [f"Ch{i}" for i in range(n_channels)]

        # Flatten spatial dims – subsample for speed
        flat = X.reshape(-1, n_channels)
        if flat.shape[0] > 100_000:
            rng = np.random.default_rng(0)
            flat = flat[rng.choice(flat.shape[0], 100_000, replace=False)]

        corr = np.corrcoef(flat.T)

        fig, ax = plt.subplots(figsize=(max(8, n_channels), max(6, n_channels - 1)))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="Pearson r")
        ax.set_xticks(range(n_channels))
        ax.set_yticks(range(n_channels))
        ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(channel_names, fontsize=8)
        ax.set_title("Input Channel Correlation Matrix", fontsize=13, fontweight="bold")

        for i in range(n_channels):
            for j in range(n_channels):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(corr[i, j]) < 0.7 else "white")

        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # 9. Fusion quality (Landsat vs Sentinel-2 vs fused)
    # ------------------------------------------------------------------
    def plot_fusion_comparison(self, landsat_data: dict, sentinel2_data: dict,
                                fused_data: dict, index: str = "NDVI",
                                filename: str = "09_fusion_comparison.png"):
        """Three-panel spatial comparison: Landsat | Sentinel-2 | Fused."""
        datasets = [
            (landsat_data,  "Landsat"),
            (sentinel2_data,"Sentinel-2"),
            (fused_data,    "Fused"),
        ]
        available = [(d, n) for d, n in datasets if index in d]
        if len(available) < 2:
            logger.warning(f"[Diagnostics] plot_fusion_comparison: need ≥2 sources for {index}.")
            return

        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
        if len(available) == 1:
            axes = [axes]

        all_vals = np.concatenate([d[index][np.isfinite(d[index])].ravel()
                                   for d, _ in available])
        vmin, vmax = np.nanpercentile(all_vals, [2, 98])

        for ax, (data, name) in zip(axes, available):
            im = ax.imshow(data[index], cmap="RdYlGn", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}\n{index}", fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Sensor Fusion Comparison – {index}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # 10. Full pipeline summary dashboard
    # ------------------------------------------------------------------
    def plot_pipeline_summary(self, splits: dict, metadata: dict,
                               filename: str = "10_pipeline_summary.png"):
        """One-page summary dashboard of the entire preprocessing run."""
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

        colors = ["#2196F3", "#FF9800", "#4CAF50"]

        # ── Sample counts ─────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, 0])
        labels = ["Train", "Val", "Test"]
        counts = [len(splits.get("X_train", [])), len(splits.get("X_val", [])),
                  len(splits.get("X_test", []))]
        ax0.bar(labels, counts, color=colors, alpha=0.85)
        for bar, cnt in zip(ax0.patches, counts):
            ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{cnt:,}", ha="center", va="bottom", fontsize=9)
        ax0.set_title("Sample Counts")
        ax0.set_ylabel("# Samples")
        ax0.grid(True, alpha=0.3, axis="y")

        # ── Target LST distributions ───────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1:3])
        for (yk, label), col in zip(
                [("y_train", "Train"), ("y_val", "Val"), ("y_test", "Test")],
                colors):
            if yk in splits:
                vals = splits[yk].ravel()
                ax1.hist(vals, bins=60, alpha=0.55, color=col, label=label, density=True)
        ax1.set_xlabel("LST (normalised)")
        ax1.set_ylabel("Density")
        ax1.set_title("Target (LST) Distribution per Split")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Metadata text block ────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 3])
        ax2.axis("off")
        tr = metadata.get("temperature_range", {})
        fi = metadata.get("fusion_info", {})
        info = (
            f"Pipeline Summary\n"
            f"{'─'*28}\n"
            f"Fusion:  {fi.get('fusion_strategy','N/A')}\n"
            f"Landsat files:  {fi.get('landsat_files', 'N/A')}\n"
            f"Sentinel-2 files: {fi.get('sentinel2_files', 'N/A')}\n"
            f"Fused datasets: {fi.get('fused_datasets', 'N/A')}\n"
            f"{'─'*28}\n"
            f"LST mean:  {tr.get('mean', float('nan')):.4f}\n"
            f"LST std:   {tr.get('std',  float('nan')):.4f}\n"
            f"LST min:   {tr.get('min',  float('nan')):.4f}\n"
            f"LST max:   {tr.get('max',  float('nan')):.4f}\n"
            f"{'─'*28}\n"
            f"Patches: {metadata.get('n_samples','N/A'):,}\n"
            f"Channels: {metadata.get('n_channels','N/A')}\n"
            f"Patch size: {metadata.get('patch_size','N/A')} px"
        )
        ax2.text(0.05, 0.95, info, transform=ax2.transAxes,
                 fontsize=8, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4ff", alpha=0.8))

        # ── Per-channel mean across splits ─────────────────────────────
        ax3 = fig.add_subplot(gs[1, :2])
        n_ch = splits["X_train"].shape[-1] if "X_train" in splits else 0
        ch_names = metadata.get("channel_order", [f"Ch{i}" for i in range(n_ch)])
        x_pos = np.arange(n_ch)
        width = 0.27
        for offset, (xk, label), col in zip(
                [-width, 0, width],
                [("X_train","Train"),("X_val","Val"),("X_test","Test")],
                colors):
            if xk in splits:
                means = splits[xk].mean(axis=(0, 1, 2))
                ax3.bar(x_pos + offset, means, width=width, color=col, alpha=0.8, label=label)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(ch_names[:n_ch], rotation=45, ha="right", fontsize=7)
        ax3.set_title("Per-Channel Mean (normalised) across Splits")
        ax3.set_ylabel("Mean value")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis="y")

        # ── Per-channel std across splits ──────────────────────────────
        ax4 = fig.add_subplot(gs[1, 2:])
        for offset, (xk, label), col in zip(
                [-width, 0, width],
                [("X_train","Train"),("X_val","Val"),("X_test","Test")],
                colors):
            if xk in splits:
                stds = splits[xk].std(axis=(0, 1, 2))
                ax4.bar(x_pos + offset, stds, width=width, color=col, alpha=0.8, label=label)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(ch_names[:n_ch], rotation=45, ha="right", fontsize=7)
        ax4.set_title("Per-Channel Std (normalised) across Splits")
        ax4.set_ylabel("Std value")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")

        # ── Patch LST variance distribution ───────────────────────────
        ax5 = fig.add_subplot(gs[2, :2])
        for (yk, label), col in zip(
                [("y_train","Train"),("y_val","Val"),("y_test","Test")],
                colors):
            if yk in splits:
                patch_stds = splits[yk].std(axis=(1, 2, 3))
                ax5.hist(patch_stds, bins=50, alpha=0.55, color=col,
                         label=f"{label} μ={patch_stds.mean():.3f}", density=True)
        ax5.set_xlabel("Per-patch LST Std")
        ax5.set_ylabel("Density")
        ax5.set_title("Patch LST Variance Distribution per Split")
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # ── Normalisation check ────────────────────────────────────────
        ax6 = fig.add_subplot(gs[2, 2:])
        split_info = []
        for xk, yk, label in [("X_train","y_train","Train"),
                                ("X_val","y_val","Val"),
                                ("X_test","y_test","Test")]:
            if xk in splits and yk in splits:
                split_info.append({
                    "Split": label,
                    "X mean": f"{splits[xk].mean():.4f}",
                    "X std":  f"{splits[xk].std():.4f}",
                    "y mean": f"{splits[yk].mean():.4f}",
                    "y std":  f"{splits[yk].std():.4f}",
                })
        ax6.axis("off")
        if split_info:
            col_labels = list(split_info[0].keys())
            cell_text  = [[row[c] for c in col_labels] for row in split_info]
            tbl = ax6.table(cellText=cell_text, colLabels=col_labels,
                            loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.1, 2.0)
        ax6.set_title("Normalisation Verification", fontweight="bold")

        fig.suptitle("Preprocessing Pipeline – Summary Dashboard",
                     fontsize=15, fontweight="bold")
        self._save(fig, filename)
        logger.info(f"[Diagnostics] ✅ Pipeline summary saved → {self.out / filename}")

    # ==================================================================
    # SENTINEL-2 SPECIFIC DIAGNOSTICS
    # ==================================================================

    # ------------------------------------------------------------------
    # S2-1. Sentinel-2 raw band overview (10 m + 20 m bands side-by-side)
    # ------------------------------------------------------------------

    # Canonical S2 band aliases: maps logical name → list of possible npz keys
    _S2_BAND_ALIASES: dict = {
        "B2":  ["B2",  "B02", "blue",  "BLUE",  "b2",  "band2",  "band02"],
        "B3":  ["B3",  "B03", "green", "GREEN", "b3",  "band3",  "band03"],
        "B4":  ["B4",  "B04", "red",   "RED",   "b4",  "band4",  "band04"],
        "B8":  ["B8",  "B08", "nir",   "NIR",   "b8",  "band8",  "band08"],
        "B11": ["B11", "swir1","SWIR1", "b11", "band11"],
        "B12": ["B12", "swir2","SWIR2", "b12", "band12"],
    }

    _S2_BAND_META: dict = {
        "B2":  ("Blue – 490 nm",   "Blues_r",  "10 m"),
        "B3":  ("Green – 560 nm",  "Greens_r", "10 m"),
        "B4":  ("Red – 665 nm",    "Reds_r",   "10 m"),
        "B8":  ("NIR – 842 nm",    "YlGn",     "10 m"),
        "B11": ("SWIR1 – 1610 nm", "YlOrBr",   "20 m"),
        "B12": ("SWIR2 – 2190 nm", "copper",   "20 m"),
    }

    def _resolve_s2_bands(self, raw_data: dict) -> dict:
        """
        Resolve actual npz keys to canonical S2 band names.
        Handles key variants like B2/B02/blue/BLUE etc.
        Also auto-detects any 2-D array that looks like a spectral band
        (i.e. not an index we computed).
        Returns {canonical_name: array}.
        """
        resolved = {}
        available_keys = set(raw_data.keys())

        # Step 1: try known aliases
        for canonical, aliases in self._S2_BAND_ALIASES.items():
            for alias in aliases:
                if alias in available_keys:
                    arr = np.asarray(raw_data[alias], dtype=float)
                    if arr.ndim == 2:
                        resolved[canonical] = arr
                        break

        # Step 2: log all keys and what we found / missed
        logger.info(f"[Diagnostics] S2 npz keys found: {sorted(available_keys)}")
        logger.info(f"[Diagnostics] S2 bands resolved: {list(resolved.keys())}")

        # Step 3: if we resolved nothing, fall back to ALL 2-D arrays
        if not resolved:
            logger.warning(
                "[Diagnostics] No canonical S2 bands matched — "
                "falling back to all 2-D arrays in npz."
            )
            for k, v in raw_data.items():
                arr = np.asarray(v, dtype=float)
                if arr.ndim == 2:
                    resolved[k] = arr

        return resolved

    @staticmethod
    def _safe_stretch(arr: np.ndarray,
                      lo_pct: float = 2.0,
                      hi_pct: float = 98.0,
                      min_spread: float = 1e-4) -> tuple:
        """
        Compute vmin/vmax for imshow with a guaranteed minimum spread.
        Falls back to absolute min/max when percentile range collapses.
        Also warns if the array looks like it contains only fill values.
        """
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            return 0.0, 1.0

        lo, hi = np.nanpercentile(valid, [lo_pct, hi_pct])
        spread = hi - lo

        if spread < min_spread:
            # Percentile stretch collapsed → use full range
            lo_full, hi_full = valid.min(), valid.max()
            spread_full = hi_full - lo_full
            logger.warning(
                f"[Diagnostics] Band has near-zero contrast "
                f"(p{lo_pct:.0f}={lo:.6f}, p{hi_pct:.0f}={hi:.6f}). "
                f"Full range: [{lo_full:.6f}, {hi_full:.6f}]."
            )
            if spread_full < min_spread:
                # Truly uniform — centre around the constant value
                centre = valid.mean()
                return centre - 0.05, centre + 0.05
            return lo_full, hi_full

        return lo, hi

    @staticmethod
    def _normalise_band(arr: np.ndarray,
                        lo: float, hi: float) -> np.ndarray:
        """Linearly stretch arr to [0, 1] given lo/hi, clip to [0, 1]."""
        spread = hi - lo
        if spread < 1e-10:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr.astype(np.float32) - np.float32(lo)) / np.float32(spread), 0.0, 1.0)

    def plot_s2_raw_bands(self, raw_data: dict,
                          filename: str = "s2_01_raw_bands.png"):
        """
        Display all Sentinel-2 spectral bands grouped by native resolution.
        10 m bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
        20 m bands: B11 (SWIR1), B12 (SWIR2)
        Also renders a false-colour NIR/Red/Green composite.

        Robust against:
        - Different npz key naming conventions (B2/B02/blue/BLUE…)
        - Near-zero variance (uniform fill tiles)
        - Different band value scales (DN, reflectance 0-1, reflectance ×10000)
        """
        bands = self._resolve_s2_bands(raw_data)

        if not bands:
            logger.warning("[Diagnostics] plot_s2_raw_bands: no 2-D bands found in raw_data.")
            return

        # ── Data-level diagnostic: print per-band statistics ───────────
        logger.info("[Diagnostics] S2 band statistics:")
        for name, arr in bands.items():
            valid = arr[np.isfinite(arr)]
            if valid.size:
                logger.info(
                    f"  {name:5s}: shape={arr.shape}  "
                    f"min={valid.min():.4f}  max={valid.max():.4f}  "
                    f"mean={valid.mean():.4f}  std={valid.std():.6f}  "
                    f"nonzero={np.count_nonzero(valid):,}/{valid.size:,}"
                )
            else:
                logger.warning(f"  {name:5s}: all NaN/Inf!")
        # ───────────────────────────────────────────────────────────────

        ordered = [b for b in ["B2", "B3", "B4", "B8", "B11", "B12"] if b in bands]
        extras  = [b for b in bands if b not in ordered]  # fallback unnamed bands
        ordered += extras

        n_panels = len(ordered) + 1  # +1 for composite
        ncols = min(4, n_panels)
        nrows = (n_panels + ncols - 1) // ncols

        # Use larger per-panel size and constrained_layout so colorbars
        # never get clipped regardless of how many bands are present.
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 5.5, nrows * 5.5),
            constrained_layout=True,
        )
        axes = np.array(axes).flatten()
        panel = 0

        # ── False-colour composite (NIR/Red/Green) ─────────────────────
        rgb_canonical = ["B8", "B4", "B3"]
        rgb_available = [b for b in rgb_canonical if b in bands]

        ax = axes[panel]; panel += 1
        if len(rgb_available) == 3:
            channels = []
            for b in rgb_available:
                arr = bands[b]
                lo, hi = self._safe_stretch(arr)
                channels.append(self._normalise_band(arr, lo, hi))

            h = min(c.shape[0] for c in channels)
            w = min(c.shape[1] for c in channels)
            composite = np.dstack([c[:h, :w] for c in channels])
            ax.imshow(composite, interpolation="bilinear")
            ax.set_title("False-Colour Composite\n(NIR / Red / Green)", fontweight="bold")
        else:
            ax.text(0.5, 0.5,
                    f"NIR/Red/Green unavailable\n(found: {list(bands.keys())})",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="gray")
            ax.set_title("False-Colour Composite\n(unavailable)", fontweight="bold")
        ax.axis("off")

        # ── Individual band panels ─────────────────────────────────────
        for b in ordered:
            ax = axes[panel]; panel += 1
            arr = bands[b]
            lo, hi = self._safe_stretch(arr)
            meta = self._S2_BAND_META.get(b, (b, "viridis", "?"))
            label, cmap, res = meta

            # Annotate if the stretch had to fall back to full range
            stretch_note = ""
            p2, p98 = np.nanpercentile(arr[np.isfinite(arr)], [2, 98]) \
                if arr[np.isfinite(arr)].size else (0, 0)
            if (p98 - p2) < 1e-4:
                stretch_note = "\n⚠ near-uniform (full range shown)"

            im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear")
            ax.set_title(f"{b} – {label}\n({res}){stretch_note}", fontsize=8)
            ax.axis("off")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=7)

        for ax in axes[panel:]:
            ax.set_visible(False)

        # ── Figure-level stats annotation ──────────────────────────────
        n_bands_found = len(bands)
        n_bands_canonical = sum(1 for b in ["B2","B3","B4","B8","B11","B12"] if b in bands)
        fig.suptitle(
            f"Sentinel-2 Raw Bands  "
            f"({n_bands_canonical}/6 canonical bands resolved, "
            f"{n_bands_found} total arrays)",
            fontsize=13, fontweight="bold"
        )
        # constrained_layout handles spacing — no tight_layout needed
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # S2-2. Sentinel-2 spectral indices with UHI-relevant interpretation
    # ------------------------------------------------------------------
    def plot_s2_spectral_indices(self, indices: dict,
                                  filename: str = "s2_02_spectral_indices.png"):
        """
        Spatial maps of all S2-derived indices with colourbar annotations
        and UHI-relevant class boundaries overlaid as contour lines.
        """
        index_cfg = {
            "NDVI":   ("Vegetation",        "RdYlGn",  [0.2, 0.5]),
            "NDBI":   ("Built-up",          "RdBu_r",  [0.0]),
            "MNDWI":  ("Water",             "Blues",   [0.0]),
            "BSI":    ("Bare Soil",         "YlOrBr",  [0.0]),
            "UI":     ("Urban Index",       "hot_r",   [0.0]),
            "albedo": ("Surface Albedo",    "gray",    [0.15, 0.25]),
        }

        keys = [k for k in index_cfg if k in indices]
        if not keys:
            logger.warning("[Diagnostics] plot_s2_spectral_indices: no indices found.")
            return

        ncols = 3
        nrows = (len(keys) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4.5))
        axes = np.array(axes).flatten()

        for ax, key in zip(axes, keys):
            arr = indices[key]
            valid = arr[np.isfinite(arr)]
            vmin, vmax = np.nanpercentile(valid, [2, 98])
            label, cmap, contour_levels = index_cfg[key]
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            # Overlay threshold contours
            try:
                ax.contour(arr, levels=contour_levels, colors="white",
                           linewidths=0.8, alpha=0.7)
            except Exception:
                pass
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{key} – {label}\nrange [{vmin:.2f}, {vmax:.2f}]",
                         fontsize=8, fontweight="bold")
            ax.axis("off")

        for ax in axes[len(keys):]:
            ax.set_visible(False)

        fig.suptitle("Sentinel-2 Spectral Indices with Class Boundaries",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # S2-3. Sentinel-2 band statistics & value distributions
    # ------------------------------------------------------------------
    def plot_s2_band_statistics(self, raw_data: dict,
                                 filename: str = "s2_03_band_statistics.png"):
        """
        Box-and-whisker + histogram for each S2 band. Highlights outlier
        pixels that may indicate cloud/shadow contamination.
        """
        bands = self._resolve_s2_bands(raw_data)
        band_order = [b for b in ["B2", "B3", "B4", "B8", "B11", "B12"] if b in bands]
        if not band_order:
            # Fall back to whatever was resolved
            band_order = list(bands.keys())
        if not band_order:
            logger.warning("[Diagnostics] plot_s2_band_statistics: no bands found.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 9))

        # --- Top: box plots ---
        ax = axes[0]
        data_lists = []
        for b in band_order:
            arr = bands[b].ravel()
            arr = arr[np.isfinite(arr)]
            # Subsample for speed
            if len(arr) > 100_000:
                arr = np.random.default_rng(42).choice(arr, 100_000, replace=False)
            data_lists.append(arr)

        bp = ax.boxplot(data_lists, tick_labels=band_order, patch_artist=True,
                        showfliers=False, notch=False)
        band_colors = ["#1565C0","#388E3C","#C62828","#7B1FA2","#E65100","#4E342E"]
        for patch, col in zip(bp["boxes"], band_colors[:len(band_order)]):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        ax.set_title("Band Value Distributions (box = IQR, whiskers = 1.5×IQR)")
        ax.set_ylabel("Digital Number / Reflectance")
        ax.grid(True, alpha=0.3, axis="y")

        # --- Bottom: overlaid KDE-style histograms ---
        ax2 = axes[1]
        for b, col in zip(band_order, band_colors):
            arr = bands[b].ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            lo, hi = np.nanpercentile(arr, [0.5, 99.5])
            spread = hi - lo
            if spread < 1e-6:
                lo, hi = arr.min(), arr.max()
            arr_clip = arr[(arr >= lo) & (arr <= hi)]
            if len(arr_clip) > 0:
                ax2.hist(arr_clip, bins=100, alpha=0.45, color=col,
                         label=b, density=True)
        ax2.set_xlabel("Pixel Value")
        ax2.set_ylabel("Density")
        ax2.set_title("Band Pixel-Value Histograms (0.5–99.5 percentile range)")
        ax2.legend(ncol=3, fontsize=8)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Sentinel-2 Band-Level Statistics", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # S2-4. Cloud / shadow quality-flag heatmap
    # ------------------------------------------------------------------
    def plot_s2_data_quality(self, raw_data: dict,
                              filename: str = "s2_04_data_quality.png"):
        """
        Derive a simple pixel-level quality score from reflectance values:
          - Saturated (>1.0 reflectance after scaling) → likely cloud top
          - Very dark (<0.01) in Blue → likely cloud shadow
          - NDVI < -0.1 AND Blue > 0.3 → possible cloud
        Also shows NaN/fill-value masks per band.
        """
        bands = self._resolve_s2_bands(raw_data)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: NaN coverage per band
        ax = axes[0]
        band_names = [b for b in ["B2","B3","B4","B8","B11","B12"] if b in bands]
        if not band_names:
            band_names = list(bands.keys())
        nan_pcts   = [np.sum(~np.isfinite(bands[b])) / bands[b].size * 100
                      for b in band_names]
        colors_q   = ["#EF5350" if p > 5 else "#66BB6A" for p in nan_pcts]
        bars = ax.barh(band_names, nan_pcts, color=colors_q, alpha=0.85)
        ax.set_xlabel("NaN / Fill pixels (%)")
        ax.set_title("Missing Data per Band")
        ax.axvline(5, color="red", linestyle="--", lw=1.2, label="5% threshold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")

        # Panel 2: Derived cloud probability proxy
        ax2 = axes[1]
        b2 = bands.get("B2"); b4 = bands.get("B4"); b8 = bands.get("B8")
        if b2 is not None and b4 is not None and b8 is not None:
            blue = b2.astype(np.float32)
            red  = b4.astype(np.float32)
            nir  = b8.astype(np.float32)
            eps  = np.float32(1e-8)
            ndvi = (nir - red) / (nir + red + eps)

            blue_norm = np.clip(blue / (np.nanmax(blue) + eps), 0, 1)
            cloud_score = blue_norm * np.clip(1 - ndvi, 0, 1)
            im2 = ax2.imshow(cloud_score, cmap="RdYlGn_r", vmin=0, vmax=1)
            ax2.set_title("Cloud Probability Proxy\n(high blue × low NDVI)")
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2, label="Score [0–1]")

            # Panel 3: Shadow proxy
            ax3 = axes[2]
            nir_norm = np.clip(nir / (np.nanmax(nir) + eps), 0, 1)
            shadow_score = np.clip(1 - nir_norm, 0, 1) * np.clip(1 - cloud_score, 0, 1)
            im3 = ax3.imshow(shadow_score, cmap="Purples", vmin=0, vmax=1)
            ax3.set_title("Shadow Probability Proxy\n(dark NIR, low cloud score)")
            ax3.axis("off")
            plt.colorbar(im3, ax=ax3, label="Score [0–1]")
        else:
            axes[1].text(0.5, 0.5,
                         f"B2/B4/B8 required\nAvailable: {list(bands.keys())}",
                         ha="center", va="center", transform=axes[1].transAxes,
                         fontsize=9, color="gray")
            axes[2].set_visible(False)

        fig.suptitle("Sentinel-2 Pixel Quality Assessment",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # S2-5. Sentinel-2 band-ratio analysis (UHI-relevant ratios)
    # ------------------------------------------------------------------
    def plot_s2_band_ratios(self, raw_data: dict,
                             filename: str = "s2_05_band_ratios.png"):
        """
        Compute and display five band-ratio images commonly used in
        urban / vegetation / water studies.
        """
        bands = self._resolve_s2_bands(raw_data)
        eps = 1e-8
        ratios = {}
        labels = {}

        b2 = bands.get("B2"); b3 = bands.get("B3"); b4 = bands.get("B4")
        b8 = bands.get("B8"); b11 = bands.get("B11"); b12 = bands.get("B12")

        if b8 is not None and b4 is not None:
            ratios["NIR/Red"]   = b8.astype(np.float32) / (b4.astype(np.float32) + eps)
            labels["NIR/Red"]   = "Vegetation Vigour"
        if b11 is not None and b8 is not None:
            ratios["SWIR1/NIR"] = b11.astype(np.float32) / (b8.astype(np.float32) + eps)
            labels["SWIR1/NIR"] = "Urban Heat / Soil Moisture Proxy"
        if b4 is not None and b2 is not None:
            ratios["Red/Blue"]  = b4.astype(np.float32) / (b2.astype(np.float32) + eps)
            labels["Red/Blue"]  = "Aerosol / Dust Proxy"
        if b8 is not None and b11 is not None:
            ratios["NIR/SWIR1"] = b8.astype(np.float32) / (b11.astype(np.float32) + eps)
            labels["NIR/SWIR1"] = "Soil Moisture Index"
        if b2 is not None and b3 is not None and b4 is not None and b8 is not None:
            num = b2.astype(np.float32) + b3.astype(np.float32)
            den = b4.astype(np.float32) + b8.astype(np.float32)
            ratios["(B+G)/(R+NIR)"] = num / (den + eps)
            labels["(B+G)/(R+NIR)"] = "Simple Urban Index"

        if not ratios:
            logger.warning(
                f"[Diagnostics] plot_s2_band_ratios: insufficient bands. "
                f"Available: {list(bands.keys())}"
            )
            return

        n = len(ratios)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
        axes = np.array(axes).flatten()

        cmaps = ["YlGn", "hot_r", "RdBu", "Blues", "RdYlBu"]
        for ax, (key, arr), cmap in zip(axes, ratios.items(), cmaps):
            arr = arr.astype(np.float32)
            vmin, vmax = self._safe_stretch(arr)
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
            ax.set_title(f"{key}\n{labels[key]}", fontsize=8, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle("Sentinel-2 Band Ratio Analysis", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # S2-6. Land cover classification proxy from S2 indices
    # ------------------------------------------------------------------
    def plot_s2_landcover_proxy(self, indices: dict,
                                 filename: str = "s2_06_landcover_proxy.png"):
        """
        Assign each pixel to one of five UHI-relevant land cover classes
        using threshold rules on NDVI, NDBI, and MNDWI, then show:
          - Spatial map of classified pixels
          - Class area statistics (bar chart)
          - Class-wise LST distribution (if LST available)
        """
        required = ["NDVI", "NDBI", "MNDWI"]
        if not all(k in indices for k in required):
            logger.warning("[Diagnostics] plot_s2_landcover_proxy: need NDVI/NDBI/MNDWI.")
            return

        ndvi  = indices["NDVI"]
        ndbi  = indices["NDBI"]
        mndwi = indices["MNDWI"]

        # Classification rules (Jakarta tropical urban context)
        # 0=Water, 1=Vegetation, 2=Bare Soil, 3=Impervious/Urban, 4=Mixed/Other
        h, w = ndvi.shape
        lc = np.full((h, w), 4, dtype=np.int8)
        lc[mndwi > 0.0]  = 0          # Water
        lc[(ndvi > 0.4) & (mndwi <= 0.0)]  = 1    # Dense vegetation
        lc[(ndvi < 0.2) & (ndbi < 0.0) & (mndwi <= 0.0)] = 2  # Bare soil
        lc[(ndbi > 0.0) & (mndwi <= 0.0)]  = 3    # Impervious / built-up

        class_names  = ["Water", "Vegetation", "Bare Soil", "Urban/Impervious", "Mixed"]
        class_colors = ["#2196F3","#4CAF50","#FFC107","#F44336","#9E9E9E"]
        cmap_lc = mcolors.ListedColormap(class_colors)

        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # Spatial map
        ax0 = fig.add_subplot(gs[:, :2])
        im  = ax0.imshow(lc, cmap=cmap_lc, vmin=-0.5, vmax=4.5,
                         interpolation="nearest")
        legend_patches = [Patch(color=c, label=n)
                          for c, n in zip(class_colors, class_names)]
        ax0.legend(handles=legend_patches, loc="lower right",
                   fontsize=8, framealpha=0.8)
        ax0.set_title("Land Cover Classification Proxy\n(NDVI / NDBI / MNDWI rules)",
                      fontweight="bold")
        ax0.axis("off")

        # Area bar chart
        ax1 = fig.add_subplot(gs[0, 2])
        counts = [(lc == i).sum() for i in range(5)]
        pcts   = [c / lc.size * 100 for c in counts]
        bars   = ax1.bar(range(5), pcts, color=class_colors, alpha=0.85)
        ax1.set_xticks(range(5))
        ax1.set_xticklabels(class_names, rotation=30, ha="right", fontsize=7)
        ax1.set_ylabel("Area (%)")
        ax1.set_title("Class Area Distribution")
        ax1.grid(True, alpha=0.3, axis="y")
        for bar, pct in zip(bars, pcts):
            if pct > 1:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f"{pct:.1f}%", ha="center", va="bottom", fontsize=7)

        # Class-wise LST violin (if available)
        ax2 = fig.add_subplot(gs[1, 2])
        if "LST" in indices:
            lst = indices["LST"]
            lst_data = []
            valid_labels = []
            for i, name in enumerate(class_names):
                mask = (lc == i) & np.isfinite(lst)
                vals = lst[mask].ravel()
                if len(vals) >= 10:
                    lst_data.append(vals[:10_000] if len(vals) > 10_000 else vals)
                    valid_labels.append(name)

            if lst_data:
                vp = ax2.violinplot(lst_data, showmedians=True, showextrema=True)
                for body, col in zip(vp["bodies"],
                                     [class_colors[class_names.index(l)]
                                      for l in valid_labels]):
                    body.set_facecolor(col)
                    body.set_alpha(0.7)
                ax2.set_xticks(range(1, len(valid_labels) + 1))
                ax2.set_xticklabels(valid_labels, rotation=30, ha="right", fontsize=7)
                ax2.set_ylabel("LST (°C)")
                ax2.set_title("LST Distribution by Land Cover")
                ax2.grid(True, alpha=0.3, axis="y")
        else:
            ax2.text(0.5, 0.5, "LST not available\nfor this dataset",
                     ha="center", va="center", transform=ax2.transAxes,
                     fontsize=10, color="gray")
            ax2.set_title("LST by Land Cover")
            ax2.axis("off")

        fig.suptitle("Sentinel-2 Land Cover Proxy & UHI Class Analysis",
                     fontsize=13, fontweight="bold")
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # S2-7. Temporal trend of S2 indices across all processed scenes
    # ------------------------------------------------------------------
    def plot_s2_temporal_trends(self,
                                 processed_scenes: list,
                                 filename: str = "s2_07_temporal_trends.png"):
        """
        Plot how scene-level statistics (mean/std) of each index evolve
        over time across all Sentinel-2 acquisitions.

        Args:
            processed_scenes: list of (pd.Timestamp, processed_dict) tuples
        """
        if len(processed_scenes) < 2:
            logger.warning("[Diagnostics] plot_s2_temporal_trends: need ≥2 scenes.")
            return

        indices_to_track = ["NDVI", "NDBI", "MNDWI", "BSI", "UI", "albedo"]
        timestamps = [ts for ts, _ in processed_scenes]
        scene_dicts = [d for _, d in processed_scenes]

        present = [k for k in indices_to_track
                   if any(k in d for d in scene_dicts)]
        if not present:
            return

        ncols = 2
        nrows = (len(present) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(14, nrows * 3.5), sharex=True)
        axes = np.array(axes).flatten()

        date_strs = [ts.strftime("%Y-%m") for ts in timestamps]
        x = np.arange(len(timestamps))

        for ax, key in zip(axes, present):
            means, stds, medians = [], [], []
            for d in scene_dicts:
                if key in d:
                    vals = d[key][np.isfinite(d[key])].ravel()
                    means.append(vals.mean() if len(vals) else np.nan)
                    stds.append(vals.std()  if len(vals) else np.nan)
                    medians.append(np.median(vals) if len(vals) else np.nan)
                else:
                    means.append(np.nan); stds.append(np.nan); medians.append(np.nan)

            means   = np.array(means,   dtype=float)
            stds    = np.array(stds,    dtype=float)
            medians = np.array(medians, dtype=float)

            ax.fill_between(x, means - stds, means + stds,
                            alpha=0.2, color="#2196F3", label="±1 σ")
            ax.plot(x, means,   "o-", color="#2196F3", lw=1.8, ms=4, label="mean")
            ax.plot(x, medians, "s--", color="#FF9800", lw=1.2, ms=3, label="median")
            ax.set_xticks(x)
            ax.set_xticklabels(date_strs, rotation=45, ha="right", fontsize=7)
            ax.set_title(key, fontsize=9, fontweight="bold")
            ax.set_ylabel("Value")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for ax in axes[len(present):]:
            ax.set_visible(False)

        fig.suptitle("Sentinel-2 Index Temporal Trends (all scenes)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # S2-8. Landsat vs Sentinel-2 index agreement (pre-fusion)
    # ------------------------------------------------------------------
    def plot_sensor_agreement(self,
                               landsat_data: dict,
                               sentinel2_data: dict,
                               filename: str = "s2_08_sensor_agreement.png"):
        """
        For each common index, overlay histograms and compute a simple
        agreement score (1 – Wasserstein-normalised distance) between
        the Landsat and Sentinel-2 distributions before fusion.
        """
        common = [k for k in ["NDVI","NDBI","MNDWI","BSI","UI","albedo"]
                  if k in landsat_data and k in sentinel2_data]
        if not common:
            logger.warning("[Diagnostics] plot_sensor_agreement: no common indices.")
            return

        ncols = 3
        nrows = (len(common) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
        axes = np.array(axes).flatten()

        for ax, key in zip(axes, common):
            ls_vals = landsat_data[key][np.isfinite(landsat_data[key])].ravel()
            s2_vals = sentinel2_data[key][np.isfinite(sentinel2_data[key])].ravel()

            # Subsample
            if len(ls_vals) > 50_000:
                ls_vals = np.random.default_rng(0).choice(ls_vals, 50_000, replace=False)
            if len(s2_vals) > 50_000:
                s2_vals = np.random.default_rng(0).choice(s2_vals, 50_000, replace=False)

            # ── Per-index sensible display range ─────────────────────────
            # Standard spectral indices live in [-1, 1] (or [-1, 2] for NDVI).
            # Albedo lives in [0, 1].  Use a tight union of the p1–p99 ranges
            # from BOTH sensors so outliers / wrong-scale data don't collapse
            # the histogram.  If a sensor's data lies entirely outside the
            # display window it means a scaling problem upstream; we still
            # show the clipped distribution so the mismatch is visible.
            _INDEX_HARD_LIMITS = {
                "NDVI":   (-1.0,  1.0),
                "NDBI":   (-1.0,  1.0),
                "MNDWI":  (-1.0,  1.0),
                "BSI":    (-1.0,  1.0),
                "UI":     (-1.0,  1.0),
                "albedo": ( 0.0,  1.0),
            }
            hard_lo, hard_hi = _INDEX_HARD_LIMITS.get(key, (-1.5, 1.5))

            # Percentile range within hard limits
            def _pct_range(arr, hard_lo, hard_hi):
                clipped = arr[(arr >= hard_lo) & (arr <= hard_hi)]
                if clipped.size < 10:
                    return hard_lo, hard_hi
                return float(np.percentile(clipped, 1)), float(np.percentile(clipped, 99))

            ls_lo, ls_hi = _pct_range(ls_vals, hard_lo, hard_hi)
            s2_lo, s2_hi = _pct_range(s2_vals, hard_lo, hard_hi)
            lo = max(hard_lo, min(ls_lo, s2_lo))
            hi = min(hard_hi, max(ls_hi, s2_hi))
            if hi - lo < 1e-6:
                lo, hi = hard_lo, hard_hi

            bins = np.linspace(lo, hi, 80)

            # Clip values to display window for histogram
            ls_plot = np.clip(ls_vals, lo, hi)
            s2_plot = np.clip(s2_vals, lo, hi)

            ax.hist(ls_plot, bins=bins, alpha=0.55, color="#1565C0",
                    density=True, label="Landsat")
            ax.hist(s2_plot, bins=bins, alpha=0.55, color="#E65100",
                    density=True, label="Sentinel-2")
            ax.set_xlim(lo, hi)

            # Simple agreement: correlation of binned counts
            ls_h, _ = np.histogram(ls_plot, bins=bins, density=True)
            s2_h, _ = np.histogram(s2_plot, bins=bins, density=True)
            if ls_h.std() > 0 and s2_h.std() > 0:
                agreement = np.corrcoef(ls_h, s2_h)[0, 1]
                ax.set_title(f"{key}  agreement r={agreement:.3f}", fontsize=9,
                             fontweight="bold")
            else:
                ax.set_title(key, fontsize=9)

            # Mean lines (clamped to display window)
            ls_mean = float(np.clip(ls_vals.mean(), lo, hi))
            s2_mean = float(np.clip(s2_vals.mean(), lo, hi))
            ax.axvline(ls_mean, color="#1565C0", linestyle="--", lw=1.5,
                       label=f"LS μ={ls_vals.mean():.3f}")
            ax.axvline(s2_mean, color="#E65100", linestyle="--", lw=1.5,
                       label=f"S2 μ={s2_vals.mean():.3f}")
            ax.set_xlabel(key, fontsize=8)
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for ax in axes[len(common):]:
            ax.set_visible(False)

        fig.suptitle("Landsat vs Sentinel-2 Index Distribution Agreement (pre-fusion)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # S2-9. Spatial resolution comparison (10 m vs 30 m)
    # ------------------------------------------------------------------
    def plot_resolution_comparison(self,
                                    s2_data: dict,
                                    ls_data: dict,
                                    index: str = "NDVI",
                                    filename: str = "s2_09_resolution_comparison.png"):
        """
        Show the same index at Sentinel-2 native 10 m resolution versus
        the Landsat 30 m equivalent.  Includes a profile transect along
        the central row to illustrate spatial detail loss.
        """
        if index not in s2_data or index not in ls_data:
            logger.warning(f"[Diagnostics] plot_resolution_comparison: {index} not in both sensors.")
            return

        s2_arr = s2_data[index].astype(np.float32)
        ls_arr = ls_data[index].astype(np.float32)

        fig = plt.figure(figsize=(18, 11))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

        # ── Per-panel colour ranges (independent) so a blank/uniform S2
        #    array is still rendered with contrast.
        def _safe_vrange(arr):
            valid = arr[np.isfinite(arr)]
            if valid.size == 0:
                return 0.0, 1.0
            lo, hi = np.nanpercentile(valid, [2, 98])
            if (hi - lo) < 1e-6:
                lo, hi = valid.min(), valid.max()
            if (hi - lo) < 1e-6:
                lo, hi = lo - 0.05, hi + 0.05
            return float(lo), float(hi)

        s2_vmin, s2_vmax = _safe_vrange(s2_arr)
        ls_vmin, ls_vmax = _safe_vrange(ls_arr)

        ax0 = fig.add_subplot(gs[0, 0])
        im0 = ax0.imshow(s2_arr, cmap="RdYlGn", vmin=s2_vmin, vmax=s2_vmax)
        ax0.set_title(f"Sentinel-2  {index}\n(10 m native)", fontweight="bold")
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0)

        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(ls_arr, cmap="RdYlGn", vmin=ls_vmin, vmax=ls_vmax)
        ax1.set_title(f"Landsat  {index}\n(30 m native)", fontweight="bold")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1)

        # Difference map (resample S2 to LS shape)
        ax2 = fig.add_subplot(gs[0, 2])
        try:
            s2_resampled = zoom(s2_arr,
                                (ls_arr.shape[0]/s2_arr.shape[0],
                                 ls_arr.shape[1]/s2_arr.shape[1]),
                                order=1)
            h = min(s2_resampled.shape[0], ls_arr.shape[0])
            w = min(s2_resampled.shape[1], ls_arr.shape[1])
            diff = s2_resampled[:h, :w] - ls_arr[:h, :w]
            lim  = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98)
            imd  = ax2.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax2.set_title(f"Difference\n(S2 resampled − Landsat)", fontweight="bold")
            ax2.axis("off")
            plt.colorbar(imd, ax=ax2, label="Δ value")
        except Exception as e:
            ax2.text(0.5, 0.5, f"Diff failed:\n{e}", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=8)
            ax2.axis("off")

        # Transect profiles
        ax3 = fig.add_subplot(gs[1, :])
        row_s2 = s2_arr.shape[0] // 2
        row_ls = ls_arr.shape[0] // 2
        profile_s2 = s2_arr[row_s2, :]
        profile_ls = ls_arr[row_ls, :]

        x_s2 = np.linspace(0, 1, len(profile_s2))
        x_ls = np.linspace(0, 1, len(profile_ls))

        # Mask NaNs for clean plotting
        s2_valid = np.isfinite(profile_s2)
        ls_valid = np.isfinite(profile_ls)

        if s2_valid.any():
            ax3.plot(x_s2[s2_valid], profile_s2[s2_valid], color="#E65100", lw=1.2,
                     alpha=0.8, label=f"Sentinel-2 (10 m, {len(profile_s2)} px)")
        else:
            ax3.text(0.5, 0.5, "S2 central row: all NaN",
                     ha="center", va="center", transform=ax3.transAxes,
                     fontsize=9, color="#E65100")

        if ls_valid.any():
            ax3.plot(x_ls[ls_valid], profile_ls[ls_valid], color="#1565C0", lw=2.0,
                     alpha=0.9, label=f"Landsat (30 m, {len(profile_ls)} px)")
        else:
            ax3.text(0.5, 0.4, "Landsat central row: all NaN",
                     ha="center", va="center", transform=ax3.transAxes,
                     fontsize=9, color="#1565C0")
        ax3.set_xlabel("Relative position along central row")
        ax3.set_ylabel(index)
        ax3.set_title("Central Row Profile – Spatial Detail Comparison")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        fig.suptitle(f"Resolution Comparison – {index}  (10 m vs 30 m)",
                     fontsize=13, fontweight="bold")
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # S2-10. Multi-scene Sentinel-2 index variability heatmap
    # ------------------------------------------------------------------
    def plot_s2_scene_variability(self,
                                   processed_scenes: list,
                                   filename: str = "s2_10_scene_variability.png"):
        """
        Compute per-pixel coefficient of variation (CV = std/mean) across
        all Sentinel-2 scenes.  High CV areas are most temporally variable
        (e.g. seasonal vegetation, flooding, construction).
        Also shows the pixel-wise mean map.
        """
        indices_show = ["NDVI", "NDBI", "MNDWI"]
        scene_dicts  = [d for _, d in processed_scenes]

        available = [k for k in indices_show
                     if all(k in d for d in scene_dicts)]
        if not available:
            logger.warning("[Diagnostics] plot_s2_scene_variability: no common indices across scenes.")
            return

        ncols = 2
        nrows = len(available)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(10, nrows * 4.5))
        if nrows == 1:
            axes = axes[np.newaxis, :]

        for row, key in enumerate(available):
            # Use Welford's online algorithm to compute mean and variance
            # incrementally — one scene at a time — so we never allocate a
            # stack of all scenes simultaneously (avoids the ~132 MiB spike).
            shapes = [d[key].shape for d in scene_dicts]
            h = min(s[0] for s in shapes)
            w = min(s[1] for s in shapes)

            count    = np.zeros((h, w), dtype=np.float32)
            mean     = np.zeros((h, w), dtype=np.float32)
            M2       = np.zeros((h, w), dtype=np.float32)

            for d in scene_dicts:
                scene = d[key][:h, :w].astype(np.float32)
                valid = np.isfinite(scene)
                count[valid] += 1
                delta         = np.where(valid, scene - mean, 0.0)
                mean         += np.where(valid, delta / np.maximum(count, 1), 0.0)
                delta2        = np.where(valid, scene - mean, 0.0)
                M2           += np.where(valid, delta * delta2, 0.0)
                del scene, valid, delta, delta2

            with np.errstate(invalid="ignore", divide="ignore"):
                pixel_mean = np.where(count > 0, mean, np.nan)
                pixel_std  = np.where(count > 1,
                                      np.sqrt(M2 / np.maximum(count - 1, 1)),
                                      np.nan)
                pixel_cv   = pixel_std / (np.abs(pixel_mean) + 1e-8)

            del count, mean, M2

            # Mean map
            ax0 = axes[row, 0]
            vmin, vmax = np.nanpercentile(pixel_mean[np.isfinite(pixel_mean)], [2, 98])
            im0 = ax0.imshow(pixel_mean, cmap="RdYlGn" if key == "NDVI" else "viridis",
                             vmin=vmin, vmax=vmax)
            ax0.set_title(f"{key} – Pixel-wise Mean\n(across {len(scene_dicts)} scenes)",
                          fontsize=9, fontweight="bold")
            ax0.axis("off")
            plt.colorbar(im0, ax=ax0)

            # CV map
            ax1 = axes[row, 1]
            cv_finite = pixel_cv[np.isfinite(pixel_cv)]
            cv_max = float(np.nanpercentile(cv_finite, 98)) if cv_finite.size > 0 else 1.0
            im1 = ax1.imshow(pixel_cv, cmap="YlOrRd", vmin=0, vmax=max(cv_max, 1e-6))
            ax1.set_title(f"{key} – Coefficient of Variation\n(temporal instability)",
                          fontsize=9, fontweight="bold")
            ax1.axis("off")
            plt.colorbar(im1, ax=ax1, label="CV = σ / |μ|")

        fig.suptitle("Sentinel-2 Multi-Scene Index Variability",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # S2-11. Fusion weight map (how much each sensor contributes)
    # ------------------------------------------------------------------
    def plot_fusion_weight_map(self,
                                landsat_data: dict,
                                sentinel2_data: dict,
                                fused_data: dict,
                                time_diff: int,
                                filename: str = "s2_11_fusion_weights.png"):
        """
        Visualise the per-pixel effective contribution of Landsat vs
        Sentinel-2 after temporal-weighted fusion.  Also validates that
        the fused values fall within the expected range of the two sensors.
        """
        index = next((k for k in ["NDVI","NDBI","MNDWI"]
                      if k in landsat_data and k in sentinel2_data and k in fused_data),
                     None)
        if index is None:
            logger.warning("[Diagnostics] plot_fusion_weight_map: no common index found.")
            return

        # Re-compute the weights exactly as done in MultiSensorFusion.fuse_data
        time_weight_s2 = 1.0 / (1.0 + time_diff / 16.0)
        time_weight_ls = 1.0 - time_weight_s2

        ls_arr = landsat_data[index].astype(np.float32)
        s2_arr = sentinel2_data[index].astype(np.float32)
        fu_arr = fused_data[index].astype(np.float32)

        # Crop to minimum common shape
        h = min(ls_arr.shape[0], s2_arr.shape[0], fu_arr.shape[0])
        w = min(ls_arr.shape[1], s2_arr.shape[1], fu_arr.shape[1])
        ls_arr = ls_arr[:h, :w]
        s2_arr = s2_arr[:h, :w]
        fu_arr = fu_arr[:h, :w]

        # Validate: fused should ≈ w_s2*S2 + w_ls*LS
        expected = time_weight_s2 * s2_arr + time_weight_ls * ls_arr
        residual = fu_arr - expected

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        panels = [
            (ls_arr, f"Landsat {index}\n(weight={time_weight_ls:.2f})", "RdYlGn"),
            (s2_arr, f"Sentinel-2 {index}\n(weight={time_weight_s2:.2f})", "RdYlGn"),
            (fu_arr, f"Fused {index}\n(Δt={time_diff}d)", "RdYlGn"),
        ]
        all_vals = np.concatenate([ls_arr[np.isfinite(ls_arr)].ravel(),
                                    s2_arr[np.isfinite(s2_arr)].ravel()])
        vmin, vmax = np.nanpercentile(all_vals, [2, 98])

        for ax, (arr, title, cmap) in zip(axes[0], panels):
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontweight="bold", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax)

        # Difference maps
        diff_s2_ls = s2_arr - ls_arr
        lim = np.nanpercentile(np.abs(diff_s2_ls[np.isfinite(diff_s2_ls)]), 98)

        for ax, arr, title in zip(
                axes[1],
                [diff_s2_ls, residual,
                 np.abs(s2_arr - ls_arr)],
                ["S2 − Landsat\n(raw sensor discrepancy)",
                 "Fused − Expected\n(fusion residual error)",
                 "|S2 − Landsat|\n(absolute discrepancy)"]):
            lim_use = np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 98) + 1e-8
            im = ax.imshow(arr, cmap="RdBu_r",
                           vmin=-lim_use, vmax=lim_use)
            ax.set_title(title, fontweight="bold", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax)

        fig.suptitle(
            f"Fusion Weight Analysis  —  {index}\n"
            f"Landsat weight={time_weight_ls:.2f}, Sentinel-2 weight={time_weight_s2:.2f}  "
            f"(Δt={time_diff} days)",
            fontsize=12, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # S2-12. Impervious surface fraction analysis
    # ------------------------------------------------------------------
    def plot_impervious_surface_analysis(self,
                                          fused_data: dict,
                                          filename: str = "s2_12_impervious_surface.png"):
        """
        Detailed analysis of the impervious surface fraction (ISF) layer,
        showing its spatial distribution, histogram, and cross-plots
        against NDVI and LST (key UHI relationships).
        """
        if "impervious_surface" not in fused_data:
            logger.warning("[Diagnostics] plot_impervious_surface_analysis: ISF not found.")
            return

        isf = fused_data["impervious_surface"].astype(np.float32)

        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # ISF spatial map
        ax0 = fig.add_subplot(gs[:, 0])
        im0 = ax0.imshow(isf, cmap="hot_r", vmin=0, vmax=1)
        ax0.set_title("Impervious Surface Fraction\n(0=pervious, 1=fully impervious)",
                      fontweight="bold")
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0, label="ISF [0–1]")

        # ISF histogram
        ax1 = fig.add_subplot(gs[0, 1])
        valid_isf = isf[np.isfinite(isf)].ravel()
        ax1.hist(valid_isf, bins=60, color="#E53935", alpha=0.8,
                 edgecolor="white", density=True)
        ax1.axvline(valid_isf.mean(), color="black", lw=1.5, linestyle="--",
                    label=f"μ={valid_isf.mean():.3f}")
        ax1.axvline(np.median(valid_isf), color="navy", lw=1.5, linestyle=":",
                    label=f"median={np.median(valid_isf):.3f}")
        ax1.set_xlabel("ISF")
        ax1.set_ylabel("Density")
        ax1.set_title("ISF Distribution")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Cumulative distribution
        ax2 = fig.add_subplot(gs[1, 1])
        sorted_isf = np.sort(valid_isf)
        ax2.plot(sorted_isf, np.linspace(0, 100, len(sorted_isf)),
                 color="#E53935", lw=2)
        # Mark urban thresholds
        for thresh, label in [(0.3,"Low"), (0.6,"Medium"), (0.85,"High")]:
            pct = np.searchsorted(sorted_isf, thresh) / len(sorted_isf) * 100
            ax2.axvline(thresh, linestyle="--", lw=1, alpha=0.6)
            ax2.text(thresh, pct + 2, f"{label}\n{thresh:.0%}", fontsize=7, ha="center")
        ax2.set_xlabel("ISF threshold")
        ax2.set_ylabel("Pixels below threshold (%)")
        ax2.set_title("Cumulative ISF Distribution")
        ax2.grid(True, alpha=0.3)

        # ISF vs NDVI scatter
        ax3 = fig.add_subplot(gs[0, 2])
        if "NDVI" in fused_data:
            ndvi = fused_data["NDVI"]
            mask = np.isfinite(isf) & np.isfinite(ndvi)
            x, y = isf[mask].ravel(), ndvi[mask].ravel()
            if len(x) > 50_000:
                sel = np.random.default_rng(0).choice(len(x), 50_000, replace=False)
                x, y = x[sel], y[sel]
            ax3.hexbin(x, y, gridsize=50, cmap="YlOrRd", mincnt=1)
            r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else float("nan")
            ax3.set_xlabel("ISF")
            ax3.set_ylabel("NDVI")
            ax3.set_title(f"ISF vs NDVI  (r={r:.3f})")
            ax3.grid(True, alpha=0.3)

        # ISF vs LST scatter
        ax4 = fig.add_subplot(gs[1, 2])
        if "LST" in fused_data:
            lst = fused_data["LST"]
            mask = np.isfinite(isf) & np.isfinite(lst)
            x, y = isf[mask].ravel(), lst[mask].ravel()
            if len(x) > 50_000:
                sel = np.random.default_rng(0).choice(len(x), 50_000, replace=False)
                x, y = x[sel], y[sel]
            ax4.hexbin(x, y, gridsize=50, cmap="inferno", mincnt=1)
            try:
                z = np.polyfit(x, y, 1)
                xr = np.linspace(x.min(), x.max(), 200)
                ax4.plot(xr, np.poly1d(z)(xr), "cyan", lw=1.5,
                         label=f"slope={z[0]:.2f}°C/unit")
                ax4.legend(fontsize=8)
            except Exception:
                pass
            r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else float("nan")
            ax4.set_xlabel("ISF")
            ax4.set_ylabel("LST (°C)")
            ax4.set_title(f"ISF vs LST  (r={r:.3f})\n[Core UHI relationship]")
            ax4.grid(True, alpha=0.3)

        fig.suptitle("Impervious Surface Fraction – Spatial & UHI Analysis",
                     fontsize=13, fontweight="bold")
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # P-MOS. Full LST mosaic reconstruction (preprocessing-time)
    # ------------------------------------------------------------------
    def plot_lst_mosaic_reconstruction(
        self,
        y: np.ndarray,
        positions_all: np.ndarray,
        patch_grid_cols: int,
        all_grid_rows: list,
        stride: int = 24,
        geo_bounds: dict = None,
        filename: str = "P_mosaic_lst_reconstruction.png",
    ):
        """
        Reconstruct the full spatial LST mosaic from the extracted patches
        *before* normalisation, and display it in the same multi-panel style
        used by the inference pipeline (03b).

        This is the preprocessing equivalent of the inference mosaic: it lets
        you immediately see whether the patch extraction, epoch-row offsets,
        and spatial grid layout are correct — without having to run the full
        model training and inference cycle.

        Layout (always produced):
          Panel 0 — Temporal mean LST  (nanmean across all stacked epochs)
          Panel 1 — Inter-epoch spread (nanstd across epochs, same shape)
          Panel 2 — Coverage map       (fraction of epochs that contributed
                                        a valid (non-NaN) patch to each cell)

        A second row shows one representative epoch slice per preview slot so
        you can visually check whether individual epochs look spatially coherent.

        Multi-epoch handling mirrors the inference plotter:
          • epoch aspect ratio > 4 → collapse epochs via nanmean/nanstd
          • aspect ratio ≤ 4       → single mosaic rendered directly

        Args:
            y               : (N, H, W, 1) or (N, H, W) raw LST patches
                              (PRE-normalisation, values in °C).
            positions_all   : (N, 2) int32 array of (grid_row, grid_col),
                              as built by main() from patch["_grid_row/_col"].
            patch_grid_cols : authoritative column count (from _all_grid_cols mode).
            all_grid_rows   : list of per-epoch row counts from _fuse_extract_free
                              (used to derive n_epochs and per_epoch_rows).
            stride          : pixel stride used during patch extraction (default 24).
                              Canvas pixel coords: r0 = grid_row * stride,
                              c0 = grid_col * stride.  Overlapping patches are
                              blended via a Hann window.
            geo_bounds      : optional dict with keys min_lon, max_lon, min_lat,
                              max_lat (WGS-84 degrees).  When supplied the panels
                              are rendered with real geographic axes (lat/lon ticks)
                              instead of raw pixel coordinates, making it easy to
                              verify alignment against the configured study area.
                              Pass STUDY_AREA["bounds"] from config.py.
            filename        : output filename (saved to self.out).
        """
        try:
            # ── Normalise y to (N, H, W) ──────────────────────────────────────
            y_arr = np.asarray(y, dtype=np.float32)
            if y_arr.ndim == 4 and y_arr.shape[-1] == 1:
                y_arr = y_arr[:, :, :, 0]
            elif y_arr.ndim == 4 and y_arr.shape[1] == 1:
                y_arr = y_arr[:, 0]
            if y_arr.ndim != 3:
                logger.warning(
                    f"[Diagnostics] plot_lst_mosaic_reconstruction: "
                    f"unexpected y shape {y_arr.shape}, skipping."
                )
                return

            N, H, W = y_arr.shape
            logger.info(
                f"[Diagnostics] Reconstructing LST mosaic from {N} patches "
                f"({H}×{W} px each), grid_cols={patch_grid_cols} …"
            )

            # ── Build the full tall mosaic using overlap-blended placement ─────
            # Each grid cell maps to a stride-sized step but patches are H×W,
            # so adjacent cells overlap by (H - stride) pixels (e.g. 64-24=40 px).
            # Hard writes (last patch wins) create visible seam lines at every
            # patch boundary.  Instead we accumulate a weighted sum and a weight
            # map so overlapping patches are feathered together:
            #   blended[r,c] = Σ(weight[r,c] * patch[r,c]) / Σ(weight[r,c])
            # The per-patch weight is a 2-D raised-cosine (Hann) window that
            # falls smoothly to 0 at the patch edges and peaks at the centre.
            # This eliminates the grid-line artefact with zero extra RAM beyond
            # two float32 arrays the same size as the canvas.
            pos = np.asarray(positions_all, dtype=np.int32)
            valid_mask = (pos[:, 0] >= 0) & (pos[:, 1] >= 0)

            if not valid_mask.any():
                logger.warning(
                    "[Diagnostics] plot_lst_mosaic_reconstruction: "
                    "no valid patch positions found (all -1). "
                    "Cannot reconstruct spatial mosaic — check that _grid_row/"
                    "_grid_col were set during extract_patches()."
                )
                return

            # ── Canvas sizing uses STRIDE-based pixel coordinates ─────────────
            # Patches were extracted with stride < patch_size (e.g. stride=24,
            # patch_size=64), meaning adjacent patches OVERLAP by (H - stride)
            # pixels.  The correct pixel origin for grid cell (gr, gc) is:
            #   r0 = gr * stride,   c0 = gc * stride
            # NOT gr * H / gc * W — that placed patches patch_size apart, creating
            # a gap of (H - stride) = 40 pixels between origins and making the
            # mosaic look cut-off and misaligned.
            #
            # Canvas must be large enough to contain the last patch fully:
            #   canvas_h = max_grid_row * stride + H
            #   canvas_w = max_grid_col * stride + W
            max_grid_row  = int(pos[valid_mask, 0].max())
            max_grid_col  = int(pos[valid_mask, 1].max())
            n_cols_canvas = int(patch_grid_cols)   # kept for epoch-slice logic

            canvas_h = max_grid_row * stride + H
            canvas_w = max_grid_col * stride + W

            # Weighted accumulator arrays (no NaN — start at 0)
            acc_sum = np.zeros((canvas_h, canvas_w), dtype=np.float64)
            acc_wgt = np.zeros((canvas_h, canvas_w), dtype=np.float64)

            # 2-D Hann (raised cosine) window — smooth taper to 0 at all edges
            _hann_1d_r = np.hanning(H).astype(np.float64)
            _hann_1d_c = np.hanning(W).astype(np.float64)
            _patch_win = np.outer(_hann_1d_r, _hann_1d_c)   # shape (H, W)

            for idx in range(N):
                gr, gc = int(pos[idx, 0]), int(pos[idx, 1])
                if gr < 0 or gc < 0:
                    continue
                # Stride-based pixel origin — KEY FIX: use stride, not H/W
                r0 = gr * stride;  r1 = r0 + H
                c0 = gc * stride;  c1 = c0 + W
                if r1 > canvas_h or c1 > canvas_w:
                    continue   # safety: skip if out of bounds
                patch = y_arr[idx].astype(np.float64)
                # Only blend pixels that have valid data in this patch
                finite = np.isfinite(patch)
                w = _patch_win * finite          # zero weight where NaN
                acc_sum[r0:r1, c0:c1] += np.where(finite, patch * w, 0.0)
                acc_wgt[r0:r1, c0:c1] += w

            # Normalise: pixels with zero total weight remain NaN
            with np.errstate(invalid="ignore", divide="ignore"):
                canvas = np.where(acc_wgt > 0,
                                  (acc_sum / acc_wgt).astype(np.float32),
                                  np.nan).astype(np.float32)
            del acc_sum, acc_wgt

            # ── Detect multi-epoch stacking ────────────────────────────────────
            # The canvas is built by placing each patch at pixel origin:
            #   r0 = grid_row * stride
            # Epoch N's patches have grid_row values in [prior_rows, prior_rows+ep_grid_rows),
            # where prior_rows = sum(all_grid_rows[0..N-1]).
            # Therefore epoch N's first pixel row = prior_rows * stride,
            # and its slice height in the canvas = ep_grid_rows * stride pixels.
            # (The last patch's tail — the final H-stride rows — overlaps into the
            # next epoch's territory in the canvas, but that bleed is feathered to
            # near-zero by the Hann window so it doesn't corrupt the slice.)
            # This gives perfectly aligned epoch boundaries for stacking.
            _ep_px_list = [r * stride for r in all_grid_rows]
            _patch_rows_total = sum(_ep_px_list) if _ep_px_list else canvas.shape[0]
            _patch_cols_total = max_grid_col + 1 if max_grid_col >= 0 else 1
            _aspect = _patch_rows_total / max(_patch_cols_total, 1)

            is_multi_epoch = (_aspect > 4) and (len(all_grid_rows) > 1)

            slices = []
            if is_multi_epoch:
                n_epochs = len(all_grid_rows)

                logger.info(
                    f"[Diagnostics] Multi-epoch mosaic: {n_epochs} epochs, "
                    f"per-epoch patch-rows={all_grid_rows}, "
                    f"per-epoch pixel-rows={_ep_px_list}, "
                    f"cols={n_cols_canvas}. Collapsing via nanmean …"
                )

                # Reference shape = largest epoch so np.stack gets uniform arrays
                max_ep_px = max(_ep_px_list) if _ep_px_list else H
                ref_cols  = canvas.shape[1]

                row_cursor = 0  # running pixel-row offset into the canvas
                for ep, ep_grid_rows in enumerate(all_grid_rows):
                    ep_px = _ep_px_list[ep]  # = ep_grid_rows * stride
                    r0    = row_cursor
                    r1    = r0 + ep_px
                    if r0 >= canvas.shape[0]:
                        logger.warning(
                            f"[Diagnostics] Epoch {ep}: row_cursor={r0} exceeds "
                            f"canvas height {canvas.shape[0]} — skipping."
                        )
                        row_cursor += ep_px
                        continue
                    r1_clamped = min(r1, canvas.shape[0])
                    ep_slice   = canvas[r0:r1_clamped, :]

                    # Pad to reference shape so all slices stack uniformly
                    if ep_slice.shape != (max_ep_px, ref_cols):
                        padded = np.full((max_ep_px, ref_cols), np.nan,
                                         dtype=np.float32)
                        h_copy = min(ep_slice.shape[0], max_ep_px)
                        w_copy = min(ep_slice.shape[1], ref_cols)
                        padded[:h_copy, :w_copy] = ep_slice[:h_copy, :w_copy]
                        slices.append(padded)
                    else:
                        slices.append(ep_slice)

                    row_cursor += ep_px

                if not slices:
                    logger.warning(
                        "[Diagnostics] plot_lst_mosaic_reconstruction: "
                        "epoch slicing produced no slices — rendering raw canvas."
                    )
                    is_multi_epoch = False

            if is_multi_epoch and slices:
                stack    = np.stack(slices, axis=0)   # (n_ep, H_ep, W)
                lst_mean = np.nanmean(stack, axis=0)
                lst_std  = np.nanstd(stack,  axis=0)
                coverage = np.sum(np.isfinite(stack), axis=0) / n_epochs
            else:
                lst_mean = canvas
                lst_std  = np.zeros_like(canvas)
                coverage = np.isfinite(canvas).astype(np.float32)
                n_epochs = 1
                slices   = [canvas]

            # ── Diagnostic gap-fill: propagate valid neighbours into NaN holes ──
            # Pixels that are NaN in lst_mean are positions where no patch passed
            # QC in any epoch (cloud/water/edge gaps).  For a cleaner diagnostic
            # image we interpolate them from surrounding valid pixels.
            # This is ONLY applied to lst_mean for the plot — the training arrays
            # y/X are not touched.  The coverage map still shows true coverage.
            #
            # Implementation: vectorised nan-aware local mean via uniform_filter.
            # Replace NaN with 0, compute sum-of-values and sum-of-weights
            # (valid pixel count) separately, then divide.  This is O(N) in C
            # and runs in milliseconds even on a 2 k×2 k canvas — unlike
            # generic_filter with a Python callback which would take minutes.
            # Two passes with increasing kernel sizes fill progressively wider
            # gaps (e.g. narrow cloud strips then larger coastal voids) without
            # over-smoothing the filled region.
            _nan_holes = ~np.isfinite(lst_mean)
            if _nan_holes.any():
                try:
                    from scipy.ndimage import uniform_filter as _uf
                    _filled = lst_mean.copy()
                    for _ksize in (max(5, H // 8), max(9, H // 4)):
                        _still_nan = ~np.isfinite(_filled)
                        if not _still_nan.any():
                            break
                        _vals  = np.where(np.isfinite(_filled), _filled, 0.0).astype(np.float64)
                        _wgts  = np.isfinite(_filled).astype(np.float64)
                        _vsum  = _uf(_vals,  size=_ksize, mode="reflect")
                        _wsum  = _uf(_wgts,  size=_ksize, mode="reflect")
                        with np.errstate(invalid="ignore", divide="ignore"):
                            _local = np.where(_wsum > 0, _vsum / _wsum, np.nan)
                        # Only write into originally-NaN positions
                        _filled = np.where(_still_nan, _local.astype(np.float32), _filled)
                    lst_mean = _filled
                    del _filled, _vals, _wgts, _vsum, _wsum, _local
                    _filled_n = int(_nan_holes.sum()) - int((~np.isfinite(lst_mean)).sum())
                    logger.info(
                        f"[Diagnostics] Gap-fill: {_filled_n:,} / {int(_nan_holes.sum()):,} "
                        f"NaN pixels interpolated from neighbours. "
                        f"Coverage map unchanged."
                    )
                except Exception as _gf_err:
                    logger.warning(
                        f"[Diagnostics] Gap-fill skipped: {_gf_err}"
                    )
            del _nan_holes

            # ── Shared colour scale (robust to outliers) ───────────────────────
            finite_mean = lst_mean[np.isfinite(lst_mean)]
            if finite_mean.size < 2:
                logger.warning(
                    "[Diagnostics] plot_lst_mosaic_reconstruction: "
                    "LST mean map is almost entirely NaN — "
                    "check that y contains pre-normalisation Celsius values."
                )
                return
            vmin = float(np.nanpercentile(finite_mean, 2))
            vmax = float(np.nanpercentile(finite_mean, 98))

            # ── Figure layout ──────────────────────────────────────────────────
            n_preview_epochs = min(4, n_epochs) if is_multi_epoch else 0
            nrows_fig = 1 + (1 if n_preview_epochs > 0 else 0)
            ncols_fig = 3

            _ep_h = lst_mean.shape[0]
            _ep_w = lst_mean.shape[1]
            _map_aspect = _ep_h / max(_ep_w, 1)
            _panel_w = 5.5
            _panel_h = max(3.5, _panel_w * _map_aspect)

            fig = plt.figure(
                figsize=(ncols_fig * _panel_w, nrows_fig * _panel_h + 1.2),
                layout="constrained",
            )
            gs_outer = gridspec.GridSpec(nrows_fig, 1, figure=fig,
                                         hspace=0.08 if n_preview_epochs else 0)
            gs_top = gridspec.GridSpecFromSubplotSpec(
                1, ncols_fig, subplot_spec=gs_outer[0], wspace=0.08
            )

            # ── Geographic extent for imshow ───────────────────────────────────
            # When geo_bounds is supplied we pass it as the `extent` argument to
            # imshow so axes show real lon/lat coordinates instead of raw pixels.
            # extent = [left, right, bottom, top] = [min_lon, max_lon, min_lat, max_lat]
            # origin="upper" means row 0 = top of image = max_lat (north), correct
            # for north-up rasters after the vflip normalisation in load_tif_as_bands.
            _has_geo = (
                geo_bounds is not None
                and all(k in geo_bounds for k in ("min_lon", "max_lon", "min_lat", "max_lat"))
            )
            _extent = (
                [geo_bounds["min_lon"], geo_bounds["max_lon"],
                 geo_bounds["min_lat"], geo_bounds["max_lat"]]
                if _has_geo else None
            )

            def _setup_geo_ax(ax):
                if _has_geo:
                    ax.set_xlabel("Longitude (°E)", fontsize=7)
                    ax.set_ylabel("Latitude (°N)", fontsize=7)
                    ax.tick_params(axis="both", labelsize=6)
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.6)
                else:
                    ax.axis("off")

            # ── Panel 0: temporal mean LST ─────────────────────────────────────
            ax0 = fig.add_subplot(gs_top[0, 0])
            im0 = ax0.imshow(lst_mean, cmap="RdYlBu_r",
                             vmin=vmin, vmax=vmax,
                             interpolation="nearest", origin="upper",
                             extent=_extent)
            ax0.set_title("Temporal Mean LST", fontsize=10, fontweight="bold")
            _setup_geo_ax(ax0)
            cb0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
            cb0.set_label("LST (°C)", fontsize=8)
            _nan_pct = float(np.isnan(lst_mean).mean()) * 100
            ax0.text(
                0.02, 0.02,
                f"mean={np.nanmean(lst_mean):.1f}°C\n"
                f"std={np.nanstd(lst_mean):.2f}°C\n"
                f"range=[{vmin:.1f}, {vmax:.1f}]\n"
                f"NaN={_nan_pct:.1f}%\n"
                f"n_epochs={n_epochs}",
                transform=ax0.transAxes, fontsize=7, color="white", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
            )

            # ── Panel 1: inter-epoch spread ────────────────────────────────────
            ax1 = fig.add_subplot(gs_top[0, 1])
            std_finite = lst_std[np.isfinite(lst_std)]
            std_vmax = float(np.nanpercentile(std_finite, 98)) if std_finite.size > 0 else 1.0
            im1 = ax1.imshow(lst_std, cmap="YlOrRd",
                             vmin=0, vmax=max(std_vmax, 0.1),
                             interpolation="nearest", origin="upper",
                             extent=_extent)
            ax1.set_title(
                "Inter-epoch Spread (σ)" if n_epochs > 1 else "Spread (single epoch)",
                fontsize=10, fontweight="bold",
            )
            _setup_geo_ax(ax1)
            cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cb1.set_label("σ (°C)", fontsize=8)
            ax1.text(
                0.02, 0.02,
                f"mean σ={np.nanmean(lst_std):.2f}°C\n"
                f"max σ={np.nanmax(lst_std):.2f}°C",
                transform=ax1.transAxes, fontsize=7, color="white", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
            )

            # ── Panel 2: coverage map ──────────────────────────────────────────
            ax2 = fig.add_subplot(gs_top[0, 2])
            im2 = ax2.imshow(coverage, cmap="Blues",
                             vmin=0, vmax=1,
                             interpolation="nearest", origin="upper",
                             extent=_extent)
            ax2.set_title("Patch Coverage\n(fraction of epochs with valid data)",
                          fontsize=10, fontweight="bold")
            _setup_geo_ax(ax2)
            cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cb2.set_label("Fraction", fontsize=8)
            _gap_pct  = float((coverage == 0).mean()) * 100
            _full_pct = float((coverage == 1).mean()) * 100
            ax2.text(
                0.02, 0.02,
                f"zero coverage: {_gap_pct:.1f}%\n"
                f"full coverage: {_full_pct:.1f}%\n"
                f"mean coverage: {float(np.nanmean(coverage)):.2f}",
                transform=ax2.transAxes, fontsize=7, color="white", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
            )

            # ── Row 1: individual epoch previews (multi-epoch only) ────────────
            if n_preview_epochs > 0:
                gs_bot = gridspec.GridSpecFromSubplotSpec(
                    1, n_preview_epochs,
                    subplot_spec=gs_outer[1], wspace=0.06,
                )
                ep_step = max(1, n_epochs // n_preview_epochs)
                ep_indices = [i * ep_step for i in range(n_preview_epochs)]

                for col_i, ep_i in enumerate(ep_indices):
                    ax_ep = fig.add_subplot(gs_bot[0, col_i])
                    ep_slice = slices[ep_i]
                    im_ep = ax_ep.imshow(
                        ep_slice, cmap="RdYlBu_r",
                        vmin=vmin, vmax=vmax,
                        interpolation="nearest", origin="upper",
                        extent=_extent,
                    )
                    ep_nan  = float(np.isnan(ep_slice).mean()) * 100
                    ep_mean = (float(np.nanmean(ep_slice))
                               if np.any(np.isfinite(ep_slice)) else float("nan"))
                    ax_ep.set_title(
                        f"Epoch {ep_i + 1}/{n_epochs}\n"
                        f"μ={ep_mean:.1f}°C  NaN={ep_nan:.0f}%",
                        fontsize=8, fontweight="bold",
                    )
                    _setup_geo_ax(ax_ep)
                    plt.colorbar(im_ep, ax=ax_ep, fraction=0.046, pad=0.04,
                                 label="LST (°C)")

            # ── Diagnostic flags ───────────────────────────────────────────────
            flags = []
            if _nan_pct > 30:
                flags.append(f"⚠ HIGH NaN ({_nan_pct:.0f}%)")
            if _gap_pct > 20:
                flags.append(f"⚠ GAPS in coverage ({_gap_pct:.0f}%)")
            if np.nanstd(lst_mean) < 0.5:
                flags.append("⚠ LOW LST VARIANCE — check scaling")
            flag_str = "   " + "  |  ".join(flags) if flags else ""

            fig.suptitle(
                f"Preprocessing – LST Mosaic Reconstruction  "
                f"({N} patches, {n_epochs} epoch{'s' if n_epochs > 1 else ''}, "
                f"{n_cols_canvas} patch-cols, "
                f"{lst_mean.shape[0]}×{lst_mean.shape[1]} px)"
                f"{flag_str}",
                fontsize=11, fontweight="bold",
            )
            self._save(fig, filename)
            logger.info(
                f"[Diagnostics] ✅ LST mosaic reconstruction saved → "
                f"{self.out / filename}"
            )

        except Exception as exc:
            logger.warning(
                f"[Diagnostics] plot_lst_mosaic_reconstruction failed: {exc}",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    def _save(self, fig: plt.Figure, filename: str):
        path = self.out / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Diagnostics] Saved → {path}")

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
        # mode='reflect' avoids the hard-edge padding artefact that 'nearest'
        # produces at tile boundaries when downsampling Sentinel-2 10m → 30m.
        # 'nearest' clamps boundary pixels to the edge value, which can create
        # a visible stripe of constant value at the right/bottom of any tile
        # that doesn't exactly divide by the zoom factor.  'reflect' mirrors
        # the boundary instead, which blends smoothly and is the standard
        # convention in image processing for convolution-based resamplers.
        resampled = zoom(src_array, zoom_factor, order=order, mode='reflect')
        
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
        
        # Detect whether ST_B10 is already in Kelvin (GeoTIFF, scale applied
        # at export) or raw Collection-2 DN integers.
        # Jakarta surface temps: ~295–330 K. Raw DN values are ~7 000–15 000.
        sample     = thermal_band[np.isfinite(thermal_band)]
        median_val = float(np.nanmedian(sample)) if sample.size > 0 else 0.0

        if 200.0 < median_val < 400.0:
            # Already in Kelvin — GeoTIFF path
            bt_kelvin = thermal_band.astype(np.float32)
        else:
            # Raw DN — apply C2 L2 scale factor (float32)
            bt_kelvin = thermal_band.astype(np.float32) * np.float32(0.00341802) + np.float32(149.0)

        bt_celsius = bt_kelvin - 273.15
        
        # Calculate land surface emissivity from NDVI (stay in float32)
        ndvi_f32 = ndvi.astype(np.float32) if ndvi.dtype != np.float32 else ndvi
        epsilon = np.where(
            ndvi_f32 < np.float32(0.2),
            np.float32(0.973),   # Bare soil
            np.where(
                ndvi_f32 > np.float32(0.5),
                np.float32(0.986),  # Full vegetation
                np.float32(0.973) + np.float32(0.047) * ((ndvi_f32 - np.float32(0.2)) / np.float32(0.3))
            )
        ).astype(np.float32)
        del ndvi_f32

        # Apply emissivity correction using Planck's law (float32 constants)
        wavelength = np.float32(10.9e-6)  # Band 10 wavelength (meters)
        rho = np.float32(1.438e-2)        # h*c/k_B  m·K (precomputed)

        # LST with emissivity correction (all float32)
        lst_celsius = bt_celsius / (np.float32(1.0) + (wavelength * bt_kelvin / rho) * np.log(epsilon).astype(np.float32))
        
        return lst_celsius
    
    def validate_lst(self, lst: np.ndarray,
                    min_temp: float = 10,
                    max_temp: float = 65) -> Tuple[np.ndarray, Dict]:
        """
        Validate and clean LST data for Jakarta climate.

        Bounds rationale (temperature gate tier 1 — pixel level):
          min_temp = 10°C  Sub-10°C is unambiguously cloud shadow / instrument
                           noise at this latitude; Jakarta water bodies stay >20°C.
          max_temp = 65°C  Tropical impervious surfaces (dark asphalt, metal
                           roofing) can reach ~60-63°C at midday dry season.
                           Keeping these pixels lets the model learn the correct
                           hot-end mapping and corrects slope compression
                           (observed slope=0.855 → target 1.0).
                           Values above 65°C are retrieval failures.

        NaN (not clip): clipping would compress real extremes to the boundary
        value, creating artificial pile-up and biasing UHI magnitude estimates.

        Args:
            lst:      Land Surface Temperature array (°C)
            min_temp: Minimum physically plausible pixel temperature (°C)
            max_temp: Maximum physically plausible pixel temperature (°C)

        Returns:
            Tuple of (cleaned LST, validation stats)
        """
        lst_clean = lst.copy()

        # Count original valid pixels
        original_valid = np.isfinite(lst).sum()

        # Null out physically impossible values — NaN, not clip.
        lst_clean[(lst < min_temp) | (lst > max_temp)] = np.nan
        
        # Calculate statistics
        valid_pixels = np.isfinite(lst_clean).sum()
        valid_ratio = valid_pixels / lst_clean.size
        
        stats = {
            "original_valid": int(original_valid),
            "filtered_valid": int(valid_pixels),
            "valid_ratio": float(valid_ratio),
            "mean": float(np.nanmean(lst_clean)) if valid_pixels > 1 else np.nan,
            "std": float(np.nanstd(lst_clean)) if valid_pixels > 1 else np.nan,
            "min": float(np.nanmin(lst_clean)) if valid_pixels > 1 else np.nan,
            "max": float(np.nanmax(lst_clean)) if valid_pixels > 1 else np.nan
        }
        
        logger.info(f"  LST validation: {valid_ratio*100:.1f}% valid pixels")
        logger.info(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]°C")
        logger.info(f"    Mean: {stats['mean']:.2f}°C, Std: {stats['std']:.2f}°C")

        # ── DIAGNOSTIC: extended per-percentile breakdown ──────────────
        if np.any(np.isfinite(lst_clean)):
            pctls = np.nanpercentile(lst_clean[np.isfinite(lst_clean)], [5, 25, 50, 75, 95])
            logger.info(
                f"    Percentiles [5,25,50,75,95]: "
                f"[{pctls[0]:.2f}, {pctls[1]:.2f}, {pctls[2]:.2f}, "
                f"{pctls[3]:.2f}, {pctls[4]:.2f}]°C"
            )
            removed_pixels = int(original_valid) - int(valid_pixels)
            logger.info(
                f"    Removed {removed_pixels:,} out-of-range pixels "
                f"({removed_pixels / max(lst.size, 1) * 100:.2f}% of total)"
            )
        # ──────────────────────────────────────────────────────────────

        return lst_clean, stats
    
    def calculate_spectral_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate spectral indices from raw bands.

        Memory-efficient implementation:
          - All intermediate arrays stay in float32 (half the size of float64).
          - Band copies are freed as soon as they are no longer needed.
          - Indices are computed one at a time to minimise peak live allocations.

        Args:
            bands: Dictionary of band arrays (float32 preferred)

        Returns:
            Dictionary of calculated indices (all float32)
        """
        indices = {}
        eps = np.float32(1e-8)

        # ── Helper: ensure float32 copy without double-allocating ──────────
        def _f32(arr: np.ndarray) -> np.ndarray:
            """Return a float32 array; reuse memory if already float32."""
            if arr.dtype == np.float32:
                return arr.copy()
            return arr.astype(np.float32)

        # ── Band extraction ─────────────────────────────────────────────────
        if self.satellite_type == "landsat":
            blue  = _f32(bands.get("SR_B2", np.zeros_like(bands["SR_B4"])))
            green = _f32(bands.get("SR_B3", np.zeros_like(bands["SR_B4"])))
            red   = _f32(bands["SR_B4"])
            nir   = _f32(bands["SR_B5"])
            swir1 = _f32(bands["SR_B6"])
            swir2 = _f32(bands["SR_B7"])

            # Detect DN (legacy export) vs pre-scaled reflectance (GeoTIFF).
            # Pre-scaled values are in [0, 1]; raw DN medians are ~5 000–30 000.
            #
            # BUG FIX: use the 75th percentile of *positive* pixels, not the
            # overall median.  When the raster has a large water/fill region
            # (zeros from unmask(0) in GEE), the overall median is pulled toward
            # 0 and the DN-detection condition evaluates False — leaving bands in
            # raw DN values and collapsing all spectral indices (the wide blank
            # areas seen in 01_raw_bands_landsat.png / 02_spectral_indices_landsat.png).
            _s = red[np.isfinite(red) & (red > 0)]
            if _s.size > 0 and float(np.percentile(_s, 75)) > 2.0:
                logger.info(
                    "  [LS scale] DN values detected (p75 of positive red pixels = "
                    f"{float(np.percentile(_s, 75)):.1f}) — applying C2 L2 scale factors."
                )
                # Legacy DN — convert to reflectance in-place
                scale = np.float32(2.75e-5)
                offset = np.float32(0.2)
                for arr in (blue, green, red, nir, swir1, swir2):
                    arr *= scale
                    arr -= offset
                np.clip(blue,  -0.5, 1.5, out=blue)
                np.clip(green, -0.5, 1.5, out=green)
                np.clip(red,   -0.5, 1.5, out=red)
                np.clip(nir,   -0.5, 1.5, out=nir)
                np.clip(swir1, -0.5, 1.5, out=swir1)
                np.clip(swir2, -0.5, 1.5, out=swir2)
            else:
                logger.info(
                    "  [LS scale] Reflectance values already in [0, 1] — no scaling needed."
                )
            del _s
            # GeoTIFF path: already in [0, 1] — no conversion needed

            # Replace all-zero fill pixels (ocean/tile-edge from GEE unmask(0))
            # with NaN so they don't bias index statistics or the colour stretch.
            _fill_mask = (
                (blue == 0) & (green == 0) & (red == 0) &
                (nir  == 0) & (swir1 == 0) & (swir2 == 0)
            )
            if _fill_mask.any():
                logger.info(
                    f"  [LS fill] Replacing {_fill_mask.sum():,} all-zero fill pixels "
                    f"({_fill_mask.mean()*100:.1f}% of image) with NaN."
                )
                for arr in (blue, green, red, nir, swir1, swir2):
                    arr[_fill_mask] = np.nan
            del _fill_mask

        else:  # Sentinel-2
            blue  = _f32(bands["B2"])
            green = _f32(bands["B3"])
            red   = _f32(bands["B4"])
            nir   = _f32(bands["B8"])
            swir1 = _f32(bands["B11"])
            swir2 = _f32(bands["B12"])

            # S2_SR_HARMONIZED: reflectance × 10 000 — divide in-place if needed.
            #
            # BUG FIX: the original code used np.nanmedian() on ALL finite pixels.
            # When the AOI contains a large ocean/fill region (zeros from
            # earth_engine_loader's unmask(0)), the median is dragged to ~0,
            # so the condition `median > 2.0` evaluates False and the DN-to-
            # reflectance scaling is SKIPPED.  This leaves bands at raw DN values
            # (~0–10 000), making every spectral index collapse to a near-constant
            # value — the wide blank/flat areas visible in s2_01_raw_bands.png and
            # s2_02_spectral_indices.png.
            #
            # FIX: use the 75th percentile of *positive* (land/vegetated) pixels
            # as the scale-detection probe.  Ocean fill values are exactly 0 and
            # are excluded; real DN land reflectances are always >> 2.0.
            _s = red[np.isfinite(red) & (red > 0)]
            if _s.size > 0 and float(np.percentile(_s, 75)) > 2.0:
                logger.info(
                    "  [S2 scale] DN values detected (p75 of positive red pixels = "
                    f"{float(np.percentile(_s, 75)):.1f}) — dividing by 10 000."
                )
                scale = np.float32(1.0 / 10000.0)
                for arr in (blue, green, red, nir, swir1, swir2):
                    arr *= scale
            else:
                logger.info(
                    "  [S2 scale] Reflectance values already in [0, 1] — no scaling needed."
                )
            del _s

            # Replace fill zeros (ocean/tile-edge, set by GEE unmask(0)) with NaN
            # so they do not corrupt index statistics or the colour stretch.
            # We only null pixels where ALL optical bands are exactly 0 to avoid
            # masking genuine dark-water pixels that may be near-zero legitimately.
            _fill_mask = (
                (blue  == 0) & (green == 0) & (red  == 0) &
                (nir   == 0) & (swir1 == 0) & (swir2 == 0)
            )
            if _fill_mask.any():
                logger.info(
                    f"  [S2 fill] Replacing {_fill_mask.sum():,} all-zero fill pixels "
                    f"({_fill_mask.mean()*100:.1f}% of image) with NaN."
                )
                for arr in (blue, green, red, nir, swir1, swir2):
                    arr[_fill_mask] = np.nan
            del _fill_mask

        # ── Index computation — one at a time, in-place where possible ─────
        # NDVI = (nir - red) / (nir + red + eps)
        _num = nir - red                          # float32 temp
        _den = nir + red; _den += eps
        np.divide(_num, _den, out=_num)
        indices["NDVI"] = _num
        del _den

        # NDBI = (swir1 - nir) / (swir1 + nir + eps)
        _num = swir1 - nir
        _den = swir1 + nir; _den += eps
        np.divide(_num, _den, out=_num)
        indices["NDBI"] = _num
        del _den

        # MNDWI = (green - swir1) / (green + swir1 + eps)
        _num = green - swir1
        _den = green + swir1; _den += eps
        np.divide(_num, _den, out=_num)
        indices["MNDWI"] = _num
        del _den

        # BSI = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)
        _a = swir1 + red          # swir1 + red
        _b = nir   + blue         # nir   + blue
        _num = _a - _b
        _den = _a + _b; _den += eps
        del _a, _b
        np.divide(_num, _den, out=_num)
        indices["BSI"] = _num
        del _den

        # UI = (swir2 - nir) / (swir2 + nir + eps)
        _num = swir2 - nir
        _den = swir2 + nir; _den += eps
        np.divide(_num, _den, out=_num)
        indices["UI"] = _num
        del _den

        # Albedo (Liang 2001) — computed in-place to avoid a large temp array
        albedo = np.float32(0.356) * blue
        albedo += np.float32(0.130) * red
        albedo += np.float32(0.373) * nir
        albedo += np.float32(0.085) * swir1
        albedo += np.float32(0.072) * swir2
        albedo -= np.float32(0.0018)
        np.clip(albedo, 0, 1, out=albedo)
        indices["albedo"] = albedo

        # Free working band copies — the originals in `bands` are unaffected
        del blue, green, red, nir, swir1, swir2, albedo
        gc.collect()
        
        logger.info(f"  Calculated {len(indices)} spectral indices")

        # ── DIAGNOSTIC: per-index statistics ──────────────────────────
        for name, arr in indices.items():
            valid = arr[np.isfinite(arr)]
            if valid.size > 0:
                logger.info(
                    f"    {name:8s}: mean={valid.mean():.4f}  std={valid.std():.4f}"
                    f"  range=[{valid.min():.4f}, {valid.max():.4f}]"
                )
        # ──────────────────────────────────────────────────────────────

        return indices
    
    def process_raw_file(self, raw_file: Path,
                        calculate_lst: bool = True) -> Optional[Dict[str, np.ndarray]]:
        """
        Process a single raw GeoTIFF file.
        
        Args:
            raw_file: Path to raw .tif file
            calculate_lst: Whether to calculate LST from thermal band
            
        Returns:
            Dictionary of processed features or None if failed
        """
        logger.info(f"Processing: {raw_file.name}")
        
        try:
            # Load raw data from GeoTIFF
            raw_data = load_tif_as_bands(raw_file)
            if raw_data is None:
                return None
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
                        
                        # ── Soft spatial smoothing of LST ────────────────────────
                        # Landsat thermal band is at 30 m native resolution.  At
                        # patch-preview scale the pixel grid is plainly visible as
                        # hard square blocks.  A light Gaussian blur (σ ≈ 0.8 px)
                        # suppresses the inter-pixel step without smearing genuine
                        # thermal gradients (urban heat islands are tens–hundreds of
                        # meters wide, i.e. >> 1 pixel). We preserve NaN positions
                        # by smoothing only finite pixels, then writing results back.
                        try:
                            from scipy.ndimage import gaussian_filter as _gf
                            _nan_mask = ~np.isfinite(lst_clean)
                            if np.isfinite(lst_clean).any():
                                fill_value = float(np.nanmedian(lst_clean))
                            else:
                                # fallback if everything is NaN
                                fill_value = 35.0   # reasonable tropical surface temperature

                            _tmp = lst_clean.copy()
                            _tmp[_nan_mask] = fill_value

                            _smoothed = _gf(_tmp.astype(np.float64), sigma=0.8).astype(np.float32)
                            _smoothed[_nan_mask] = np.nan
                            lst_clean = _smoothed
                            del _tmp, _smoothed, _nan_mask
                            logger.info("  Applied Gaussian LST smoothing (σ=0.8 px) to reduce 30m pixel-grid artefacts")
                        except Exception as _se:
                            logger.warning(f"  LST Gaussian smoothing skipped: {_se}")
                        # ─────────────────────────────────────────────────────────
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
                    stride: int = 32,
                    min_valid_ratio: float = 0.95,  # RAISED from 0.8: eliminates NaN-heavy patches
                                                    # that caused the central spike at normalised LST ≈ 0
                    min_variance: float = 0.5,
                    min_temp: float = 15.0,         # Patch-mean lower bound (°C) — LOWERED from 20°C
                                                    # Jakarta coastal/vegetated patches can reach
                                                    # 15–20°C in wet-season mornings; the old 20°C
                                                    # floor was truncating the cool tail and hurting
                                                    # cool-end calibration. Pixel-level Tier-1
                                                    # guard (10°C) still catches instrument artifacts.
                    max_temp: float = 58.0) -> List[Dict]:  # Patch-mean upper bound (°C)
        """
        Extract patches from raster data with quality control for Jakarta climate.

        Three-tier temperature gate:
          Tier 1 — pixel level  (validate_lst):    10–65°C  → NaN impossible pixels
          Tier 2 — patch mean   (here):            20–58°C  → reject artifact-dominated patches
          Tier 3 — safety clip  (create_training): 10–65°C  → final safety net

        Ceiling rationale (58°C patch mean):
          The patch mean ceiling was previously 48°C, which excluded genuinely
          hot dense-urban patches in Jakarta's industrial zones.  Those patches
          are exactly where UHI signal is strongest; losing them contributes to
          the observed slope compression (slope=0.855 instead of 1.0).
          No legitimate 64×64 patch averages above 58°C over Jakarta.

        Args:
            raster_data:     Dictionary of raster arrays
            patch_size:      Size of patches in pixels
            stride:          Stride between patches
            min_valid_ratio: Minimum fraction of finite LST pixels (0.95 cuts
                             edge/cloud patches that caused NaN spike at 0)
            min_variance:    Minimum LST std (°C); rejects uniform patches
            min_temp:        Minimum acceptable patch-mean LST (°C)
            max_temp:        Maximum acceptable patch-mean LST (°C)

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
        filtered_by_temp = 0
        filtered_by_variance = 0

        lst = raster_data["LST"]   # reference view — no copy

        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                patch_lst = lst[i:i+patch_size, j:j+patch_size]

                if patch_lst.shape != (patch_size, patch_size):
                    continue

                valid_pixels = np.isfinite(patch_lst).sum()
                valid_ratio  = valid_pixels / (patch_size * patch_size)

                if valid_ratio < min_valid_ratio:
                    continue

                valid_temps = patch_lst[np.isfinite(patch_lst)]
                if len(valid_temps) == 0:
                    continue

                lst_mean = float(valid_temps.mean())
                lst_std  = float(valid_temps.std())

                if lst_std < min_variance or lst_std < 0.3:
                    filtered_by_variance += 1
                    continue

                if lst_mean < min_temp or lst_mean > max_temp:
                    filtered_by_temp += 1
                    continue

                extreme_ratio = np.sum((valid_temps < 10.0) | (valid_temps > 65.0)) / len(valid_temps)
                if extreme_ratio > 0.20:
                    filtered_by_temp += 1
                    continue

                # Store position + lightweight stats only — NOT data copies.
                # create_training_samples re-slices from the raster on demand.
                patches.append({
                    "position":  (i, j),
                    "_lst_mean": lst_mean,
                    "_lst_std":  lst_std,
                })
        
        # ── Compute the full spatial grid dimensions ──────────────────
        # These are the number of candidate patch positions in each axis
        # BEFORE quality filtering.  The key insight: patches are generated
        # in row-major order (i outer loop, j inner loop), and the mosaic
        # assembler must know how many columns were in the original scan grid
        # so it can reconstruct the 2-D layout.
        #
        # grid_cols = number of j steps = len(range(0, width-patch_size+1, stride))
        # grid_rows = number of i steps = len(range(0, height-patch_size+1, stride))
        #
        # We attach both to each patch dict as "_grid_row" / "_grid_col" so
        # the downstream assembler can build the grid without needing the
        # raster dimensions.  We also attach them as top-level scalars for
        # _fuse_extract_free to pick up easily.
        grid_cols = len(range(0, width  - patch_size + 1, stride))
        grid_rows = len(range(0, height - patch_size + 1, stride))

        # Tag each patch with its grid position (row-index, col-index in the
        # full scan grid), preserving spatial order even after QC filtering.
        # The grid_row/col pair uniquely identifies the patch's position in the
        # mosaic regardless of how many patches are later kept or discarded.
        _row_counter: Dict[int, int] = {}   # i → count of j positions
        _patch_grid_pos = {}   # (i, j) → (grid_row, grid_col)
        for _gr, _i in enumerate(range(0, height - patch_size + 1, stride)):
            for _gc, _j in enumerate(range(0, width  - patch_size + 1, stride)):
                _patch_grid_pos[(_i, _j)] = (_gr, _gc)

        for p in patches:
            _pos = p["position"]
            p["_grid_row"], p["_grid_col"] = _patch_grid_pos.get(_pos, (-1, -1))

        logger.info(f"  Extracted {len(patches)} valid patches")
        logger.info(f"  Filtered by temperature: {filtered_by_temp} patches")
        logger.info(f"  Filtered by variance: {filtered_by_variance} patches")
        logger.info(f"  Patch scan grid : {grid_rows} rows × {grid_cols} cols "                    f"= {grid_rows * grid_cols} candidate positions")

        # ── DIAGNOSTIC: patch LST statistics ──────────────────────────
        if patches:
            patch_means = np.array([p["_lst_mean"] for p in patches])
            patch_stds  = np.array([p["_lst_std"]  for p in patches])
            logger.info(
                f"  Patch LST mean – μ={patch_means.mean():.2f}°C  "
                f"σ={patch_means.std():.2f}°C  "
                f"range=[{patch_means.min():.2f}, {patch_means.max():.2f}]°C"
            )
            logger.info(
                f"  Patch LST std  – μ={patch_stds.mean():.2f}°C  "
                f"range=[{patch_stds.min():.2f}, {patch_stds.max():.2f}]°C"
            )
            total_candidates = grid_rows * grid_cols
            accept_rate = len(patches) / max(total_candidates, 1) * 100
            logger.info(
                f"  Patch acceptance rate: {accept_rate:.1f}%  "
                f"(rejected: temp={filtered_by_temp}, var={filtered_by_variance})"
            )
        # ──────────────────────────────────────────────────────────────

        # Return the grid shape alongside the patch list so callers can
        # record it in dataset metadata without needing raster dimensions.
        # Backward-compatible: callers that only unpack the list still work
        # because we return a named-tuple-style pair.
        from collections import namedtuple
        _PatchResult = namedtuple("PatchResult", ["patches", "grid_rows", "grid_cols"])
        return _PatchResult(patches, grid_rows, grid_cols)
    
    def create_training_samples(self, patches: List[Dict],
                                temporal_features: Dict,
                                channel_order: List[str] = None,
                                raster_data: Dict[str, np.ndarray] = None
                                ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create training samples from patches.

        Memory-efficient two-pass strategy:
          Pass 1 — lightweight QC scan on _lst_mean/_lst_std (no array alloc).
          Pass 2 — allocate exactly-sized X / y and fill by slicing raster_data.

        Accepts either:
          - Position-only patches ({"position": (r,c), "_lst_mean": …}) with
            raster_data provided for slicing.
          - Legacy data-carrying patches ({"position": …, "data": {…}}) when
            raster_data is None.

        Args:
            patches:          List of patch dicts.
            temporal_features: Temporal feature dict for metadata.
            channel_order:    Ordered list of feature names to stack.
            raster_data:      Raster dict to slice from (position-only mode).

        Returns:
            Tuple of (X, y, metadata).
        """
        if not patches:
            raise ValueError("No patches provided")

        if channel_order is None:
            channel_order = [
                "SR_B4", "SR_B5", "SR_B6", "SR_B7",
                "NDVI", "NDBI", "MNDWI", "BSI", "UI", "albedo",
            ]

        position_only = "data" not in patches[0]
        # Derive patch_size robustly
        if position_only and raster_data is not None:
            r0, c0   = patches[0]["position"]
            patch_size = min(64, raster_data["LST"].shape[0] - r0,
                             raster_data["LST"].shape[1] - c0)
        elif not position_only:
            patch_size = patches[0]["data"]["LST"].shape[0]
        else:
            patch_size = 64  # fallback

        n_channels = len(channel_order)

        # ── QC scan for legacy data-carrying patches only ─────────────────────
        # Position-only patches are already fully QC'd by extract_patches(),
        # so we skip the redundant scan and use all indices directly.
        if position_only:
            valid_indices = list(range(len(patches)))
        else:
            valid_indices = []
            for idx, patch in enumerate(patches):
                lst_patch = patch["data"]["LST"].astype(np.float32)
                fin       = np.isfinite(lst_patch)
                if fin.mean() < 0.95:
                    continue
                vt = lst_patch[fin]
                if vt.size == 0 or not (15.0 <= float(vt.mean()) <= 58.0):
                    continue
                valid_indices.append(idx)

        n_samples = len(valid_indices)
        if n_samples == 0:
            raise ValueError("No valid patches after QC filtering")

        skipped = len(patches) - n_samples
        if skipped:
            logger.info(f"  QC removed {skipped}/{len(patches)} patches")

        # ── Pass 2: allocate exact-sized arrays and fill directly ─────────────
        X = np.zeros((n_samples, patch_size, patch_size, n_channels), dtype=np.float32)
        y = np.zeros((n_samples, patch_size, patch_size, 1),           dtype=np.float32)
        # Collect per-patch spatial grid positions so the inference mosaic
        # assembler can place each patch at its original (grid_row, grid_col)
        # even after QC filtering removed some positions.  Shape: (N, 2).
        patch_positions = np.full((n_samples, 2), -1, dtype=np.int32)

        for out_idx, src_idx in enumerate(valid_indices):
            patch = patches[src_idx]
            r, c  = patch["position"]
            patch_positions[out_idx, 0] = patch.get("_grid_row", -1)
            patch_positions[out_idx, 1] = patch.get("_grid_col", -1)

            if position_only and raster_data is not None:
                for ch_idx, feat in enumerate(channel_order):
                    arr = raster_data.get(feat)
                    if arr is not None:
                        X[out_idx, :, :, ch_idx] = arr[r:r+patch_size, c:c+patch_size]
                lst_arr = raster_data.get("LST")
                if lst_arr is not None:
                    y[out_idx, :, :, 0] = lst_arr[r:r+patch_size, c:c+patch_size]
            else:
                for ch_idx, feat in enumerate(channel_order):
                    if feat in patch["data"]:
                        X[out_idx, :, :, ch_idx] = patch["data"][feat]
                y[out_idx, :, :, 0] = patch["data"]["LST"].astype(np.float32)

        # ── In-place NaN fill (distance-weighted Gaussian inpainting) ────────
        # WHY NOT MEAN FILL:
        #   Filling NaN pixels with the patch-global mean collapses local spatial
        #   structure — it creates flat "blobs" at the mean value wherever data is
        #   missing (typically at scan-line edges and cloud shadows).
        #
        # WHY NOT SQUARE MEDIAN FILTER (old approach):
        #   scipy.ndimage.median_filter uses a square kernel, which tends to leave
        #   blocky square artefacts at the border between filled and valid pixels,
        #   particularly for large contiguous NaN holes (cloud shadows, scan-line
        #   gaps).  The square kernel boundary is visible in the patch previews as
        #   step-edges aligned to pixel rows/columns.
        #
        # IMPROVED STRATEGY — iterative Gaussian-weighted inpainting:
        #   1. Each pass: convolve with a Gaussian kernel (isotropic → no square
        #      edge artefacts) on a NaN-replaced copy, then write results back ONLY
        #      for pixels that were originally NaN.
        #   2. Kernel σ grows each pass (1→2→3 px) so small holes fill first
        #      using tight local context, and large holes fill later using wider
        #      neighbourhood — gradients stay smooth across hole boundaries.
        #   3. Up to MAX_PASSES=8 to handle moderately-sized holes.
        #   4. Any pixels still NaN fall back to the patch median as a last resort.
        #
        # Performance: gaussian_filter is a separable O(N·σ) operation and runs
        # entirely in C — comparable speed to median_filter for these kernel sizes.
        from scipy.ndimage import gaussian_filter as _gf

        _INPAINT_SIGMAS   = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]   # px
        _MAX_INPAINT_PASS = len(_INPAINT_SIGMAS)

        def _gaussian_inpaint_pass(arr2d: np.ndarray, sigma: float) -> int:
            """One isotropic Gaussian inpainting pass.

            NaN pixels are temporarily replaced with the patch median (a
            neutral value that doesn't bias the convolution for *adjacent*
            valid pixels), then the Gaussian-blurred result is written back
            only where the original was NaN.

            Args:
                arr2d : 2-D float32 array modified in-place.
                sigma : Gaussian σ in pixels.

            Returns:
                Number of NaN pixels that received a fill value this pass
                (0 means the array is fully finite — stop iterating).
            """
            nan_mask = ~np.isfinite(arr2d)
            n_nan = int(nan_mask.sum())
            if n_nan == 0:
                return 0
            finite_vals = arr2d[~nan_mask]
            placeholder = float(np.median(finite_vals)) if finite_vals.size > 0 else 0.0
            tmp = arr2d.copy()
            tmp[nan_mask] = placeholder
            blurred = _gf(tmp.astype(np.float64), sigma=sigma, mode='reflect').astype(np.float32)
            # Accept blurred values only for originally-NaN positions.
            # This ensures valid pixels are never altered by the inpainting.
            arr2d[nan_mask] = blurred[nan_mask]
            return n_nan

        for s in range(n_samples):
            for ch in range(n_channels):
                sl = X[s, :, :, ch]
                if not np.isfinite(sl).all():
                    for _sigma in _INPAINT_SIGMAS:
                        if _gaussian_inpaint_pass(sl, _sigma) == 0:
                            break
                    # Last-resort fallback: patch median (not mean)
                    still_nan = ~np.isfinite(sl)
                    if still_nan.any():
                        finite_vals = sl[~still_nan]
                        sl[still_nan] = float(np.median(finite_vals)) if finite_vals.size > 0 else 0.0

            sl = y[s, :, :, 0]
            if not np.isfinite(sl).all():
                for _sigma in _INPAINT_SIGMAS:
                    if _gaussian_inpaint_pass(sl, _sigma) == 0:
                        break
                still_nan = ~np.isfinite(sl)
                if still_nan.any():
                    finite_vals = sl[~still_nan]
                    sl[still_nan] = float(np.median(finite_vals)) if finite_vals.size > 0 else 35.0

        # Tier 3 safety clip — in-place
        np.clip(y, 10.0, 65.0, out=y)

        # ── Sample weights ────────────────────────────────────────────────────
        patch_means   = y[:, :, :, 0].reshape(n_samples, -1).mean(axis=1)
        n_wb          = 10
        bin_edges     = np.linspace(patch_means.min(), patch_means.max() + 1e-6, n_wb + 1)
        bin_ids       = np.clip(np.digitize(patch_means, bin_edges) - 1, 0, n_wb - 1)
        bin_counts    = np.maximum(np.bincount(bin_ids, minlength=n_wb).astype(np.float32), 1)
        raw_weights   = 1.0 / bin_counts[bin_ids]
        sample_weights = raw_weights / raw_weights.mean()

        logger.info(f"  Sample weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

        metadata = {
            "n_samples":       n_samples,
            "patch_size":      patch_size,
            "n_channels":      n_channels,
            "channel_order":   channel_order,
            "temporal_features": temporal_features,
            "temperature_range": {
                "min":  float(np.min(y)),
                "max":  float(np.max(y)),
                "mean": float(np.mean(y)),
                "std":  float(np.std(y)),
            },
            "sample_weights":   sample_weights,
            # (N, 2) int32 — (grid_row, grid_col) for every kept patch.
            # -1 entries mean the patch dict had no _grid_row/_grid_col tag
            # (e.g. legacy data-carrying patches without spatial metadata).
            "patch_positions":  patch_positions,
        }

        logger.info(f"Created training samples: X={X.shape}, y={y.shape}")
        logger.info(f"Temperature range: [{metadata['temperature_range']['min']:.2f}, "
                    f"{metadata['temperature_range']['max']:.2f}]°C")

        logger.info("  Per-channel feature statistics (X):")
        for ch_idx, feat in enumerate(channel_order):
            ch_data = X[:, :, :, ch_idx]
            logger.info(f"    [{ch_idx:2d}] {feat:12s}: mean={ch_data.mean():.4f} "
                        f"std={ch_data.std():.4f} "
                        f"range=[{ch_data.min():.4f}, {ch_data.max():.4f}]")

        return X, y, metadata
    
    def compute_and_save_normalization_stats(self, X_train: np.ndarray, 
                                            y_train: np.ndarray,
                                            output_dir: Path) -> Dict:
        """
        Compute normalization statistics from training data and save them
        
        Args:
            X_train: Training features (N, H, W, C)
            y_train: Training targets (N, H, W, 1)
            output_dir: Directory to save statistics
            
        Returns:
            Normalization statistics dictionary
        """
        logger.info("\n" + "="*70)
        logger.info("COMPUTING NORMALIZATION STATISTICS")
        logger.info("="*70)
        
        n_channels = X_train.shape[-1]
        
        # Compute per-channel statistics for features
        feature_stats = {}
        for ch in range(n_channels):
            channel_data = X_train[:, :, :, ch]
            
            feature_stats[f'channel_{ch}'] = {
                'mean': float(channel_data.mean()),
                'std': float(channel_data.std()),
                'min': float(channel_data.min()),
                'max': float(channel_data.max())
            }
            
            logger.info(f"  Channel {ch}: mean={feature_stats[f'channel_{ch}']['mean']:.2f}, "
                    f"std={feature_stats[f'channel_{ch}']['std']:.2f}")
        
        # Compute target statistics
        target_stats = {
            'mean': float(y_train.mean()),
            'std': float(y_train.std()),
            'min': float(y_train.min()),
            'max': float(y_train.max())
        }
        
        logger.info(f"\n  Target LST: mean={target_stats['mean']:.2f}°C, "
                f"std={target_stats['std']:.2f}°C")
        
        # Compile statistics
        normalization_stats = {
            'features': feature_stats,
            'target': target_stats,
            'n_channels': n_channels,
            'description': 'Normalization statistics computed from training data'
        }
        
        # Save to file
        output_dir.mkdir(parents=True, exist_ok=True)
        stats_path = output_dir / "normalization_stats.json"
        
        import json
        with open(stats_path, 'w') as f:
            json.dump(normalization_stats, f, indent=2)
        
        logger.info(f"\n✅ Saved normalization statistics to: {stats_path}")
        logger.info("="*70)
        
        return normalization_stats
    
    def normalize_data(self, X: np.ndarray, y: np.ndarray, 
                    norm_stats: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features and targets in-place — no second array allocated.

        Args:
            X: Features (N, H, W, C) — modified in place
            y: Targets (N, H, W, 1) — modified in place
            norm_stats: Normalization statistics dictionary

        Returns:
            Tuple of (X, y) — same objects, now normalized
        """
        logger.info("Normalizing data (in-place)...")

        n_channels = X.shape[-1]
        for ch in range(n_channels):
            ch_key = f'channel_{ch}'
            if ch_key in norm_stats['features']:
                mean = norm_stats['features'][ch_key]['mean']
                std  = norm_stats['features'][ch_key]['std']
                if std > 1e-8:
                    X[:, :, :, ch] -= mean
                    X[:, :, :, ch] /= std
                else:
                    logger.warning(f"  Channel {ch} has zero std, skipping normalization")

        target_mean = norm_stats['target']['mean']
        target_std  = norm_stats['target']['std']
        y -= target_mean
        y /= target_std

        logger.info(f"  X normalized: mean={X.mean():.4f}, std={X.std():.4f}")
        logger.info(f"  y normalized: mean={y.mean():.4f}, std={y.std():.4f}")

        return X, y
    
    def verify_no_data_leakage(self, X_train, y_train, X_val, y_val, X_test, y_test,
                            norm_stats: Dict):
        """
        Verify normalization stats only come from training data
        """
        logger.info("\n" + "="*70)
        logger.info("DATA LEAKAGE CHECK")
        logger.info("="*70)
        
        # Check 1: Verify normalization stats match training data
        actual_train_mean = float(np.mean(y_train))
        actual_train_std = float(np.std(y_train, ddof=0))

        if actual_train_std < 1e-6:
            logger.warning("Training target std is near zero — skipping leakage check")
            return
        
        stats_mean = norm_stats['target']['mean']
        stats_std = norm_stats['target']['std']
        
        mean_diff = abs(actual_train_mean - stats_mean)
        std_diff = abs(actual_train_std - stats_std)
        
        logger.info("Normalization stats vs actual training data:")
        logger.info(f"  Stats mean: {stats_mean:.4f}, Actual: {actual_train_mean:.4f}, "
                f"Diff: {mean_diff:.6f}")
        logger.info(f"  Stats std: {stats_std:.4f}, Actual: {actual_train_std:.4f}, "
                f"Diff: {std_diff:.6f}")
        
        if mean_diff > 0.01 or std_diff > 0.01:
            logger.error("❌ POTENTIAL DATA LEAKAGE: Stats don't match training data!")
            raise ValueError("Normalization stats mismatch")
        else:
            logger.info("✅ Normalization stats match training data")
        
        # Check 2: Verify val/test distributions are different
        val_mean = y_val.mean()
        test_mean = y_test.mean()
        
        logger.info(f"\nSplit distributions (normalized):")
        logger.info(f"  Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        logger.info(f"  Val: mean={val_mean:.4f}, std={y_val.std():.4f}")
        logger.info(f"  Test: mean={test_mean:.4f}, std={y_test.std():.4f}")
        
        # Check 3: Verify no sample overlap
        logger.info(f"\nSample overlap check:")
        logger.info(f"  Train samples: {len(X_train)}")
        logger.info(f"  Val samples: {len(X_val)}")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Total: {len(X_train) + len(X_val) + len(X_test)}")
        
        logger.info("="*70)
        
    def save_dataset(self, splits: Dict, output_dir: Path, metadata: Dict, 
                    norm_stats: Optional[Dict] = None):
        """
        Save dataset to disk.

        Memory-efficient: saves each split then immediately removes it from
        the splits dict so Python can reclaim the memory before the next
        split is processed.

        Args:
            splits: Dictionary with train/val/test splits (mutated in place)
            output_dir: Output directory
            metadata: Metadata dictionary
            norm_stats: Normalization statistics (optional)
        """
        output_dir = Path(output_dir)

        for split_name in ['train', 'val', 'test']:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            X_key = f"X_{split_name}"
            y_key = f"y_{split_name}"
            d_key = f"dates_{split_name}"
            p_key = f"positions_{split_name}"

            if X_key in splits:
                np.save(split_dir / "X.npy", splits[X_key])
                del splits[X_key]
            if y_key in splits:
                np.save(split_dir / "y.npy", splits[y_key])
                del splits[y_key]
            if d_key in splits:
                np.save(split_dir / "dates.npy", splits[d_key])
                del splits[d_key]
            if p_key in splits and splits[p_key] is not None:
                np.save(split_dir / "patch_positions.npy", splits[p_key])
                del splits[p_key]
                logger.info(f"  ✓ Saved patch_positions.npy → {split_dir}")

            gc.collect()
            logger.info(f"  ✓ Saved and freed {split_name} split → {split_dir}")

        import json
        metadata_clean = {}
        for k, v in metadata.items():
            if isinstance(v, (np.integer, np.floating)):
                metadata_clean[k] = float(v)
            elif isinstance(v, np.ndarray):
                metadata_clean[k] = v.tolist()
            else:
                metadata_clean[k] = v
        # Save as both metadata.json (legacy) AND dataset_metadata.json
        # (_load_patch_grid_shape in pipeline_manager looks for dataset_metadata.json)
        for _mname in ["metadata.json", "dataset_metadata.json"]:
            with open(output_dir / _mname, "w") as f:
                json.dump(metadata_clean, f, indent=2)
        metadata_file = output_dir / "metadata.json"

        if norm_stats is not None:
            stats_file = output_dir / "normalization_stats.json"
            with open(stats_file, "w") as f:
                json.dump(norm_stats, f, indent=2)
            logger.info(f"✅ Saved normalization stats to {stats_file}")

        # ── Save patch grid shape so the inference mosaic assembler can
        # reconstruct the spatially-correct full-area map without needing
        # the original raster dimensions or a CLI argument.
        #
        # Written to BOTH the dataset root AND each split sub-directory so
        # _load_patch_grid_shape() finds it regardless of which path the
        # user passes to --test-data.
        grid_rows = metadata.get("patch_grid_rows")
        grid_cols = metadata.get("patch_grid_cols")
        rows_per_epoch = metadata.get("patch_grid_rows_per_epoch")
        n_epochs_meta  = metadata.get("n_epochs", 1)
        if grid_rows is not None and grid_cols is not None:
            # Store [total_rows, cols, rows_per_epoch, n_epochs] so the
            # inference mosaic can collapse epochs correctly.
            _rpe = int(rows_per_epoch) if rows_per_epoch is not None else int(grid_rows)
            _ne  = int(n_epochs_meta) if n_epochs_meta is not None else 1
            grid_arr = np.array([int(grid_rows), int(grid_cols), _rpe, _ne], dtype=np.int32)
            # Root dataset dir
            np.save(output_dir / "patch_grid_shape.npy", grid_arr)
            # Each split sub-directory
            for _sname in ["train", "val", "test"]:
                _sdir = output_dir / _sname
                if _sdir.exists():
                    np.save(_sdir / "patch_grid_shape.npy", grid_arr)
            logger.info(
                f"✅ Saved patch grid shape [{grid_rows}rows, {grid_cols}cols, "
                f"{_rpe}rows/epoch, {_ne}epochs] to "
                f"{output_dir} (and all split sub-directories)"
            )
        else:
            logger.warning(
                "  ⚠ patch_grid_rows/cols missing from metadata — "
                "patch_grid_shape.npy NOT saved.  The inference mosaic will "                "fall back to sqrt(N) layout, which may jumble patches.  "                "Check that extract_patches() returned a PatchResult and that "                "_fuse_extract_free populated _all_grid_rows/_all_grid_cols."
            )

        logger.info(f"✅ Dataset saved to {output_dir}")

class EnhancedDatasetCreator(DatasetCreator):
    """Extended DatasetCreator with better splitting"""
    
    def create_stratified_split(self, X: np.ndarray, y: np.ndarray,
                               dates: np.ndarray,
                               split_ratios: Tuple[float, float, float] = (0.65, 0.15, 0.20),
                               random_seed: int = 42,
                               patch_positions: np.ndarray = None) -> Dict:
        """
        Create train/val/test split that maintains:
        1. Temporal distribution (all seasons in all splits)
        2. Spatial distribution (all areas in all splits)
        
        Args:
            X: Features array (N, H, W, C)
            y: Target array (N, H, W, 1)
            dates: Array of dates for each sample
            split_ratios: (train, val, test) ratios
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("\n" + "="*70)
        logger.info("CREATING STRATIFIED SPLIT WITH SPATIAL-TEMPORAL DISTRIBUTION")
        logger.info("="*70)
        
        np.random.seed(random_seed)
        
        # Extract data dimensions
        n_samples = len(X)
        train_ratio, val_ratio, test_ratio = split_ratios
        
        # Create season labels (0=winter, 1=spring, 2=summer, 3=fall)
        # Winter: Dec-Feb (12,1,2), Spring: Mar-May (3,4,5), 
        # Summer: Jun-Aug (6,7,8), Fall: Sep-Nov (9,10,11)
        dates_dt = pd.to_datetime(dates)
        seasons = (dates_dt.month % 12 // 3).values
        
        logger.info(f"\nTemporal distribution:")
        season_names = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Fall (SON)']
        for season_idx in range(4):
            count = np.sum(seasons == season_idx)
            logger.info(f"  {season_names[season_idx]}: {count} samples ({count/n_samples*100:.1f}%)")
        
        # ── Spatial features — vectorised (no Python loop over samples) ─────
        logger.info(f"\nCreating spatial blocks...")
        n_feat_cols = min(3, X.shape[-1])
        spatial_features = X[:, :, :, :n_feat_cols].mean(axis=(1, 2))  # (N, ≤3)
        
        # Cluster locations into spatial blocks
        from sklearn.cluster import KMeans
        n_spatial_blocks = 5
        kmeans = KMeans(n_clusters=n_spatial_blocks, random_state=random_seed)
        spatial_blocks = kmeans.fit_predict(spatial_features)
        
        logger.info(f"Spatial distribution:")
        for block_idx in range(n_spatial_blocks):
            count = np.sum(spatial_blocks == block_idx)
            logger.info(f"  Block {block_idx}: {count} samples ({count/n_samples*100:.1f}%)")
        
        # ── LST VARIANCE STRATIFICATION ───────────────────────────────────────
        # Bin each patch by LST std so all splits share the same variance profile.
        # Without this the test set had ~16% wider LST range than val.
        # Vectorised — no Python loop over samples (was the biggest memory/speed bottleneck)
        lst_stds = y[:, :, :, 0].reshape(n_samples, -1).std(axis=1)
        lst_std_bins = pd.qcut(lst_stds, q=4, labels=False, duplicates='drop')
        n_variance_bins = len(np.unique(lst_std_bins))
        logger.info(f"\nLST variance bins ({n_variance_bins} quartile bins):")
        for b in range(n_variance_bins):
            mask = lst_std_bins == b
            logger.info(f"  Bin {b}: {mask.sum()} patches, "
                       f"LST std range [{lst_stds[mask].min():.2f}, {lst_stds[mask].max():.2f}]°C")

        # ── VALID-PIXEL RATIO STRATIFICATION ──────────────────────────────────
        # NaN-heavy patches (cloud edges, raster borders) were not included in the
        # stratification key, causing them to cluster non-uniformly in val/test.
        # After global-constant NaN fill (→ median ≈ 0 normalised) this produced
        # the central spike in val/test LST histograms. Including a valid-ratio
        # bin ensures NaN-heavy patches are distributed evenly across splits.
        # Vectorised — no Python loop over samples
        valid_ratios = np.isfinite(y[:, :, :, 0]).reshape(n_samples, -1).mean(axis=1)
        valid_ratio_bins = (valid_ratios < 0.99).astype(int)  # 0=fully-valid, 1=has-NaN
        n_valid_bins = len(np.unique(valid_ratio_bins))
        logger.info(f"\nValid-pixel ratio bins ({n_valid_bins} bins):")
        for b in range(n_valid_bins):
            mask = valid_ratio_bins == b
            label = "has-NaN" if b == 1 else "fully-valid"
            logger.info(f"  Bin {b} ({label}): {mask.sum()} patches")
        # ──────────────────────────────────────────────────────────────────────

        # Combine season × spatial × LST-variance × valid-pixel-ratio into stratum key
        strata = (seasons * n_spatial_blocks * n_variance_bins * n_valid_bins
                + spatial_blocks * n_variance_bins * n_valid_bins
                + lst_std_bins * n_valid_bins
                + valid_ratio_bins)

        unique_strata = len(np.unique(strata))
        logger.info(f"\nCombined stratification (season × spatial × LST-var): {unique_strata} unique strata")

        from collections import Counter

        class_counts = Counter(strata)

        valid_indices = np.array([
            i for i, label in enumerate(strata)
            if class_counts[label] >= 2
        ])

        if len(valid_indices) < len(strata):
            removed = len(strata) - len(valid_indices)
            logger.warning(
                f"Removing {removed} samples from rare strata (<2 samples)"
            )

        # Filter every parallel array consistently — missing a slice here causes
        # an IndexError or silent shape mismatch in the logging / split code below.
        # Free originals immediately after slicing to avoid holding both in RAM.
        _X_tmp = X[valid_indices]; del X; X = _X_tmp; del _X_tmp
        _y_tmp = y[valid_indices]; del y; y = _y_tmp; del _y_tmp; gc.collect()
        dates = dates[valid_indices]
        seasons = seasons[valid_indices]
        spatial_blocks = spatial_blocks[valid_indices]
        lst_stds = lst_stds[valid_indices]
        lst_std_bins = lst_std_bins[valid_indices]
        valid_ratio_bins = valid_ratio_bins[valid_indices]
        strata = strata[valid_indices]
        if patch_positions is not None:
            patch_positions = patch_positions[valid_indices]

        # Update sample count after filtering
        n_samples = len(X)

        logger.info(f"Samples after rare-strata removal: {n_samples}")
        # ==========================================================

        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=test_ratio,
            stratify=strata,
            random_state=random_seed
        )

        logger.info(f"\nInitial split:")
        logger.info(f"  Train+Val: {len(train_val_idx)} samples")
        logger.info(f"  Test: {len(test_idx)} samples")
        logger.info(f"  LST std — all: {lst_stds.mean():.3f}°C, "
                   f"test: {lst_stds[test_idx].mean():.3f}°C")

        # Second split: train vs val
        strata_train_val = strata[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio / (train_ratio + val_ratio),
            stratify=strata_train_val,
            random_state=random_seed
        )
        
        # Create the splits — free full X/y as soon as all slices are taken.
        # Fancy indexing creates copies, so after three slices the originals
        # serve no further purpose.  Deleting them here reduces peak RSS by
        # ~33% compared with holding both the originals and the three copies.
        # We extract all index arrays first, then slice and free in one pass.
        X_train = X[train_idx]
        X_val   = X[val_idx]
        X_test  = X[test_idx]
        del X; gc.collect()

        y_train = y[train_idx]
        y_val   = y[val_idx]
        y_test  = y[test_idx]
        del y; gc.collect()

        dates_train = dates[train_idx]
        dates_val   = dates[val_idx]
        dates_test  = dates[test_idx]

        # Slice patch positions in lock-step with X/y so each split's
        # patch_positions.npy correctly maps to that split's X.npy rows.
        pos_train = patch_positions[train_idx] if patch_positions is not None else None
        pos_val   = patch_positions[val_idx]   if patch_positions is not None else None
        pos_test  = patch_positions[test_idx]  if patch_positions is not None else None
        
        # Verify distribution
        logger.info("\n" + "="*70)
        logger.info("DATA SPLIT VERIFICATION")
        logger.info("="*70)
        logger.info(f"Train: {len(train_idx)} samples ({len(train_idx)/n_samples*100:.1f}%)")
        logger.info(f"Val:   {len(val_idx)} samples ({len(val_idx)/n_samples*100:.1f}%)")
        logger.info(f"Test:  {len(test_idx)} samples ({len(test_idx)/n_samples*100:.1f}%)")
        
        # Check season distribution for each split
        for split_name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            split_seasons = seasons[idx]
            season_dist = pd.Series(split_seasons).value_counts(normalize=True).sort_index()
            logger.info(f"\n{split_name} season distribution:")
            for season_idx, pct in season_dist.items():
                logger.info(f"  {season_names[season_idx]}: {pct*100:.1f}%")
        
        # Check spatial distribution for each split
        for split_name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            split_blocks = spatial_blocks[idx]
            block_dist = pd.Series(split_blocks).value_counts(normalize=True).sort_index()
            logger.info(f"\n{split_name} spatial block distribution:")
            for block_idx, pct in block_dist.items():
                logger.info(f"  Block {block_idx}: {pct*100:.1f}%")
        
        # Check temperature distribution — now includes per-patch LST std balance check
        train_mean = y_train.mean()
        val_mean   = y_val.mean()
        test_mean  = y_test.mean()

        logger.info(f"\nTemperature distribution per split:")
        logger.info(f"  Train: mean={train_mean:.2f}°C  std={y_train.std():.3f}°C  "
                   f"patch-std-mean={lst_stds[train_idx].mean():.3f}°C")
        logger.info(f"  Val:   mean={val_mean:.2f}°C  std={y_val.std():.3f}°C  "
                   f"patch-std-mean={lst_stds[val_idx].mean():.3f}°C")
        logger.info(f"  Test:  mean={test_mean:.2f}°C  std={y_test.std():.3f}°C  "
                   f"patch-std-mean={lst_stds[test_idx].mean():.3f}°C")

        # Warn if LST variance is still imbalanced (>5% difference)
        patch_stds = [lst_stds[train_idx].mean(), lst_stds[val_idx].mean(), lst_stds[test_idx].mean()]
        pct_range = (max(patch_stds) - min(patch_stds)) / np.mean(patch_stds) * 100
        if pct_range > 5:
            logger.warning(f"  LST patch-std spread: {pct_range:.1f}% — consider more variance bins")
        else:
            logger.info(f"  LST variance balanced across splits (spread: {pct_range:.1f}%) ✓")
        
        logger.info("="*70 + "\n")
        
        splits = {
            "X_train": X_train,
            "y_train": y_train,
            "dates_train": dates_train,
            "X_val": X_val,
            "y_val": y_val,
            "dates_val": dates_val,
            "X_test": X_test,
            "y_test": y_test,
            "dates_test": dates_test,
            # Per-patch (grid_row, grid_col) arrays — None when patch dicts
            # lacked _grid_row/_grid_col tags (legacy datasets).
            "positions_train": pos_train,
            "positions_val":   pos_val,
            "positions_test":  pos_test,
        }
        
        return splits
    
    def create_temporal_cv_splits(self, X: np.ndarray, y: np.ndarray,
                                  dates: np.ndarray,
                                  n_splits: int = 5) -> List[Dict]:
        """
        Create time-aware cross-validation splits
        Ensures temporal ordering is preserved
        
        Args:
            X, y, dates: Data arrays
            n_splits: Number of CV folds
            
        Returns:
            List of split dictionaries
        """
        logger.info("\n" + "="*70)
        logger.info(f"CREATING {n_splits}-FOLD TEMPORAL CV SPLITS")
        logger.info("="*70)
        
        # Sort by date
        sort_idx = np.argsort(dates)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        dates_sorted = dates[sort_idx]
        
        n_samples = len(X)
        fold_size = n_samples // n_splits
        
        cv_splits = []
        
        for fold in range(n_splits):
            # Validation indices for this fold
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_splits - 1 else n_samples
            
            # Training indices (all data before validation)
            train_end = val_start
            
            if train_end < fold_size:
                # Not enough training data, skip this fold
                logger.warning(f"Fold {fold}: Insufficient training data, skipping")
                continue
            
            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)
            
            split = {
                "X_train": X_sorted[train_indices],
                "y_train": y_sorted[train_indices],
                "dates_train": dates_sorted[train_indices],
                "X_val": X_sorted[val_indices],
                "y_val": y_sorted[val_indices],
                "dates_val": dates_sorted[val_indices],
                "fold": fold
            }
            
            cv_splits.append(split)
            
            logger.info(f"Fold {fold}:")
            logger.info(f"  Train: {len(train_indices)} samples "
                       f"({dates_sorted[train_indices[0]]} to {dates_sorted[train_indices[-1]]})")
            logger.info(f"  Val: {len(val_indices)} samples "
                       f"({dates_sorted[val_indices[0]]} to {dates_sorted[val_indices[-1]]})")
        
        return cv_splits

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

    # ── DIAGNOSTICS: initialise visualisation helper ────────────────────────
    output_dataset_dir_early = PROCESSED_DATA_DIR / "cnn_dataset"
    diag = PreprocessingDiagnostics(output_dir=output_dataset_dir_early)
    logger.info("[Diagnostics] Diagnostics module initialised")
    
    # Step 1: Process Landsat data
    landsat_processed = []
    if has_landsat:
        logger.info("\n" + "="*70)
        logger.info("STEP 1A: Process Landsat data")
        logger.info("="*70)
        
        landsat_files = sorted(
            list(landsat_dir.glob("Landsat_*.tif")) +   # GEE export (new)
            list(landsat_dir.glob("landsat_*.tif"))     # lower-case variant
        )
        logger.info(f"Found {len(landsat_files)} Landsat files")
        
        for raw_file in landsat_files:
            parts = raw_file.stem.split('_')
            try:
                year  = int(parts[-2])
                month = int(parts[-1])
            except (IndexError, ValueError):
                logger.warning(f"  Cannot parse date from {raw_file.name}, skipping")
                continue
            timestamp = pd.Timestamp(year=year, month=month, day=15)
            
            # Process the file
            processed = landsat_preprocessor.process_raw_file(raw_file, calculate_lst=True)
            
            if processed is None:
                continue
            
            # Validate LST
            if "LST" not in processed:
                logger.warning(f"  No LST in {raw_file.name}, skipping")
                continue
            
            lst_finite = processed["LST"][np.isfinite(processed["LST"])]
            lst_std = float(np.nanstd(lst_finite)) if lst_finite.size > 1 else 0.0
            lst_valid_ratio = lst_finite.size / processed["LST"].size
            
            if lst_std < 0.5:
                logger.warning(f"  LST variance too low ({lst_std:.2f}°C), skipping")
                continue
            
            if lst_valid_ratio < 0.1:
                logger.warning(f"  LST valid ratio too low ({lst_valid_ratio*100:.1f}%), skipping")
                continue
            
            landsat_processed.append((timestamp, processed))
            logger.info(f"  ✓ {raw_file.name}")

            # ── DIAGNOSTIC PLOTS (first file only) ────────────────────
            if len(landsat_processed) == 1:
                try:
                    raw_data_for_plot = load_tif_as_bands(raw_file) or {}
                    diag.plot_raw_bands(
                        raw_data_for_plot,
                        title=f"Raw Bands – {raw_file.name}",
                        filename="01_raw_bands_landsat.png",
                    )
                    diag.plot_spectral_indices(
                        processed,
                        filename="02_spectral_indices_landsat.png",
                    )
                    # LST validation plot (raw vs clean already computed above)
                    if "LST" in processed:
                        raw_lst = raw_data_for_plot.get("ST_B10")
                        if raw_lst is not None:
                            ndvi_tmp = processed.get("NDVI", np.zeros_like(raw_lst, dtype=float))
                            lst_raw_calc = landsat_preprocessor.calculate_lst_from_thermal(
                                raw_lst, ndvi_tmp
                            )
                            if lst_raw_calc is not None:
                                _, tmp_stats = landsat_preprocessor.validate_lst(lst_raw_calc)
                                diag.plot_lst_validation(
                                    lst_raw_calc,
                                    processed["LST"],
                                    tmp_stats,
                                    filename="03_lst_validation.png",
                                )
                        diag.plot_lst_vs_indices(
                            processed,
                            filename="04_lst_vs_indices_landsat.png",
                        )
                except Exception as _e:
                    logger.warning(f"[Diagnostics] plot failed for {raw_file.name}: {_e}")
            # ──────────────────────────────────────────────────────────
        
        logger.info(f"\n✓ Processed {len(landsat_processed)} Landsat files")
    
    # Step 2: Process Sentinel-2 data
    sentinel2_processed = []
    if has_sentinel2:
        logger.info("\n" + "="*70)
        logger.info("STEP 1B: Process Sentinel-2 data")
        logger.info("="*70)
        
        sentinel2_files = sorted(
            list(sentinel2_dir.glob("Sentinel2_*.tif")) +   # GEE export (new)
            list(sentinel2_dir.glob("sentinel2_*.tif"))     # lower-case variant
        )
        logger.info(f"Found {len(sentinel2_files)} Sentinel-2 files")
        
        for raw_file in sentinel2_files:
            parts = raw_file.stem.split('_')
            try:
                year  = int(parts[-2])
                month = int(parts[-1])
            except (IndexError, ValueError):
                logger.warning(f"  Cannot parse date from {raw_file.name}, skipping")
                continue
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

            # ── DIAGNOSTIC PLOTS (first S2 file only) ─────────────────
            if len(sentinel2_processed) == 1:
                try:
                    raw_s2_data = load_tif_as_bands(raw_file) or {}
                    diag.plot_s2_raw_bands(
                        raw_s2_data,
                        filename="s2_01_raw_bands.png",
                    )
                    diag.plot_s2_spectral_indices(
                        processed,
                        filename="s2_02_spectral_indices.png",
                    )
                    diag.plot_s2_band_statistics(
                        raw_s2_data,
                        filename="s2_03_band_statistics.png",
                    )
                    diag.plot_s2_data_quality(
                        raw_s2_data,
                        filename="s2_04_data_quality.png",
                    )
                    diag.plot_s2_band_ratios(
                        raw_s2_data,
                        filename="s2_05_band_ratios.png",
                    )
                    diag.plot_s2_landcover_proxy(
                        processed,
                        filename="s2_06_landcover_proxy.png",
                    )
                except Exception as _e:
                    logger.warning(f"[Diagnostics] S2 plots failed for {raw_file.name}: {_e}")
            # ──────────────────────────────────────────────────────────
        
        logger.info(f"\n✓ Processed {len(sentinel2_processed)} Sentinel-2 files")

        # ── DIAGNOSTIC: S2 temporal trends & resolution comparison ────
        try:
            if len(sentinel2_processed) >= 2:
                diag.plot_s2_temporal_trends(
                    sentinel2_processed,
                    filename="s2_07_temporal_trends.png",
                )
            if len(sentinel2_processed) >= 1 and len(landsat_processed) >= 1:
                diag.plot_sensor_agreement(
                    landsat_processed[0][1],
                    sentinel2_processed[0][1],
                    filename="s2_08_sensor_agreement.png",
                )
                diag.plot_resolution_comparison(
                    sentinel2_processed[0][1],
                    landsat_processed[0][1],
                    index="NDVI",
                    filename="s2_09_resolution_comparison.png",
                )
            if len(sentinel2_processed) >= 2:
                diag.plot_s2_scene_variability(
                    sentinel2_processed,
                    filename="s2_10_scene_variability.png",
                )
        except Exception as _e:
            logger.warning(f"[Diagnostics] S2 multi-scene plots failed: {_e}")
        # ──────────────────────────────────────────────────────────────
    
    # Steps 2 + 3 (combined): fuse and immediately extract patches,
    # freeing each raster as soon as its patches are done.
    # Previously all_fused_data held every raster in RAM before patch
    # extraction even began — for 10 years of monthly data this was several GB.
    logger.info("\n" + "="*70)
    logger.info("STEP 2+3: Fuse data and extract patches (streaming)")
    logger.info("="*70)

    all_patches: List[Dict] = []
    all_dates:   List       = []
    all_rasters: List       = []   # (raster_dict, [position_patches]) pairs

    # ── Patch grid shape tracking ─────────────────────────────────────────────
    # We record the grid dimensions from every raster and take the most common
    # value (mode) so a single anomalous raster doesn't pollute the metadata.
    # For a well-configured pipeline all rasters share the same spatial extent
    # and therefore the same grid shape; the mode is just a safety net.
    _all_grid_rows: List[int] = []
    _all_grid_cols: List[int] = []

    # ── Epoch counter — used to give each temporal epoch a unique row-offset
    # in the combined patch_positions array.  Without this, patches from
    # different dates share the same (grid_row, grid_col) values and overwrite
    # each other in the mosaic assembler, creating a checkerboard artefact.
    # Epoch N's patches are placed at rows [N * grid_rows ... (N+1)*grid_rows - 1].
    _epoch_index: List[int] = []   # epoch number for each entry in all_rasters

    first_fused_diag = True

    def _fuse_extract_free(fused_data: Dict, date, label: str) -> None:
        """Extract position-only patches, store raster ref, free bands later."""
        result = dataset_creator.extract_patches(
            fused_data,
            patch_size=64, stride=24, min_valid_ratio=0.95,
            min_variance=0.3, min_temp=10.0, max_temp=65.0,
        )
        # extract_patches now returns a named PatchResult(patches, grid_rows, grid_cols)
        patches = result.patches

        # FIX: Skip entirely-cloud-covered / all-NaN epochs (zero valid patches).
        # Previously an epoch that passed LST-std/valid-ratio checks at the scene
        # level but yielded 0 acceptable patches (e.g. Epoch 31 = 100% cloud) was
        # still appended to _all_grid_rows, inflating the cumulative row offset for
        # every subsequent epoch and misaligning their mosaic slices.  By returning
        # early we keep _all_grid_rows in sync with the epochs that actually have
        # canvas rows, so the slicer boundaries remain correct.
        if not patches:
            logger.warning(
                f"  {label}: 0 patches extracted "
                f"(grid {result.grid_rows}r×{result.grid_cols}c) — "
                f"epoch skipped (all-cloud or all-NaN scene)."
            )
            # Still record the raster so it can be freed, but mark it as empty.
            all_rasters.append((fused_data, []))
            all_dates.append(date)
            return

        _all_grid_rows.append(result.grid_rows)
        _all_grid_cols.append(result.grid_cols)

        # ── Offset grid_row by epoch index so patches from different dates
        # never share the same grid position.  The epoch's row-offset equals
        # the cumulative row count of all preceding (non-empty) epochs.
        epoch_idx = len(_epoch_index)
        _epoch_index.append(epoch_idx)
        prior_rows = sum(_all_grid_rows[:-1])   # total rows from previous epochs
        for p in patches:
            p["date"] = date
            # Shift _grid_row by the epoch offset so each epoch occupies its own
            # row band in the combined grid.  _grid_col stays unchanged.
            if p.get("_grid_row", -1) >= 0:
                p["_grid_row"] = p["_grid_row"] + prior_rows

        all_patches.extend(patches)
        all_dates.append(date)
        # Keep a reference to the raster alongside its patches so
        # create_training_samples can slice from it without re-loading.
        all_rasters.append((fused_data, patches))
        logger.info(f"  {label}: {len(patches)} patches "
                    f"(grid {result.grid_rows}r×{result.grid_cols}c, "
                    f"epoch row-offset={prior_rows}, "
                    f"running total: {len(all_patches)})")

    if has_landsat and has_sentinel2 and len(landsat_processed) > 0 and len(sentinel2_processed) > 0:
        logger.info("Performing multi-sensor fusion (streaming)...")
        matches = fusion.temporal_match(
            landsat_processed, sentinel2_processed, max_time_diff_days=16
        )
        del landsat_processed, sentinel2_processed
        gc.collect()

        for mi, (avg_date, ls_data, s2_data, time_diff) in enumerate(matches):
            logger.info(f"\nFusing pair {mi+1}/{len(matches)} (Δt={time_diff}d):")
            fused_data = fusion.fuse_data(ls_data, s2_data, time_diff, target_resolution=30)

            if all(k in fused_data for k in ("NDVI", "NDBI", "MNDWI")):
                fused_data["impervious_surface"] = feature_engineer.calculate_impervious_surface(
                    fused_data["NDVI"], fused_data["NDBI"], fused_data["MNDWI"]
                )

            if first_fused_diag:
                try:
                    diag.plot_fusion_comparison(ls_data, s2_data, fused_data,
                                                index="NDVI",
                                                filename="09_fusion_comparison_NDVI.png")
                    diag.plot_fusion_comparison(ls_data, s2_data, fused_data,
                                                index="NDBI",
                                                filename="09b_fusion_comparison_NDBI.png")
                    diag.plot_fusion_weight_map(ls_data, s2_data, fused_data,
                                                time_diff=time_diff,
                                                filename="s2_11_fusion_weights.png")
                    diag.plot_impervious_surface_analysis(fused_data,
                                                          filename="s2_12_impervious_surface.png")
                except Exception as _e:
                    logger.warning(f"[Diagnostics] fusion plots failed: {_e}")
                first_fused_diag = False

            del ls_data, s2_data
            gc.collect()

            _fuse_extract_free(fused_data, avg_date, f"Pair {mi+1}/{len(matches)}")

        logger.info(f"\n✓ Processed {len(matches)} fused pairs, "
                    f"{len(all_patches)} total patches")

    elif has_landsat and len(landsat_processed) > 0:
        logger.info("Using Landsat data only (no Sentinel-2 available)")
        n_ls = len(landsat_processed)
        for li, (timestamp, ls_data) in enumerate(landsat_processed):
            if all(k in ls_data for k in ("NDVI", "NDBI", "MNDWI")):
                ls_data["impervious_surface"] = feature_engineer.calculate_impervious_surface(
                    ls_data["NDVI"], ls_data["NDBI"], ls_data["MNDWI"]
                )
            _fuse_extract_free(ls_data, timestamp, f"Landsat {li+1}/{n_ls}")
        del landsat_processed
        gc.collect()

    else:
        logger.error("No valid data available for training")
        return

    if not all_patches:
        logger.error("No patches extracted")
        return
    
    # Patch quality diagnostic (before patches are freed)
    # Pass the first raster's LST to enable visual patch thumbnails.
    # This lets the plot reveal NaN-fill artefacts (flat cyan blobs) directly.
    try:
        _diag_raster = all_rasters[0][0] if all_rasters else None
        diag.plot_patch_diagnostics(
            all_patches,
            raster_data=_diag_raster,
            filename="05_patch_diagnostics.png",
        )
    except Exception as _e:
        logger.warning(f"[Diagnostics] patch plot failed: {_e}")

    # Step 4: Create training samples (streaming — one raster at a time)
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Create training samples")
    logger.info("="*70)

    temporal_features = feature_engineer.encode_temporal_features(all_dates[0])
    dates_all = np.array([patch["date"] for patch in all_patches])

    # Build X/y by streaming through each (raster, patches) pair, freeing
    # the raster dict immediately after its patches are written into X/y.
    channel_order = [
        "SR_B4", "SR_B5", "SR_B6", "SR_B7",
        "NDVI", "NDBI", "MNDWI", "BSI", "UI", "albedo",
    ]
    n_channels = len(channel_order)
    patch_size = 64
    n_total    = len(all_patches)

    X = np.zeros((n_total, patch_size, patch_size, n_channels), dtype=np.float32)
    y = np.zeros((n_total, patch_size, patch_size, 1),           dtype=np.float32)
    # Per-patch spatial grid positions — (grid_row, grid_col) — needed by the
    # mosaic assembler to place each patch at its correct canvas cell even after
    # QC filtering removes some patches from the full scan grid.
    positions_all = np.full((n_total, 2), -1, dtype=np.int32)

    write_idx = 0
    for raster_data, raster_patches in all_rasters:
        for p in raster_patches:
            r, c = p["position"]
            for ch_idx, feat in enumerate(channel_order):
                arr = raster_data.get(feat)
                if arr is not None:
                    X[write_idx, :, :, ch_idx] = arr[r:r+patch_size, c:c+patch_size]
            lst_arr = raster_data.get("LST")
            if lst_arr is not None:
                y[write_idx, :, :, 0] = lst_arr[r:r+patch_size, c:c+patch_size]
            positions_all[write_idx, 0] = p.get("_grid_row", -1)
            positions_all[write_idx, 1] = p.get("_grid_col", -1)
            write_idx += 1
        # Free this raster immediately — its data is now in X/y
        raster_data.clear()
        gc.collect()

    del all_rasters
    gc.collect()

    # Free the patch list — no longer needed once X/y are filled
    del all_patches
    gc.collect()
    logger.info("  Freed rasters and patch list from memory")

    # QC + NaN fill + safety clip
    # Temperature bounds aligned with validate_lst (Tier-1 pixel gate: 10–65 °C).
    valid_mask = np.ones(n_total, dtype=bool)
    for s in range(n_total):
        lst_p = y[s, :, :, 0]
        fin   = np.isfinite(lst_p)
        if fin.mean() < 0.95:
            valid_mask[s] = False; continue
        lst_mean = float(lst_p[fin].mean()) if fin.any() else 0.0
        if not (10.0 <= lst_mean <= 65.0):
            valid_mask[s] = False

    if not valid_mask.all():
        dropped = int((~valid_mask).sum())
        logger.info(f"  QC dropped {dropped}/{n_total} samples")
        X             = X[valid_mask]
        y             = y[valid_mask]
        dates_all     = dates_all[valid_mask]
        positions_all = positions_all[valid_mask]

    n_samples = len(X)

    # ── Resolve the canonical patch grid shape ───────────────────────────────
    # _all_grid_rows / _all_grid_cols were populated by _fuse_extract_free.
    # grid_rows = TOTAL stacked rows across all epochs.
    # grid_cols = derived from actual patch positions (max col index + 1).
    #
    # FIX: Previously grid_cols was taken as the MODE of _all_grid_cols.
    # When epochs are loaded at different downsample factors their grid widths
    # differ, so the mode may be wrong for some epochs.  Patch placement in
    # the canvas uses the actual _grid_col values, so the true canvas width
    # is always max(_grid_col) + 1 — read directly from positions_all.
    # This avoids the "16 patch-cols" bug where mode undercount caused patches
    # to wrap into incorrect canvas columns.
    if _all_grid_cols:
        patch_grid_rows = int(sum(_all_grid_rows))

        # FIX: derive cols from observed positions, not the _all_grid_cols mode
        if np.any(positions_all[:, 1] >= 0):
            patch_grid_cols = int(positions_all[positions_all[:, 1] >= 0, 1].max()) + 1
        else:
            # Nothing survived QC — fall back to mode of recorded col counts
            from collections import Counter as _Counter
            patch_grid_cols = int(_Counter(_all_grid_cols).most_common(1)[0][0])

        logger.info(
            f"  Patch grid shape (stacked across {len(_all_grid_rows)} epochs): "
            f"{patch_grid_rows} total rows × {patch_grid_cols} cols"
        )
        logger.info(
            f"  Per-epoch row counts: {_all_grid_rows}"
        )
        if len(set(_all_grid_cols)) > 1:
            logger.warning(
                f"  ⚠ Rasters have different grid widths: {sorted(set(_all_grid_cols))}. "
                f"patch_grid_cols resolved from positions_all={patch_grid_cols}."
            )
    else:
        # Fallback: estimate from positions_all when _all_grid_cols is empty.
        # (all_rasters is already freed at this point, so we cannot re-read it.)
        if np.any(positions_all >= 0):
            valid_pos = positions_all[positions_all[:, 1] >= 0]
            patch_grid_cols = int(valid_pos[:, 1].max()) + 1
            patch_grid_rows = int(positions_all[positions_all[:, 0] >= 0, 0].max()) + 1
        else:
            patch_grid_cols = patch_grid_rows = None
        logger.warning("  ⚠ _all_grid_cols is empty — patch grid shape estimated from positions_all.")

    # ── DIAGNOSTIC: LST mosaic reconstruction ────────────────────────────────
    # IMPORTANT: called here, BEFORE the NaN-fill loop below, so that ocean/cloud
    # pixels still carry their true NaN values.  If called after NaN fill, those
    # pixels get replaced with the patch mean (a warm land temperature for coastal
    # patches), which places spuriously-warm values at ocean grid cells and creates
    # a red stripe along the coastline in the temporal mean panel.
    # The mosaic function has its own gap-fill that interpolates NaN holes from
    # spatially neighbouring valid pixels, which is geographically correct.
    _mosaic_cols = patch_grid_cols if patch_grid_cols else int(np.ceil(np.sqrt(n_samples)))
    try:
        diag.plot_lst_mosaic_reconstruction(
            y               = y,
            positions_all   = positions_all,
            patch_grid_cols = _mosaic_cols,
            all_grid_rows   = _all_grid_rows,
            stride          = 24,
            geo_bounds      = STUDY_AREA.get("bounds"),
            filename        = "P_mosaic_lst_reconstruction.png",
        )
        logger.info(
            f"[Diagnostics] LST mosaic reconstruction: "
            f"{len(_all_grid_rows)} non-empty epochs, "
            f"grid_cols={_mosaic_cols}, "
            f"n_patches={len(y)}"
        )
    except Exception as _e:
        logger.warning(f"[Diagnostics] LST mosaic reconstruction plot failed: {_e}")
    # ─────────────────────────────────────────────────────────────────────────

    # NaN fill + safety clip (runs AFTER mosaic so the diagnostic sees raw NaN)
    for s in range(n_samples):
        for ch in range(n_channels):
            sl = X[s, :, :, ch]; nm = ~np.isfinite(sl)
            if nm.any(): sl[nm] = float(sl[~nm].mean()) if (~nm).any() else 0.0
        sl = y[s, :, :, 0]; nm = ~np.isfinite(sl)
        if nm.any(): sl[nm] = float(sl[~nm].mean()) if (~nm).any() else 35.0

    np.clip(y, 10.0, 65.0, out=y)

    # Sample weights
    patch_means   = y[:, :, :, 0].reshape(n_samples, -1).mean(axis=1)
    n_wb          = 10
    be_           = np.linspace(patch_means.min(), patch_means.max() + 1e-6, n_wb + 1)
    bids_         = np.clip(np.digitize(patch_means, be_) - 1, 0, n_wb - 1)
    bc_           = np.maximum(np.bincount(bids_, minlength=n_wb).astype(np.float32), 1)
    rw_           = 1.0 / bc_[bids_]
    sample_weights = rw_ / rw_.mean()

    metadata = {
        "n_samples":        n_samples,
        "patch_size":       patch_size,
        "n_channels":       n_channels,
        "channel_order":    channel_order,
        "temporal_features": temporal_features,
        "temperature_range": {
            "min": float(np.min(y)), "max": float(np.max(y)),
            "mean": float(np.mean(y)), "std": float(np.std(y)),
        },
        "sample_weights": sample_weights,
        # ── Patch grid shape — consumed by uhi_pipeline_manager._mosaic_patches
        # to reconstruct the spatially-correct full-area mosaic at inference time.
        "patch_grid_rows": patch_grid_rows,
        "patch_grid_cols": patch_grid_cols,
        "patch_stride":    24,   # keep in sync with stride used above
        # ── Per-epoch grid dimensions — required for correct multi-epoch collapse.
        # patch_grid_rows is the TOTAL stacked rows (sum of all per-epoch rows).
        # The inference mosaic uses the full per-epoch list to slice correctly,
        # because epochs may have different row counts when raster extents vary.
        #
        # FIX: Store the complete list of per-epoch row counts instead of only
        # _all_grid_rows[0].  The old single-value assumption caused the inference
        # plotter to use a fixed stride that drifted when epochs had different
        # heights, producing the same blocky mis-slice seen in preprocessing.
        "patch_grid_rows_per_epoch": int(_all_grid_rows[0]) if _all_grid_rows else None,
        "patch_grid_rows_per_epoch_list": [int(r) for r in _all_grid_rows] if _all_grid_rows else [],
        "n_epochs": int(len(_all_grid_rows)) if _all_grid_rows else 1,
        # ── Per-patch (grid_row, grid_col) positions — shape (N, 2) int32.
        # Stored here so create_stratified_split can propagate them into the
        # splits dict alongside X/y, letting save_dataset write
        # patch_positions.npy per split directory.
        "patch_positions": positions_all,
    }

    logger.info(f"  Created X={X.shape}, y={y.shape}")
    logger.info(f"  Temp range: [{metadata['temperature_range']['min']:.2f}, "
                f"{metadata['temperature_range']['max']:.2f}]°C")

    dates = dates_all[:n_samples]

    # Step 6: Create splits
    logger.info("\n" + "="*70)
    logger.info("STEP 6: Create train/val/test splits")
    logger.info("="*70)
    
    # Use enhanced dataset creator
    dataset_creator = EnhancedDatasetCreator()
    
    # Stratified split with spatial-temporal distribution
    # NEW: 65% train, 15% val, 20% test (was 70/15/15)
    splits = dataset_creator.create_stratified_split(
        X, y, dates,
        split_ratios=(0.65, 0.15, 0.20),
        random_seed=42,
        patch_positions=metadata.get("patch_positions"),
    )

    # Step 6.5: Compute normalization statistics from training data
    logger.info("\n" + "="*70)
    logger.info("STEP 6.5: Compute normalization statistics")
    logger.info("="*70)

    output_dataset_dir = PROCESSED_DATA_DIR / "cnn_dataset"
    norm_stats = dataset_creator.compute_and_save_normalization_stats(
        splits['X_train'], 
        splits['y_train'],
        output_dataset_dir
    )

    logger.info("\n" + "="*70)
    logger.info("STEP 6.5.1: Verify no data leakage (RAW)")
    logger.info("="*70)

    dataset_creator.verify_no_data_leakage(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val'],
        splits['X_test'], splits['y_test'],
        norm_stats
    )

    # Step 6.6: Normalize all splits
    logger.info("\n" + "="*70)
    logger.info("STEP 6.6: Normalize data")
    logger.info("="*70)

    # Snapshot a small raw sample before normalisation for the diagnostics plot.
    # 200 patches is negligible memory cost and avoids the approximate
    # un-normalisation reconstruction that was used previously.
    _snap_size = min(200, splits['X_train'].shape[0])
    X_raw_snap = splits['X_train'][:_snap_size].copy()
    y_raw_snap = splits['y_train'][:_snap_size].copy()

    logger.info("Normalizing training data...")
    splits['X_train'], splits['y_train'] = dataset_creator.normalize_data(
        splits['X_train'], splits['y_train'], norm_stats
    )

    logger.info("Normalizing validation data...")
    splits['X_val'], splits['y_val'] = dataset_creator.normalize_data(
        splits['X_val'], splits['y_val'], norm_stats
    )

    logger.info("Normalizing test data...")
    splits['X_test'], splits['y_test'] = dataset_creator.normalize_data(
        splits['X_test'], splits['y_test'], norm_stats
    )

    # Verify normalization on training data
    logger.info("\n" + "="*70)
    logger.info("NORMALIZATION VERIFICATION")
    logger.info("="*70)
    logger.info("Training data after normalization:")
    logger.info(f"  X_train: mean={splits['X_train'].mean():.4f}, std={splits['X_train'].std():.4f}")
    logger.info(f"  y_train: mean={splits['y_train'].mean():.4f}, std={splits['y_train'].std():.4f}")

    if not (-0.1 < splits['X_train'].mean() < 0.1):
        logger.warning("⚠️ X_train mean is not close to 0!")
    if not (0.9 < splits['X_train'].std() < 1.1):
        logger.warning("⚠️ X_train std is not close to 1!")
    if not (-0.1 < splits['y_train'].mean() < 0.1):
        logger.warning("⚠️ y_train mean is not close to 0!")
    if not (0.9 < splits['y_train'].std() < 1.1):
        logger.warning("⚠️ y_train std is not close to 1!")

    logger.info("="*70)

    # ── DIAGNOSTIC PLOTS: splits, normalisation, channel correlation ───────
    try:
        channel_names = metadata.get("channel_order", None)

        diag.plot_split_distributions(splits, filename="06_split_distributions.png")
        logger.info("[Diagnostics] Split distribution plot saved.")
    except Exception as _e:
        logger.warning(f"[Diagnostics] split distribution plot failed: {_e}")

    # ── Normalisation diagnostics — uses the pre-norm snapshot taken above ──
    try:
        X_train_norm = splits.get("X_train")
        y_train_norm = splits.get("y_train")
        if X_train_norm is not None and y_train_norm is not None:
            snap_size = X_raw_snap.shape[0]
            diag.plot_normalization_diagnostics(
                X_raw_snap, X_train_norm[:snap_size],
                y_raw_snap, y_train_norm[:snap_size],
                channel_names=channel_names,
                filename="07_normalization.png",
            )
            logger.info("[Diagnostics] Normalisation diagnostics plot saved.")
            del X_raw_snap, y_raw_snap
    except Exception as _e:
        logger.warning(f"[Diagnostics] normalisation plot failed: {_e}")

    try:
        X_train_norm = splits.get("X_train")
        if X_train_norm is not None:
            diag.plot_channel_correlation(
                X_train_norm,
                channel_names=channel_names,
                filename="08_channel_correlation.png",
            )
            logger.info("[Diagnostics] Channel correlation plot saved.")
    except Exception as _e:
        logger.warning(f"[Diagnostics] channel correlation plot failed: {_e}")

    try:
        diag.plot_pipeline_summary(splits, metadata, filename="10_pipeline_summary.png")
        logger.info("[Diagnostics] Pipeline summary dashboard saved.")
    except Exception as _e:
        logger.warning(f"[Diagnostics] pipeline summary plot failed: {_e}")
    # ──────────────────────────────────────────────────────────────────────


    # Step 6.7: Update metadata with fusion info
    logger.info("\n" + "="*70)
    logger.info("STEP 6.7: Update metadata")
    logger.info("="*70)

    fusion_strategy = (
        'multi-sensor' if (has_landsat and has_sentinel2)
        else 'landsat-only'
    )
    metadata['fusion_info'] = {
        'fusion_strategy': fusion_strategy,
        'landsat_available':   has_landsat,
        'sentinel2_available': has_sentinel2,
        'total_samples':       metadata["n_samples"],
    }

    # Save sample weights for the training split
    sample_weights = metadata.pop("sample_weights", None)
    if sample_weights is not None:
        y_train_raw = splits['y_train']
        patch_means_train = y_train_raw[:, :, :, 0].reshape(len(y_train_raw), -1).mean(axis=1)
        n_wb = 10
        be = np.linspace(patch_means_train.min(), patch_means_train.max() + 1e-6, n_wb + 1)
        bids = np.clip(np.digitize(patch_means_train, be) - 1, 0, n_wb - 1)
        bc = np.maximum(np.bincount(bids, minlength=n_wb).astype(np.float32), 1)
        rw = 1.0 / bc[bids]
        train_weights = rw / rw.mean()
        weights_path = output_dataset_dir / "train" / "weights_train.npy"
        np.save(weights_path, train_weights.astype(np.float32))
        logger.info(f"✅ Saved sample weights → {weights_path}")
        logger.info(f"   Weight range: [{train_weights.min():.3f}, {train_weights.max():.3f}]")

    # save_dataset streams one split at a time, deleting each from `splits` after saving
    dataset_creator.save_dataset(splits, output_dataset_dir, metadata, norm_stats)

    # ── Final summary (splits dict is now empty — use metadata) ─────────────
    logger.info("\n" + "="*70)
    logger.info("✓ MULTI-SENSOR PREPROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(f"Data sources:")
    logger.info(f"  Landsat available:    {has_landsat}")
    logger.info(f"  Sentinel-2 available: {has_sentinel2}")
    logger.info(f"  Fusion strategy:      {fusion_strategy}")
    logger.info(f"Training data:")
    logger.info(f"  Total samples (after QC): {metadata['n_samples']}")
    logger.info(f"Output:")
    logger.info(f"  Dataset saved to: {output_dataset_dir}")
    _pgr = metadata.get("patch_grid_rows", "unknown")
    _pgc = metadata.get("patch_grid_cols", "unknown")
    logger.info(f"  Patch grid shape: {_pgr} rows × {_pgc} cols")
    logger.info(f"    → patch_grid_shape.npy saved in dataset root + all split dirs")
    logger.info(f"    → inference mosaic will reconstruct full area automatically")
    logger.info(f"Normalization:")
    logger.info(f"  Stats saved: ✅")
    logger.info(f"  All splits normalized in-place: ✅")
    logger.info("="*70)
    logger.info("\nNext step: Run model training")
    logger.info(f"  (mosaic grid: {_pgr}r × {_pgc}c — no --patch-grid-cols flag needed)")


if __name__ == "__main__":
    main()