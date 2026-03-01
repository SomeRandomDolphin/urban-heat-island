"""
Data preprocessing pipeline for Urban Heat Island detection
Purpose: Process Landsat 8 and 9 data
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
from scipy.ndimage import uniform_filter, generic_filter, zoom
import pyproj
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, StratifiedKFold

import rasterio
from rasterio.enums import Resampling as RasterioResampling

from config import *

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
# GeoTIFF / NPZ unified band loader
# ---------------------------------------------------------------------------

_LANDSAT_BAND_ORDER = [
    "SR_B1", "SR_B2", "SR_B3", "SR_B4",
    "SR_B5", "SR_B6", "SR_B7", "ST_B10", "QA_PIXEL",
]


# ---------------------------------------------------------------------------
# Memory-budget configuration
# ---------------------------------------------------------------------------
# Maximum number of pixels (rows × cols) allowed per band before adaptive
# downsampling kicks in.  At float32 each pixel costs 4 bytes, so:
#   12_000_000 px  →  ~46 MiB per band  →  ~410 MiB for 9 Landsat bands
# Raise this if your machine has more headroom; lower it if you still OOM.
# The value can also be overridden via an environment variable:
#   export UHI_MAX_PIXELS=8000000
import os as _os
_DEFAULT_MAX_PIXELS = 8_000_000
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
    {band_name: 2-D float32 array}, mirroring the format of np.load(npz).

    Band identification priority:
      1. Rasterio band description stored in the file (set by GEE export).
      2. Positional fallback using the known export order (inferred from
         the number of bands in the file).
      3. Generic ``band_N`` keys as a last resort.

    The earth_engine_loader applies Landsat C2 L2 scale factors before
    exporting, so TIF values are already physically meaningful:
      - SR bands  : surface reflectance [0.0 – 1.0]
      - ST_B10    : brightness temperature in Kelvin (~280–330 K)
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

            if not has_desc:
                if n_bands == len(_LANDSAT_BAND_ORDER):
                    descriptions = _LANDSAT_BAND_ORDER

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
                bands[name] = arr

        actual_rows, actual_cols = next(iter(bands.values())).shape
        logger.info(
            f"  Loaded {n_bands} bands from {tif_path.name} "
            f"[{actual_rows}×{actual_cols}]: {list(bands.keys())}"
        )
        return bands
    except Exception as exc:
        logger.error(f"  Failed to load {tif_path}: {exc}")
        return None


def _downsample_npz_bands(data: Dict[str, np.ndarray],
                           max_pixels: int = MAX_PIXELS) -> Dict[str, np.ndarray]:
    """
    Apply adaptive downsampling to a dict of arrays loaded from a legacy .npz.

    Uses scipy.ndimage.zoom with order=1 (bilinear) for continuous bands and
    order=0 (nearest-neighbour) for QA/mask bands.

    Args:
        data       : {band_name: 2-D ndarray} loaded from np.load(npz)
        max_pixels : pixel budget

    Returns:
        Same dict with arrays resampled in-place (or original if no downsample).
    """
    _MASK_BAND_NAMES = {"QA_PIXEL", "SCL", "QA60"}

    # Determine shape from the first 2-D array
    ref_shape = None
    for arr in data.values():
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            ref_shape = arr.shape
            break
    if ref_shape is None:
        return data

    native_rows, native_cols = ref_shape
    n_bands = sum(1 for v in data.values() if isinstance(v, np.ndarray) and v.ndim == 2)
    factor = _compute_downsample_factor(native_rows, native_cols, n_bands, max_pixels)

    if factor >= 1.0:
        return data  # nothing to do

    logger.warning(
        f"  [Downsample/npz] native {native_rows}×{native_cols} "
        f"→ factor={factor:.3f}. Downsampling all 2-D bands."
    )

    out: Dict[str, np.ndarray] = {}
    for name, arr in data.items():
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            out[name] = arr   # pass through scalars / 1-D metadata as-is
            continue
        # Use the same factor even if this particular band has a different shape
        # (e.g. a higher-res ancillary band) — keep relative shapes consistent.
        band_factor_r = factor * (native_rows / arr.shape[0])
        band_factor_c = factor * (native_cols / arr.shape[1])
        zoom_order = 0 if name in _MASK_BAND_NAMES else 1
        out[name] = zoom(
            arr.astype(np.float32),
            (band_factor_r, band_factor_c),
            order=zoom_order,
            prefilter=False,
        ).astype(np.float32)
    return out


def load_raw_file(raw_file: Path,
                   max_pixels: int = MAX_PIXELS) -> Optional[Dict[str, np.ndarray]]:
    """
    Unified loader: GeoTIFF (.tif/.tiff) or legacy NumPy archive (.npz).
    Returns {band_name: float32 np.ndarray} or None on failure.

    Adaptive downsampling is applied automatically when the raster exceeds
    *max_pixels* per band — see load_tif_as_bands / _downsample_npz_bands.
    """
    suffix = raw_file.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return load_tif_as_bands(raw_file, max_pixels=max_pixels)
    elif suffix == ".npz":
        try:
            data = dict(np.load(raw_file))
            data = _downsample_npz_bands(data, max_pixels=max_pixels)
            logger.info(f"  Loaded {len(data)} arrays from {raw_file.name} (npz)")
            return data
        except Exception as exc:
            logger.error(f"  Failed to load {raw_file}: {exc}")
            return None
    else:
        logger.error(f"  Unsupported file format: {raw_file.suffix}")
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
            vmin, vmax = np.nanpercentile(arr, [2, 98])
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
            vmin, vmax = np.nanpercentile(arr[np.isfinite(arr)], [2, 98])
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
        fig = plt.figure(figsize=(16, 10))
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
                                filename: str = "05_patch_diagnostics.png"):
        """Distributions of per-patch LST statistics.

        Patches in this pipeline are position-only — pixel data is held in the
        raster dict, not in the patch object.  We therefore use the lightweight
        metadata (_lst_mean, _lst_std) that extract_patches stores on every patch
        instead of trying to access p["data"]["LST"].
        """
        if not patches:
            logger.warning("[Diagnostics] plot_patch_diagnostics: no patches.")
            return

        # Use pre-computed per-patch metadata (always present)
        means = np.array([p["_lst_mean"] for p in patches], dtype=np.float32)
        stds  = np.array([p["_lst_std"]  for p in patches], dtype=np.float32)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. LST mean distribution
        ax = axes[0]
        ax.hist(means, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(float(means.mean()), color="black", linestyle="--", lw=1.5,
                   label=f"μ={means.mean():.1f}°C")
        ax.axvline(20.0, color="red",  linestyle=":", lw=1.2, label="QC min 20°C")
        ax.axvline(58.0, color="red",  linestyle=":", lw=1.2, label="QC max 58°C")
        ax.set_xlabel("Patch LST Mean (°C)", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Patch LST Mean Distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. LST std distribution
        ax = axes[1]
        ax.hist(stds, bins=50, color="darkorange", edgecolor="white", alpha=0.85)
        ax.axvline(float(stds.mean()), color="black", linestyle="--", lw=1.5,
                   label=f"μ={stds.mean():.2f}°C")
        ax.axvline(0.3, color="red", linestyle=":", lw=1.2, label="QC min σ=0.3")
        ax.set_xlabel("Patch LST Std (°C)", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Patch LST Std Distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Mean vs Std scatter (quality overview)
        ax = axes[2]
        ax.hexbin(means, stds, gridsize=40, cmap="YlOrRd", mincnt=1)
        ax.axvline(20.0, color="red",  linestyle=":", lw=1.0, label="QC temp bounds")
        ax.axvline(58.0, color="red",  linestyle=":", lw=1.0)
        ax.axhline(0.3,  color="blue", linestyle=":", lw=1.0, label="QC min σ")
        ax.set_xlabel("Patch LST Mean (°C)", fontsize=9)
        ax.set_ylabel("Patch LST Std (°C)", fontsize=9)
        ax.set_title("LST Mean vs Std per Patch")
        ax.legend(fontsize=8)

        fig.suptitle(f"Patch Quality Diagnostics  (n={len(patches):,} patches)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
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
            bar_h = bar.get_height()
            txt_y = bar_h * 0.5 if bar_h != 0 else 0.01
            ax.text(bar.get_x() + bar.get_width() / 2, txt_y,
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
    # 10. Full pipeline summary dashboard
    # ------------------------------------------------------------------
    def plot_pipeline_summary(self, splits: dict, metadata: dict,
                               filename: str = "10_pipeline_summary.png"):
        """One-page summary dashboard of the entire preprocessing run.

        Args:
            splits:   dict with keys "y_train", "y_val", "y_test" (4D arrays,
                      N×H×W×1, normalised) and "split_counts" (dict of ints).
            metadata: the metadata dict written by preprocessing (patch_size,
                      n_channels, channel_order, temperature_range, qc_info, …).
        """
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

        colors = ["#2196F3", "#FF9800", "#4CAF50"]
        split_keys = [("y_train", "Train"), ("y_val", "Val"), ("y_test", "Test")]

        # ── 1. Sample count bar ───────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, 0])
        sc = splits.get("split_counts", metadata.get("split_counts", {}))
        labels = ["Train", "Val", "Test"]
        counts = [sc.get("train", 0), sc.get("val", 0), sc.get("test", 0)]
        bars = ax0.bar(labels, counts, color=colors, alpha=0.85)
        for bar, cnt in zip(bars, counts):
            ax0.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(counts) * 0.01,
                     f"{cnt:,}", ha="center", va="bottom", fontsize=9)
        ax0.set_title("Sample Counts per Split")
        ax0.set_ylabel("# Patches")
        ax0.grid(True, alpha=0.3, axis="y")

        # ── 2. LST distributions ──────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1])
        for (yk, label), col in zip(split_keys, colors):
            if yk in splits:
                vals = np.asarray(splits[yk]).ravel()
                ax1.hist(vals, bins=60, alpha=0.55, color=col, label=label, density=True)
        ax1.set_xlabel("LST (normalised)")
        ax1.set_ylabel("Density")
        ax1.set_title("LST Distribution per Split")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── 3. QC info ────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis("off")
        qc = metadata.get("qc_info", {})
        tr = metadata.get("temperature_range", {})
        fi = metadata.get("fusion_info", {})
        info_lines = [
            "Pipeline Summary",
            "─" * 26,
            f"Strategy:   {fi.get('fusion_strategy','N/A')}",
            f"Scanned:    {qc.get('scanned', 'N/A'):,}" if isinstance(qc.get('scanned'), int) else f"Scanned:    {qc.get('scanned','N/A')}",
            f"Rejected:   {qc.get('rejected', 'N/A'):,}" if isinstance(qc.get('rejected'), int) else f"Rejected:   {qc.get('rejected','N/A')}",
            f"Accepted:   {qc.get('accepted', 'N/A'):,}" if isinstance(qc.get('accepted'), int) else f"Accepted:   {qc.get('accepted','N/A')}",
            "─" * 26,
            f"LST mean:   {tr.get('mean', float('nan')):.2f}°C",
            f"LST std:    {tr.get('std',  float('nan')):.2f}°C",
            f"LST range:  [{tr.get('min', float('nan')):.1f}, {tr.get('max', float('nan')):.1f}]°C",
            "─" * 26,
            f"Patches:    {metadata.get('n_samples','N/A'):,}" if isinstance(metadata.get('n_samples'), int) else f"Patches:    {metadata.get('n_samples','N/A')}",
            f"Channels:   {metadata.get('n_channels','N/A')}",
            f"Patch size: {metadata.get('patch_size','N/A')} px",
        ]
        ax2.text(0.05, 0.95, "\n".join(info_lines), transform=ax2.transAxes,
                 fontsize=8, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4ff", alpha=0.8))
        ax2.set_title("Run Info", fontweight="bold")

        # ── 4. Per-split mean ± std bar ───────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        s_labels, s_means, s_stds = [], [], []
        for yk, label in split_keys:
            if yk in splits:
                v = np.asarray(splits[yk]).ravel()
                s_labels.append(label)
                s_means.append(float(v.mean()))
                s_stds.append(float(v.std()))
        if s_labels:
            errbars = ax3.bar(s_labels, s_means, yerr=s_stds,
                              color=colors[:len(s_labels)], capsize=6, alpha=0.8)
            ax3.set_ylabel("Mean LST ± Std (normalised)")
            ax3.set_title("Split LST Statistics")
            ax3.grid(True, alpha=0.3, axis="y")

        # ── 5. Patch LST variance per split ──────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        for (yk, label), col in zip(split_keys, colors):
            if yk in splits:
                arr = np.asarray(splits[yk])    # (N, H, W, 1)
                patch_stds = arr.reshape(arr.shape[0], -1).std(axis=1)
                ax4.hist(patch_stds, bins=50, alpha=0.55, color=col,
                         label=f"{label} μ={patch_stds.mean():.3f}", density=True)
        ax4.set_xlabel("Per-patch LST Std (normalised)")
        ax4.set_ylabel("Density")
        ax4.set_title("Patch LST Variance per Split")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # ── 6. Channel order ─────────────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")
        ch = metadata.get("channel_order", [])
        ch_text = "Input channels:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(ch))
        ax5.text(0.05, 0.95, ch_text, transform=ax5.transAxes,
                 fontsize=9, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", alpha=0.9))
        ax5.set_title("Channel Order", fontweight="bold")

        fig.suptitle("Preprocessing Pipeline – Summary Dashboard",
                     fontsize=15, fontweight="bold")
        plt.tight_layout()
        self._save(fig, filename)
        logger.info(f"[Diagnostics] ✅ Pipeline summary saved → {self.out / filename}")

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
            satellite_type: 'landsat'
        """
        self.satellite_type = satellite_type
        self.config = LANDSAT_CONFIG
        
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
        
        # Detect whether ST_B10 is already in Kelvin (GeoTIFF, scale applied
        # at export) or raw Collection-2 DN integers (legacy .npz).
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
            "mean": float(np.nanmean(lst_clean)) if valid_pixels > 0 else np.nan,
            "std": float(np.nanstd(lst_clean)) if valid_pixels > 0 else np.nan,
            "min": float(np.nanmin(lst_clean)) if valid_pixels > 0 else np.nan,
            "max": float(np.nanmax(lst_clean)) if valid_pixels > 0 else np.nan
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

            # Detect DN (legacy .npz) vs pre-scaled reflectance (GeoTIFF).
            # Pre-scaled values are in [0, 1]; raw DN medians are ~5 000–30 000.
            _s = red[np.isfinite(red)]
            if _s.size > 0 and float(np.nanmedian(_s)) > 2.0:
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
            del _s
            # GeoTIFF path: already in [0, 1] — no conversion needed


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
        Process a single raw data file
        
        Args:
            raw_file: Path to raw .npz file
            calculate_lst: Whether to calculate LST from thermal band
            
        Returns:
            Dictionary of processed features or None if failed
        """
        logger.info(f"Processing: {raw_file.name}")
        
        try:
            # Load raw data — supports both .tif (new) and .npz (legacy)
            raw_data = load_raw_file(raw_file)
            if raw_data is None:
                return None
            logger.info(f"  Loaded {len(raw_data)} raw bands")
            
            # Extract band data
            bands = {}
            for key, arr in raw_data.items():
                if (key.startswith('SR_B') or key.startswith('B')) and key not in ['BSI', 'B11', 'B12']:
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
                    min_temp: float = 20.0,         # Patch-mean lower bound (°C)
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
        
        logger.info(f"  Extracted {len(patches)} valid patches")
        logger.info(f"  Filtered by temperature: {filtered_by_temp} patches")
        logger.info(f"  Filtered by variance: {filtered_by_variance} patches")

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
            total_candidates = (height // stride) * (width // stride)
            accept_rate = len(patches) / max(total_candidates, 1) * 100
            logger.info(
                f"  Patch acceptance rate: {accept_rate:.1f}%  "
                f"(rejected: temp={filtered_by_temp}, var={filtered_by_variance})"
            )
        # ──────────────────────────────────────────────────────────────

        return patches
    
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
        patch_size    = patches[0]["data"]["LST"].shape[0] if not position_only \
                        else raster_data["LST"][
                            patches[0]["position"][0]:patches[0]["position"][0]+64,
                            patches[0]["position"][1]:patches[0]["position"][1]+64,
                        ].shape[0]
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

        # ── Pass 1: lightweight QC scan ───────────────────────────────────────
        valid_indices = []
        for idx, patch in enumerate(patches):
            if position_only:
                lst_mean = patch["_lst_mean"]
                lst_std  = patch["_lst_std"]
                if lst_std < 0.3 or lst_mean < 20.0 or lst_mean > 58.0:
                    continue
                valid_indices.append(idx)
            else:
                lst_patch = patch["data"]["LST"].astype(np.float32)
                fin       = np.isfinite(lst_patch)
                if fin.mean() < 0.95:
                    continue
                vt = lst_patch[fin]
                if vt.size == 0 or not (20.0 <= float(vt.mean()) <= 58.0):
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

        for out_idx, src_idx in enumerate(valid_indices):
            patch = patches[src_idx]
            r, c  = patch["position"]

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

        # ── In-place NaN fill ─────────────────────────────────────────────────
        for s in range(n_samples):
            for ch in range(n_channels):
                sl = X[s, :, :, ch]
                nm = ~np.isfinite(sl)
                if nm.any():
                    sl[nm] = float(sl[~nm].mean()) if (~nm).any() else 0.0
            sl = y[s, :, :, 0]
            nm = ~np.isfinite(sl)
            if nm.any():
                sl[nm] = float(sl[~nm].mean()) if (~nm).any() else 35.0

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
            "sample_weights": sample_weights,
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

            if X_key in splits:
                np.save(split_dir / "X.npy", splits[X_key])
                del splits[X_key]
            if y_key in splits:
                np.save(split_dir / "y.npy", splits[y_key])
                del splits[y_key]
            if d_key in splits:
                np.save(split_dir / "dates.npy", splits[d_key])
                del splits[d_key]

            gc.collect()
            logger.info(f"  ✓ Saved and freed {split_name} split → {split_dir}")

        import json
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            metadata_clean = {}
            for k, v in metadata.items():
                if isinstance(v, (np.integer, np.floating)):
                    metadata_clean[k] = float(v)
                elif isinstance(v, np.ndarray):
                    metadata_clean[k] = v.tolist()
                else:
                    metadata_clean[k] = v
            json.dump(metadata_clean, f, indent=2)

        if norm_stats is not None:
            stats_file = output_dir / "normalization_stats.json"
            with open(stats_file, "w") as f:
                json.dump(norm_stats, f, indent=2)
            logger.info(f"✅ Saved normalization stats to {stats_file}")

        logger.info(f"✅ Dataset saved to {output_dir}")

class EnhancedDatasetCreator(DatasetCreator):
    """Extended DatasetCreator with better splitting"""
    
    def create_stratified_split(self, X: np.ndarray, y: np.ndarray,
                               dates: np.ndarray,
                               split_ratios: Tuple[float, float, float] = (0.65, 0.15, 0.20),
                               random_seed: int = 42) -> Dict:
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
            "dates_test": dates_test
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
    """Main preprocessing pipeline - processes Landsat 8 and 9 data"""
    logger.info("="*70)
    logger.info("LANDSAT PREPROCESSING PIPELINE")
    logger.info("="*70)
    
    # Define input directories
    raw_data_dir = RAW_DATA_DIR
    landsat_dir = raw_data_dir / "landsat"
    # Check what data is available
    has_landsat = landsat_dir.exists()
    
    if not has_landsat:
        logger.error(f"No raw data found in {raw_data_dir}")
        logger.error("Please run earth_engine_loader.py first to download data")
        return
    
    logger.info(f"Data availability:")
    logger.info(f"  Landsat: {'✓' if has_landsat else '✗'}")
    
    # Initialize components
    landsat_preprocessor = SatellitePreprocessor(satellite_type="landsat")
    dataset_creator = DatasetCreator()
    feature_engineer = FeatureEngineer()

    # ── DIAGNOSTICS: initialise visualisation helper ────────────────────────
    # Diagnostics go to the project dataset dir (same as output)
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
            list(landsat_dir.glob("landsat_*.tif")) +   # lower-case variant
            list(landsat_dir.glob("landsat_*.npz"))     # legacy npz
        )
        logger.info(f"Found {len(landsat_files)} Landsat files")
        
        for raw_file in landsat_files:
            # Handles both Landsat_YYYY_MM.tif and landsat_YYYY_MM.npz
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

            # ── DIAGNOSTIC PLOTS (first file only) ────────────────────
            if len(landsat_processed) == 1:
                try:
                    raw_data_for_plot = load_raw_file(raw_file) or {}
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
    first_fused_diag = True

    def _fuse_extract_free(fused_data: Dict, date, label: str) -> None:
        """Extract position-only patches, store raster ref, free bands later."""
        # UHI_STRIDE: controls patch overlap.  Default 56 gives ~12% overlap
        # between 64-px patches — enough variety without massive redundancy.
        # stride=24 (old default) produced ~988 K patches; stride=56 → ~50-150 K.
        # UHI_MAX_PATCHES_PER_SCENE caps per-scene contribution (0 = no cap).
        _stride    = int(_os.environ.get("UHI_STRIDE", 56))
        _scene_cap = int(_os.environ.get("UHI_MAX_PATCHES_PER_SCENE", 3000))

        patches = dataset_creator.extract_patches(
            fused_data,
            patch_size=64, stride=_stride, min_valid_ratio=0.95,
            min_variance=0.3, min_temp=20.0, max_temp=58.0,
        )
        # Spatially subsample if over the per-scene cap
        if _scene_cap > 0 and len(patches) > _scene_cap:
            step = max(1, len(patches) // _scene_cap)
            patches = patches[::step][:_scene_cap]

        for p in patches:
            p["date"] = date
        all_patches.extend(patches)
        all_dates.append(date)
        all_rasters.append((fused_data, patches))
        logger.info(f"  {label}: {len(patches)} patches "
                    f"(running total: {len(all_patches)})")

    if has_landsat and len(landsat_processed) > 0:
        logger.info("Processing Landsat data...")
        n_ls = len(landsat_processed)
        for li, (timestamp, ls_data) in enumerate(landsat_processed):
            if all(k in ls_data for k in ("NDVI", "NDBI", "MNDWI")):
                ls_data["impervious_surface"] = feature_engineer.calculate_impervious_surface(
                    ls_data["NDVI"], ls_data["NDBI"], ls_data["MNDWI"]
                )
            _fuse_extract_free(ls_data, timestamp, f"Landsat {li+1}/{n_ls}")
        del landsat_processed
        gc.collect()

    if not all_patches:
        logger.error("No patches extracted")
        return
    
    # Patch quality diagnostic (before patches are freed)
    try:
        diag.plot_patch_diagnostics(all_patches, filename="05_patch_diagnostics.png")
    except Exception as _e:
        logger.warning(f"[Diagnostics] patch plot failed: {_e}")

    # ── Step 4: QC-inline stream → project data dir, no temp files ───────────
    # ─────────────────────────────────────────────────────────────────────────
    # DESIGN:
    #  • Phase 4a: lightweight QC scan using _lst_mean/_lst_std already stored
    #    by extract_patches — no raster pixels are read here.  Counts accepted
    #    patches and prints disk estimate before touching the drive.
    #  • Phase 4b: only QC-passing patches are written to disk (no rejected data
    #    ever hits the drive).  Output goes to PROCESSED_DATA_DIR/cnn_dataset/_raw
    #    inside the project tree — not to OneDrive or a temp folder.
    #  • All subsequent passes (norm stats, split, normalise) are chunked so
    #    peak RAM stays O(CHUNK × patch_bytes).
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Stream patches to disk (QC-inline, project data dir)")
    logger.info("="*70)

    import json as _json, shutil as _shutil

    channel_order = [
        "SR_B4", "SR_B5", "SR_B6", "SR_B7",
        "NDVI", "NDBI", "MNDWI", "BSI", "UI", "albedo",
    ]
    n_channels = len(channel_order)
    patch_size  = 64
    temporal_features = feature_engineer.encode_temporal_features(all_dates[0])

    # Chunk size: patches held in RAM at once.
    # Default 1024 ≈ ~160 MiB.  Lower with:  set UHI_CHUNK=256
    CHUNK = int(_os.environ.get("UHI_CHUNK", 1024))

    # All output goes into the project data directory — same location as before,
    # but now memmaps also live here so no writes hit OneDrive quota or /tmp.
    output_dataset_dir = PROCESSED_DATA_DIR / "cnn_dataset"
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Output directory: {output_dataset_dir}")

    # ── 4a. QC scan — count accepted patches, estimate disk ───────────────────
    logger.info("  Phase 4a: QC scan (no disk writes)...")
    valid_patch_info = []   # (raster_id, position, date, lst_mean)
    raster_by_id     = {}   # id(raster) → raster dict

    for raster_data, raster_patches in all_rasters:
        rid = id(raster_data)
        raster_by_id[rid] = raster_data
        for p in raster_patches:
            lm, ls = p["_lst_mean"], p["_lst_std"]
            if ls < 0.3 or not (20.0 <= lm <= 58.0):
                continue
            valid_patch_info.append((rid, p["position"], p["date"], lm))

    n_valid    = len(valid_patch_info)
    n_scanned  = len(all_patches)
    n_rejected = n_scanned - n_valid

    bytes_raw = n_valid * patch_size * patch_size * (n_channels + 1) * 4
    logger.info(f"  QC: {n_scanned:,} scanned  {n_rejected:,} rejected  "
                f"{n_valid:,} accepted")
    logger.info(f"  Raw memmap size estimate: {bytes_raw / 1024**3:.1f} GiB")

    if n_valid == 0:
        logger.error("No QC-passing patches — aborting.")
        return

    # ── 4b. Write accepted patches to project _raw dir ────────────────────────
    logger.info("  Phase 4b: writing accepted patches...")
    raw_dir = output_dataset_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    X_path = raw_dir / "X_all.dat"
    y_path = raw_dir / "y_all.dat"
    X_mm = np.memmap(X_path, dtype=np.float32, mode="w+",
                     shape=(n_valid, patch_size, patch_size, n_channels))
    y_mm = np.memmap(y_path, dtype=np.float32, mode="w+",
                     shape=(n_valid, patch_size, patch_size, 1))

    valid_dates = np.empty(n_valid, dtype=object)
    valid_lm    = np.empty(n_valid, dtype=np.float32)
    write_idx   = 0

    for rid, (row, col), date, lm in valid_patch_info:
        rd = raster_by_id[rid]
        for ch_idx, feat in enumerate(channel_order):
            arr = rd.get(feat)
            if arr is not None:
                sl = arr[row:row+patch_size, col:col+patch_size].astype(np.float32)
                nm = ~np.isfinite(sl)
                if nm.any():
                    sl[nm] = float(sl[~nm].mean()) if (~nm).any() else 0.0
                X_mm[write_idx, :, :, ch_idx] = sl
        lst_arr = rd.get("LST")
        if lst_arr is not None:
            sl = lst_arr[row:row+patch_size, col:col+patch_size].astype(np.float32)
            nm = ~np.isfinite(sl)
            if nm.any():
                sl[nm] = float(sl[~nm].mean()) if (~nm).any() else 35.0
            np.clip(sl, 10.0, 65.0, out=sl)
            y_mm[write_idx, :, :, 0] = sl
        valid_dates[write_idx] = date
        valid_lm[write_idx]    = lm
        write_idx += 1
        if write_idx % 10_000 == 0:
            logger.info(f"    Written {write_idx:,}/{n_valid:,} patches...")

    X_mm.flush(); y_mm.flush()

    for rd in raster_by_id.values():
        rd.clear()
    del all_rasters, all_patches, raster_by_id, valid_patch_info
    gc.collect()
    logger.info(f"  Written {write_idx:,} patches → {raw_dir}")

    # ── 4c. LST temperature stats (chunked) ────────────────────────────────────
    y_sum = 0.0; y_sq = 0.0; y_min = np.inf; y_max = -np.inf
    n_pix = n_valid * patch_size * patch_size
    for start in range(0, n_valid, CHUNK):
        end = min(start + CHUNK, n_valid)
        yc  = np.asarray(y_mm[start:end]).ravel().astype(np.float64)
        y_sum += float(yc.sum()); y_sq += float((yc**2).sum())
        y_min = min(y_min, float(yc.min())); y_max = max(y_max, float(yc.max()))
        del yc; gc.collect()
    y_mean = y_sum / n_pix
    y_std  = float(np.sqrt(max(y_sq / n_pix - y_mean**2, 0.0)))
    logger.info(f"  LST: [{y_min:.2f}, {y_max:.2f}]°C  "
                f"mean={y_mean:.2f}  std={y_std:.2f}")

    # ── 4d. Sample weights ─────────────────────────────────────────────────────
    n_wb  = 10
    be_   = np.linspace(valid_lm.min(), valid_lm.max() + 1e-6, n_wb + 1)
    bids_ = np.clip(np.digitize(valid_lm, be_) - 1, 0, n_wb - 1)
    bc_   = np.maximum(np.bincount(bids_, minlength=n_wb).astype(np.float32), 1)
    rw_   = 1.0 / bc_[bids_]
    sample_weights = (rw_ / rw_.mean()).astype(np.float32)
    del valid_lm; gc.collect()

    # ── Step 5: Index-only stratified split ────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Create train/val/test splits (index-only)")
    logger.info("="*70)

    dates_dt = pd.to_datetime(valid_dates)
    seasons  = (dates_dt.month % 12 // 3).values

    # Spatial clustering on per-channel means (chunked)
    logger.info("  Computing spatial clustering features...")
    n_sp = min(3, n_channels)
    sp_feats = np.zeros((n_valid, n_sp), dtype=np.float32)
    for ch in range(n_sp):
        for start in range(0, n_valid, CHUNK):
            end = min(start + CHUNK, n_valid)
            sp_feats[start:end, ch] = np.asarray(
                X_mm[start:end, :, :, ch]).mean(axis=(1, 2))

    from sklearn.cluster import KMeans
    sp_blocks = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(sp_feats)
    del sp_feats; gc.collect()

    # LST variance bins (chunked)
    logger.info("  Computing LST variance bins...")
    lst_stds = np.empty(n_valid, dtype=np.float32)
    for start in range(0, n_valid, CHUNK):
        end = min(start + CHUNK, n_valid)
        lst_stds[start:end] = (np.asarray(y_mm[start:end, :, :, 0])
                               .reshape(end - start, -1).std(axis=1))
    lst_bins = pd.qcut(lst_stds, q=4, labels=False, duplicates="drop").astype(int)
    del lst_stds; gc.collect()

    strata = seasons * 5 * 4 + sp_blocks * 4 + lst_bins

    from collections import Counter
    counts = Counter(strata)
    keep   = np.array([counts[s] >= 2 for s in strata])
    if not keep.all():
        removed = int((~keep).sum())
        logger.warning(f"  Removing {removed} singleton-strata samples")
        keep_idx       = np.where(keep)[0]
        strata         = strata[keep]
        seasons        = seasons[keep]
        sp_blocks      = sp_blocks[keep]
        valid_dates    = valid_dates[keep]
        sample_weights = sample_weights[keep]

        # Remap memmaps (chunked copy)
        n_kept  = int(keep.sum())
        Xk_path = raw_dir / "X_kept.dat"
        yk_path = raw_dir / "y_kept.dat"
        Xk_mm = np.memmap(Xk_path, dtype=np.float32, mode="w+",
                          shape=(n_kept, patch_size, patch_size, n_channels))
        yk_mm = np.memmap(yk_path, dtype=np.float32, mode="w+",
                          shape=(n_kept, patch_size, patch_size, 1))
        for out_s in range(0, n_kept, CHUNK):
            out_e    = min(out_s + CHUNK, n_kept)
            src_rows = keep_idx[out_s:out_e]
            Xk_mm[out_s:out_e] = X_mm[src_rows]
            yk_mm[out_s:out_e] = y_mm[src_rows]
        Xk_mm.flush(); yk_mm.flush()
        X_mm._mmap.close(); y_mm._mmap.close()
        _os.remove(X_path); _os.remove(y_path)
        X_mm, y_mm     = Xk_mm, yk_mm
        X_path, y_path = Xk_path, yk_path
        n_valid = n_kept
        gc.collect()

    train_r, val_r, test_r = 0.65, 0.15, 0.20
    tv_idx, test_idx = train_test_split(
        np.arange(n_valid), test_size=test_r, stratify=strata, random_state=42)
    train_idx, val_idx = train_test_split(
        tv_idx, test_size=val_r / (train_r + val_r),
        stratify=strata[tv_idx], random_state=42)
    logger.info(f"  Train:{len(train_idx):,}  Val:{len(val_idx):,}  "
                f"Test:{len(test_idx):,}")

    # ── Step 5.5: Normalisation stats (chunked, train only) ───────────────────
    logger.info("\n" + "="*70)
    logger.info("STEP 5.5: Compute normalisation statistics")
    logger.info("="*70)

    n_train = len(train_idx)
    n_px_tr = n_train * patch_size * patch_size
    f_sum   = np.zeros(n_channels, dtype=np.float64)
    f_sq    = np.zeros(n_channels, dtype=np.float64)
    t_sum   = 0.0; t_sq = 0.0

    for start in range(0, n_train, CHUNK):
        end    = min(start + CHUNK, n_train)
        idx_ch = train_idx[start:end]
        Xc     = np.asarray(X_mm[idx_ch], dtype=np.float64)
        yc     = np.asarray(y_mm[idx_ch], dtype=np.float64)
        f_sum += Xc.sum(axis=(0, 1, 2))
        f_sq  += (Xc**2).sum(axis=(0, 1, 2))
        t_sum += float(yc.sum()); t_sq += float((yc**2).sum())
        del Xc, yc; gc.collect()

    f_mean = f_sum / n_px_tr
    f_std  = np.sqrt(np.maximum(f_sq / n_px_tr - f_mean**2, 1e-10))
    t_mean = t_sum / n_px_tr
    t_std  = float(np.sqrt(max(t_sq / n_px_tr - t_mean**2, 1e-10)))

    norm_stats = {
        "features": {"mean": f_mean.tolist(), "std": f_std.tolist()},
        "target":   {"mean": float(t_mean),   "std": float(t_std)},
        "channel_order": channel_order,
    }
    with open(output_dataset_dir / "normalization_stats.json", "w") as _f:
        _json.dump(norm_stats, _f, indent=2)
    logger.info(f"  Feature mean range: [{f_mean.min():.4f}, {f_mean.max():.4f}]")
    logger.info(f"  Feature std  range: [{f_std.min():.4f},  {f_std.max():.4f}]")
    logger.info(f"  Target: mean={t_mean:.4f}  std={t_std:.4f}")
    logger.info(f"  ✅ Norm stats saved")

    f_mean_f32 = f_mean.astype(np.float32)
    f_std_f32  = f_std.astype(np.float32)
    t_mean_f32 = np.float32(t_mean)
    t_std_f32  = np.float32(t_std)

    # ── Step 5.6: Normalise + save splits (chunked) ───────────────────────────
    logger.info("\n" + "="*70)
    logger.info("STEP 5.6: Normalise and save splits")
    logger.info("="*70)

    def _save_split(name: str, indices: np.ndarray, sw=None):
        n         = len(indices)
        split_dir = output_dataset_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        Xo = np.memmap(split_dir / "X.npy", dtype=np.float32, mode="w+",
                       shape=(n, patch_size, patch_size, n_channels))
        yo = np.memmap(split_dir / "y.npy", dtype=np.float32, mode="w+",
                       shape=(n, patch_size, patch_size, 1))
        xm_acc = np.zeros(n_channels, dtype=np.float64)
        ym_acc = 0.0; ys_acc = 0.0
        for start in range(0, n, CHUNK):
            end    = min(start + CHUNK, n)
            idx_ch = indices[start:end]
            Xc = (np.asarray(X_mm[idx_ch]) - f_mean_f32) / f_std_f32
            yc = (np.asarray(y_mm[idx_ch]) - t_mean_f32) / t_std_f32
            Xo[start:end] = Xc
            yo[start:end] = yc
            k = end - start
            xm_acc += Xc.mean(axis=(0, 1, 2)).astype(np.float64) * k
            ym_acc += float(yc.mean()) * k
            ys_acc += float(yc.std())  * k
            del Xc, yc; gc.collect()
        Xo.flush(); yo.flush()
        np.save(split_dir / "dates.npy", valid_dates[indices])
        if sw is not None:
            np.save(split_dir / "weights_train.npy", sw)
            logger.info(f"    Weights: [{sw.min():.3f}, {sw.max():.3f}]")
        xm = xm_acc / n; ym = ym_acc / n; ys = ys_acc / n
        logger.info(f"  ✓ {name:5s}: {n:,} samples | "
                    f"X̄≈{xm.mean():.3f}  ȳ≈{ym:.3f}  σy≈{ys:.3f}")
        if name == "train" and not (-0.15 < float(xm.mean()) < 0.15):
            logger.warning("⚠️  X_train mean far from 0 — check norm stats")

    _save_split("train", train_idx, sw=sample_weights[train_idx])
    _save_split("val",   val_idx)
    _save_split("test",  test_idx)

    # ── Metadata ───────────────────────────────────────────────────────────────
    metadata = {
        "n_samples":    n_valid,
        "patch_size":   patch_size,
        "n_channels":   n_channels,
        "channel_order": channel_order,
        "temporal_features": temporal_features,
        "temperature_range": {
            "min": float(y_min), "max": float(y_max),
            "mean": float(y_mean), "std": float(y_std),
        },
        "fusion_info": {
            "fusion_strategy":   "landsat-only",
            "landsat_available": has_landsat,
            "total_samples":     n_valid,
        },
        "split_counts": {
            "train": int(len(train_idx)),
            "val":   int(len(val_idx)),
            "test":  int(len(test_idx)),
        },
        "qc_info": {
            "scanned":  int(n_scanned),
            "rejected": int(n_rejected),
            "accepted": int(n_valid),
        },
    }
    with open(output_dataset_dir / "metadata.json", "w") as _f:
        _json.dump(metadata, _f, indent=2)

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    _MD = 5_000   # max patches to load into RAM for plotting

    # 06 — Split LST distributions
    try:
        def _ysamp(idx):
            chosen = idx[:_MD]
            return np.asarray(y_mm[chosen])   # normalised, shape (n, 64, 64, 1)
        diag.plot_split_distributions(
            {"y_train": _ysamp(train_idx),
             "y_val":   _ysamp(val_idx),
             "y_test":  _ysamp(test_idx)},
            filename="06_split_distributions.png")
        logger.info("[Diagnostics] 06_split_distributions saved.")
    except Exception as _e:
        logger.warning(f"[Diagnostics] split distribution plot failed: {_e}")

    # 07 — Normalisation diagnostics (raw vs normalised, sample of training data)
    try:
        _samp_idx = train_idx[:min(_MD, len(train_idx))]
        _X_raw_s  = np.asarray(X_mm[_samp_idx])            # unnormalised
        _y_raw_s  = np.asarray(y_mm[_samp_idx])
        _X_norm_s = (_X_raw_s - f_mean_f32) / f_std_f32    # normalised
        _y_norm_s = (_y_raw_s - t_mean_f32) / t_std_f32
        diag.plot_normalization_diagnostics(
            _X_raw_s, _X_norm_s, _y_raw_s, _y_norm_s,
            channel_names=channel_order,
            filename="07_normalization.png")
        logger.info("[Diagnostics] 07_normalization saved.")
        del _X_raw_s, _y_raw_s, _X_norm_s, _y_norm_s; gc.collect()
    except Exception as _e:
        logger.warning(f"[Diagnostics] normalization plot failed: {_e}")

    # 08 — Channel correlation heatmap (sample of training data, normalised)
    try:
        _samp_idx2 = train_idx[:min(_MD, len(train_idx))]
        _X_corr = np.asarray(X_mm[_samp_idx2])
        _X_corr = (_X_corr - f_mean_f32) / f_std_f32
        diag.plot_channel_correlation(
            _X_corr, channel_names=channel_order,
            filename="08_channel_correlation.png")
        logger.info("[Diagnostics] 08_channel_correlation saved.")
        del _X_corr; gc.collect()
    except Exception as _e:
        logger.warning(f"[Diagnostics] channel correlation plot failed: {_e}")

    # 10 — Pipeline summary dashboard
    try:
        _splits_for_summary = {
            "y_train":     np.asarray(y_mm[train_idx[:_MD]]),
            "y_val":       np.asarray(y_mm[val_idx[:_MD]]),
            "y_test":      np.asarray(y_mm[test_idx[:_MD]]),
            "split_counts": {
                "train": int(len(train_idx)),
                "val":   int(len(val_idx)),
                "test":  int(len(test_idx)),
            },
        }
        diag.plot_pipeline_summary(
            _splits_for_summary, metadata,
            filename="10_pipeline_summary.png")
        logger.info("[Diagnostics] 10_pipeline_summary saved.")
        del _splits_for_summary; gc.collect()
    except Exception as _e:
        logger.warning(f"[Diagnostics] pipeline summary plot failed: {_e}")

    # ── Clean up raw memmaps ───────────────────────────────────────────────────
    try:
        X_mm._mmap.close(); y_mm._mmap.close()
        _shutil.rmtree(raw_dir, ignore_errors=True)
        logger.info(f"  Cleaned up raw memmaps from {raw_dir}")
    except Exception as _e:
        logger.warning(f"  Could not remove raw memmaps: {_e}")
    gc.collect()



    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("✓ LANDSAT PREPROCESSING COMPLETE")
    logger.info("="*70)
    logger.info(f"  Patches: {n_valid:,} accepted / {n_scanned:,} scanned "
                f"({n_rejected:,} rejected by QC)")
    logger.info(f"  Train / Val / Test: "
                f"{len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}")
    logger.info(f"  Dataset saved to: {output_dataset_dir}")
    logger.info(f"  Norm stats:       ✅")
    logger.info("="*70)
    logger.info("\nNext step: Run model training")
if __name__ == "__main__":
    main()