"""
UHI Analysis - Calculate UHI intensity, detect hotspots, and generate reports
Enhanced with comprehensive diagnostics and matplotlib visualizations
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
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


def _safe_kde(data: np.ndarray, x_grid: np.ndarray):
    """Return KDE values on x_grid, or None if data has near-zero variance or contains inf/NaN."""
    flat = np.asarray(data, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.std() < 1e-6 or len(flat) < 2:
        logger.warning("_safe_kde: near-zero variance or insufficient data — skipping KDE")
        return None
    x_grid = np.asarray(x_grid, dtype=np.float64)
    x_grid = x_grid[np.isfinite(x_grid)]
    if len(x_grid) == 0:
        return None
    try:
        return stats.gaussian_kde(flat)(x_grid)
    except Exception as _e:
        logger.warning(f"_safe_kde: gaussian_kde failed ({_e}) — skipping KDE")
        return None


def _safe_kde_2d(x: np.ndarray, y: np.ndarray):
    """Return 2-D KDE density values, or None on degenerate input."""
    try:
        if np.asarray(x).std() < 1e-6 or np.asarray(y).std() < 1e-6:
            raise ValueError("Near-zero variance")
        xy = np.vstack([x, y])
        return stats.gaussian_kde(xy)(xy)
    except Exception as _e:
        logger.warning(f"_safe_kde_2d: 2-D KDE failed ({_e}) — falling back to uniform density")
        return None


# ─── Shared style ────────────────────────────────────────────────────────────
_THERMAL_COLORS = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                   '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
_THERMAL_CMAP   = LinearSegmentedColormap.from_list('thermal', _THERMAL_COLORS, N=256)
_UHI_CAT_COLORS = ['#3288bd', '#99d594', '#fee08b', '#fc8d59', '#d53e4f']
_UHI_CAT_LABELS = ['No UHI / Cooling', 'Weak (0–1 °C)',
                   'Moderate (1–2 °C)', 'Strong (2–3 °C)', 'Very Strong (>3 °C)']

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
})


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic helper
# ══════════════════════════════════════════════════════════════════════════════

def _diag_header(title: str, width: int = 70) -> None:
    bar = "═" * width
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def _diag_array(name: str, arr: np.ndarray) -> None:
    """Log rich per-array statistics."""
    flat = arr[~np.isnan(arr)]
    q1, median, q3 = np.percentile(flat, [25, 50, 75])
    # Guard against catastrophic cancellation on near-constant arrays
    if flat.std() > 1e-6:
        skew = float(stats.skew(flat))
        kurt = float(stats.kurtosis(flat))
    else:
        skew, kurt = 0.0, 0.0
    logger.info(f"  {name}:")
    logger.info(f"    shape={arr.shape}  dtype={arr.dtype}  NaN={np.isnan(arr).sum()}")
    logger.info(f"    min={flat.min():.4f}  max={flat.max():.4f}  mean={flat.mean():.4f}  std={flat.std():.4f}")
    logger.info(f"    Q1={q1:.4f}  median={median:.4f}  Q3={q3:.4f}  IQR={q3-q1:.4f}")
    logger.info(f"    skewness={skew:.4f}  kurtosis={kurt:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# UHIAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class UHIAnalyzer:
    """Analyze UHI intensity and patterns with rich diagnostics."""

    def __init__(self, lst_map: np.ndarray, coords: Optional[np.ndarray] = None):
        self.lst_map = lst_map
        self.coords  = coords
        self.uhi_map = None
        self.reference_temps: Dict[str, float] = {}

        _diag_header("UHI ANALYZER — INITIALISATION")
        _diag_array("LST map", lst_map)
        logger.info(f"  coords provided: {coords is not None}")

    # ── reference areas ──────────────────────────────────────────────────────

    def define_reference_areas(self, urban_mask: np.ndarray,
                               rural_mask: np.ndarray) -> Dict[str, float]:
        _diag_header("DEFINE REFERENCE AREAS")

        T_urban_vals = self.lst_map[urban_mask]
        T_rural_vals = self.lst_map[rural_mask]

        logger.info(f"  Urban pixels : {urban_mask.sum():,}  ({urban_mask.mean()*100:.2f}%)")
        logger.info(f"  Rural pixels : {rural_mask.sum():,}  ({rural_mask.mean()*100:.2f}%)")

        _diag_array("Urban LST", T_urban_vals.reshape(-1))
        _diag_array("Rural LST", T_rural_vals.reshape(-1))

        T_urban = float(T_urban_vals.mean())
        T_rural = float(T_rural_vals.mean())

        # Welch t-test
        t_stat, p_val = stats.ttest_ind(T_urban_vals, T_rural_vals, equal_var=False)
        logger.info(f"  Welch t-test: t={t_stat:.4f}  p={p_val:.2e}  "
                    f"{'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'} (α=0.05)")

        self.reference_temps = {
            "T_urban":    T_urban,
            "T_rural":    T_rural,
            "T_diff":     T_urban - T_rural,
            "urban_std":  float(T_urban_vals.std()),
            "rural_std":  float(T_rural_vals.std()),
            "t_stat":     float(t_stat),
            "p_value":    float(p_val),
        }
        return self.reference_temps

    # ── UHI intensity ─────────────────────────────────────────────────────────

    def calculate_uhi_intensity(self, rural_reference: Optional[float] = None) -> np.ndarray:
        _diag_header("CALCULATE UHI INTENSITY")

        if rural_reference is None:
            if "T_rural" not in self.reference_temps:
                raise ValueError("Call define_reference_areas() first.")
            rural_reference = self.reference_temps["T_rural"]

        logger.info(f"  Rural reference baseline : {rural_reference:.4f} °C")
        self.uhi_map = self.lst_map - rural_reference

        _diag_array("UHI intensity", self.uhi_map)
        positive = self.uhi_map[self.uhi_map > 0]
        logger.info(f"  Pixels > 0 °C  : {len(positive):,}  "
                    f"({len(positive)/self.uhi_map.size*100:.2f}%)")
        logger.info(f"  Pixels > 2 °C  : {(self.uhi_map > 2).sum():,}")
        logger.info(f"  Pixels > 3 °C  : {(self.uhi_map > 3).sum():,}")

        return self.uhi_map

    # ── classification ────────────────────────────────────────────────────────

    def classify_uhi_intensity(self) -> Tuple[np.ndarray, Dict[str, int]]:
        if self.uhi_map is None:
            raise ValueError("Call calculate_uhi_intensity() first.")

        _diag_header("CLASSIFY UHI INTENSITY")

        classified = np.zeros_like(self.uhi_map, dtype=np.int8)
        classified[self.uhi_map < 0]                                   = 0
        classified[(self.uhi_map >= 0) & (self.uhi_map < 1)]          = 1
        classified[(self.uhi_map >= 1) & (self.uhi_map < 2)]          = 2
        classified[(self.uhi_map >= 2) & (self.uhi_map < 3)]          = 3
        classified[self.uhi_map >= 3]                                  = 4

        categories: Dict[str, int] = {}
        for idx, label in enumerate(_UHI_CAT_LABELS):
            cnt = int((classified == idx).sum())
            pct = cnt / classified.size * 100
            categories[label] = cnt
            logger.info(f"  [{idx}] {label:<25s}  {cnt:>8,} px  ({pct:6.2f}%)")

        # Entropy of classification distribution
        counts = np.array(list(categories.values()), dtype=float)
        probs  = counts / counts.sum()
        probs  = probs[probs > 0]
        entropy = -float((probs * np.log2(probs)).sum())
        logger.info(f"  Classification entropy : {entropy:.4f} bits "
                    f"(max={np.log2(5):.4f})")

        return classified, categories

    # ── statistics ────────────────────────────────────────────────────────────

    def calculate_statistics(self) -> Dict[str, float]:
        if self.uhi_map is None:
            raise ValueError("Call calculate_uhi_intensity() first.")

        _diag_header("UHI STATISTICS")

        # Use NaN-safe functions throughout — the mosaicked map may contain
        # NaN-padded cells in grid positions where QC filtering removed patches.
        uhi_flat = self.uhi_map.ravel()
        uhi_finite = uhi_flat[np.isfinite(uhi_flat)]
        uhi_pos = uhi_finite[uhi_finite > 0]

        q5, q25, q75, q95 = np.nanpercentile(self.uhi_map, [5, 25, 75, 95])
        skew = float(stats.skew(uhi_finite)) if len(uhi_finite) > 1 else float("nan")
        kurt = float(stats.kurtosis(uhi_finite)) if len(uhi_finite) > 1 else float("nan")

        stat_dict: Dict[str, float] = {
            "max_intensity":          float(np.nanmax(self.uhi_map)),
            "min_intensity":          float(np.nanmin(self.uhi_map)),
            "mean_intensity":         float(np.nanmean(self.uhi_map)),
            "mean_positive_intensity":float(uhi_pos.mean()) if len(uhi_pos) else 0.0,
            "median_intensity":       float(np.nanmedian(self.uhi_map)),
            "std_intensity":          float(np.nanstd(self.uhi_map)),
            "p5":  float(q5),
            "p25": float(q25),
            "p75": float(q75),
            "p95": float(q95),
            "skewness":               skew,
            "kurtosis":               kurt,
            "spatial_extent_km2":     float((self.uhi_map > 2).sum() * 0.0025),
            "magnitude":              float(uhi_pos.sum()),
            "pct_positive":           float(len(uhi_pos) / max(len(uhi_finite), 1) * 100),
        }

        for k, v in stat_dict.items():
            logger.info(f"  {k:<30s}: {v:.4f}")

        return stat_dict

    # ─────────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC PLOTS
    # ─────────────────────────────────────────────────────────────────────────

    def plot_lst_distribution(self, output_path: Path,
                              urban_mask: Optional[np.ndarray] = None,
                              rural_mask: Optional[np.ndarray] = None) -> None:
        """
        Diagnostic: LST histogram with KDE, optionally split urban vs rural.
        Includes normality test annotation.
        """
        _diag_header("PLOT — LST DISTRIBUTION")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("LST Distribution Diagnostics", fontsize=15, fontweight="bold")

        # ── left: full histogram + KDE ────────────────────────────────────────
        ax = axes[0]
        flat = self.lst_map.ravel()
        ax.hist(flat, bins=80, color="#4393c3", edgecolor="none",
                density=True, alpha=0.7, label="All pixels")
        kde_x = np.linspace(flat.min(), flat.max(), 500)
        kde_vals = _safe_kde(flat, kde_x)
        if kde_vals is not None:
            ax.plot(kde_x, kde_vals, color="#b2182b", lw=2, label="KDE")
        # normal overlay
        mu, sigma = flat.mean(), flat.std()
        ax.plot(kde_x, stats.norm.pdf(kde_x, mu, sigma),
                color="grey", lw=1.5, ls="--", label=f"Normal(μ={mu:.1f}, σ={sigma:.2f})")
        # percentile lines
        for pct, col in [(5, "#92c5de"), (25, "#4393c3"), (75, "#d6604d"), (95, "#b2182b")]:
            val = np.percentile(flat, pct)
            ax.axvline(val, color=col, lw=0.9, ls=":", alpha=0.9, label=f"P{pct}={val:.1f}°C")
        # Shapiro-Wilk (sample)
        sample = flat[np.random.choice(len(flat), min(5000, len(flat)), replace=False)]
        sw_stat, sw_p = stats.shapiro(sample)
        ax.text(0.02, 0.97, f"Shapiro-Wilk: W={sw_stat:.4f}  p={sw_p:.2e}\n"
                             f"skew={stats.skew(flat):.3f}  kurt={stats.kurtosis(flat):.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                bbox=dict(fc="white", ec="grey", alpha=0.8))
        ax.set_xlabel("LST (°C)")
        ax.set_ylabel("Density")
        ax.set_title("Full LST Histogram + KDE")
        ax.legend(fontsize=8, ncol=2)

        # ── right: urban vs rural comparison ─────────────────────────────────
        ax = axes[1]
        if urban_mask is not None and rural_mask is not None:
            u_vals = self.lst_map[urban_mask].ravel()
            r_vals = self.lst_map[rural_mask].ravel()
            for vals, color, label in [(r_vals, "#4393c3", "Rural"),
                                       (u_vals, "#d6604d", "Urban")]:
                ax.hist(vals, bins=60, color=color, density=True,
                        alpha=0.55, edgecolor="none", label=f"{label} (n={len(vals):,})")
                x_v = np.linspace(vals.min(), vals.max(), 500)
                kde_v = _safe_kde(vals, x_v)
                if kde_v is not None:
                    ax.plot(x_v, kde_v, color=color, lw=2)
            t_stat, p_val = stats.ttest_ind(u_vals, r_vals, equal_var=False)
            _pooled_std = np.sqrt((u_vals.std()**2 + r_vals.std()**2) / 2)
            d_cohen = ((u_vals.mean() - r_vals.mean()) / _pooled_std
                       if _pooled_std > 1e-8 else float("nan"))
            ax.text(0.02, 0.97,
                    f"Welch t={t_stat:.3f}  p={p_val:.2e}\nCohen's d={d_cohen:.3f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                    bbox=dict(fc="white", ec="grey", alpha=0.8))
            ax.legend(fontsize=9)
        else:
            # Q–Q plot
            osm, osr = stats.probplot(flat, dist="norm")
            ax.plot(osm[0], osm[1], ".", color="#4393c3", ms=2, alpha=0.5)
            ax.plot(osm[0], osm[0] * osr[0] + osr[1], color="#b2182b", lw=2)
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.set_title("Q–Q Plot vs Normal")

        ax.set_xlabel("LST (°C)")
        ax.set_ylabel("Density")
        ax.set_title("Urban vs Rural LST Distributions")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def plot_uhi_intensity_diagnostics(self, output_path: Path) -> None:
        """
        Diagnostic 2×3 grid:
          [0,0] UHI spatial map          [0,1] UHI histogram + thresholds
          [1,0] Empirical CDF            [1,1] Box-plot per category
          [2,0] Spatial row-profile      [2,1] Spatial col-profile
        """
        if self.uhi_map is None:
            raise ValueError("Call calculate_uhi_intensity() first.")

        _diag_header("PLOT — UHI INTENSITY DIAGNOSTICS")
        fig = plt.figure(figsize=(16, 14))
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle("UHI Intensity Diagnostics", fontsize=16, fontweight="bold")

        uhi = self.uhi_map
        flat = uhi.ravel()

        # ── [0,0] UHI spatial map ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        vabs = max(float(np.percentile(np.abs(flat), 98)), 1e-6)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im   = ax.imshow(uhi, cmap="RdBu_r", norm=norm,
                         interpolation="nearest", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="UHI (°C)")
        ax.set_title("UHI Intensity Map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # ── [0,1] Histogram + thresholds ─────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        ax.hist(flat, bins=100, color="#4393c3", edgecolor="none",
                density=True, alpha=0.7, label="UHI values")
        x_k = np.linspace(flat.min(), flat.max(), 500)
        kde_vals = _safe_kde(flat, x_k)
        if kde_vals is not None:
            ax.plot(x_k, kde_vals, color="#b2182b", lw=2, label="KDE")
        for thr, col, lbl in [(0, "#99d594", "0 °C"),
                               (1, "#fee08b", "1 °C"),
                               (2, "#fc8d59", "2 °C"),
                               (3, "#d53e4f", "3 °C")]:
            ax.axvline(thr, color=col, lw=1.5, ls="--", label=f"+{lbl}")
        ax.set_xlabel("UHI Intensity (°C)")
        ax.set_ylabel("Density")
        ax.set_title("UHI Intensity Histogram")
        ax.legend(fontsize=8)

        # ── [1,0] Empirical CDF ───────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        sorted_uhi = np.sort(flat)
        cdf = np.arange(1, len(sorted_uhi) + 1) / len(sorted_uhi)
        ax.plot(sorted_uhi, cdf, color="#4393c3", lw=2)
        for thr, col in [(0, "#99d594"), (1, "#fee08b"),
                         (2, "#fc8d59"), (3, "#d53e4f")]:
            ax.axvline(thr, color=col, lw=1.2, ls="--")
            pct_above = 100 * (1 - np.interp(thr, sorted_uhi, cdf))
            ax.text(thr + 0.05, 0.05, f"{pct_above:.1f}%\nabove",
                    fontsize=7.5, color=col, va="bottom")
        ax.set_xlabel("UHI Intensity (°C)")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("Empirical CDF of UHI Intensity")
        ax.grid(True, alpha=0.3)

        # ── [1,1] Box-plot per UHI category ──────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        boundaries = [(-99, 0), (0, 1), (1, 2), (2, 3), (3, 99)]
        box_data   = []
        for lo, hi in boundaries:
            mask = (uhi >= lo) & (uhi < hi) if hi != 99 else uhi >= lo
            vals = uhi[mask].ravel()
            box_data.append(vals if len(vals) else np.array([np.nan]))
        bp = ax.boxplot(box_data, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=2))
        for patch, col in zip(bp["boxes"], _UHI_CAT_COLORS):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        short_labels = ["≤0", "0–1", "1–2", "2–3", "≥3"]
        ax.set_xticklabels(short_labels)
        ax.set_xlabel("UHI Category (°C)")
        ax.set_ylabel("UHI Intensity (°C)")
        ax.set_title("Box-Plot per UHI Category")
        ax.grid(True, axis="y", alpha=0.3)

        # ── [2,0] Row-mean spatial profile ────────────────────────────────────
        ax = fig.add_subplot(gs[2, 0])
        row_means = uhi.mean(axis=1)
        row_stds  = uhi.std(axis=1)
        y_idx     = np.arange(len(row_means))
        ax.plot(row_means, y_idx, color="#4393c3", lw=1.5, label="Row mean")
        ax.fill_betweenx(y_idx, row_means - row_stds, row_means + row_stds,
                         alpha=0.2, color="#4393c3", label="±1 σ")
        ax.axvline(0, color="grey", lw=1, ls="--")
        ax.set_xlabel("Mean UHI (°C)")
        ax.set_ylabel("Row index (N→S)")
        ax.set_title("North–South UHI Profile (row mean)")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [2,1] Column-mean spatial profile ────────────────────────────────
        ax = fig.add_subplot(gs[2, 1])
        col_means = uhi.mean(axis=0)
        col_stds  = uhi.std(axis=0)
        x_idx     = np.arange(len(col_means))
        ax.plot(x_idx, col_means, color="#d6604d", lw=1.5, label="Col mean")
        ax.fill_between(x_idx, col_means - col_stds, col_means + col_stds,
                        alpha=0.2, color="#d6604d", label="±1 σ")
        ax.axhline(0, color="grey", lw=1, ls="--")
        ax.set_xlabel("Column index (W→E)")
        ax.set_ylabel("Mean UHI (°C)")
        ax.set_title("West–East UHI Profile (column mean)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def plot_classification_breakdown(self, classified: np.ndarray,
                                      categories: Dict[str, int],
                                      output_path: Path) -> None:
        """
        3-panel figure:
          [left]  classified spatial map (colour-coded)
          [mid]   stacked-bar / pie comparison
          [right] per-category mean UHI ± std bar
        """
        if self.uhi_map is None:
            raise ValueError("Call calculate_uhi_intensity() first.")

        _diag_header("PLOT — UHI CLASSIFICATION BREAKDOWN")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("UHI Classification Breakdown", fontsize=15, fontweight="bold")

        # ── left: spatial ─────────────────────────────────────────────────────
        ax = axes[0]
        cmap_cls = mcolors.ListedColormap(_UHI_CAT_COLORS)
        norm_cls = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap_cls.N)
        im       = ax.imshow(classified, cmap=cmap_cls, norm=norm_cls,
                             interpolation="nearest", aspect="auto")
        cbar     = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                                boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                                ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(["No UHI", "Weak", "Moderate", "Strong", "V.Strong"],
                                fontsize=8)
        ax.set_title("Classification Map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # ── middle: donut / pie ───────────────────────────────────────────────
        ax = axes[1]
        total  = sum(categories.values())
        labels = list(categories.keys())
        sizes  = [v for v in categories.values()]
        pcts   = [v / total * 100 for v in sizes]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None,
            colors=_UHI_CAT_COLORS[:len(labels)],
            autopct="%1.1f%%", startangle=140,
            wedgeprops=dict(width=0.55, edgecolor="white", lw=1.5),
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax.set_title("Category Distribution (%)")
        legend_patches = [Patch(fc=_UHI_CAT_COLORS[i], label=f"{labels[i]}\n({pcts[i]:.1f}%)")
                          for i in range(len(labels))]
        ax.legend(handles=legend_patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.20), fontsize=7.5, ncol=2)

        # ── right: per-category mean ± std ────────────────────────────────────
        ax = axes[2]
        boundaries = [(-99, 0), (0, 1), (1, 2), (2, 3), (3, 99)]
        means, stds, counts = [], [], []
        for lo, hi in boundaries:
            mask  = (self.uhi_map >= lo) & (self.uhi_map < hi) if hi != 99 else self.uhi_map >= lo
            vals  = self.uhi_map[mask].ravel()
            means.append(vals.mean() if len(vals) else 0)
            stds.append(vals.std()   if len(vals) else 0)
            counts.append(len(vals))
        short_labels = ["≤0", "0–1", "1–2", "2–3", "≥3"]
        x = np.arange(len(short_labels))
        bars = ax.bar(x, means, yerr=stds, color=_UHI_CAT_COLORS,
                      alpha=0.8, capsize=5, edgecolor="black", lw=0.8)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"n={cnt:,}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels)
        ax.set_xlabel("UHI Category")
        ax.set_ylabel("Mean UHI Intensity (°C)")
        ax.set_title("Mean ± Std per Category")
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def plot_spatial_autocorrelation(self, output_path: Path,
                                     max_lag: int = 30) -> None:
        """
        Moran's I approximation via row/column lag autocorrelation.
        Plots ACF in both spatial directions.
        Uses numpy-only correlation to avoid repeated stats.pearsonr on large maps.
        """
        if self.uhi_map is None:
            raise ValueError("Call calculate_uhi_intensity() first.")

        _diag_header("PLOT — SPATIAL AUTOCORRELATION (LAG PROFILE)")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Spatial Autocorrelation of UHI", fontsize=14, fontweight="bold")

        uhi = self.uhi_map

        def _fast_corr(a: np.ndarray, b: np.ndarray) -> float:
            """Pearson r without scipy overhead."""
            a = a - a.mean(); b = b - b.mean()
            denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
            return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0

        for ax, direction, data in [
            (axes[0], "N–S (row lag)", uhi),
            (axes[1], "W–E (col lag)", uhi.T),
        ]:
            lags, acf_vals, ci_upper = [], [], []
            for lag in range(0, max_lag + 1):
                if lag == 0:
                    acf_vals.append(1.0)
                    ci_upper.append(1.0)
                else:
                    a = data[:-lag, :].ravel()
                    b = data[lag:,  :].ravel()
                    acf_vals.append(_fast_corr(a, b))
                    ci_upper.append(1.96 / np.sqrt(len(a)))
                lags.append(lag)

            ax.bar(lags, acf_vals, color="#4393c3", alpha=0.7, width=0.6)
            ax.plot(lags, ci_upper,  color="red", ls="--", lw=1.2, label="95% CI")
            ax.plot(lags, [-v for v in ci_upper], color="red", ls="--", lw=1.2)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xlabel(f"Lag (pixels) — {direction}")
            ax.set_ylabel("Autocorrelation")
            ax.set_title(f"ACF {direction}")
            ax.legend(fontsize=8)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, alpha=0.3)

            logger.info(f"  {direction}: lag-1={acf_vals[1]:.4f}  "
                        f"lag-5={acf_vals[min(5,max_lag)]:.4f}  "
                        f"lag-10={acf_vals[min(10,max_lag)]:.4f}")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# HotspotDetector
# ══════════════════════════════════════════════════════════════════════════════

class HotspotDetector:
    """Detect and analyze UHI hotspots using Getis-Ord Gi* statistic."""

    def __init__(self, lst_map: np.ndarray, resolution: float = 50.0):
        self.lst_map    = lst_map
        self.resolution = resolution
        self.hotspot_map: Optional[np.ndarray] = None

        _diag_header("HOTSPOT DETECTOR — INITIALISATION")
        _diag_array("Input LST", lst_map)
        logger.info(f"  Resolution: {resolution} m/pixel")

    def calculate_gi_star(self, search_radius: float = 500.0) -> np.ndarray:
        """
        Vectorised Getis-Ord Gi* using scipy.ndimage.uniform_filter.

        The original implementation used an O(N²) nested Python loop — one full
        (H×W) distance matrix per pixel — which freezes on any map larger than
        ~128×128.  This version runs in O(N) time by replacing the per-pixel loop
        with two uniform_filter passes (sum of values, sum of weights) over the
        entire array at once, which is equivalent to a binary disc kernel Gi*.
        """
        from scipy.ndimage import uniform_filter

        logger.info(f"Calculating Gi* statistic (radius={search_radius}m, vectorised)...")

        height, width = self.lst_map.shape

        # Kernel half-width in pixels (equivalent to the search radius disc)
        kernel_px = max(1, int(search_radius / self.resolution))
        # uniform_filter size must be odd
        ksize = 2 * kernel_px + 1

        logger.info(f"  Kernel size : {ksize}×{ksize} px  "
                    f"({ksize * self.resolution:.0f}×{ksize * self.resolution:.0f} m)")

        X_bar = float(np.nanmean(self.lst_map))
        S     = float(np.nanstd(self.lst_map))

        if S < 1e-6:
            logger.warning("  ⚠️ LST map has near-zero variance — Gi* will be all zeros")
            gi_star = np.zeros_like(self.lst_map, dtype=np.float32)
            self.hotspot_map = gi_star
            return gi_star

        # Replace NaN cells with the global mean before filtering so boundary
        # NaN-padding (from QC-filtered mosaic cells) does not pull the local
        # sums toward zero and suppress Gi* values near the edges.
        lst_filled = np.where(np.isfinite(self.lst_map), self.lst_map, X_bar)

        # Count of valid (non-NaN) pixels for the global n
        n_valid = int(np.isfinite(self.lst_map).sum())
        n       = max(n_valid, 1)

        # uniform_filter computes a local mean; multiply by ksize² to get local sum
        w_sum    = float(ksize * ksize)          # all weights = 1 (binary disc approx.)
        w_sum_sq = w_sum                         # sum of w² = sum of 1² = w_sum

        local_sum = uniform_filter(lst_filled.astype(np.float64),
                                   size=ksize, mode="reflect") * w_sum

        numerator   = local_sum - X_bar * w_sum
        denominator = S * np.sqrt((n * w_sum_sq - w_sum ** 2) / max(n - 1, 1))

        if denominator > 1e-6:
            gi_star = (numerator / denominator).astype(np.float32)
        else:
            gi_star = np.zeros_like(self.lst_map, dtype=np.float32)

        self.hotspot_map = gi_star

        _diag_header("GI* STATISTICS")
        _diag_array("Gi* map", gi_star)
        for thr, ci in [(1.65, "90%"), (1.96, "95%"), (2.58, "99%")]:
            cnt = (gi_star > thr).sum()
            logger.info(f"  Gi* > {thr} ({ci} CI) : {cnt:,} px  "
                        f"({cnt/gi_star.size*100:.2f}%)")

        return gi_star

    def identify_hotspots(self, confidence_level: float = 0.95
                          ) -> Tuple[np.ndarray, List[Dict]]:
        if self.hotspot_map is None:
            raise ValueError("Call calculate_gi_star() first.")

        threshold = 1.96 if confidence_level == 0.95 else 2.58
        logger.info(f"Identifying hotspots at {confidence_level*100:.0f}% CI "
                    f"(Gi* > {threshold})...")

        hotspots = self.hotspot_map > threshold

        from scipy.ndimage import label
        labeled_hotspots, n_hotspots = label(hotspots)
        logger.info(f"  {n_hotspots} connected regions found")

        hotspot_list: List[Dict] = []
        for region_id in range(1, n_hotspots + 1):
            region_mask  = labeled_hotspots == region_id
            region_pixels = region_mask.sum()
            if region_pixels < 4:
                continue
            y_idx, x_idx = np.where(region_mask)
            hotspot_list.append({
                "id":         region_id,
                "n_pixels":   int(region_pixels),
                "area_km2":   float(region_pixels * (self.resolution / 1000) ** 2),
                "centroid_y": float(y_idx.mean()),
                "centroid_x": float(x_idx.mean()),
                "mean_lst":   float(self.lst_map[region_mask].mean()),
                "max_lst":    float(self.lst_map[region_mask].max()),
                "mean_gi_star": float(self.hotspot_map[region_mask].mean()),
                "max_gi_star":  float(self.hotspot_map[region_mask].max()),
            })

        hotspot_list.sort(key=lambda x: x["area_km2"], reverse=True)

        _diag_header("HOTSPOT IDENTIFICATION SUMMARY")
        logger.info(f"  Qualifying regions (≥4 px): {len(hotspot_list)}")
        if hotspot_list:
            areas = [h["area_km2"] for h in hotspot_list]
            logger.info(f"  Total hotspot area  : {sum(areas):.3f} km²")
            logger.info(f"  Largest hotspot     : {areas[0]:.3f} km²")
            logger.info(f"  Median hotspot area : {np.median(areas):.3f} km²")
            logger.info(f"  Max LST in hotspots : "
                        f"{max(h['max_lst'] for h in hotspot_list):.2f} °C")

        return hotspots, hotspot_list

    def prioritize_hotspots(self, hotspot_list: List[Dict],
                            population_density: Optional[np.ndarray] = None,
                            vulnerability_index: Optional[np.ndarray] = None
                            ) -> pd.DataFrame:
        logger.info("Prioritizing hotspots for intervention...")
        df = pd.DataFrame(hotspot_list)

        if len(df) == 0:
            return df

        # Score each criterion, guarding against zero-range (single hotspot)
        def _norm(series: pd.Series) -> pd.Series:
            rng = series.max() - series.min()
            if rng < 1e-9:
                # All values equal → assign 0.5 uniformly (not 0/NaN)
                return pd.Series(0.5, index=series.index)
            return ((series - series.min()) / rng).clip(0, 1)

        df["intensity_score"]    = _norm(df["mean_lst"])
        df["extent_score"]       = _norm(df["area_km2"])
        sig_raw                  = (df["max_gi_star"] - 1.96).clip(lower=0)
        df["significance_score"] = _norm(sig_raw)

        df["priority_score"] = (
            0.4 * df["intensity_score"] +
            0.3 * df["extent_score"] +
            0.3 * df["significance_score"]
        ).clip(0, 1)   # final clamp — float arithmetic can produce tiny negatives

        # Warn if scores are degenerate
        if df["priority_score"].std() < 0.01:
            logger.warning(
                "  ⚠ Priority scores are nearly constant — all hotspots have "
                "similar characteristics (common with only 1 hotspot detected). "
                "Consider lowering --hotspot-confidence or --hotspot-radius."
            )

        df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        _diag_header("TOP-5 PRIORITY HOTSPOTS")
        for _, row in df.head(5).iterrows():
            logger.info(f"  #{int(row['rank'])}  area={row['area_km2']:.3f} km²  "
                        f"LST={row['mean_lst']:.1f}°C  score={row['priority_score']:.3f}")

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # DIAGNOSTIC PLOTS
    # ─────────────────────────────────────────────────────────────────────────

    def plot_gi_star_diagnostics(self, output_path: Path) -> None:
        """
        2×2 diagnostic figure:
          [0,0] Gi* spatial heat-map
          [0,1] Gi* histogram with significance thresholds
          [1,0] Gi* empirical CDF
          [1,1] Scatter: LST vs Gi*
        """
        if self.hotspot_map is None:
            raise ValueError("Call calculate_gi_star() first.")

        _diag_header("PLOT — GI* DIAGNOSTICS")
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle("Getis-Ord Gi* Diagnostic Plots", fontsize=15, fontweight="bold")

        gi = self.hotspot_map

        # [0,0] spatial
        ax = axes[0, 0]
        vmax = np.percentile(np.abs(gi), 98)
        im   = ax.imshow(gi, cmap="RdYlBu_r", vmin=-vmax, vmax=vmax,
                         interpolation="nearest", aspect="auto")
        for thr, col, ls in [(1.96, "black", "--"), (2.58, "black", "-")]:
            cs = ax.contour(gi, levels=[thr], colors=col, linewidths=1.2, linestyles=ls)
            ax.clabel(cs, fmt=f"{thr}", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Gi*")
        ax.set_title("Gi* Map (contours at 1.96 & 2.58)")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

        # [0,1] histogram
        ax = axes[0, 1]
        flat = gi.ravel()
        ax.hist(flat, bins=100, color="#92c5de", edgecolor="none",
                density=True, alpha=0.8)
        x_k = np.linspace(flat.min(), flat.max(), 500)
        kde_vals = _safe_kde(flat, x_k)
        if kde_vals is not None:
            ax.plot(x_k, kde_vals, color="#b2182b", lw=2, label="KDE")
        for thr, lbl, col in [(1.96, "95% CI", "orange"),
                               (2.58, "99% CI", "red"),
                               (-1.96, "Cold-spot 95%", "steelblue")]:
            ax.axvline(thr, color=col, lw=1.5, ls="--", label=f"{lbl} ({thr})")
        # standard normal overlay
        ax.plot(x_k, stats.norm.pdf(x_k, 0, 1), color="grey",
                lw=1.2, ls=":", label="Std Normal")
        ax.set_xlabel("Gi* Value")
        ax.set_ylabel("Density")
        ax.set_title("Gi* Distribution")
        ax.legend(fontsize=7.5)

        # [1,0] CDF
        ax = axes[1, 0]
        sorted_gi = np.sort(flat)
        cdf = np.linspace(0, 1, len(sorted_gi))
        ax.plot(sorted_gi, cdf, color="#4393c3", lw=2)
        for thr, col in [(1.96, "orange"), (2.58, "red"), (-1.96, "steelblue")]:
            pct = 100 * np.interp(thr, sorted_gi, cdf)
            ax.axvline(thr, color=col, ls="--", lw=1.2,
                       label=f"Gi*={thr}: {pct:.1f}%ile")
        ax.set_xlabel("Gi* Value")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("Empirical CDF of Gi*")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)

        # [1,1] Scatter: LST vs Gi*
        ax = axes[1, 1]
        # downsample for speed
        idx = np.random.choice(gi.size, min(10000, gi.size), replace=False)
        ax.scatter(self.lst_map.ravel()[idx], gi.ravel()[idx],
                   c=gi.ravel()[idx], cmap="RdYlBu_r",
                   vmin=-vmax, vmax=vmax, s=4, alpha=0.5)
        # trend line — guard against constant arrays (ConstantInputWarning → nan r)
        _lst_s = self.lst_map.ravel()[idx]
        _gi_s  = gi.ravel()[idx]
        if _lst_s.std() > 1e-8 and _gi_s.std() > 1e-8:
            r, p = stats.pearsonr(_lst_s, _gi_s)
            _r_label = f"r={r:.3f}, p={p:.2e}"
        else:
            r, p = float("nan"), float("nan")
            _r_label = "r=N/A (constant input)"
        ax.set_xlabel("LST (°C)")
        ax.set_ylabel("Gi*")
        ax.set_title(f"LST vs Gi*  ({_r_label})")
        ax.axhline(1.96, color="orange", ls="--", lw=1, label="95% CI")
        ax.axhline(2.58, color="red",    ls="--", lw=1, label="99% CI")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def plot_hotspot_ranking(self, hotspots_df: pd.DataFrame,
                             output_path: Path, top_n: int = 20) -> None:
        """
        3-panel ranking figure:
          [left]  horizontal bar: top-N by area
          [mid]   scatter: area vs mean LST, coloured by priority score
          [right] cumulative area curve
        """
        _diag_header("PLOT — HOTSPOT RANKING")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Hotspot Ranking & Prioritisation", fontsize=14, fontweight="bold")

        df = hotspots_df.head(top_n).copy()

        # ── left: horizontal bar ──────────────────────────────────────────────
        ax = axes[0]
        colors = plt.cm.RdYlGn_r(
            np.linspace(0, 1, len(df)))[::-1]
        ax.barh(range(len(df)), df["area_km2"].values,
                color=colors, alpha=0.85, edgecolor="none")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([f"#{int(r)}" for r in df["rank"]], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Area (km²)")
        ax.set_title(f"Top-{top_n} Hotspots by Area")
        ax.grid(True, axis="x", alpha=0.3)
        for i, (area, lst) in enumerate(zip(df["area_km2"], df["mean_lst"])):
            ax.text(area + area * 0.01, i,
                    f"{lst:.1f}°C", va="center", fontsize=7)

        # ── middle: scatter area vs LST ────────────────────────────────────────
        ax = axes[1]
        if "priority_score" in hotspots_df.columns:
            sc = ax.scatter(hotspots_df["area_km2"], hotspots_df["mean_lst"],
                            c=hotspots_df["priority_score"], cmap="RdYlGn_r",
                            s=40, alpha=0.8, edgecolors="none")
            fig.colorbar(sc, ax=ax, label="Priority Score")
        else:
            ax.scatter(hotspots_df["area_km2"], hotspots_df["mean_lst"],
                       color="#4393c3", s=30, alpha=0.7)
        # annotate top-5
        for _, row in hotspots_df.head(5).iterrows():
            ax.annotate(f"#{int(row['rank'])}",
                        (row["area_km2"], row["mean_lst"]),
                        fontsize=8, color="black",
                        xytext=(4, 2), textcoords="offset points")
        ax.set_xlabel("Area (km²)")
        ax.set_ylabel("Mean LST (°C)")
        ax.set_title("Area vs Mean LST")
        ax.grid(True, alpha=0.3)

        # ── right: cumulative area ─────────────────────────────────────────────
        ax = axes[2]
        sorted_df = hotspots_df.sort_values("area_km2", ascending=False)
        cum_area  = sorted_df["area_km2"].cumsum().values
        x_idx     = np.arange(1, len(cum_area) + 1)
        ax.plot(x_idx, cum_area, color="#d6604d", lw=2)
        ax.fill_between(x_idx, cum_area, alpha=0.2, color="#d6604d")
        total = cum_area[-1] if len(cum_area) else 0
        for pct in [0.5, 0.8, 0.9]:
            thr_idx = np.searchsorted(cum_area, pct * total)
            if thr_idx < len(cum_area):
                ax.axvline(thr_idx + 1, color="grey", ls="--", lw=1,
                           label=f"{int(pct*100)}% of area: top-{thr_idx+1}")
        ax.set_xlabel("Number of Hotspots (sorted by area)")
        ax.set_ylabel("Cumulative Area (km²)")
        ax.set_title("Cumulative Hotspot Area")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# ValidationAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class ValidationAnalyzer:
    """Validate predictions against ground truth with extended diagnostics."""

    def __init__(self, predictions: np.ndarray, ground_truth: np.ndarray):
        self.predictions  = predictions.flatten()
        self.ground_truth = ground_truth.flatten()

        mask = ~(np.isnan(self.predictions) | np.isnan(self.ground_truth))
        self.predictions  = self.predictions[mask]
        self.ground_truth = self.ground_truth[mask]

        _diag_header("VALIDATION ANALYZER — INITIALISATION")
        logger.info(f"  Valid pixels : {len(self.predictions):,}")
        _diag_array("Predictions",  self.predictions)
        _diag_array("Ground truth", self.ground_truth)

    def calculate_metrics(self) -> Dict[str, float]:
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        _diag_header("VALIDATION METRICS")

        r2   = r2_score(self.ground_truth, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.ground_truth, self.predictions))
        mae  = mean_absolute_error(self.ground_truth, self.predictions)
        mbe  = float(np.mean(self.predictions - self.ground_truth))

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            self.ground_truth, self.predictions)

        residuals = self.predictions - self.ground_truth
        sw_stat, sw_p = stats.shapiro(
            residuals[np.random.choice(len(residuals),
                                       min(5000, len(residuals)), replace=False)])

        metrics = {
            "r2":          float(r2),
            "rmse":        float(rmse),
            "mae":         float(mae),
            "mbe":         float(mbe),
            "slope":       float(slope),
            "intercept":   float(intercept),
            "correlation": float(r_value),
            "p_value":     float(p_value),
            "std_err":     float(std_err),
            "residual_std":    float(residuals.std()),
            "residual_skew":   float(stats.skew(residuals)),
            "residual_kurt":   float(stats.kurtosis(residuals)),
            "shapiro_stat":    float(sw_stat),
            "shapiro_p":       float(sw_p),
        }

        for k, v in metrics.items():
            logger.info(f"  {k:<22s}: {v:.6f}")

        logger.info(f"\n  Regression : ŷ = {slope:.4f}·x + {intercept:.4f}")
        logger.info(f"  Residuals are {'NORMAL' if sw_p > 0.05 else 'NON-NORMAL'} "
                    f"(Shapiro-Wilk p={sw_p:.2e})")
        return metrics

    def plot_validation(self, output_path: Path) -> None:
        """
        Extended 2×3 validation figure:
          [0,0] Predicted vs Observed scatter
          [0,1] Residual vs Observed scatter
          [0,2] Residual histogram + KDE
          [1,0] Residual Q–Q plot
          [1,1] Absolute error heatmap (if 2-D input shapes preserved)
          [1,2] Error by temperature bin
        """
        _diag_header("PLOT — EXTENDED VALIDATION")
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle("Prediction Validation Diagnostics", fontsize=15, fontweight="bold")

        pred = self.predictions
        obs  = self.ground_truth
        res  = pred - obs
        abs_err = np.abs(res)

        mn = min(obs.min(), pred.min())
        mx = max(obs.max(), pred.max())

        slope, intercept, r_val, p_val, _ = stats.linregress(obs, pred)
        r2 = r_val ** 2

        # ── [0,0] Scatter ─────────────────────────────────────────────────────
        ax = axes[0, 0]
        # density-coloured scatter
        idx = np.random.choice(len(obs), min(5000, len(obs)), replace=False)
        z = _safe_kde_2d(obs[idx], pred[idx])
        scat  = ax.scatter(obs[idx], pred[idx],
                           c=z if z is not None else pred[idx],
                           cmap="plasma", s=6, alpha=0.7)
        fig.colorbar(scat, ax=ax, label="Density")
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="1:1 line")
        x_fit = np.array([mn, mx])
        ax.plot(x_fit, slope * x_fit + intercept, "b-", lw=1.5,
                label=f"Fit: y={slope:.3f}x+{intercept:.3f}")
        ax.set_xlabel("Observed LST (°C)")
        ax.set_ylabel("Predicted LST (°C)")
        ax.set_title(f"Predicted vs Observed  R²={r2:.4f}  RMSE={np.sqrt(np.mean(res**2)):.4f}°C")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [0,1] Residual vs Observed ────────────────────────────────────────
        ax = axes[0, 1]
        ax.scatter(obs[idx], res[idx], c="#4393c3", s=5, alpha=0.5)
        ax.axhline(0, color="red", lw=1.5, ls="--")
        ax.axhline(np.mean(res),  color="orange", lw=1.2, ls="--",
                   label=f"MBE={np.mean(res):.3f}°C")
        ax.axhline(np.std(res),   color="grey",   lw=1,   ls=":",
                   label=f"+1σ={np.std(res):.3f}°C")
        ax.axhline(-np.std(res),  color="grey",   lw=1,   ls=":")
        ax.set_xlabel("Observed LST (°C)")
        ax.set_ylabel("Residual (°C)")
        ax.set_title("Residuals vs Observed")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [0,2] Residual histogram ──────────────────────────────────────────
        ax = axes[0, 2]
        ax.hist(res, bins=80, color="#92c5de", edgecolor="none",
                density=True, alpha=0.75, label="Residuals")
        x_k = np.linspace(res.min(), res.max(), 500)
        kde_vals = _safe_kde(res, x_k)
        if kde_vals is not None:
            ax.plot(x_k, kde_vals, color="#b2182b", lw=2, label="KDE")
        mu, sigma = res.mean(), res.std()
        ax.plot(x_k, stats.norm.pdf(x_k, mu, sigma),
                color="grey", lw=1.5, ls="--",
                label=f"Normal(μ={mu:.3f}, σ={sigma:.3f})")
        ax.axvline(0, color="red", lw=1.2, ls="--")
        ax.set_xlabel("Residual (°C)")
        ax.set_ylabel("Density")
        ax.set_title("Residual Distribution")
        ax.legend(fontsize=8)

        # ── [1,0] Q–Q plot ────────────────────────────────────────────────────
        ax = axes[1, 0]
        osm, osr = stats.probplot(res, dist="norm")
        ax.plot(osm[0], osm[1], ".", color="#4393c3", ms=2, alpha=0.5)
        ax.plot(osm[0], osm[0] * osr[0] + osr[1], color="#b2182b", lw=2,
                label=f"y={osr[0]:.3f}x+{osr[1]:.3f}")
        sw_stat, sw_p = stats.shapiro(
            res[np.random.choice(len(res), min(5000, len(res)), replace=False)])
        ax.text(0.03, 0.95, f"Shapiro-Wilk: W={sw_stat:.4f}  p={sw_p:.2e}",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(fc="white", ec="grey", alpha=0.8))
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title("Q–Q Plot of Residuals")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [1,1] Absolute error vs predicted ────────────────────────────────
        ax = axes[1, 1]
        ax.scatter(pred[idx], abs_err[idx], c="#fc8d59", s=5, alpha=0.5)
        # running median
        sort_idx  = np.argsort(pred[idx])
        win       = max(1, len(sort_idx) // 30)
        run_med   = pd.Series(abs_err[idx][sort_idx]).rolling(win, center=True).median()
        ax.plot(pred[idx][sort_idx], run_med, color="#b2182b", lw=2,
                label=f"Running median (w={win})")
        ax.set_xlabel("Predicted LST (°C)")
        ax.set_ylabel("|Residual| (°C)")
        ax.set_title("Absolute Error vs Predicted")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [1,2] Error by temperature bin ───────────────────────────────────
        ax = axes[1, 2]
        n_bins   = 10
        bin_edges = np.percentile(obs, np.linspace(0, 100, n_bins + 1))
        bin_mids, bin_rmse, bin_mae, bin_mbe = [], [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask  = (obs >= lo) & (obs <= hi)
            if mask.sum() < 5:
                continue
            bin_mids.append((lo + hi) / 2)
            bin_rmse.append(np.sqrt(np.mean(res[mask] ** 2)))
            bin_mae.append(np.mean(np.abs(res[mask])))
            bin_mbe.append(np.mean(res[mask]))
        ax.plot(bin_mids, bin_rmse, "o-", color="#d6604d", lw=1.8, label="RMSE")
        ax.plot(bin_mids, bin_mae,  "s-", color="#4393c3", lw=1.8, label="MAE")
        ax.plot(bin_mids, bin_mbe,  "^-", color="#66c2a5", lw=1.8, label="MBE")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Observed LST bin midpoint (°C)")
        ax.set_ylabel("Error (°C)")
        ax.set_title("Error Metrics by Temperature Bin")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {output_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Report generator
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(uhi_stats: Dict, hotspots_df: pd.DataFrame,
                    validation_metrics: Optional[Dict], output_path: Path) -> None:
    """Generate comprehensive UHI analysis report (JSON)."""

    _diag_header("GENERATING REPORT")

    if hotspots_df is not None and not hotspots_df.empty and "area_km2" in hotspots_df.columns:
        total_hotspot_area = f"{hotspots_df['area_km2'].sum():.2f} km²"
    else:
        total_hotspot_area = "0.00 km²"

    report = {
        "title": "Urban Heat Island Analysis Report",
        "date":  pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uhi_characterization": {
            "maximum_intensity":  f"{uhi_stats['max_intensity']:.2f}°C",
            "minimum_intensity":  f"{uhi_stats['min_intensity']:.2f}°C",
            "mean_intensity":     f"{uhi_stats['mean_intensity']:.2f}°C",
            "median_intensity":   f"{uhi_stats['median_intensity']:.2f}°C",
            "std_intensity":      f"{uhi_stats['std_intensity']:.2f}°C",
            "p5_p95":             f"{uhi_stats['p5']:.2f}–{uhi_stats['p95']:.2f}°C",
            "skewness":           f"{uhi_stats['skewness']:.4f}",
            "kurtosis":           f"{uhi_stats['kurtosis']:.4f}",
            "spatial_extent":     f"{uhi_stats['spatial_extent_km2']:.2f} km²",
            "pct_positive":       f"{uhi_stats['pct_positive']:.2f}%",
            "magnitude":          f"{uhi_stats['magnitude']:.2f}°C·pixels",
        },
        "hotspot_summary": {
            "total_hotspots":     len(hotspots_df),
            "total_hotspot_area": total_hotspot_area,
            "largest_hotspot":    f"{hotspots_df.iloc[0]['area_km2']:.3f} km²"
                                  if len(hotspots_df) > 0 else "N/A",
            "highest_temperature":f"{hotspots_df['max_lst'].max():.2f}°C"
                                  if len(hotspots_df) > 0 else "N/A",
        },
        "top_priority_hotspots": hotspots_df.head(20).to_dict("records")
                                 if len(hotspots_df) > 0 else [],
    }

    if validation_metrics:
        report["validation"] = {
            "r2_score":  f"{validation_metrics['r2']:.4f}",
            "rmse":      f"{validation_metrics['rmse']:.4f}°C",
            "mae":       f"{validation_metrics['mae']:.4f}°C",
            "bias":      f"{validation_metrics['mbe']:.4f}°C",
            "slope":     f"{validation_metrics['slope']:.4f}",
            "intercept": f"{validation_metrics['intercept']:.4f}",
            "residual_normality": "NORMAL" if validation_metrics.get("shapiro_p", 0) > 0.05
                                  else "NON-NORMAL",
        }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"  Report saved: {output_path}")
    _diag_header("UHI ANALYSIS SUMMARY")
    logger.info(f"  Max UHI intensity : {uhi_stats['max_intensity']:.2f}°C")
    logger.info(f"  Spatial extent    : {uhi_stats['spatial_extent_km2']:.2f} km²")
    logger.info(f"  Hotspot regions   : {len(hotspots_df)}")
    if validation_metrics:
        logger.info(f"  Validation R²     : {validation_metrics['r2']:.4f}")
        logger.info(f"  Validation RMSE   : {validation_metrics['rmse']:.4f}°C")


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: run all diagnostic plots from one call
# ══════════════════════════════════════════════════════════════════════════════

def run_all_diagnostics(
        lst_map: np.ndarray,
        uhi_map: Optional[np.ndarray],
        gi_star: Optional[np.ndarray],
        hotspots_df: Optional[pd.DataFrame],
        predictions: Optional[np.ndarray],
        ground_truth: Optional[np.ndarray],
        output_dir: Path,
        urban_mask: Optional[np.ndarray] = None,
        rural_mask: Optional[np.ndarray] = None,
) -> List[Path]:
    """
    Convenience function — run all diagnostic plots and return saved paths.

    Args:
        lst_map      : Raw LST array (H, W)
        uhi_map      : UHI intensity array (H, W) or None
        gi_star      : Gi* statistic array (H, W) or None
        hotspots_df  : Hotspot prioritisation DataFrame or None
        predictions  : Predicted LST flat array or None
        ground_truth : Observed LST flat array or None
        output_dir   : Directory to save all plots
        urban_mask   : Optional boolean urban mask
        rural_mask   : Optional boolean rural mask

    Returns:
        List of Path objects for saved figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # Delegate to the new composable plotter (same interface, inference-style output)
    plotter = AnalysisDiagnosticsPlotter(save_dir=output_dir)
    saved = plotter.plot_all(
        lst_map=lst_map,
        uhi_map=uhi_map,
        gi_star=gi_star,
        hotspots_df=hotspots_df,
        predictions=predictions,
        ground_truth=ground_truth,
        urban_mask=urban_mask,
        rural_mask=rural_mask,
    )
    return saved

    # ── Legacy per-class calls kept below for reference (unreachable) ──────────
    analyzer = UHIAnalyzer(lst_map)
    if uhi_map is not None:
        analyzer.uhi_map = uhi_map

    # 1. LST distribution
    p = output_dir / "diag_lst_distribution.png"
    analyzer.plot_lst_distribution(p, urban_mask, rural_mask)
    saved.append(p)

    # 2. UHI intensity diagnostics
    if uhi_map is not None:
        p = output_dir / "diag_uhi_intensity.png"
        analyzer.plot_uhi_intensity_diagnostics(p)
        saved.append(p)

        classified, categories = analyzer.classify_uhi_intensity()
        p = output_dir / "diag_uhi_classification.png"
        analyzer.plot_classification_breakdown(classified, categories, p)
        saved.append(p)

        p = output_dir / "diag_uhi_autocorrelation.png"
        analyzer.plot_spatial_autocorrelation(p)
        saved.append(p)

    # 3. Gi* diagnostics
    if gi_star is not None:
        detector = HotspotDetector(lst_map)
        detector.hotspot_map = gi_star
        p = output_dir / "diag_gi_star.png"
        detector.plot_gi_star_diagnostics(p)
        saved.append(p)

    # 4. Hotspot ranking
    if hotspots_df is not None and len(hotspots_df) > 0:
        detector = HotspotDetector(lst_map)
        detector.hotspot_map = gi_star
        p = output_dir / "diag_hotspot_ranking.png"
        detector.plot_hotspot_ranking(hotspots_df, p)
        saved.append(p)

    # 5. Validation
    if predictions is not None and ground_truth is not None:
        val = ValidationAnalyzer(predictions, ground_truth)
        p = output_dir / "diag_validation.png"
        val.plot_validation(p)
        saved.append(p)

    logger.info(f"\n✅  All diagnostic plots saved ({len(saved)} figures) → {output_dir}")
    return saved




# ══════════════════════════════════════════════════════════════════════════════
# AnalysisDiagnosticsPlotter
# ══════════════════════════════════════════════════════════════════════════════

class AnalysisDiagnosticsPlotter:
    """
    Centralised matplotlib diagnostics for the UHI analysis pipeline.

    Mirrors the InferenceDiagnosticsPlotter style from uhi_inference.py, wrapping
    all UHIAnalyzer, HotspotDetector, and ValidationAnalyzer plot methods into a
    single, composable class.

    All figures are saved under `save_dir`
    (default: OUTPUT_DIR / \"analysis_diagnostics\").

    Usage::

        plotter = AnalysisDiagnosticsPlotter()
        plotter.plot_all(
            lst_map=lst, uhi_map=uhi, gi_star=gi,
            hotspots_df=hotspots,
            predictions=preds, ground_truth=gt,
            urban_mask=urban, rural_mask=rural,
        )
        # — or call individual plot_* methods as needed —
    """

    _STYLE_CANDIDATES = [
        "seaborn-v0_8-darkgrid",
        "seaborn-darkgrid",
        "ggplot",
        "default",
    ]

    def __init__(self, save_dir: Path = None):
        self.save_dir = Path(save_dir) if save_dir else OUTPUT_DIR / "analysis_diagnostics"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._style = self._resolve_style()
        logger.info(
            f"📊 AnalysisDiagnosticsPlotter initialised — "
            f"style='{self._style}' → {self.save_dir}"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @classmethod
    def _resolve_style(cls) -> str:
        available = set(plt.style.available)
        for style in cls._STYLE_CANDIDATES:
            if style in available:
                return style
        return "default"

    def _use_style(self):
        try:
            plt.style.use(self._style)
        except Exception:
            pass

    def _save(self, fig, name: str) -> Path:
        path = self.save_dir / f"{name}.png"
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"  ✅ Saved: {path.name}")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not save {path.name}: {e}")
        finally:
            plt.close(fig)
        return path

    # ── 1. LST distribution ───────────────────────────────────────────────────

    def plot_lst_distribution(self,
                              lst_map: np.ndarray,
                              urban_mask: np.ndarray = None,
                              rural_mask: np.ndarray = None):
        """
        Histogram + KDE of the full LST array; optionally overlaid with
        separate urban and rural distributions.

        Args:
            lst_map:    2-D LST array (H, W) in °C.
            urban_mask: Optional boolean mask selecting urban pixels.
            rural_mask: Optional boolean mask selecting rural pixels.
        """
        try:
            self._use_style()
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Analysis – LST Distribution", fontsize=14, fontweight="bold")

            flat = lst_map.flatten()
            flat = flat[np.isfinite(flat)]

            # Left: full histogram + KDE
            ax = axes[0]
            ax.hist(flat, bins=60, density=True, color="#EF5350", alpha=0.65,
                    edgecolor="white", linewidth=0.3, label="All pixels")
            x_grid = np.linspace(flat.min(), flat.max(), 300)
            kde_vals = _safe_kde(flat, x_grid)
            if kde_vals is not None:
                ax.plot(x_grid, kde_vals, color="#B71C1C", lw=1.8, label="KDE")
            ax.axvline(float(np.nanmean(flat)), color="navy", ls="--", lw=1.2,
                       label=f"Mean {np.nanmean(flat):.1f}°C")
            ax.axvline(float(np.nanmedian(flat)), color="darkgreen", ls=":", lw=1.2,
                       label=f"Median {np.nanmedian(flat):.1f}°C")
            ax.set_xlabel("LST (°C)"); ax.set_ylabel("Density")
            ax.set_title("Full LST Histogram + KDE"); ax.legend(fontsize=8)
            ax.text(0.97, 0.97,
                    f"n={len(flat):,}\nmean={np.nanmean(flat):.2f}°C\n"
                    f"std={np.nanstd(flat):.2f}°C\n"
                    f"[{flat.min():.1f}, {flat.max():.1f}]",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            # Right: urban vs rural if masks provided
            ax = axes[1]
            if urban_mask is not None and rural_mask is not None:
                u_vals = lst_map[urban_mask]
                r_vals = lst_map[rural_mask]
                u_vals = u_vals[np.isfinite(u_vals)]
                r_vals = r_vals[np.isfinite(r_vals)]
                for vals, color, label in [
                    (u_vals, "#E53935", "Urban"),
                    (r_vals, "#43A047", "Rural"),
                ]:
                    ax.hist(vals, bins=50, density=True, alpha=0.55,
                            color=color, edgecolor="white", linewidth=0.3, label=label)
                    xg = np.linspace(vals.min(), vals.max(), 200)
                    kd = _safe_kde(vals, xg)
                    if kd is not None:
                        ax.plot(xg, kd, color=color, lw=1.6)
                # Cohen's d
                _pooled = np.sqrt((u_vals.std()**2 + r_vals.std()**2) / 2)
                d_cohen = ((u_vals.mean() - r_vals.mean()) / _pooled
                           if _pooled > 1e-8 else float("nan"))
                ax.set_title(f"Urban vs Rural LST  (Cohen d={d_cohen:.2f})")
                ax.legend(fontsize=8)
            else:
                # Fallback: spatial heatmap as an image
                im = ax.imshow(lst_map, cmap=_THERMAL_CMAP, interpolation="bilinear")
                plt.colorbar(im, ax=ax, label="LST (°C)", shrink=0.85)
                ax.set_title("LST Spatial Map"); ax.axis("off")
            ax.set_xlabel("LST (°C)"); ax.set_ylabel("Density")

            plt.tight_layout()
            return self._save(fig, "01_lst_distribution")
        except Exception as e:
            logger.warning(f"plot_lst_distribution failed: {e}")

    # ── 2. UHI intensity diagnostics ─────────────────────────────────────────

    def plot_uhi_intensity(self, uhi_map: np.ndarray):
        """
        Histogram, spatial map, and cumulative distribution of UHI intensity.

        Args:
            uhi_map: 2-D UHI intensity array (H, W) in °C (urban − rural).
        """
        try:
            self._use_style()
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            fig.suptitle("Analysis – UHI Intensity Diagnostics",
                         fontsize=14, fontweight="bold")

            flat = uhi_map.flatten()
            flat = flat[np.isfinite(flat)]

            # [0] Histogram + KDE
            ax = axes[0]
            ax.hist(flat, bins=60, density=True, color="#FB8C00", alpha=0.7,
                    edgecolor="white", linewidth=0.3)
            x_grid = np.linspace(flat.min(), flat.max(), 300)
            kde_vals = _safe_kde(flat, x_grid)
            if kde_vals is not None:
                ax.plot(x_grid, kde_vals, color="#E65100", lw=1.8, label="KDE")
            ax.axvline(0, color="black", ls="--", lw=1.2, label="Neutral (0°C)")
            ax.axvline(float(np.nanmean(flat)), color="navy", ls=":", lw=1.2,
                       label=f"Mean {np.nanmean(flat):.2f}°C")
            ax.set_xlabel("UHI Intensity (°C)"); ax.set_ylabel("Density")
            ax.set_title("UHI Distribution"); ax.legend(fontsize=8)
            ax.text(0.97, 0.97,
                    f"mean={np.nanmean(flat):.2f}°C\nstd={np.nanstd(flat):.2f}°C\n"
                    f"% > 0°C: {(flat > 0).mean()*100:.1f}%\n"
                    f"% > 2°C: {(flat > 2).mean()*100:.1f}%",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            # [1] Spatial map
            ax = axes[1]
            vmax_abs = max(abs(float(np.nanpercentile(flat, 2))),
                           abs(float(np.nanpercentile(flat, 98))))
            norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)
            im = ax.imshow(uhi_map, cmap="RdBu_r", norm=norm,
                           interpolation="bilinear")
            plt.colorbar(im, ax=ax, label="UHI (°C)", shrink=0.85)
            ax.set_title("UHI Spatial Map"); ax.axis("off")

            # [2] Cumulative distribution
            ax = axes[2]
            sorted_vals = np.sort(flat)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            ax.plot(sorted_vals, cdf * 100, color="#5E35B1", lw=1.8)
            for threshold, color, ls in [(0, "gray", "--"), (2, "orange", "-."),
                                         (3, "red", ":")]:
                pct = float((flat <= threshold).mean() * 100)
                ax.axvline(threshold, color=color, ls=ls, lw=1.0,
                           label=f"{threshold}°C → {pct:.0f}%ile")
            ax.set_xlabel("UHI Intensity (°C)"); ax.set_ylabel("Cumulative % of pixels")
            ax.set_title("CDF of UHI Intensity"); ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return self._save(fig, "02_uhi_intensity")
        except Exception as e:
            logger.warning(f"plot_uhi_intensity failed: {e}")

    # ── 3. UHI classification breakdown ──────────────────────────────────────

    def plot_uhi_classification(self, uhi_map: np.ndarray):
        """
        Pie chart + classified spatial map of UHI intensity categories.

        Args:
            uhi_map: 2-D UHI intensity array (H, W).
        """
        try:
            self._use_style()
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("Analysis – UHI Classification Breakdown",
                         fontsize=14, fontweight="bold")

            classified = np.zeros_like(uhi_map, dtype=np.int8)
            classified[uhi_map < 0]                            = 0
            classified[(uhi_map >= 0) & (uhi_map < 1)]        = 1
            classified[(uhi_map >= 1) & (uhi_map < 2)]        = 2
            classified[(uhi_map >= 2) & (uhi_map < 3)]        = 3
            classified[uhi_map >= 3]                           = 4

            counts = [int((classified == i).sum()) for i in range(5)]
            labels = _UHI_CAT_LABELS
            colors = _UHI_CAT_COLORS

            # [0] Pie
            ax = axes[0]
            non_zero = [(c, l, col) for c, l, col in zip(counts, labels, colors) if c > 0]
            if non_zero:
                c_vals, l_vals, col_vals = zip(*non_zero)
                ax.pie(c_vals, labels=l_vals, colors=col_vals, autopct="%1.1f%%",
                       startangle=90, pctdistance=0.8)
                ax.set_title("Area Fraction by Category")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)

            # [1] Spatial classified map
            ax = axes[1]
            from matplotlib.colors import ListedColormap, BoundaryNorm as BN
            cmap_cls = ListedColormap(colors)
            bnorm    = BN([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap_cls.N)
            im = ax.imshow(classified, cmap=cmap_cls, norm=bnorm,
                           interpolation="nearest")
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], shrink=0.85)
            cbar.ax.set_yticklabels(labels, fontsize=7)
            ax.set_title("UHI Classification Map"); ax.axis("off")

            plt.tight_layout()
            return self._save(fig, "03_uhi_classification")
        except Exception as e:
            logger.warning(f"plot_uhi_classification failed: {e}")

    # ── 4. Gi* hotspot diagnostics ────────────────────────────────────────────

    def plot_gi_star_diagnostics(self, lst_map: np.ndarray,
                                 gi_star: np.ndarray):
        """
        Gi* spatial map, histogram, and scatter vs LST.

        Args:
            lst_map: 2-D LST array (H, W).
            gi_star: 2-D Gi* statistic array (H, W).
        """
        try:
            self._use_style()
            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            fig.suptitle("Analysis – Getis-Ord Gi* Hotspot Diagnostics",
                         fontsize=14, fontweight="bold")

            gi_flat  = gi_star.flatten()
            lst_flat = lst_map.flatten()
            mask     = np.isfinite(gi_flat) & np.isfinite(lst_flat)
            gi_flat  = gi_flat[mask]
            lst_flat = lst_flat[mask]

            # [0] Gi* spatial map
            ax = axes[0]
            vmax = float(np.abs(np.nanpercentile(gi_flat, [2, 98])).max())
            # Guard: TwoSlopeNorm requires vmin < vcenter < vmax strictly.
            # When all Gi* values are the same sign (or near-zero), vmax can be
            # 0 or equal to |vmin|, collapsing the norm → ValueError.
            if vmax < 1e-6:
                vmax = 1e-6
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            im = ax.imshow(gi_star, cmap="RdBu_r", norm=norm,
                           interpolation="bilinear")
            plt.colorbar(im, ax=ax, label="Gi* statistic", shrink=0.85)
            ax.set_title("Gi* Spatial Distribution"); ax.axis("off")

            # [1] Histogram + significance lines
            ax = axes[1]
            ax.hist(gi_flat, bins=60, density=True, color="#26C6DA", alpha=0.7,
                    edgecolor="white", linewidth=0.3)
            x_grid  = np.linspace(gi_flat.min(), gi_flat.max(), 300)
            kde_vals = _safe_kde(gi_flat, x_grid)
            if kde_vals is not None:
                ax.plot(x_grid, kde_vals, color="#00838F", lw=1.8)
            for z, label, color in [(1.65, "90%", "gold"),
                                     (1.96, "95%", "orange"),
                                     (2.58, "99%", "red")]:
                ax.axvline( z, color=color, ls="--", lw=1.0, label=f"+{z} ({label})")
                ax.axvline(-z, color=color, ls="--", lw=1.0)
            ax.set_xlabel("Gi* statistic"); ax.set_ylabel("Density")
            ax.set_title("Gi* Distribution + Significance Thresholds")
            ax.legend(fontsize=8)
            pct_hot = float((gi_flat > 2.58).mean() * 100)
            ax.text(0.97, 0.97,
                    f"n={len(gi_flat):,}\nmean={gi_flat.mean():.2f}\n"
                    f"std={gi_flat.std():.2f}\n% > 2.58: {pct_hot:.1f}%",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            # [2] Scatter: LST vs Gi*
            ax = axes[2]
            idx = np.random.choice(len(gi_flat), min(8000, len(gi_flat)), replace=False)
            density = _safe_kde_2d(lst_flat[idx], gi_flat[idx])
            if density is not None:
                sc = ax.scatter(lst_flat[idx], gi_flat[idx], c=density,
                                cmap="plasma", s=6, alpha=0.7)
                plt.colorbar(sc, ax=ax, label="Density")
            else:
                ax.scatter(lst_flat[idx], gi_flat[idx],
                           c=gi_flat[idx], cmap="plasma", s=6, alpha=0.7)
            # Guard: pearsonr raises ConstantInputWarning (and returns nan r/p)
            # when either array has zero variance — e.g. all Gi* = 0 because
            # the LST map had near-zero variance going into calculate_gi_star().
            if lst_flat.std() > 1e-8 and gi_flat.std() > 1e-8:
                r, p = stats.pearsonr(lst_flat, gi_flat)
                _r_label = f"r={r:.3f}, p={p:.2e}"
            else:
                r, p = float("nan"), float("nan")
                _r_label = "r=N/A (constant input)"
            ax.set_xlabel("LST (°C)"); ax.set_ylabel("Gi* statistic")
            ax.set_title(f"LST vs Gi*  ({_r_label})")
            ax.axhline(2.58, color="red", ls="--", lw=0.8, label="99% threshold")
            ax.legend(fontsize=8)

            plt.tight_layout()
            return self._save(fig, "04_gi_star_diagnostics")
        except Exception as e:
            logger.warning(f"plot_gi_star_diagnostics failed: {e}")

    # ── 5. Hotspot ranking ────────────────────────────────────────────────────

    def plot_hotspot_ranking(self, hotspots_df: pd.DataFrame):
        """
        Top-N hotspot bar chart and area-vs-LST scatter.

        Args:
            hotspots_df: DataFrame produced by HotspotDetector.prioritize_hotspots().
        """
        try:
            if hotspots_df is None or len(hotspots_df) == 0:
                logger.warning("plot_hotspot_ranking: empty hotspots_df, skipping")
                return
            self._use_style()
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Analysis – Hotspot Ranking", fontsize=14, fontweight="bold")

            top = hotspots_df.sort_values("priority_score", ascending=False).head(15)

            # [0] Bar chart
            ax = axes[0]
            ranks = [f"#{int(r)}" for r in top["rank"]] if "rank" in top else \
                    [str(i + 1) for i in range(len(top))]
            scores = top["priority_score"].values
            bars = ax.barh(ranks[::-1], scores[::-1],
                           color=plt.cm.Reds(scores[::-1] / max(scores.max(), 1e-8)),  # type: ignore
                           edgecolor="white")
            ax.set_xlabel("Priority Score"); ax.set_title("Top-15 Hotspots by Priority")
            ax.set_xlim(0, 1.05)

            # [1] Area vs mean LST scatter
            ax = axes[1]
            if "area_km2" in hotspots_df and "mean_lst" in hotspots_df:
                sc = ax.scatter(hotspots_df["area_km2"],
                                hotspots_df["mean_lst"],
                                c=hotspots_df["priority_score"],
                                cmap="hot", s=40, alpha=0.8, edgecolors="gray", lw=0.3)
                plt.colorbar(sc, ax=ax, label="Priority score")
                ax.set_xlabel("Area (km²)"); ax.set_ylabel("Mean LST (°C)")
                ax.set_title("Hotspot Area vs Mean LST")
            else:
                ax.text(0.5, 0.5, "area_km2 / mean_lst columns not found",
                        ha="center", va="center", transform=ax.transAxes)

            plt.tight_layout()
            return self._save(fig, "05_hotspot_ranking")
        except Exception as e:
            logger.warning(f"plot_hotspot_ranking failed: {e}")

    # ── 6. Spatial autocorrelation ────────────────────────────────────────────

    def plot_spatial_autocorrelation(self, uhi_map: np.ndarray,
                                     max_lag: int = 20):
        """
        Moran's I autocorrelogram showing spatial dependency of UHI intensity.

        Args:
            uhi_map:  2-D UHI intensity array (H, W).
            max_lag:  Maximum pixel lag for autocorrelation.
        """
        try:
            self._use_style()
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Analysis – Spatial Autocorrelation of UHI",
                         fontsize=14, fontweight="bold")

            def _fast_corr(a: np.ndarray, b: np.ndarray) -> float:
                a = a - a.mean(); b = b - b.mean()
                denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
                return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0

            # [0] 2-D UHI map
            ax = axes[0]
            im = ax.imshow(uhi_map, cmap="hot", interpolation="bilinear")
            plt.colorbar(im, ax=ax, label="UHI (°C)", shrink=0.85)
            ax.set_title("UHI Intensity Map"); ax.axis("off")

            # [1] Autocorrelogram — numpy only, no stats.pearsonr overhead
            ax = axes[1]
            lags, corrs = [], []
            for lag in range(1, max_lag + 1):
                h, w = uhi_map.shape
                a = uhi_map[:h - lag, :w - lag].ravel()
                b = uhi_map[lag:, lag:].ravel()
                finite = np.isfinite(a) & np.isfinite(b)
                if finite.sum() > 10:
                    lags.append(lag)
                    corrs.append(_fast_corr(a[finite], b[finite]))
            ax.bar(lags, corrs, color="#42A5F5", edgecolor="white", linewidth=0.4)
            ax.axhline(0, color="black", lw=0.8)
            ax.axhline( 1.96 / np.sqrt(uhi_map.size), color="red",
                        ls="--", lw=0.9, label="95% CI")
            ax.axhline(-1.96 / np.sqrt(uhi_map.size), color="red", ls="--", lw=0.9)
            ax.set_xlabel("Pixel Lag"); ax.set_ylabel("Pearson r")
            ax.set_title("UHI Spatial Autocorrelogram"); ax.legend(fontsize=8)

            plt.tight_layout()
            return self._save(fig, "06_spatial_autocorrelation")
        except Exception as e:
            logger.warning(f"plot_spatial_autocorrelation failed: {e}")

    # ── 7. Validation ─────────────────────────────────────────────────────────

    def plot_validation(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Predicted vs observed scatter, residual distribution, and calibration curve.

        Args:
            predictions:  Flat predicted LST array (°C).
            ground_truth: Flat observed LST array (°C).
        """
        try:
            self._use_style()

            preds = np.asarray(predictions).flatten()
            gt    = np.asarray(ground_truth).flatten()
            mask  = np.isfinite(preds) & np.isfinite(gt)
            preds, gt = preds[mask], gt[mask]

            if len(preds) < 2:
                logger.warning("plot_validation: fewer than 2 finite samples, skipping")
                return

            # Subsample for scatter / KDE — avoids OOM / freeze on 19M-element arrays.
            # linregress and metrics still use the full arrays for accuracy.
            MAX_SCATTER = 10_000
            if len(preds) > MAX_SCATTER:
                idx = np.random.choice(len(preds), MAX_SCATTER, replace=False)
                preds_plot = preds[idx]
                gt_plot    = gt[idx]
                logger.info(f"  plot_validation: subsampled {len(preds):,} → {MAX_SCATTER:,} "
                            f"points for scatter/KDE (metrics use full array)")
            else:
                preds_plot = preds
                gt_plot    = gt

            from scipy.stats import linregress
            slope, intercept, r_val, *_ = linregress(gt, preds)
            residuals      = preds      - gt
            residuals_plot = preds_plot - gt_plot

            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            fig.suptitle("Analysis – Validation: Predicted vs Observed",
                         fontsize=14, fontweight="bold")

            # [0] Scatter
            ax = axes[0]
            density = _safe_kde_2d(gt_plot, preds_plot)
            if density is not None:
                sc = ax.scatter(gt_plot, preds_plot, c=density, cmap="plasma",
                                s=8, alpha=0.7)
                plt.colorbar(sc, ax=ax, label="Density")
            else:
                ax.scatter(gt_plot, preds_plot, c=np.abs(residuals_plot),
                           cmap="RdYlGn_r", s=8, alpha=0.6)
            lo, hi = min(gt.min(), preds.min()), max(gt.max(), preds.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect")
            fit_x = np.linspace(lo, hi, 200)
            ax.plot(fit_x, slope * fit_x + intercept, "r-", lw=1.4,
                    label=f"Fit slope={slope:.3f}")
            ax.set_xlabel("Observed (°C)"); ax.set_ylabel("Predicted (°C)")
            ax.set_title("Predicted vs Observed"); ax.legend(fontsize=8)
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            mae  = float(np.mean(np.abs(residuals)))
            mbe  = float(np.mean(residuals))
            ax.text(0.03, 0.97,
                    f"R²={r_val**2:.4f}\nRMSE={rmse:.3f}°C\n"
                    f"MAE={mae:.3f}°C\nMBE={mbe:.3f}°C\nslope={slope:.3f}\n"
                    f"n={len(preds):,}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            # [1] Residuals histogram + KDE (subsample)
            ax = axes[1]
            ax.hist(residuals_plot, bins=60, density=True, color="#5C6BC0",
                    alpha=0.7, edgecolor="white", linewidth=0.3)
            xr = np.linspace(residuals_plot.min(), residuals_plot.max(), 300)
            kde_r = _safe_kde(residuals_plot, xr)
            if kde_r is not None:
                ax.plot(xr, kde_r, color="#283593", lw=1.8)
            ax.axvline(0, color="black", lw=1.2, ls="--", label="Zero bias")
            ax.axvline( residuals.std(), color="orange", ls=":", lw=1.0, label="±1σ")
            ax.axvline(-residuals.std(), color="orange", ls=":", lw=1.0)
            ax.set_xlabel("Residual (°C)"); ax.set_ylabel("Density")
            ax.set_title("Residual Distribution"); ax.legend(fontsize=8)

            # [2] Calibration curve — quantile–quantile on full arrays
            ax = axes[2]
            n_bins = 20
            quantiles = np.linspace(0, 100, n_bins + 1)
            obs_bins  = np.percentile(gt,    quantiles)
            pred_bins = np.percentile(preds, quantiles)
            ax.plot(obs_bins, pred_bins, color="#E53935", lw=1.6, marker="o",
                    ms=4, label="Predicted quantiles")
            lo_q = min(obs_bins.min(), pred_bins.min())
            hi_q = max(obs_bins.max(), pred_bins.max())
            ax.plot([lo_q, hi_q], [lo_q, hi_q], "k--", lw=1.1,
                    label="Perfect calibration")
            ax.set_xlabel("Observed LST bin midpoint (°C)")
            ax.set_ylabel("Predicted LST (°C)")
            ax.set_title("Calibration Curve (quantile–quantile)")
            ax.legend(fontsize=8)

            plt.tight_layout()
            return self._save(fig, "07_validation")
        except Exception as e:
            logger.warning(f"plot_validation failed: {e}", exc_info=True)

    # ── 8. Summary dashboard ──────────────────────────────────────────────────

    def plot_summary_dashboard(self,
                               lst_map: np.ndarray,
                               uhi_map: np.ndarray = None,
                               gi_star: np.ndarray = None,
                               reference_temps: Dict = None):
        """
        Single-page overview: LST map · UHI map · Gi* map · reference temperature
        bar chart.

        Args:
            lst_map:         2-D LST array (H, W).
            uhi_map:         Optional 2-D UHI intensity array.
            gi_star:         Optional 2-D Gi* statistic array.
            reference_temps: Optional dict with keys T_urban, T_rural, T_diff
                             (as returned by UHIAnalyzer.define_reference_areas()).
        """
        try:
            self._use_style()
            n_maps = 1 + (uhi_map is not None) + (gi_star is not None)
            has_ref = reference_temps is not None and "T_urban" in reference_temps

            n_cols = n_maps + (1 if has_ref else 0)
            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
            if n_cols == 1:
                axes = [axes]
            fig.suptitle("Analysis – Summary Dashboard",
                         fontsize=15, fontweight="bold")

            col = 0

            # LST map
            im = axes[col].imshow(lst_map, cmap=_THERMAL_CMAP, interpolation="bilinear")
            plt.colorbar(im, ax=axes[col], label="LST (°C)", shrink=0.85)
            axes[col].set_title(f"LST  (mean={np.nanmean(lst_map):.1f}°C)")
            axes[col].axis("off"); col += 1

            # UHI map
            if uhi_map is not None:
                vabs = max(float(np.nanpercentile(np.abs(uhi_map), 98)), 1e-6)
                norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
                im2  = axes[col].imshow(uhi_map, cmap="RdBu_r", norm=norm,
                                        interpolation="bilinear")
                plt.colorbar(im2, ax=axes[col], label="UHI (°C)", shrink=0.85)
                axes[col].set_title(f"UHI Intensity  (mean={np.nanmean(uhi_map):.1f}°C)")
                axes[col].axis("off"); col += 1

            # Gi* map
            if gi_star is not None:
                vabs = max(float(np.nanpercentile(np.abs(gi_star), 98)), 1e-6)
                norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
                im3  = axes[col].imshow(gi_star, cmap="RdBu_r", norm=norm,
                                        interpolation="bilinear")
                plt.colorbar(im3, ax=axes[col], label="Gi*", shrink=0.85)
                axes[col].set_title("Gi* Hotspot Map")
                axes[col].axis("off"); col += 1

            # Reference temps bar
            if has_ref:
                ax = axes[col]
                t_u = float(reference_temps["T_urban"])
                t_r = float(reference_temps["T_rural"])
                t_d = float(reference_temps.get("T_diff", t_u - t_r))
                labels_bar = ["Urban", "Rural", "UHI Δ"]
                values_bar = [t_u, t_r, t_d]
                colors_bar = ["#E53935", "#43A047", "#FB8C00"]
                ax.bar(labels_bar, values_bar, color=colors_bar,
                       edgecolor="white", linewidth=0.6)
                ax.set_ylabel("Temperature (°C)")
                ax.set_title("Urban vs Rural Reference Temperatures")
                for i, v in enumerate(values_bar):
                    ax.text(i, v + 0.05 * abs(v), f"{v:.2f}°C",
                            ha="center", va="bottom", fontsize=9)

            plt.tight_layout()
            return self._save(fig, "08_summary_dashboard")
        except Exception as e:
            logger.warning(f"plot_summary_dashboard failed: {e}")

    # ── plot_all ──────────────────────────────────────────────────────────────

    def plot_all(self,
                 lst_map: np.ndarray,
                 uhi_map: np.ndarray = None,
                 gi_star: np.ndarray = None,
                 hotspots_df: pd.DataFrame = None,
                 predictions: np.ndarray = None,
                 ground_truth: np.ndarray = None,
                 urban_mask: np.ndarray = None,
                 rural_mask: np.ndarray = None,
                 reference_temps: Dict = None) -> List[Path]:
        """
        Run the full diagnostic suite and return the list of saved file paths.

        Args:
            lst_map:         2-D LST array (H, W) in °C.
            uhi_map:         2-D UHI intensity array (optional).
            gi_star:         2-D Gi* statistic array (optional).
            hotspots_df:     Hotspot prioritisation DataFrame (optional).
            predictions:     Flat predicted LST array for validation (optional).
            ground_truth:    Flat observed LST array for validation (optional).
            urban_mask:      Boolean mask for urban pixels (optional).
            rural_mask:      Boolean mask for rural pixels (optional).
            reference_temps: Dict with T_urban / T_rural / T_diff (optional).

        Returns:
            List of Path objects for each saved figure.
        """
        logger.info("=" * 70)
        logger.info("  AnalysisDiagnosticsPlotter — full diagnostic suite")
        logger.info("=" * 70)
        saved: List[Path] = []

        def _run(label, fn, *args, **kwargs):
            try:
                p = fn(*args, **kwargs)
                if p is not None:
                    saved.append(p)
            except Exception as _e:
                logger.warning(f"  ⚠️ {label} failed: {_e}")

        _run("LST distribution",       self.plot_lst_distribution,
             lst_map, urban_mask, rural_mask)

        if uhi_map is not None:
            _run("UHI intensity",       self.plot_uhi_intensity,      uhi_map)
            _run("UHI classification",  self.plot_uhi_classification,  uhi_map)
            _run("Spatial autocorr.",   self.plot_spatial_autocorrelation, uhi_map)

        if gi_star is not None:
            _run("Gi* diagnostics",     self.plot_gi_star_diagnostics, lst_map, gi_star)

        if hotspots_df is not None and len(hotspots_df) > 0:
            _run("Hotspot ranking",     self.plot_hotspot_ranking,     hotspots_df)

        if predictions is not None and ground_truth is not None:
            _run("Validation",          self.plot_validation,
                 predictions, ground_truth)

        _run("Summary dashboard",   self.plot_summary_dashboard,
             lst_map, uhi_map, gi_star, reference_temps)

        logger.info(f"\n✅  AnalysisDiagnosticsPlotter: {len(saved)} figures → {self.save_dir}")
        return saved


if __name__ == "__main__":
    logger.info("UHI Analysis module loaded — enhanced with diagnostics")
    logger.info("Classes  : UHIAnalyzer, HotspotDetector, ValidationAnalyzer")
    logger.info("Helper   : run_all_diagnostics()")