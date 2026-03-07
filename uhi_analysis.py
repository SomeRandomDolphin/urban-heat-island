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
    """Return KDE values on x_grid, or None if data has near-zero variance."""
    flat = np.asarray(data).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.std() < 1e-6 or len(flat) < 2:
        logger.warning("_safe_kde: near-zero variance or insufficient data — skipping KDE")
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

        uhi_pos = self.uhi_map[self.uhi_map > 0]

        q5, q25, q75, q95 = np.percentile(self.uhi_map, [5, 25, 75, 95])
        skew = float(stats.skew(self.uhi_map.ravel()))
        kurt = float(stats.kurtosis(self.uhi_map.ravel()))

        stat_dict: Dict[str, float] = {
            "max_intensity":          float(self.uhi_map.max()),
            "min_intensity":          float(self.uhi_map.min()),
            "mean_intensity":         float(self.uhi_map.mean()),
            "mean_positive_intensity":float(uhi_pos.mean()) if len(uhi_pos) else 0.0,
            "median_intensity":       float(np.median(self.uhi_map)),
            "std_intensity":          float(self.uhi_map.std()),
            "p5":  float(q5),
            "p25": float(q25),
            "p75": float(q75),
            "p95": float(q95),
            "skewness":               skew,
            "kurtosis":               kurt,
            "spatial_extent_km2":     float((self.uhi_map > 2).sum() * 0.0025),
            "magnitude":              float(uhi_pos.sum()),
            "pct_positive":           float(len(uhi_pos) / self.uhi_map.size * 100),
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
        vabs = np.percentile(np.abs(flat), 98)
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
        """
        if self.uhi_map is None:
            raise ValueError("Call calculate_uhi_intensity() first.")

        _diag_header("PLOT — SPATIAL AUTOCORRELATION (LAG PROFILE)")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Spatial Autocorrelation of UHI", fontsize=14, fontweight="bold")

        uhi = self.uhi_map

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
                    b = data[lag:, :].ravel()
                    r, _ = stats.pearsonr(a, b)
                    acf_vals.append(r)
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

            # Log key lags
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
        logger.info(f"Calculating Gi* statistic (radius={search_radius}m)...")

        height, width = self.lst_map.shape
        gi_star = np.zeros_like(self.lst_map)

        y_coords, x_coords = np.mgrid[0:height, 0:width] * self.resolution

        X_bar = np.mean(self.lst_map)
        S     = np.std(self.lst_map)
        n     = self.lst_map.size

        step = max(1, int(search_radius / self.resolution / 2))

        for i in tqdm(range(0, height, step), desc="Gi* calculation"):
            for j in range(0, width, step):
                dist_y    = (y_coords - y_coords[i, j]) ** 2
                dist_x    = (x_coords - x_coords[i, j]) ** 2
                distances = np.sqrt(dist_y + dist_x)

                weights = np.zeros_like(distances)
                in_r    = distances <= search_radius
                weights[in_r] = 1.0 / (distances[in_r] + 1e-6)
                w_sum   = weights.sum()
                if w_sum > 0:
                    weights /= w_sum

                weighted_sum   = np.sum(weights * self.lst_map)
                sum_weights    = np.sum(weights)
                sum_weights_sq = np.sum(weights ** 2)

                if sum_weights > 0:
                    numerator   = weighted_sum - X_bar * sum_weights
                    denominator = S * np.sqrt(
                        (n * sum_weights_sq - sum_weights ** 2) / (n - 1))
                    if denominator > 1e-6:
                        gi_star[i, j] = numerator / denominator

        if step > 1:
            from scipy.interpolate import RegularGridInterpolator
            y_sub = np.arange(0, height, step)
            x_sub = np.arange(0, width, step)
            interp = RegularGridInterpolator(
                (y_sub, x_sub), gi_star[::step, ::step],
                method="linear", bounds_error=False, fill_value=0)
            y_full, x_full = np.mgrid[0:height, 0:width]
            gi_star = interp(
                np.column_stack([y_full.ravel(), x_full.ravel()])
            ).reshape(height, width)

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
        # trend line
        r, p = stats.pearsonr(self.lst_map.ravel()[idx], gi.ravel()[idx])
        ax.set_xlabel("LST (°C)")
        ax.set_ylabel("Gi*")
        ax.set_title(f"LST vs Gi*  (r={r:.3f}, p={p:.2e})")
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


if __name__ == "__main__":
    logger.info("UHI Analysis module loaded — enhanced with diagnostics")
    logger.info("Classes  : UHIAnalyzer, HotspotDetector, ValidationAnalyzer")
    logger.info("Helper   : run_all_diagnostics()")