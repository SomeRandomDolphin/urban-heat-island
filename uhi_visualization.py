"""
UHI Visualization - Create maps, plots, and output products
Enhanced with additional diagnostic matplotlib figures
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from PIL import Image
import json
from scipy import stats

from config import *

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def _safe_kde(data: np.ndarray, x_grid: np.ndarray) -> Optional[np.ndarray]:
    """
    Evaluate a Gaussian KDE on x_grid, returning None if the data have
    near-zero variance (which makes gaussian_kde raise LinAlgError) or
    contain inf/NaN values that would propagate into the KDE bandwidth matrix.
    Callers should skip the KDE line when None is returned.
    """
    flat = np.asarray(data, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.std() < 1e-6 or len(flat) < 2:
        logger.warning("_safe_kde: data has near-zero variance or too few points — skipping KDE")
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


def _safe_kde_2d(x: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """
    Evaluate a 2-D Gaussian KDE for density-coloured scatterplots.
    Returns None when the data are degenerate (singular covariance) or
    contain non-finite values.
    """
    try:
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        # Guard: both dimensions need variance
        if x.std() < 1e-6 or y.std() < 1e-6:
            raise ValueError("Near-zero variance in one dimension")
        xy = np.vstack([x, y])
        return stats.gaussian_kde(xy)(xy)
    except Exception as _e:
        logger.warning(f"_safe_kde_2d: 2-D KDE failed ({_e}) — falling back to uniform density")
        return None


# ─── Shared style ─────────────────────────────────────────────────────────────
_THERMAL_COLORS = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
                   '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
_THERMAL_CMAP   = LinearSegmentedColormap.from_list('thermal', _THERMAL_COLORS, N=256)
_UHI_CAT_COLORS = ['#3288bd', '#99d594', '#fee08b', '#fc8d59', '#d53e4f', '#800026']
_UHI_CAT_LABELS = ['No UHI / Cooling', 'Weak (0–2 °C)',
                   'Moderate (2–4 °C)', 'Strong (4–6 °C)',
                   'Very Strong (6–8 °C)', 'Extreme (>8 °C)']

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
})


# ══════════════════════════════════════════════════════════════════════════════
# UHIVisualizer  (original + new diagnostic charts)
# ══════════════════════════════════════════════════════════════════════════════

class UHIVisualizer:
    """Create visualizations for UHI analysis – original maps + rich diagnostics."""

    def __init__(self, figsize: Tuple[int, int] = (12, 10), dpi: int = 300):
        self.figsize = figsize
        self.dpi     = dpi
        sns.set_style("whitegrid")
        plt.rcParams['font.size']          = 10
        plt.rcParams['axes.labelsize']     = 12
        plt.rcParams['axes.titlesize']     = 14
        plt.rcParams['figure.titlesize']   = 16

    # ── original maps ─────────────────────────────────────────────────────────

    def create_lst_map(self, lst_data: np.ndarray, output_path: Path,
                       title: str = "Land Surface Temperature Map",
                       vmin: Optional[float] = None, vmax: Optional[float] = None,
                       add_colorbar: bool = True,
                       water_mask: Optional[np.ndarray] = None):
        """
        Render a Land Surface Temperature map.

        Args:
            water_mask: Optional boolean array (True = water/sea pixel).
                        Water pixels are overlaid with a semi-transparent
                        steel-blue hatch so they are visually distinct from
                        cold land surfaces and not mistaken for UHI signal.
        """
        logger.info(f"Creating LST map: {title}")
        # Compute vmin/vmax from land pixels only so the colour scale is not
        # anchored to the much colder sea temperature.
        if vmin is None or vmax is None:
            land_vals = lst_data[~water_mask] if water_mask is not None else lst_data
            land_vals = land_vals[np.isfinite(land_vals)]
            if vmin is None:
                vmin = float(np.percentile(land_vals, 2))
            if vmax is None:
                vmax = float(np.percentile(land_vals, 98))
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(lst_data, cmap=_THERMAL_CMAP, vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='auto')
        # Overlay water pixels with a distinct hatched blue mask
        if water_mask is not None and water_mask.any():
            water_rgba = np.zeros((*water_mask.shape, 4), dtype=np.float32)
            water_rgba[water_mask] = [0.27, 0.51, 0.71, 0.55]   # steel-blue, semi-transparent
            ax.imshow(water_rgba, interpolation='nearest', aspect='auto')
            from matplotlib.patches import Patch as _Patch
            ax.legend(handles=[_Patch(facecolor='#456eb4', alpha=0.55, label='Sea / Water')],
                      loc='lower right', fontsize=9)
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
                                  title: str = "UHI Intensity Map",
                                  water_mask: Optional[np.ndarray] = None):
        """
        Render a classified UHI intensity map.

        Args:
            water_mask: Optional boolean array (True = sea/water pixel).
                        Water pixels are rendered as a 6th category "Sea / Water"
                        in steel-blue so they are not conflated with the
                        "No UHI / Cooling" land class.
        """
        logger.info(f"Creating UHI intensity map: {title}")
        fig, ax = plt.subplots(figsize=self.figsize)

        # Determine water pixels: explicit mask OR NaN cells in uhi_data
        if water_mask is not None:
            sea_pixels = water_mask
        else:
            sea_pixels = ~np.isfinite(uhi_data)

        # Classify only finite (land) pixels; sea gets category 5
        classified = np.full(uhi_data.shape, -1, dtype=np.int8)
        finite = np.isfinite(uhi_data)
        classified[finite & (uhi_data < 0)]                            = 0
        classified[finite & (uhi_data >= 0) & (uhi_data < 2)]         = 1
        classified[finite & (uhi_data >= 2) & (uhi_data < 4)]         = 2
        classified[finite & (uhi_data >= 4) & (uhi_data < 6)]         = 3
        classified[finite & (uhi_data >= 6) & (uhi_data < 8)]         = 4
        classified[finite & (uhi_data >= 8)]                           = 5
        classified[sea_pixels]                                         = 6

        sea_color  = '#456eb4'   # steel-blue for sea
        cat_colors = _UHI_CAT_COLORS + [sea_color]
        cat_labels = _UHI_CAT_LABELS + ['Sea / Water']

        cmap   = mcolors.ListedColormap(cat_colors)
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5]
        norm   = mcolors.BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(classified, cmap=cmap, norm=norm,
                       interpolation='nearest', aspect='auto')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)

        from matplotlib.patches import Patch as _Patch
        legend_handles = [
            _Patch(facecolor=cat_colors[i], label=cat_labels[i])
            for i in range(len(cat_labels))
            if (classified == i).any()
        ]
        ax.legend(handles=legend_handles, loc='lower right', fontsize=8,
                  title='UHI Category', title_fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved UHI intensity map: {output_path}")
        plt.close()

    def create_hotspot_map(self, lst_data: np.ndarray, hotspot_mask: np.ndarray,
                           gi_star: np.ndarray, output_path: Path,
                           title: str = "UHI Hotspot Map"):
        logger.info(f"Creating hotspot map: {title}")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax = axes[0]
        im1 = ax.imshow(lst_data, cmap=_THERMAL_CMAP, alpha=0.8,
                        interpolation='nearest', aspect='auto')
        hotspot_overlay = np.ma.masked_where(~hotspot_mask, hotspot_mask)
        ax.contour(hotspot_overlay, levels=[0.5], colors='red',
                   linewidths=2, linestyles='solid')
        ax.set_title('LST with Hotspot Boundaries', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label('LST (°C)', fontsize=11)
        ax = axes[1]
        im2 = ax.imshow(gi_star, cmap='RdYlBu_r', vmin=-3, vmax=3,
                        interpolation='nearest', aspect='auto')
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
                               output_path: Path,
                               title: str = "Prediction Uncertainty Map"):
        logger.info(f"Creating uncertainty map: {title}")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax = axes[0]
        im1 = ax.imshow(lst_data, cmap=_THERMAL_CMAP,
                        interpolation='nearest', aspect='auto')
        ax.set_title('LST Predictions', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label('Temperature (°C)', fontsize=11)
        ax = axes[1]
        im2 = ax.imshow(uncertainty, cmap='YlOrRd',
                        interpolation='nearest', aspect='auto')
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
        logger.info("Creating statistics dashboard...")
        fig = plt.figure(figsize=(16, 12))
        gs  = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        # summary text
        ax = fig.add_subplot(gs[0, :2])
        ax.axis('off')
        stats_text = (
            f"UHI CHARACTERIZATION SUMMARY\n\n"
            f"    Maximum Intensity:           {uhi_stats['max_intensity']:.2f}°C\n"
            f"    Mean Intensity:              {uhi_stats['mean_intensity']:.2f}°C\n"
            f"    Mean Positive Intensity:     {uhi_stats['mean_positive_intensity']:.2f}°C\n"
            f"    Spatial Extent (>6°C):       {uhi_stats['spatial_extent_km2']:.2f} km²\n"
            f"    UHI Magnitude:               {uhi_stats['magnitude']:.2f}°C·pixels"
        )
        ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # pie
        ax = fig.add_subplot(gs[0, 2])
        labels = [k for k, v in classification.items() if v > 0]
        sizes  = [v for v in classification.values() if v > 0]
        colors_pie = _UHI_CAT_COLORS[:len(labels)]
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 9})
        ax.set_title('UHI Classification', fontsize=12, fontweight='bold')

        # bar
        ax = fig.add_subplot(gs[1, :])
        if len(hotspots_df) > 0:
            top_n  = min(15, len(hotspots_df))
            top_hs = hotspots_df.head(top_n)
            x      = np.arange(top_n)
            ax.bar(x - 0.175, top_hs['area_km2'], 0.35,
                   label='Area (km²)', color='steelblue', alpha=0.8)
            ax.bar(x + 0.175, top_hs['mean_lst'] / 10, 0.35,
                   label='Mean LST / 10 (°C)', color='coral', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f'#{i+1}' for i in range(top_n)])
            ax.set_xlabel('Hotspot Rank')
            ax.set_ylabel('Value')
            ax.set_title(f'Top {top_n} Priority Hotspots')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')

        # histograms
        for idx, (col, color, xlabel, subtitle) in enumerate([
            ('area_km2',       'steelblue', 'Hotspot Area (km²)',    'Size Distribution'),
            ('mean_lst',       'coral',     'Mean LST (°C)',         'Temperature Distribution'),
            ('priority_score', 'green',     'Priority Score',        'Priority Distribution'),
        ]):
            ax = fig.add_subplot(gs[2, idx])
            if len(hotspots_df) > 0 and col in hotspots_df.columns:
                ax.hist(hotspots_df[col], bins=min(20, len(hotspots_df)),
                        color=color, edgecolor='black', alpha=0.7)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3, axis='y')
            ax.set_title(subtitle, fontsize=11, fontweight='bold')

        plt.suptitle('UHI Analysis Dashboard', fontsize=18, fontweight='bold')
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved statistics dashboard: {output_path}")
        plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # NEW DIAGNOSTIC PLOTS
    # ══════════════════════════════════════════════════════════════════════════

    def create_model_comparison_plot(self, results: Dict[str, np.ndarray],
                                     ground_truth: Optional[np.ndarray],
                                     output_path: Path) -> None:
        """
        Compare CNN, GBM, and Ensemble predictions side-by-side.

        Layout (always 3 columns = CNN | GBM | Ensemble):
          Row 0  : prediction maps on shared colour scale
          Row 1  : difference from Ensemble  (CNN-Ens | GBM-Ens | GT-Ens if available)
          Row 2  : scatter vs Ensemble (or vs GT)  — only if ground_truth provided

        Fixes vs previous version:
          - GBM column is always shown (was being silently dropped when GBM had
            a different spatial shape from CNN)
          - Shared vmin/vmax clamped to P2–P98 so outlier pixels don't crush the scale
          - Per-panel stats annotation (mean, std, RMSE vs ensemble)
          - Row-1 shows CNN-Ensemble and GBM-Ensemble difference to expose the
            spatial smoothing that GBM introduces
        """
        logger.info("Creating model comparison plot...")

        # Resolve which keys are available; always try all three
        all_keys = [k for k in ('cnn', 'gbm', 'ensemble') if k in results]
        if not all_keys:
            logger.warning("  No model keys found in results — skipping model comparison")
            return

        # Force to 2-D single patch if needed
        def _to_2d(arr):
            if arr.ndim == 3:
                return arr[0]
            return arr

        maps = {k: _to_2d(results[k]) for k in all_keys}
        ens_map = maps.get('ensemble', list(maps.values())[-1])

        # Shared colour scale (robust to outliers)
        all_vals = np.concatenate([m.ravel() for m in maps.values()])
        vmin, vmax = np.percentile(all_vals, [2, 98])

        has_gt  = ground_truth is not None
        gt_2d   = _to_2d(ground_truth) if has_gt else None

        n_cols  = 3          # always CNN | GBM | Ensemble
        n_rows  = 2 + (1 if has_gt else 0)

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(7 * n_cols, 5.5 * n_rows))
        fig.suptitle("Model Comparison: CNN vs GBM vs Ensemble",
                     fontsize=15, fontweight="bold")

        col_keys = ['cnn', 'gbm', 'ensemble']

        # ── Row 0: prediction maps ─────────────────────────────────────────────
        for col, key in enumerate(col_keys):
            ax = axes[0, col]
            data = maps.get(key)
            if data is None:
                ax.text(0.5, 0.5, f"{key.upper()}\nnot available",
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.axis('off')
                continue
            im = ax.imshow(data, cmap=_THERMAL_CMAP, vmin=vmin, vmax=vmax,
                           interpolation='bilinear', aspect='auto')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="LST (°C)")
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            # annotate stats
            ax.set_title(
                f"{key.upper()}\nμ={data.mean():.2f}°C  σ={data.std():.2f}°C",
                fontsize=11)

        # ── Row 1: difference from Ensemble ───────────────────────────────────
        diff_titles = ['CNN − Ensemble', 'GBM − Ensemble', 'Ground Truth − Ensemble']
        diff_sources = [maps.get('cnn'), maps.get('gbm'), gt_2d]

        diffs = []
        for col, (diff_data, dtitle) in enumerate(zip(diff_sources, diff_titles)):
            ax = axes[1, col]
            if diff_data is None:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.axis('off')
                continue
            diff = diff_data - ens_map
            diffs.append(np.abs(diff).max())
            lim  = float(np.percentile(np.abs(diff), 98)) or 0.1
            im2  = ax.imshow(diff, cmap='RdBu_r', vmin=-lim, vmax=lim,
                             interpolation='bilinear', aspect='auto')
            fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="Δ°C")
            mae  = float(np.abs(diff).mean())
            bias = float(diff.mean())
            ax.set_title(f"{dtitle}\nMAE={mae:.3f}°C  bias={bias:+.3f}°C", fontsize=11)
            ax.set_xlabel("X"); ax.set_ylabel("Y")

        # ── Row 2 (optional): scatter vs GT ────────────────────────────────────
        if has_gt and n_rows == 3:
            from sklearn.metrics import r2_score
            for col, key in enumerate(col_keys):
                ax = axes[2, col]
                data = maps.get(key)
                if data is None:
                    ax.axis('off'); continue
                pred_f = data.ravel()
                gt_f   = gt_2d.ravel()
                mask   = ~(np.isnan(pred_f) | np.isnan(gt_f))
                pf, gf = pred_f[mask], gt_f[mask]
                idx    = np.random.choice(len(pf), min(5000, len(pf)), replace=False)
                ax.scatter(gf[idx], pf[idx], s=3, alpha=0.4, color=
                           ['#4393c3','#d6604d','#66c2a5'][col])
                mn, mx = gf.min(), gf.max()
                ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label="1:1")
                sl, ic, rv, *_ = stats.linregress(gf[idx], pf[idx])
                ax.plot([mn, mx], [sl*mn+ic, sl*mx+ic], 'r-', lw=1.5,
                        label=f"fit slope={sl:.3f}")
                r2   = r2_score(gf, pf)
                rmse = float(np.sqrt(np.mean((pf - gf)**2)))
                ax.set_title(f"{key.upper()} vs GT\nR²={r2:.3f}  RMSE={rmse:.3f}°C",
                             fontsize=11)
                ax.set_xlabel("Ground Truth (°C)"); ax.set_ylabel("Predicted (°C)")
                ax.legend(fontsize=7); ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def create_uncertainty_analysis_plot(self, predictions: np.ndarray,
                                         uncertainty: np.ndarray,
                                         output_path: Path,
                                         ground_truth: Optional[np.ndarray] = None) -> None:
        """
        6-panel uncertainty deep-dive (expanded from original 4-panel):

          [0,0] Uncertainty spatial map  — with anomaly pixel overlay
          [0,1] Uncertainty histogram    — bimodal check + anomaly flag
          [1,0] Scatter: LST vs σ        — Spearman + Pearson, coloured by density
          [1,1] 95% CI width map         — capped at 3σ_data to suppress outliers
          [2,0] Calibration curve        — expected vs actual coverage (if GT given)
          [2,1] Uncertainty gradient map — d(σ)/dx  to expose artefact boundaries

        Fixes vs previous version:
          - Anomaly pixel (bottom-left corner spike) is highlighted and flagged
          - Bimodal distribution is labelled and its modes estimated
          - 95% CI width is shown in physically meaningful range
          - New calibration curve shows if uncertainty is over/under-confident
          - σ gradient exposes the boundary artefact visible in image 1
        """
        logger.info("Creating uncertainty analysis plot...")

        fig, axes = plt.subplots(3, 2, figsize=(14, 16))
        fig.suptitle("Prediction Uncertainty Analysis", fontsize=15, fontweight="bold")

        flat = uncertainty.ravel()
        p98  = float(np.percentile(flat, 98))
        p2   = float(np.percentile(flat, 2))

        # Detect outlier / anomaly pixels (> 3 IQR above Q3)
        q1, q3   = np.percentile(flat, [25, 75])
        iqr      = q3 - q1
        anom_thr = q3 + 3 * iqr
        anom_mask = uncertainty > anom_thr
        n_anom   = int(anom_mask.sum())
        if n_anom > 0:
            logger.warning(f"  ⚠ {n_anom} anomaly pixels detected (σ > {anom_thr:.2f}°C)")

        # ── [0,0] Spatial map + anomaly overlay ───────────────────────────────
        ax = axes[0, 0]
        unc_display = np.clip(uncertainty, p2, p98)   # suppress outliers visually
        im = ax.imshow(unc_display, cmap='YlOrRd', interpolation='bilinear', aspect='auto')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="σ (°C)")
        if n_anom > 0:
            ys, xs = np.where(anom_mask)
            ax.scatter(xs, ys, c='blue', s=15, marker='x', linewidths=1.2,
                       label=f"{n_anom} anomaly px (σ>{anom_thr:.1f}°C)", zorder=5)
            ax.legend(fontsize=7, loc='upper right')
        ax.set_title(f"Uncertainty Map (σ, clipped P2–P98)\nAnomaly pixels: {n_anom}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

        # ── [0,1] Histogram with bimodal analysis ─────────────────────────────
        ax = axes[0, 1]
        flat_clip = flat[flat <= p98]
        if flat_clip.size == 0:
            flat_clip = flat  # fallback: use all values if filter produced empty array
        ax.hist(flat_clip, bins=80, color="#fc8d59", edgecolor="none",
                density=True, alpha=0.75)
        x_k = np.linspace(flat_clip.min(), flat_clip.max(), 500)
        kde_vals = _safe_kde(flat_clip, x_k)
        if kde_vals is not None:
            ax.plot(x_k, kde_vals, color="#b2182b", lw=2, label="KDE")

        # find local maxima (modes) — bimodal indicator
        from scipy.signal import find_peaks
        peaks, props = find_peaks(kde_vals if kde_vals is not None else np.array([]),
                                  prominence=(kde_vals.max() * 0.1 if kde_vals is not None else 0),
                                  distance=10)
        for pk in peaks:
            ax.axvline(x_k[pk], color='darkred', lw=1.2, ls=':',
                       label=f"Mode ≈ {x_k[pk]:.2f}°C")
        if len(peaks) > 1:
            ax.text(0.5, 0.97, f"⚠ Bimodal distribution detected ({len(peaks)} modes)\n"
                               "— may indicate domain mismatch or batch artefact",
                    transform=ax.transAxes, va='top', ha='center', fontsize=8,
                    color='darkred', bbox=dict(fc='#fff3e0', ec='orange', alpha=0.9))

        for pct, col in [(50, "#fee08b"), (90, "#fc8d59"), (95, "#d53e4f")]:
            val = np.percentile(flat, pct)
            ax.axvline(val, color=col, lw=1.5, ls="--", label=f"P{pct}={val:.2f}°C")
        ax.set_xlabel("Uncertainty σ (°C)")
        ax.set_ylabel("Density")
        ax.set_title(f"Uncertainty Distribution  (mean={flat.mean():.2f}°C)")
        ax.legend(fontsize=7.5)

        # ── [1,0] Density-coloured scatter: LST vs σ ──────────────────────────
        ax = axes[1, 0]
        pred_flat = predictions.ravel()
        unc_flat  = uncertainty.ravel()
        # use only non-outlier, finite points for the scatter
        valid = (unc_flat <= anom_thr) & np.isfinite(unc_flat) & np.isfinite(pred_flat)
        pf, uf = pred_flat[valid], unc_flat[valid]

        if len(pf) < 2:
            # Not enough data after filtering — skip scatter, show a notice
            ax.text(0.5, 0.5, "Insufficient valid points\nfor scatter plot",
                    ha='center', va='center', transform=ax.transAxes, fontsize=10,
                    color='grey')
            ax.set_title("LST vs Uncertainty\n(insufficient data)")
            ax.set_xlabel("Predicted LST (°C)")
            ax.set_ylabel("Uncertainty σ (°C)")
        else:
            idx = np.random.choice(len(pf), min(8000, len(pf)), replace=False)
            density = _safe_kde_2d(pf[idx], uf[idx])
            sc = ax.scatter(pf[idx], uf[idx],
                            c=density if density is not None else uf[idx],
                            cmap="plasma", s=5, alpha=0.6)
            fig.colorbar(sc, ax=ax, label="Density")
            if len(pf[idx]) >= 2:
                r_p, _ = stats.pearsonr(pf[idx], uf[idx])
                r_s, _ = stats.spearmanr(pf[idx], uf[idx])
            else:
                r_p = r_s = float('nan')
            ax.set_xlabel("Predicted LST (°C)")
            ax.set_ylabel("Uncertainty σ (°C)")
            ax.set_title(f"LST vs Uncertainty\n"
                         f"Pearson r={r_p:.3f}  Spearman ρ={r_s:.3f}")
            ax.grid(True, alpha=0.3)
            # flag high-uncertainty / high-LST regime
            high_unc = uf > np.percentile(uf, 90)
            high_lst = pf > np.percentile(pf, 90)
            n_both   = int((high_unc & high_lst).sum())
            ax.text(0.02, 0.97,
                    f"High-σ & high-LST pixels: {n_both} ({n_both/len(pf)*100:.1f}%)",
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(fc='white', ec='grey', alpha=0.8))

        # ── [1,1] 95% CI width map — capped ───────────────────────────────────
        ax = axes[1, 1]
        ci_width = 2 * 1.96 * uncertainty
        ci_p98   = np.percentile(ci_width, 98)
        ci_disp  = np.clip(ci_width, 0, ci_p98)
        im2 = ax.imshow(ci_disp, cmap="hot_r", interpolation='bilinear', aspect='auto')
        fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="95% CI width (°C)")
        mean_ci  = float(ci_width.mean())
        ax.set_title(f"95% CI Width (mean={mean_ci:.2f}°C, clipped at P98={ci_p98:.2f}°C)\n"
                     f"{'⚠ CI > 10°C — consider retraining MC Dropout' if mean_ci > 10 else '✓ CI width in acceptable range'}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

        # ── [2,0] Calibration curve (if GT available) ─────────────────────────
        ax = axes[2, 0]
        if ground_truth is not None:
            gt_flat = ground_truth.ravel()
            err_flat = np.abs(predictions.ravel() - gt_flat)
            # for each confidence level, check what fraction of errors fall within CI
            alphas  = np.linspace(0.05, 0.99, 30)
            expected, observed = [], []
            for alpha in alphas:
                z      = stats.norm.ppf((1 + alpha) / 2)
                ci_half = z * unc_flat
                covered = (err_flat <= ci_half).mean()
                expected.append(alpha)
                observed.append(float(covered))
            ax.plot(expected, observed, 'o-', color='#4393c3', lw=2, ms=5,
                    label='Model calibration')
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
            ax.fill_between(expected, expected, observed,
                            where=[o > e for o, e in zip(observed, expected)],
                            alpha=0.2, color='green', label='Over-confident (too narrow CI)')
            ax.fill_between(expected, expected, observed,
                            where=[o < e for o, e in zip(observed, expected)],
                            alpha=0.2, color='red', label='Under-confident (too wide CI)')
            ece = float(np.mean(np.abs(np.array(observed) - np.array(expected))))
            ax.set_xlabel("Expected coverage")
            ax.set_ylabel("Observed coverage")
            ax.set_title(f"Uncertainty Calibration Curve\nECE={ece:.3f} "
                         f"({'well-calibrated' if ece < 0.05 else 'poorly calibrated'})")
            ax.legend(fontsize=7.5)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5,
                    "Calibration curve requires\nground truth data\n(pass ground_truth= argument)",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11,
                    color='grey')
            ax.set_title("Calibration Curve (GT not provided)")
            ax.axis('off')

        # ── [2,1] Spatial gradient of σ — exposes boundary artefacts ──────────
        ax = axes[2, 1]
        dy = np.gradient(uncertainty, axis=0)
        dx = np.gradient(uncertainty, axis=1)
        grad_mag = np.sqrt(dx**2 + dy**2)
        grad_p98 = np.percentile(grad_mag, 98)
        im3 = ax.imshow(np.clip(grad_mag, 0, grad_p98), cmap='hot',
                        interpolation='bilinear', aspect='auto')
        fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label="|∇σ| (°C/px)")
        ax.set_title("Uncertainty Spatial Gradient |∇σ|\n"
                     "(high values = sharp uncertainty boundaries / artefacts)")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        # annotate worst gradient location
        gy, gx = np.unravel_index(np.argmax(grad_mag), grad_mag.shape)
        ax.plot(gx, gy, 'c+', ms=14, mew=2,
                label=f"Max gradient at ({gx},{gy})")
        ax.legend(fontsize=7.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def create_feature_importance_plot(self, feature_names: List[str],
                                        importances: np.ndarray,
                                        output_path: Path,
                                        top_n: int = 30) -> None:
        """
        Horizontal bar chart of GBM feature importances.

        Args:
            feature_names : list of feature name strings
            importances   : array of importance values (same length)
            output_path   : save path
            top_n         : show top-N features
        """
        logger.info("Creating feature importance plot...")
        idx  = np.argsort(importances)[-top_n:]
        names = [feature_names[i] for i in idx]
        vals  = importances[idx]
        normed = vals / vals.max()

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        colors = plt.cm.RdYlGn(normed)
        ax.barh(range(len(names)), vals, color=colors, alpha=0.85, edgecolor="none")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top-{top_n} GBM Feature Importances", fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

        # cumulative % line on twin axis
        ax2 = ax.twiny()
        cum_pct = np.cumsum(vals[::-1])[::-1] / vals.sum() * 100
        ax2.plot(cum_pct, range(len(names)), "o-", color="#4393c3",
                 ms=4, lw=1.5, label="Cumulative %")
        ax2.set_xlabel("Cumulative Importance (%)", color="#4393c3")
        ax2.tick_params(axis="x", labelcolor="#4393c3")
        ax2.set_xlim(0, 105)
        ax2.legend(loc="lower right", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def create_temporal_analysis_plot(self, lst_series: List[np.ndarray],
                                       dates: List[str],
                                       output_path: Path,
                                       hotspot_mask: Optional[np.ndarray] = None
                                       ) -> None:
        """
        Multi-panel time-series analysis:
          [0] Mean LST over time (full extent vs hotspot)
          [1] Std deviation over time
          [2] Heatmap: date × spatial-row mean
          [3] Boxplot per date
        """
        logger.info("Creating temporal analysis plot...")
        n = len(lst_series)
        assert n == len(dates), "lst_series and dates must be same length"

        means     = [arr.mean() for arr in lst_series]
        stds      = [arr.std()  for arr in lst_series]
        hot_means = ([arr[hotspot_mask].mean() for arr in lst_series]
                     if hotspot_mask is not None else None)

        fig = plt.figure(figsize=(16, 12))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
        fig.suptitle("Temporal UHI Analysis", fontsize=15, fontweight="bold")

        # ── [0,0] mean LST time series ─────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        x  = np.arange(n)
        ax.plot(x, means, "o-", color="#d6604d", lw=2, label="Full extent")
        if hot_means:
            ax.plot(x, hot_means, "s-", color="#4393c3", lw=2, label="Hotspots")
        ax.fill_between(x, [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color="#d6604d")
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean LST (°C)")
        ax.set_title("Mean LST Over Time")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── [0,1] std deviation ─────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        ax.bar(x, stds, color="#92c5de", edgecolor="none", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Spatial Std Dev (°C)")
        ax.set_title("Spatial Variability Over Time")
        ax.grid(True, axis="y", alpha=0.3)

        # ── [1,0] heatmap: date × row ──────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        row_profiles = np.array([arr.mean(axis=1) for arr in lst_series])  # (T, H)
        im = ax.imshow(row_profiles.T, cmap=_THERMAL_CMAP,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Row (N→S)")
        ax.set_title("LST Heatmap: Date × N–S Profile")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="LST (°C)")

        # ── [1,1] boxplot per date ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        box_data = [arr.ravel() for arr in lst_series]
        bp = ax.boxplot(box_data, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=2))
        cmap_t = plt.cm.coolwarm(np.linspace(0, 1, n))
        for patch, col in zip(bp["boxes"], cmap_t):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("LST (°C)")
        ax.set_title("LST Distribution per Date")
        ax.grid(True, axis="y", alpha=0.3)

        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def create_urban_rural_comparison_plot(self,
                                            lst_map: np.ndarray,
                                            urban_mask: np.ndarray,
                                            rural_mask: np.ndarray,
                                            output_path: Path) -> None:
        """
        5-panel urban vs rural diagnostic:
          [0,0] Violin plot
          [0,1] Empirical CDF comparison
          [1,0] Temperature gradient (Urban→Rural profile)
          [1,1] Q–Q urban vs rural
          [bottom centre] KDE overlay
        """
        logger.info("Creating urban vs rural comparison plot...")
        u_vals = lst_map[urban_mask].ravel()
        r_vals = lst_map[rural_mask].ravel()

        fig = plt.figure(figsize=(16, 11))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
        fig.suptitle("Urban vs Rural LST Comparison", fontsize=15, fontweight="bold")

        t_stat, p_val = stats.ttest_ind(u_vals, r_vals, equal_var=False)
        _pooled_std = np.sqrt((u_vals.std()**2 + r_vals.std()**2) / 2)
        d_cohen = ((u_vals.mean() - r_vals.mean()) / _pooled_std
                   if _pooled_std > 1e-8 else float("nan"))

        # ── [0,0] Violin ───────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        vp = ax.violinplot([r_vals, u_vals], positions=[1, 2],
                           showmedians=True, showextrema=True)
        for pc, col in zip(vp['bodies'], ["#4393c3", "#d6604d"]):
            pc.set_facecolor(col); pc.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Rural", "Urban"])
        ax.set_ylabel("LST (°C)")
        ax.set_title("Violin: Rural vs Urban")
        ax.text(0.5, 0.97, f"Δ = {u_vals.mean()-r_vals.mean():.2f}°C\n"
                            f"t={t_stat:.2f}  p={p_val:.2e}\nCohen d={d_cohen:.3f}",
                transform=ax.transAxes, va="top", ha="center", fontsize=8.5,
                bbox=dict(fc="white", ec="grey", alpha=0.85))
        ax.grid(True, axis="y", alpha=0.3)

        # ── [0,1] CDF ─────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        for vals, col, lbl in [(r_vals, "#4393c3", "Rural"),
                               (u_vals, "#d6604d", "Urban")]:
            sorted_v = np.sort(vals)
            cdf_v    = np.linspace(0, 1, len(sorted_v))
            ax.plot(sorted_v, cdf_v, lw=2, color=col, label=lbl)
        ax.set_xlabel("LST (°C)")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("Empirical CDF Comparison")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── [0,2] KDE overlay ─────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 2])
        for vals, col, lbl in [(r_vals, "#4393c3", "Rural"),
                               (u_vals, "#d6604d", "Urban")]:
            ax.hist(vals, bins=60, color=col, density=True,
                    alpha=0.35, edgecolor="none", label=f"{lbl} hist")
            x_k = np.linspace(vals.min(), vals.max(), 400)
            kde_vals = _safe_kde(vals, x_k)
            if kde_vals is not None:
                ax.plot(x_k, kde_vals, color=col, lw=2, label=f"{lbl} KDE")
        ax.set_xlabel("LST (°C)")
        ax.set_ylabel("Density")
        ax.set_title("KDE Overlay")
        ax.legend(fontsize=7.5)

        # ── [1,0] Spatial gradient ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        row_urban = np.where(urban_mask.any(axis=1))[0]
        row_rural = np.where(rural_mask.any(axis=1))[0]
        row_means = lst_map.mean(axis=1)
        ax.plot(range(len(row_means)), row_means, color="grey", lw=1.5,
                label="All rows")
        ax.scatter(row_urban, row_means[row_urban], color="#d6604d", s=8,
                   alpha=0.6, label="Urban rows")
        ax.scatter(row_rural, row_means[row_rural], color="#4393c3", s=8,
                   alpha=0.6, label="Rural rows")
        ax.set_xlabel("Row (N→S)")
        ax.set_ylabel("Mean LST (°C)")
        ax.set_title("N–S Spatial Gradient")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [1,1] Q–Q urban vs rural ──────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        # match lengths by quantile
        q_pts = np.linspace(0, 100, 200)
        u_q   = np.percentile(u_vals, q_pts)
        r_q   = np.percentile(r_vals, q_pts)
        ax.plot(r_q, u_q, ".", color="#7b3294", ms=4, alpha=0.7)
        mn = min(r_q.min(), u_q.min())
        mx = max(r_q.max(), u_q.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="1:1 line")
        ax.set_xlabel("Rural LST Quantiles (°C)")
        ax.set_ylabel("Urban LST Quantiles (°C)")
        ax.set_title("Q–Q: Urban vs Rural")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── [1,2] Summary stats table ──────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 2])
        ax.axis("off")
        rows = [
            ["Metric", "Rural", "Urban", "Δ (U–R)"],
            ["Mean (°C)",   f"{r_vals.mean():.2f}", f"{u_vals.mean():.2f}",
             f"{u_vals.mean()-r_vals.mean():.2f}"],
            ["Median (°C)", f"{np.median(r_vals):.2f}", f"{np.median(u_vals):.2f}",
             f"{np.median(u_vals)-np.median(r_vals):.2f}"],
            ["Std (°C)",    f"{r_vals.std():.2f}", f"{u_vals.std():.2f}", "—"],
            ["P5 (°C)",     f"{np.percentile(r_vals,5):.2f}",
             f"{np.percentile(u_vals,5):.2f}", "—"],
            ["P95 (°C)",    f"{np.percentile(r_vals,95):.2f}",
             f"{np.percentile(u_vals,95):.2f}", "—"],
            ["n pixels",    f"{len(r_vals):,}", f"{len(u_vals):,}", "—"],
        ]
        tab = ax.table(cellText=rows[1:], colLabels=rows[0],
                       loc="center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1.2, 1.6)
        for (row, col), cell in tab.get_celld().items():
            if row == 0:
                cell.set_facecolor("#4393c3")
                cell.set_text_props(color="white", fontweight="bold")
            elif col == 3:
                cell.set_facecolor("#fddbc7")
        ax.set_title("Summary Statistics", fontsize=11, fontweight="bold", pad=10)

        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def create_ensemble_weights_plot(self,
                                      weights_history: Optional[List[Dict]] = None,
                                      final_weights: Optional[Dict] = None,
                                      output_path: Optional[Path] = None) -> None:
        """
        Visualise ensemble model weights.

        Args:
            weights_history : list of {cnn, gbm} dicts over validation iterations
            final_weights   : final {cnn, gbm} weight dict
            output_path     : save path
        """
        if output_path is None:
            output_path = Path("ensemble_weights.png")
        logger.info("Creating ensemble weights plot...")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Ensemble Weight Diagnostics", fontsize=14, fontweight="bold")

        # ── left: final weights pie ────────────────────────────────────────────
        ax = axes[0]
        if final_weights:
            # Filter to numeric-only entries — 'cnn_residual' mode string causes ValueError
            numeric_weights = {k: v for k, v in final_weights.items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool)}
            keys  = list(numeric_weights.keys())
            vals  = [numeric_weights[k] for k in keys]
            cols  = ["#4393c3", "#d6604d", "#66c2a5"][:len(keys)]
            if keys and sum(vals) > 0:
                wedges, texts, autos = ax.pie(
                    vals, labels=keys, colors=cols,
                    autopct="%1.2f%%", startangle=90,
                    wedgeprops=dict(edgecolor="white", lw=2))
                for at in autos:
                    at.set_fontsize(10)
            else:
                mode_note = final_weights.get("mode", "")
                ax.text(0.5, 0.5,
                        f"Non-numeric weights\n({mode_note or 'cnn_residual mode'})",
                        ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.axis("off")
            ax.set_title("Final Ensemble Weights")
        else:
            ax.text(0.5, 0.5, "No weight data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")

        # ── right: weight evolution ────────────────────────────────────────────
        ax = axes[1]
        if weights_history:
            keys = list(weights_history[0].keys())
            cols = ["#4393c3", "#d6604d", "#66c2a5"][:len(keys)]
            for key, col in zip(keys, cols):
                vals = [w[key] for w in weights_history]
                ax.plot(vals, "o-", color=col, lw=2, ms=5, label=key.upper())
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Weight")
            ax.set_title("Weight Evolution During Optimisation")
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No history data\n(single run)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=11)
            ax.set_title("Weight History")

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def create_error_heatmap(self, predictions: np.ndarray,
                              ground_truth: np.ndarray,
                              output_path: Path,
                              patch_size: int = 16) -> None:
        """
        Aggregate per-patch errors into a spatial heatmap to reveal
        systematic spatial bias of the model.

        Layout (2×3):
          [0,0] Signed Bias map          [0,1] Absolute Error map
          [0,2] Global Error histogram
          [1,0] Row-mean bias profile    [1,1] Col-mean bias profile
          [1,2] Systematic vs Random variance decomposition

        Fixes vs previous version:
          - Axis labels were showing fractional patch indices (0.5 steps)
            due to imshow default extent — now corrected to integer patch coords
          - Row/column bias profiles added to immediately diagnose the top-bottom
            gradient visible in the original plot
          - Systematic variance fraction added to quantify directional bias
        """
        logger.info("Creating spatial error heatmap...")

        def _to_2d(arr):
            return arr[0] if arr.ndim == 3 else arr

        pred_mean = _to_2d(predictions) if predictions.ndim <= 3 else predictions.mean(0)
        gt_mean   = _to_2d(ground_truth) if ground_truth.ndim <= 3 else ground_truth.mean(0)

        # Align shapes if needed
        hmin = min(pred_mean.shape[0], gt_mean.shape[0])
        wmin = min(pred_mean.shape[1], gt_mean.shape[1])
        pred_mean = pred_mean[:hmin, :wmin]
        gt_mean   = gt_mean[:hmin, :wmin]

        err     = pred_mean - gt_mean
        abs_err = np.abs(err)
        H, W    = err.shape
        ph      = max(1, (H + patch_size - 1) // patch_size)
        pw      = max(1, (W + patch_size - 1) // patch_size)
        err_map     = np.full((ph, pw), np.nan)
        abs_err_map = np.full((ph, pw), np.nan)

        for i in range(ph):
            for j in range(pw):
                tile     = err[i*patch_size:(i+1)*patch_size,
                               j*patch_size:(j+1)*patch_size]
                tile_abs = abs_err[i*patch_size:(i+1)*patch_size,
                                   j*patch_size:(j+1)*patch_size]
                if tile.size:
                    err_map[i, j]     = tile.mean()
                    abs_err_map[i, j] = tile_abs.mean()

        lim = float(np.nanpercentile(np.abs(err_map), 98)) or 0.1

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Spatial Error Heatmap (model bias)",
                     fontsize=14, fontweight="bold")

        # ── [0,0] Signed bias ─────────────────────────────────────────────────
        ax = axes[0, 0]
        im = ax.imshow(err_map, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       interpolation="nearest", aspect="auto",
                       extent=[-0.5, pw-0.5, ph-0.5, -0.5])   # integer ticks
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Bias (°C)")
        ax.set_title(f"Signed Bias (patch={patch_size}px)")
        ax.set_xlabel("Patch column"); ax.set_ylabel("Patch row")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # ── [0,1] Absolute error ──────────────────────────────────────────────
        ax = axes[0, 1]
        im2 = ax.imshow(abs_err_map, cmap="YlOrRd",
                        interpolation="nearest", aspect="auto",
                        extent=[-0.5, pw-0.5, ph-0.5, -0.5])
        fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="|Error| (°C)")
        ax.set_title("Absolute Error per Patch")
        ax.set_xlabel("Patch column"); ax.set_ylabel("Patch row")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # ── [0,2] Global error histogram ─────────────────────────────────────
        ax = axes[0, 2]
        flat = err.ravel()
        ax.hist(flat, bins=80, color="#92c5de", edgecolor="none",
                density=True, alpha=0.75, label="Pixel errors")
        x_k = np.linspace(flat.min(), flat.max(), 400)
        kde_vals = _safe_kde(flat, x_k)
        if kde_vals is not None:
            ax.plot(x_k, kde_vals, color="#b2182b", lw=2, label="KDE")
        mu, sigma = flat.mean(), flat.std()
        ax.plot(x_k, stats.norm.pdf(x_k, mu, sigma),
                color="grey", lw=1.5, ls="--", label="Normal fit")
        ax.axvline(0,  color="red",    ls="--", lw=1.3, label="Zero bias")
        ax.axvline(mu, color="orange", ls="-",  lw=1.3, label=f"μ={mu:.3f}°C")
        ax.text(0.03, 0.96,
                f"μ={mu:.3f}°C\nσ={sigma:.3f}°C\n"
                f"RMSE={np.sqrt(np.mean(flat**2)):.3f}°C\n"
                f"Skew={float(stats.skew(flat)):.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="white", ec="grey", alpha=0.85))
        ax.set_xlabel("Error (°C)"); ax.set_ylabel("Density")
        ax.set_title("Global Error Distribution")
        ax.legend(fontsize=7.5)

        # ── [1,0] Row-mean bias profile (N→S) ────────────────────────────────
        ax = axes[1, 0]
        row_bias  = np.nanmean(err_map, axis=1)
        row_std   = np.nanstd(err_map, axis=1)
        row_idx   = np.arange(ph)
        ax.barh(row_idx, row_bias, color=np.where(row_bias >= 0, '#d6604d', '#4393c3'),
                alpha=0.8, edgecolor='none')
        ax.fill_betweenx(row_idx, row_bias - row_std, row_bias + row_std,
                         alpha=0.2, color='grey', label='±1 σ')
        ax.axvline(0, color='black', lw=1, ls='--')
        ax.invert_yaxis()
        ax.set_xlabel("Mean Bias (°C)")
        ax.set_ylabel("Patch row (N→S)")
        ax.set_title("N–S Bias Profile\n(red=over-prediction, blue=under-prediction)")
        # annotate trend
        if ph > 2:
            sl, ic, rv, *_ = stats.linregress(row_idx, row_bias)
            ax.plot(sl * row_idx + ic, row_idx, 'k-', lw=1.5,
                    label=f"Trend: {sl:+.3f}°C/row (r={rv:.2f})")
            if abs(rv) > 0.5:
                ax.text(0.5, 0.02,
                        f"⚠ Strong N–S gradient (r={rv:.2f})\n— possible illumination/scan-line artefact",
                        transform=ax.transAxes, ha='center', va='bottom', fontsize=8,
                        color='darkred', bbox=dict(fc='#fff3e0', ec='orange', alpha=0.9))
        ax.legend(fontsize=7.5)
        ax.grid(True, axis='x', alpha=0.3)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # ── [1,1] Column-mean bias profile (W→E) ─────────────────────────────
        ax = axes[1, 1]
        col_bias  = np.nanmean(err_map, axis=0)
        col_std   = np.nanstd(err_map, axis=0)
        col_idx   = np.arange(pw)
        ax.bar(col_idx, col_bias, color=np.where(col_bias >= 0, '#d6604d', '#4393c3'),
               alpha=0.8, edgecolor='none')
        ax.fill_between(col_idx, col_bias - col_std, col_bias + col_std,
                        alpha=0.2, color='grey', label='±1 σ')
        ax.axhline(0, color='black', lw=1, ls='--')
        ax.set_xlabel("Patch column (W→E)")
        ax.set_ylabel("Mean Bias (°C)")
        ax.set_title("W–E Bias Profile")
        if pw > 2:
            sl, ic, rv, *_ = stats.linregress(col_idx, col_bias)
            ax.plot(col_idx, sl * col_idx + ic, 'k-', lw=1.5,
                    label=f"Trend: {sl:+.3f}°C/col (r={rv:.2f})")
        ax.legend(fontsize=7.5)
        ax.grid(True, axis='y', alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # ── [1,2] Systematic vs Random variance decomposition ─────────────────
        ax = axes[1, 2]
        total_var   = float(np.var(flat))
        # systematic = variance of patch means (spatial structure)
        sys_var     = float(np.nanvar(err_map))
        rand_var    = max(0.0, total_var - sys_var)
        sys_pct     = sys_var / total_var * 100 if total_var > 0 else 0
        rand_pct    = rand_var / total_var * 100 if total_var > 0 else 0

        bars = ax.bar(['Systematic\n(spatial bias)', 'Random\n(noise)'],
                      [sys_pct, rand_pct],
                      color=['#d6604d', '#4393c3'], alpha=0.8, edgecolor='black', lw=0.8)
        for bar, val in zip(bars, [sys_pct, rand_pct]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_ylabel("% of total error variance")
        ax.set_title("Error Variance Decomposition\n"
                     "(Systematic = spatially structured bias)")
        ax.set_ylim(0, 115)
        ax.grid(True, axis='y', alpha=0.3)
        if sys_pct > 30:
            ax.text(0.5, 0.95,
                    f"⚠ {sys_pct:.0f}% systematic — model has\nlearnable spatial bias",
                    transform=ax.transAxes, ha='center', va='top', fontsize=9,
                    color='darkred', bbox=dict(fc='#fff3e0', ec='orange', alpha=0.9))

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        plt.close()

    def create_comprehensive_dashboard(self,
                                        lst_map: np.ndarray,
                                        uhi_map: np.ndarray,
                                        gi_star: np.ndarray,
                                        hotspots_df: pd.DataFrame,
                                        uhi_stats: Dict,
                                        classification: Dict,
                                        output_path: Path,
                                        uncertainty: Optional[np.ndarray] = None) -> None:
        """
        Single comprehensive 3×4 dashboard combining all key views.

        Args:
            lst_map        : (H, W) LST array
            uhi_map        : (H, W) UHI intensity array
            gi_star        : (H, W) Gi* map
            hotspots_df    : prioritised hotspot DataFrame
            uhi_stats      : statistics dict from calculate_statistics()
            classification : category count dict from classify_uhi_intensity()
            output_path    : save path
            uncertainty    : optional (H, W) prediction uncertainty
        """
        logger.info("Creating comprehensive dashboard...")
        fig = plt.figure(figsize=(24, 18))
        gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.40, wspace=0.32)
        fig.suptitle("Urban Heat Island — Comprehensive Analysis Dashboard",
                     fontsize=18, fontweight="bold", y=0.995)

        # ── [0,0] LST map ──────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(lst_map, cmap=_THERMAL_CMAP,
                       interpolation="nearest", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="°C")
        ax.set_title("Land Surface Temperature")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

        # ── [0,1] UHI classified map ───────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 1])
        classified = np.zeros_like(uhi_map)
        classified[uhi_map < 0]                               = 0
        classified[(uhi_map >= 0) & (uhi_map < 2)]           = 1
        classified[(uhi_map >= 2) & (uhi_map < 4)]           = 2
        classified[(uhi_map >= 4) & (uhi_map < 6)]           = 3
        classified[(uhi_map >= 6) & (uhi_map < 8)]           = 4
        classified[uhi_map >= 8]                              = 5
        cmap_c  = mcolors.ListedColormap(_UHI_CAT_COLORS)
        norm_c  = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap_c.N)
        im2 = ax.imshow(classified, cmap=cmap_c, norm=norm_c,
                        interpolation="nearest", aspect="auto")
        cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04,
                             boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                             ticks=[0, 1, 2, 3, 4, 5])
        cbar2.ax.set_yticklabels(["No UHI", "Weak\n(0–2)", "Mod.\n(2–4)", "Strong\n(4–6)", "V.Strong\n(6–8)", "Extreme\n(>8)"],
                                 fontsize=7)
        ax.set_title("UHI Intensity Classification")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

        # ── [0,2] Gi* map ──────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 2])
        vgi = np.percentile(np.abs(gi_star), 98)
        im3 = ax.imshow(gi_star, cmap="RdYlBu_r", vmin=-vgi, vmax=vgi,
                        interpolation="nearest", aspect="auto")
        ax.contour(gi_star, levels=[1.96, 2.58], colors=["black", "black"],
                   linewidths=[1, 1.8], linestyles=["--", "-"], alpha=0.7)
        fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label="Gi*")
        ax.set_title("Gi* Hot/Cold Spots")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

        # ── [0,3] UHI pie chart ────────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 3])
        labels = [k for k, v in classification.items() if v > 0]
        sizes  = [v for v in classification.values() if v > 0]
        wedges, texts, autos = ax.pie(
            sizes, labels=None, colors=_UHI_CAT_COLORS[:len(labels)],
            autopct="%1.1f%%", startangle=140,
            wedgeprops=dict(width=0.55, edgecolor="white"),
            pctdistance=0.75)
        for at in autos: at.set_fontsize(8)
        ax.legend([Patch(fc=_UHI_CAT_COLORS[i]) for i in range(len(labels))],
                  labels, fontsize=7, loc="lower center",
                  bbox_to_anchor=(0.5, -0.15), ncol=2)
        ax.set_title("Category Distribution")

        # ── [1,0] UHI histogram ────────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        flat = uhi_map.ravel()
        ax.hist(flat, bins=80, color="#4393c3", density=True,
                alpha=0.7, edgecolor="none")
        x_k = np.linspace(flat.min(), flat.max(), 400)
        kde_vals = _safe_kde(flat, x_k)
        if kde_vals is not None:
            ax.plot(x_k, kde_vals, color="#b2182b", lw=2)
        for thr, col in [(0, "#99d594"), (2, "#fc8d59"), (3, "#d53e4f")]:
            ax.axvline(thr, color=col, lw=1.3, ls="--")
        ax.set_xlabel("UHI (°C)"); ax.set_ylabel("Density")
        ax.set_title("UHI Intensity Distribution")

        # ── [1,1] N–S profile ─────────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        row_m = uhi_map.mean(axis=1)
        row_s = uhi_map.std(axis=1)
        y_idx = np.arange(len(row_m))
        ax.plot(row_m, y_idx, color="#4393c3", lw=1.8)
        ax.fill_betweenx(y_idx, row_m - row_s, row_m + row_s,
                         alpha=0.2, color="#4393c3")
        ax.axvline(0, color="grey", lw=1, ls="--")
        ax.set_xlabel("Mean UHI (°C)"); ax.set_ylabel("Row")
        ax.set_title("N–S UHI Profile")
        ax.invert_yaxis()

        # ── [1,2] Hotspot area bar ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 2])
        if len(hotspots_df) > 0:
            top_n  = min(12, len(hotspots_df))
            top_hs = hotspots_df.head(top_n)
            colors = plt.cm.RdYlGn_r(np.linspace(0, 1, top_n))[::-1]
            ax.barh(range(top_n), top_hs["area_km2"].values,
                    color=colors, alpha=0.85, edgecolor="none")
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([f"#{int(r)}" for r in top_hs["rank"]], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Area (km²)")
            ax.set_title(f"Top-{top_n} Hotspots by Area")
            ax.grid(True, axis="x", alpha=0.3)

        # ── [1,3] Hotspot scatter LST vs area ────────────────────────────────
        ax = fig.add_subplot(gs[1, 3])
        if len(hotspots_df) > 0 and "priority_score" in hotspots_df.columns:
            sc = ax.scatter(hotspots_df["area_km2"], hotspots_df["mean_lst"],
                            c=hotspots_df["priority_score"], cmap="RdYlGn_r",
                            s=30, alpha=0.8, edgecolors="none")
            fig.colorbar(sc, ax=ax, label="Priority")
            for _, row in hotspots_df.head(3).iterrows():
                ax.annotate(f"#{int(row['rank'])}",
                            (row["area_km2"], row["mean_lst"]),
                            fontsize=8, xytext=(4, 2),
                            textcoords="offset points")
        ax.set_xlabel("Area (km²)"); ax.set_ylabel("Mean LST (°C)")
        ax.set_title("Hotspot Area vs LST")
        ax.grid(True, alpha=0.3)

        # ── [2,0] Uncertainty map (or LST std) ────────────────────────────────
        ax = fig.add_subplot(gs[2, 0])
        unc_data = uncertainty if uncertainty is not None else np.abs(uhi_map)
        unc_lbl  = "Uncertainty (σ)" if uncertainty is not None else "|UHI| (°C)"
        im_u = ax.imshow(unc_data, cmap="YlOrRd",
                         interpolation="nearest", aspect="auto")
        fig.colorbar(im_u, ax=ax, fraction=0.046, pad=0.04, label=unc_lbl)
        ax.set_title("Prediction Uncertainty" if uncertainty is not None
                     else "|UHI Intensity|")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

        # ── [2,1] CDF UHI ─────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[2, 1])
        sorted_u = np.sort(flat)
        cdf      = np.linspace(0, 1, len(sorted_u))
        ax.plot(sorted_u, cdf, color="#4393c3", lw=2)
        for thr, col in [(0, "#99d594"), (2, "#fee08b"),
                         (4, "#fc8d59"), (6, "#d53e4f"), (8, "#800026")]:
            pct = 100 * (1 - np.interp(thr, sorted_u, cdf))
            ax.axvline(thr, color=col, lw=1.2, ls="--")
            ax.text(thr + 0.05, 0.05, f"{pct:.1f}%",
                    fontsize=7, color=col, va="bottom")
        ax.set_xlabel("UHI (°C)"); ax.set_ylabel("CDF")
        ax.set_title("UHI Empirical CDF")
        ax.grid(True, alpha=0.3)

        # ── [2,2] Key metrics table ────────────────────────────────────────────
        ax = fig.add_subplot(gs[2, 2])
        ax.axis("off")
        rows = [
            ["Metric", "Value"],
            ["Max UHI (°C)", f"{uhi_stats['max_intensity']:.2f}"],
            ["Mean UHI (°C)", f"{uhi_stats['mean_intensity']:.2f}"],
            ["Std UHI (°C)", f"{uhi_stats['std_intensity']:.2f}"],
            ["Extent >6°C (km²)", f"{uhi_stats['spatial_extent_km2']:.2f}"],
            ["% Positive pixels", f"{uhi_stats['pct_positive']:.1f}%"],
            ["Skewness", f"{uhi_stats['skewness']:.3f}"],
            ["Kurtosis", f"{uhi_stats['kurtosis']:.3f}"],
            ["Hotspot regions", f"{len(hotspots_df)}"],
        ]
        tab = ax.table(cellText=rows[1:], colLabels=rows[0],
                       loc="center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(9)
        tab.scale(1.3, 1.7)
        for (r, c), cell in tab.get_celld().items():
            if r == 0:
                cell.set_facecolor("#d6604d")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f7f7f7")
        ax.set_title("Key UHI Metrics", fontsize=11, fontweight="bold")

        # ── [2,3] Priority score distribution ────────────────────────────────
        ax = fig.add_subplot(gs[2, 3])
        if len(hotspots_df) > 0 and "priority_score" in hotspots_df.columns:
            scores = hotspots_df["priority_score"].clip(0, 1)  # guard against float arith bleed
            n_neg  = int((hotspots_df["priority_score"] < 0).sum())
            ax.hist(scores, bins=min(20, len(hotspots_df)),
                    color="#fc8d59", edgecolor="black", alpha=0.75)
            top3_scores = scores.nlargest(3)
            for s in top3_scores:
                ax.axvline(s, color="red", lw=1.2, ls="--")
            if n_neg > 0:
                ax.text(0.5, 0.95,
                        f"⚠ {n_neg} score(s) were < 0\n(clipped to 0 for display)",
                        transform=ax.transAxes, ha='center', va='top', fontsize=8,
                        color='darkred', bbox=dict(fc='#fff3e0', ec='orange', alpha=0.9))
        ax.set_xlabel("Priority Score")
        ax.set_ylabel("Count")
        ax.set_title("Priority Score Distribution")
        ax.set_xlim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# OutputGenerator  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

class OutputGenerator:
    """Generate final output products."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    def export_geotiff(self, data: np.ndarray, output_name: str,
                       bounds: Tuple[float, float, float, float],
                       crs: str = "EPSG:32748",
                       metadata: Optional[Dict] = None):
        import rasterio
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        output_path = self.output_dir / output_name
        height, width = data.shape
        transform = from_bounds(*bounds, width, height)
        with rasterio.open(output_path, 'w', driver='GTiff',
                           height=height, width=width, count=1,
                           dtype=data.dtype, crs=CRS.from_string(crs),
                           transform=transform, compress='lzw') as dst:
            dst.write(data, 1)
            if metadata:
                dst.update_tags(**metadata)
        logger.info(f"Saved: {output_path}")

    def export_shapefile(self, hotspots_df: pd.DataFrame,
                         geometry_col: str = 'geometry',
                         output_name: str = "hotspots.shp",
                         crs: str = "EPSG:32748"):
        import geopandas as gpd
        output_path = self.output_dir / output_name
        gdf = gpd.GeoDataFrame(hotspots_df, geometry=geometry_col, crs=crs)
        gdf.to_file(output_path)
        logger.info(f"Saved: {output_path}")

    def export_csv(self, data: pd.DataFrame, output_name: str):
        output_path = self.output_dir / output_name
        data.to_csv(output_path, index=False)
        logger.info(f"Saved: {output_path}")

    def create_metadata_file(self, metadata: Dict, output_name: str = "metadata.json"):
        output_path = self.output_dir / output_name
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Time-series animation helper (original)
# ══════════════════════════════════════════════════════════════════════════════

def create_time_series_animation(lst_maps: List[np.ndarray], dates: List[str],
                                  output_path: Path, fps: int = 2):
    logger.info(f"Creating time-series animation ({len(lst_maps)} frames)...")
    import imageio
    vmin = min(m.min() for m in lst_maps)
    vmax = max(m.max() for m in lst_maps)
    frames = []
    for lst_map, date in zip(lst_maps, dates):
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(lst_map, cmap=_THERMAL_CMAP, vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='auto')
        ax.set_title(f'Land Surface Temperature – {date}',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate'); ax.set_ylabel('Y Coordinate')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Temperature (°C)', fontsize=12)
        plt.tight_layout()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close()
    imageio.mimsave(output_path, frames, fps=fps)
    logger.info(f"Saved animation: {output_path}")


if __name__ == "__main__":
    logger.info("UHI Visualization module loaded — enhanced with diagnostics")
    logger.info("Classes  : UHIVisualizer, OutputGenerator")
    logger.info("New plots: create_model_comparison_plot, create_uncertainty_analysis_plot,")
    logger.info("           create_feature_importance_plot, create_temporal_analysis_plot,")
    logger.info("           create_urban_rural_comparison_plot, create_ensemble_weights_plot,")
    logger.info("           create_error_heatmap, create_comprehensive_dashboard")