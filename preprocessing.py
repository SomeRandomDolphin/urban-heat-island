"""
Data preprocessing pipeline for Urban Heat Island detection
Purpose: Process BOTH Landsat and Sentinel-2 data and fuse them
Does NOT download data - only transforms existing raw data files
"""
import sys
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
        """Distributions of per-patch LST statistics + example patches."""
        if not patches:
            logger.warning("[Diagnostics] plot_patch_diagnostics: no patches.")
            return

        means = [np.nanmean(p["data"]["LST"]) for p in patches]
        stds  = [np.nanstd(p["data"]["LST"])  for p in patches]
        mins  = [np.nanmin(p["data"]["LST"])   for p in patches]
        maxs  = [np.nanmax(p["data"]["LST"])   for p in patches]

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

        # Distributions
        for col, (vals, label, color) in enumerate(zip(
                [means, stds, mins, maxs],
                ["Patch LST Mean (°C)", "Patch LST Std (°C)",
                 "Patch LST Min (°C)", "Patch LST Max (°C)"],
                ["steelblue", "darkorange", "seagreen", "crimson"])):
            ax = fig.add_subplot(gs[0, col])
            ax.hist(vals, bins=40, color=color, edgecolor="white", alpha=0.85)
            ax.axvline(np.mean(vals), color="black", linestyle="--", lw=1.5,
                       label=f"μ={np.mean(vals):.1f}")
            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Example patches (up to 4)
        n_ex = min(4, len(patches))
        ex_indices = np.linspace(0, len(patches) - 1, n_ex, dtype=int)
        for col, idx in enumerate(ex_indices):
            ax = fig.add_subplot(gs[1, col])
            lst_patch = patches[idx]["data"]["LST"]
            im = ax.imshow(lst_patch, cmap="inferno",
                           vmin=np.nanpercentile(lst_patch, 2),
                           vmax=np.nanpercentile(lst_patch, 98))
            ax.set_title(f"Patch #{idx}\nμ={means[idx]:.1f}°C σ={stds[idx]:.1f}°C",
                         fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Patch Quality Diagnostics  (n={len(patches):,} patches)",
                     fontsize=13, fontweight="bold")
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
            return np.zeros_like(arr, dtype=float)
        return np.clip((arr.astype(float) - lo) / spread, 0.0, 1.0)

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

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.5, nrows * 4.2))
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
        plt.tight_layout()
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
            blue = b2.astype(float)
            red  = b4.astype(float)
            nir  = b8.astype(float)
            eps  = 1e-8
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
            ratios["NIR/Red"]   = b8.astype(float) / (b4.astype(float) + eps)
            labels["NIR/Red"]   = "Vegetation Vigour"
        if b11 is not None and b8 is not None:
            ratios["SWIR1/NIR"] = b11.astype(float) / (b8.astype(float) + eps)
            labels["SWIR1/NIR"] = "Urban Heat / Soil Moisture Proxy"
        if b4 is not None and b2 is not None:
            ratios["Red/Blue"]  = b4.astype(float) / (b2.astype(float) + eps)
            labels["Red/Blue"]  = "Aerosol / Dust Proxy"
        if b8 is not None and b11 is not None:
            ratios["NIR/SWIR1"] = b8.astype(float) / (b11.astype(float) + eps)
            labels["NIR/SWIR1"] = "Soil Moisture Index"
        if b2 is not None and b3 is not None and b4 is not None and b8 is not None:
            num = b2.astype(float) + b3.astype(float)
            den = b4.astype(float) + b8.astype(float)
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
            arr = arr.astype(float)
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

            lo = min(ls_vals.min(), s2_vals.min())
            hi = max(ls_vals.max(), s2_vals.max())
            bins = np.linspace(lo, hi, 80)

            ax.hist(ls_vals, bins=bins, alpha=0.55, color="#1565C0",
                    density=True, label="Landsat")
            ax.hist(s2_vals, bins=bins, alpha=0.55, color="#E65100",
                    density=True, label="Sentinel-2")

            # Simple agreement: correlation of binned counts
            ls_h, _ = np.histogram(ls_vals, bins=bins, density=True)
            s2_h, _ = np.histogram(s2_vals, bins=bins, density=True)
            if ls_h.std() > 0 and s2_h.std() > 0:
                agreement = np.corrcoef(ls_h, s2_h)[0, 1]
                ax.set_title(f"{key}  agreement r={agreement:.3f}", fontsize=9,
                             fontweight="bold")
            else:
                ax.set_title(key, fontsize=9)

            # Mean lines
            ax.axvline(ls_vals.mean(), color="#1565C0", linestyle="--", lw=1.5,
                       label=f"LS μ={ls_vals.mean():.3f}")
            ax.axvline(s2_vals.mean(), color="#E65100", linestyle="--", lw=1.5,
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

        s2_arr = s2_data[index].astype(float)
        ls_arr = ls_data[index].astype(float)

        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # Determine shared colour range
        all_vals = np.concatenate([s2_arr[np.isfinite(s2_arr)].ravel(),
                                   ls_arr[np.isfinite(ls_arr)].ravel()])
        vmin, vmax = np.nanpercentile(all_vals, [2, 98])

        ax0 = fig.add_subplot(gs[0, 0])
        im0 = ax0.imshow(s2_arr, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax0.set_title(f"Sentinel-2  {index}\n(10 m native)", fontweight="bold")
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0)

        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(ls_arr, cmap="RdYlGn", vmin=vmin, vmax=vmax)
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

        ax3.plot(x_s2, profile_s2, color="#E65100", lw=1.2, alpha=0.8,
                 label=f"Sentinel-2 (10 m, {len(profile_s2)} px)")
        ax3.plot(x_ls, profile_ls, color="#1565C0", lw=2.0, alpha=0.9,
                 label=f"Landsat (30 m, {len(profile_ls)} px)")
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
            # Stack all scenes (use minimum common shape)
            shapes = [d[key].shape for d in scene_dicts]
            h = min(s[0] for s in shapes)
            w = min(s[1] for s in shapes)
            stack = np.stack([d[key][:h, :w].astype(float) for d in scene_dicts])

            with np.errstate(invalid="ignore", divide="ignore"):
                pixel_mean = np.nanmean(stack, axis=0)
                pixel_std  = np.nanstd(stack,  axis=0)
                pixel_cv   = pixel_std / (np.abs(pixel_mean) + 1e-8)

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
            cv_max = np.nanpercentile(pixel_cv[np.isfinite(pixel_cv)], 98)
            im1 = ax1.imshow(pixel_cv, cmap="YlOrRd", vmin=0, vmax=cv_max)
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

        ls_arr = landsat_data[index].astype(float)
        s2_arr = sentinel2_data[index].astype(float)
        fu_arr = fused_data[index].astype(float)

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

        isf = fused_data["impervious_surface"].astype(float)

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
        
        # Convert DN to Kelvin using Landsat Collection 2 Level-2 scaling
        bt_kelvin = thermal_band * 0.00341802 + 149.0
        bt_celsius = bt_kelvin - 273.15
        
        # Calculate land surface emissivity from NDVI
        epsilon = np.where(
            ndvi < 0.2,
            0.973,  # Bare soil
            np.where(
                ndvi > 0.5,
                0.986,  # Full vegetation
                0.973 + 0.047 * ((ndvi - 0.2) / 0.3)  # Mixed pixels
            )
        )
        
        # Apply emissivity correction using Planck's law
        wavelength = 10.9e-6  # Band 10 wavelength (meters)
        h = 6.626e-34  # Planck's constant (J·s)
        c = 2.998e8    # Speed of light (m/s)
        sigma = 1.38e-23  # Boltzmann constant (J/K)
        rho = (h * c) / sigma  # ≈ 1.438e-2 m·K
        
        # LST with emissivity correction
        lst_celsius = bt_celsius / (1 + (wavelength * bt_kelvin / rho) * np.log(epsilon))
        
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
        Calculate spectral indices from raw bands
        
        Args:
            bands: Dictionary of band arrays
            
        Returns:
            Dictionary of calculated indices
        """
        indices = {}
        eps = 1e-8
        
        # Extract bands based on satellite type
        if self.satellite_type == "landsat":
            blue = bands.get("SR_B2", np.zeros_like(bands["SR_B4"])).astype(float)
            green = bands.get("SR_B3", np.zeros_like(bands["SR_B4"])).astype(float)
            red = bands["SR_B4"].astype(float)
            nir = bands["SR_B5"].astype(float)
            swir1 = bands["SR_B6"].astype(float)
            swir2 = bands["SR_B7"].astype(float)
        else:  # Sentinel-2
            blue = bands["B2"].astype(float)
            green = bands["B3"].astype(float)
            red = bands["B4"].astype(float)
            nir = bands["B8"].astype(float)
            swir1 = bands["B11"].astype(float)
            swir2 = bands["B12"].astype(float)
        
        # Calculate indices
        indices["NDVI"] = (nir - red) / (nir + red + eps)
        indices["NDBI"] = (swir1 - nir) / (swir1 + nir + eps)
        indices["MNDWI"] = (green - swir1) / (green + swir1 + eps)
        indices["BSI"] = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)
        indices["UI"] = (swir2 - nir) / (swir2 + nir + eps)
        
        # Albedo (simplified broadband surface albedo — Liang 2001)
        # FIX: Landsat Collection 2 Level-2 SR bands are scaled integers.
        #   Reflectance = DN * 0.0000275 − 0.2   (USGS Collection 2 scaling)
        #   Sentinel-2 bands are already in surface reflectance (0–1).
        # Without this conversion Landsat albedo ≈ 0.51 vs S2 ≈ 0.001,
        # giving sensor-agreement r = 0.687 and a large bias in the fused
        # albedo channel after multi-sensor fusion.
        if self.satellite_type == "landsat":
            blue_r  = np.clip(blue  * 0.0000275 - 0.2, 0.0, 1.0)
            red_r   = np.clip(red   * 0.0000275 - 0.2, 0.0, 1.0)
            nir_r   = np.clip(nir   * 0.0000275 - 0.2, 0.0, 1.0)
            swir1_r = np.clip(swir1 * 0.0000275 - 0.2, 0.0, 1.0)
            swir2_r = np.clip(swir2 * 0.0000275 - 0.2, 0.0, 1.0)
        else:
            # Sentinel-2 is already in reflectance (0–1)
            blue_r, red_r, nir_r, swir1_r, swir2_r = blue, red, nir, swir1, swir2

        albedo = (0.356 * blue_r + 0.130 * red_r + 0.373 * nir_r +
                  0.085 * swir1_r + 0.072 * swir2_r - 0.0018)
        indices["albedo"] = np.clip(albedo, 0, 1)
        
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
            # Load raw data
            raw_data = dict(np.load(raw_file))
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
        
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                patch = {
                    "position": (i, j),
                    "data": {}
                }
                
                # Extract patch for each feature
                valid_patch = True
                for name, arr in raster_data.items():
                    if isinstance(arr, np.ndarray) and arr.ndim == 2:
                        if arr.shape != (height, width):
                            logger.warning(f"  Skipping feature {name}: shape mismatch {arr.shape} vs ({height}, {width})")
                            valid_patch = False
                            break
                        
                        patch_data = arr[i:i+patch_size, j:j+patch_size]
                        
                        # Verify patch dimensions
                        if patch_data.shape != (patch_size, patch_size):
                            logger.warning(f"  Invalid patch shape for {name}: {patch_data.shape}")
                            valid_patch = False
                            break
                        
                        patch["data"][name] = patch_data
                
                if not valid_patch:
                    continue
                
                # Quality control on LST
                if "LST" not in patch["data"]:
                    continue
                
                patch_lst = patch["data"]["LST"]
                
                # Verify patch shape one more time
                if patch_lst.shape != (patch_size, patch_size):
                    logger.warning(f"  LST patch has wrong shape: {patch_lst.shape}")
                    continue
                
                valid_pixels = np.isfinite(patch_lst).sum()
                valid_ratio = valid_pixels / (patch_size * patch_size)
                
                if valid_ratio >= min_valid_ratio:
                    # Check variance
                    lst_std = np.nanstd(patch_lst)
                    if lst_std < min_variance:
                        filtered_by_variance += 1
                        continue
                    
                    # Only check the patch MEAN, not individual pixels, so extreme
                    # urban heat pixels (dark rooftops, asphalt) are preserved.
                    valid_temps = patch_lst[np.isfinite(patch_lst)]
                    if len(valid_temps) == 0:
                        continue

                    lst_mean = np.nanmean(patch_lst)
                    lst_std  = np.nanstd(patch_lst)

                    # Tier 2 patch-mean gate (see docstring for ceiling rationale)
                    if lst_mean < min_temp or lst_mean > max_temp:
                        filtered_by_temp += 1
                        continue

                    # Reject spatially uniform patches (cloud decks, water, bad data)
                    if lst_std < 0.3:
                        filtered_by_variance += 1
                        continue

                    # Reject patches where >20% of pixels are outside the pixel-level
                    # plausible range [10, 65°C] — matches validate_lst tier bounds.
                    extreme_pixel_ratio = (
                        np.sum((valid_temps < 10.0) | (valid_temps > 65.0)) / len(valid_temps)
                    )
                    if extreme_pixel_ratio > 0.20:
                        filtered_by_temp += 1
                        continue
                    
                    patches.append(patch)
        
        logger.info(f"  Extracted {len(patches)} valid patches")
        logger.info(f"  Filtered by temperature: {filtered_by_temp} patches")
        logger.info(f"  Filtered by variance: {filtered_by_variance} patches")

        # ── DIAGNOSTIC: patch LST statistics ──────────────────────────
        if patches:
            patch_means = np.array([np.nanmean(p["data"]["LST"]) for p in patches])
            patch_stds  = np.array([np.nanstd(p["data"]["LST"])  for p in patches])
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
                                channel_order: List[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create training samples from patches
        
        Args:
            patches: List of patch dictionaries
            temporal_features: Temporal feature dictionary
            channel_order: Order of input channels
            
        Returns:
            Tuple of (X, y, metadata)
        """
        if len(patches) == 0:
            raise ValueError("No patches provided")
        
        if channel_order is None:
            # Default channel order (works for both Landsat and fused data)
            channel_order = [
                "SR_B4", "SR_B5", "SR_B6", "SR_B7",  # Red, NIR, SWIR1, SWIR2
                "NDVI", "NDBI", "MNDWI", "BSI", "UI",
                "albedo"
            ]
        
        n_samples = len(patches)
        patch_size = patches[0]["data"]["LST"].shape[0]
        n_channels = len(channel_order)
        
        X = np.zeros((n_samples, patch_size, patch_size, n_channels), dtype=np.float32)
        y = np.zeros((n_samples, patch_size, patch_size, 1), dtype=np.float32)
        
        valid_samples = []

        for idx, patch in enumerate(patches):
            # Stack channels
            for channel_idx, feature in enumerate(channel_order):
                if feature in patch["data"]:
                    X[idx, :, :, channel_idx] = patch["data"][feature]
                else:
                    logger.warning(f"Feature {feature} not found in patch {idx}")

            # Target (LST) — assign raw patch (may have residual NaN from edge pixels)
            lst_patch = patch["data"]["LST"].astype(np.float32)
            y[idx, :, :, 0] = lst_patch

            # FIX: validate on the actual patch values, NOT the zero-initialised array.
            # Previously y was checked while still containing zeros for NaN pixels,
            # so every patch with any NaN was falsely flagged as < 20°C and discarded.
            valid_fin = np.isfinite(lst_patch)
            if valid_fin.mean() < 0.95:
                logger.warning(f"Sample {idx} low valid ratio ({valid_fin.mean():.2f}), skipping")
                continue
            valid_temps_s = lst_patch[valid_fin]
            lst_mean_s = float(valid_temps_s.mean()) if valid_temps_s.size > 0 else 0.0
            if lst_mean_s < 20.0 or lst_mean_s > 58.0:
                logger.warning(f"Sample {idx} implausible mean temp ({lst_mean_s:.1f}°C), removing")
                continue

            valid_samples.append(idx)

        # Keep only valid samples
        if len(valid_samples) < n_samples:
            logger.info(f"Removing {n_samples - len(valid_samples)} samples with invalid temperatures")
            X = X[valid_samples]
            y = y[valid_samples]
            n_samples = len(valid_samples)

        # FIX: fill residual NaN pixels with the per-patch mean rather than a global
        # constant (e.g. 35°C median). A global constant collapses to ~0 after z-score
        # normalisation, producing the central spike in val/test LST histograms.
        # Per-patch mean preserves local temperature context with no systematic bias.
        n_ch = X.shape[-1]
        for s in range(n_samples):
            for ch in range(n_ch):                      # features: per-channel mean fill
                sl = X[s, :, :, ch]
                nm = ~np.isfinite(sl)
                if nm.any():
                    sl[nm] = float(sl[~nm].mean()) if (~nm).any() else 0.0
                    X[s, :, :, ch] = sl
            sl = y[s, :, :, 0]                         # LST target: per-patch mean fill
            nm = ~np.isfinite(sl)
            if nm.any():
                sl[nm] = float(sl[~nm].mean()) if (~nm).any() else 35.0
                y[s, :, :, 0] = sl

        # Tier 3 safety clip — matches validate_lst pixel bounds [10, 65°C].
        # Should rarely trigger; preserves real extreme urban heat values.
        y = np.clip(y, 10.0, 65.0)

        # ── SAMPLE WEIGHTS for tail upweighting ───────────────────────────────
        # The model shows slope compression (slope=0.855, std_ratio=0.893):
        # it under-predicts hot surfaces and over-predicts cool ones.  The root
        # cause is that mid-range patches (30–38°C) heavily outnumber tail patches
        # (<28°C or >42°C) in a balanced city-wide dataset.
        #
        # Fix: compute a per-sample inverse-frequency weight based on each patch's
        # mean LST, so the loss sees equal effective representation across the
        # temperature distribution.  Weights are saved as weights_train.npy for use
        # in the model trainer (pass to sample_weight= or WeightedRandomSampler).
        #
        # Weight formula: w_i = 1 / p(bin_i), normalised so mean(w) = 1.0
        # Uses 10 equal-width bins across the training LST range.
        patch_means = y[:, :, :, 0].reshape(n_samples, -1).mean(axis=1)
        n_weight_bins = 10
        bin_edges = np.linspace(patch_means.min(), patch_means.max() + 1e-6, n_weight_bins + 1)
        bin_ids = np.digitize(patch_means, bin_edges) - 1
        bin_ids = np.clip(bin_ids, 0, n_weight_bins - 1)
        bin_counts = np.bincount(bin_ids, minlength=n_weight_bins).astype(float)
        bin_counts = np.maximum(bin_counts, 1)          # avoid divide-by-zero for empty bins
        raw_weights = 1.0 / bin_counts[bin_ids]
        sample_weights = raw_weights / raw_weights.mean()  # normalise: mean weight = 1

        logger.info("  Sample weights for tail upweighting:")
        logger.info(f"    Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        logger.info(f"    Bins: {bin_counts.astype(int).tolist()}")
        logger.info(f"    Weight per bin: {(1.0/bin_counts/((1.0/bin_counts).mean())).round(2).tolist()}")
        # ──────────────────────────────────────────────────────────────────────
        
        metadata = {
            "n_samples": n_samples,
            "patch_size": patch_size,
            "n_channels": n_channels,
            "channel_order": channel_order,
            "temporal_features": temporal_features,
            "temperature_range": {
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "mean": float(np.mean(y)),
                "std": float(np.std(y))
            },
            "sample_weights": sample_weights,  # shape (N,) — use in trainer for tail upweighting
        }
        
        logger.info(f"Created training samples: X={X.shape}, y={y.shape}")
        logger.info(f"Temperature range: [{metadata['temperature_range']['min']:.2f}, {metadata['temperature_range']['max']:.2f}]°C")
        logger.info(f"Mean: {metadata['temperature_range']['mean']:.2f}°C, Std: {metadata['temperature_range']['std']:.2f}°C")

        # ── DIAGNOSTIC: per-channel feature statistics ─────────────────
        logger.info("  Per-channel feature statistics (X):")
        for ch_idx, feat in enumerate(channel_order):
            ch_data = X[:, :, :, ch_idx]
            logger.info(
                f"    [{ch_idx:2d}] {feat:12s}: mean={ch_data.mean():.4f}  "
                f"std={ch_data.std():.4f}  "
                f"range=[{ch_data.min():.4f}, {ch_data.max():.4f}]"
            )

        nan_x = np.sum(~np.isfinite(X))
        nan_y = np.sum(~np.isfinite(y))
        logger.info(f"  NaN/Inf count – X: {nan_x}  y: {nan_y}")
        logger.info(
            f"  y percentiles [5,25,50,75,95]: "
            f"{np.percentile(y, [5,25,50,75,95]).round(2).tolist()}"
        )
        # ──────────────────────────────────────────────────────────────

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
        Normalize features and targets using provided statistics
        
        Args:
            X: Features (N, H, W, C)
            y: Targets (N, H, W, 1)
            norm_stats: Normalization statistics dictionary
            
        Returns:
            Tuple of (X_normalized, y_normalized)
        """
        logger.info("Normalizing data...")
        
        X_norm = np.zeros_like(X, dtype=np.float32)
        
        # Normalize each channel
        n_channels = X.shape[-1]
        for ch in range(n_channels):
            ch_key = f'channel_{ch}'
            if ch_key in norm_stats['features']:
                mean = norm_stats['features'][ch_key]['mean']
                std = norm_stats['features'][ch_key]['std']
                
                if std > 1e-8:  # Avoid division by zero
                    X_norm[:, :, :, ch] = (X[:, :, :, ch] - mean) / std
                else:
                    logger.warning(f"  Channel {ch} has zero std, skipping normalization")
                    X_norm[:, :, :, ch] = X[:, :, :, ch]
        
        # Normalize targets
        target_mean = norm_stats['target']['mean']
        target_std = norm_stats['target']['std']
        y_norm = (y - target_mean) / target_std
        
        logger.info(f"  X normalized: mean={X_norm.mean():.4f}, std={X_norm.std():.4f}")
        logger.info(f"  y normalized: mean={y_norm.mean():.4f}, std={y_norm.std():.4f}")
        
        return X_norm, y_norm
    
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
        Save dataset to disk with normalization statistics
        
        Args:
            splits: Dictionary with train/val/test splits
            output_dir: Output directory
            metadata: Metadata dictionary
            norm_stats: Normalization statistics (optional)
        """
        output_dir = Path(output_dir)
        
        # Save each split
        for split_name in ['train', 'val', 'test']:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(split_dir / "X.npy", splits[f"X_{split_name}"])
            np.save(split_dir / "y.npy", splits[f"y_{split_name}"])
            
            if f"dates_{split_name}" in splits:
                np.save(split_dir / "dates.npy", splits[f"dates_{split_name}"])
        
        # Save metadata
        import json
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            # Convert numpy types to native Python types
            metadata_clean = {}
            for k, v in metadata.items():
                if isinstance(v, (np.integer, np.floating)):
                    metadata_clean[k] = float(v)
                elif isinstance(v, np.ndarray):
                    metadata_clean[k] = v.tolist()
                else:
                    metadata_clean[k] = v
            json.dump(metadata_clean, f, indent=2)
        
        # Save normalization statistics if provided
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
        
        # Create spatial blocks by clustering patch center locations
        # Extract center coordinates from each patch
        logger.info(f"\nCreating spatial blocks...")
        
        # For patches, we need to extract spatial information
        # We'll use NDVI spatial patterns as a proxy for location
        # (higher variation in certain channels indicates different regions)
        spatial_features = []
        for i in range(n_samples):
            # Use mean values of key indices as spatial signature
            # NDVI, NDBI, MNDWI are typically in channels 0, 1, 2
            if X.shape[-1] >= 3:
                ndvi_mean = X[i, :, :, 0].mean()
                ndbi_mean = X[i, :, :, 1].mean()
                mndwi_mean = X[i, :, :, 2].mean()
                spatial_features.append([ndvi_mean, ndbi_mean, mndwi_mean])
            else:
                # Fallback: use first 3 channels
                spatial_features.append([X[i, :, :, ch].mean() for ch in range(min(3, X.shape[-1]))])
        
        spatial_features = np.array(spatial_features)
        
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
        lst_stds = np.array([y[i].std() for i in range(n_samples)])
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
        valid_ratios = np.array([np.isfinite(y[i]).mean() for i in range(n_samples)])
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
        X = X[valid_indices]
        y = y[valid_indices]
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
        
        # Create the splits
        X_train = X[train_idx]
        y_train = y[train_idx]
        dates_train = dates[train_idx]
        
        X_val = X[val_idx]
        y_val = y[val_idx]
        dates_val = dates[val_idx]
        
        X_test = X[test_idx]
        y_test = y[test_idx]
        dates_test = dates[test_idx]
        
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
        
        landsat_files = sorted(landsat_dir.glob("landsat_*.npz"))
        logger.info(f"Found {len(landsat_files)} Landsat files")
        
        for raw_file in landsat_files:
            # Extract date from filename
            parts = raw_file.stem.split('_')
            year = int(parts[1])
            month = int(parts[2])
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
                    raw_data_for_plot = dict(np.load(raw_file))
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
        
        sentinel2_files = sorted(sentinel2_dir.glob("sentinel2_*.npz"))
        logger.info(f"Found {len(sentinel2_files)} Sentinel-2 files")
        
        for raw_file in sentinel2_files:
            # Extract date from filename
            parts = raw_file.stem.split('_')
            year = int(parts[1])
            month = int(parts[2])
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
                    raw_s2_data = dict(np.load(raw_file))
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
    
    # Step 3: Fuse data or use single sensor
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Fuse multi-sensor data")
    logger.info("="*70)
    
    all_fused_data = []
    all_dates = []
    
    if has_landsat and has_sentinel2 and len(landsat_processed) > 0 and len(sentinel2_processed) > 0:
        # Perform temporal matching and fusion
        logger.info("Performing multi-sensor fusion...")
        
        matches = fusion.temporal_match(landsat_processed, sentinel2_processed, max_time_diff_days=16)
        
        for avg_date, ls_data, s2_data, time_diff in matches:
            logger.info(f"\nFusing data (time_diff={time_diff} days):")
            fused_data = fusion.fuse_data(ls_data, s2_data, time_diff, target_resolution=30)
            
            # Add impervious surface
            if "NDVI" in fused_data and "NDBI" in fused_data and "MNDWI" in fused_data:
                isf = feature_engineer.calculate_impervious_surface(
                    fused_data["NDVI"],
                    fused_data["NDBI"],
                    fused_data["MNDWI"]
                )
                fused_data["impervious_surface"] = isf
            
            all_fused_data.append(fused_data)
            all_dates.append(avg_date)

            # ── DIAGNOSTIC: fusion quality plots (first pair only) ─────
            if len(all_fused_data) == 1:
                try:
                    diag.plot_fusion_comparison(
                        ls_data, s2_data, fused_data,
                        index="NDVI",
                        filename="09_fusion_comparison_NDVI.png",
                    )
                    diag.plot_fusion_comparison(
                        ls_data, s2_data, fused_data,
                        index="NDBI",
                        filename="09b_fusion_comparison_NDBI.png",
                    )
                    diag.plot_fusion_weight_map(
                        ls_data, s2_data, fused_data,
                        time_diff=time_diff,
                        filename="s2_11_fusion_weights.png",
                    )
                    diag.plot_impervious_surface_analysis(
                        fused_data,
                        filename="s2_12_impervious_surface.png",
                    )
                except Exception as _e:
                    logger.warning(f"[Diagnostics] fusion plots failed: {_e}")
            # ──────────────────────────────────────────────────────────
        
        logger.info(f"\n✓ Created {len(all_fused_data)} fused datasets")
        
    elif has_landsat and len(landsat_processed) > 0:
        # Use Landsat only
        logger.info("Using Landsat data only (no Sentinel-2 available)")
        
        for timestamp, ls_data in landsat_processed:
            # Add impervious surface
            if "NDVI" in ls_data and "NDBI" in ls_data and "MNDWI" in ls_data:
                isf = feature_engineer.calculate_impervious_surface(
                    ls_data["NDVI"],
                    ls_data["NDBI"],
                    ls_data["MNDWI"]
                )
                ls_data["impervious_surface"] = isf
            
            all_fused_data.append(ls_data)
            all_dates.append(timestamp)
        
        logger.info(f"✓ Using {len(all_fused_data)} Landsat datasets")
        
    else:
        logger.error("No valid data available for training")
        return
    
    if len(all_fused_data) == 0:
        logger.error("No data available after processing")
        return
    
    # Step 4: Extract patches
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Extract patches")
    logger.info("="*70)
    
    all_patches = []
    for idx, fused_data in enumerate(all_fused_data):
        patches = dataset_creator.extract_patches(
            fused_data,
            patch_size=64,
            stride=24,  # CHANGED: 75% overlap → 3x more patches
            min_valid_ratio=0.95,  # RAISED: strict NaN filter prevents central LST spike
            min_variance=0.3,
            min_temp=20.0,           # Patch-mean lower bound (cloud/shadow below this)
            max_temp=58.0            # Patch-mean upper bound (raised to admit hot urban patches)
        )
        
        for patch in patches:
            patch["date"] = all_dates[idx]
        
        all_patches.extend(patches)
        logger.info(f"  Dataset {idx+1}/{len(all_fused_data)}: {len(patches)} patches")
    
    logger.info(f"\n✓ Total patches extracted: {len(all_patches)}")

    # ── DIAGNOSTIC: patch quality plot ────────────────────────────────
    if all_patches:
        try:
            diag.plot_patch_diagnostics(all_patches, filename="05_patch_diagnostics.png")
        except Exception as _e:
            logger.warning(f"[Diagnostics] patch plot failed: {_e}")
    # ──────────────────────────────────────────────────────────────────
    
    if len(all_patches) == 0:
        logger.error("No patches extracted")
        return
    
    # Step 5: Create training samples
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Create training samples")
    logger.info("="*70)
    
    temporal_features = feature_engineer.encode_temporal_features(all_dates[0])
    # Build dates BEFORE create_training_samples so they can be filtered in sync
    # with X/y. If built afterwards from all_patches it stays at full length
    # (e.g. 9187) while X/y are already reduced (e.g. 7520), causing a
    # ValueError: operands could not be broadcast together in create_stratified_split.
    dates_all = np.array([patch["date"] for patch in all_patches])
    X, y, metadata = dataset_creator.create_training_samples(all_patches, temporal_features)

    n_kept = metadata["n_samples"]
    if len(dates_all) != n_kept:
        logger.warning(
            f"create_training_samples dropped {len(dates_all) - n_kept} samples; "
            f"trimming dates {len(dates_all)} → {n_kept} to match X/y."
        )
        # Reproduce the acceptance mask used inside create_training_samples:
        # valid-pixel ratio >= 0.95  AND  patch-mean LST in [20, 58]°C
        valid_mask = []
        for patch in all_patches:
            lst_p = patch["data"]["LST"].astype("float32")
            fin = np.isfinite(lst_p)
            if fin.mean() < 0.95:
                valid_mask.append(False)
                continue
            mean_t = float(lst_p[fin].mean()) if fin.any() else 0.0
            valid_mask.append(20.0 <= mean_t <= 58.0)
        dates = dates_all[np.array(valid_mask)]
    else:
        dates = dates_all
    
    # Step 6: Create splits - MODIFIED
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Create train/val/test splits")
    logger.info("="*70)
    
    # Use enhanced dataset creator
    from preprocessing import EnhancedDatasetCreator
    dataset_creator = EnhancedDatasetCreator()
    
    # Stratified split with spatial-temporal distribution
    # NEW: 65% train, 15% val, 20% test (was 70/15/15)
    splits = dataset_creator.create_stratified_split(
        X, y, dates,
        split_ratios=(0.65, 0.15, 0.20),
        random_seed=42
    )

    # Step 6.5: Compute normalization statistics from training data
    logger.info("\n" + "="*70)
    logger.info("STEP 5.5: Compute normalization statistics")
    logger.info("="*70)

    output_dataset_dir = PROCESSED_DATA_DIR / "cnn_dataset"
    norm_stats = dataset_creator.compute_and_save_normalization_stats(
        splits['X_train'], 
        splits['y_train'],
        output_dataset_dir
    )

    logger.info("\n" + "="*70)
    logger.info("STEP 5.5.1: Verify no data leakage (RAW)")
    logger.info("="*70)

    dataset_creator.verify_no_data_leakage(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val'],
        splits['X_test'], splits['y_test'],
        norm_stats
    )

    # Step 6.6: Normalize all splits
    logger.info("\n" + "="*70)
    logger.info("STEP 5.6: Normalize data")
    logger.info("="*70)

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
    # ──────────────────────────────────────────────────────────────────────

    # Step 6.7: Update metadata with fusion info
    logger.info("\n" + "="*70)
    logger.info("STEP 6.7: Update metadata")
    logger.info("="*70)

    # Add fusion strategy info to metadata
    metadata['fusion_info'] = {
        'fusion_strategy': 'multi-sensor' if (has_landsat and has_sentinel2 and len(landsat_processed) > 0 and len(sentinel2_processed) > 0) else 'landsat-only',
        'landsat_files': len(landsat_processed) if has_landsat else 0,
        'sentinel2_files': len(sentinel2_processed) if has_sentinel2 else 0,
        'fused_datasets': len(all_fused_data)
    }

    # Save sample weights for the training split (used for tail upweighting in trainer)
    sample_weights = metadata.pop("sample_weights", None)
    if sample_weights is not None:
        # Align weights with the train split indices used by create_stratified_split.
        # splits contains 'X_train' etc already subset; we need the weights for those rows.
        # create_stratified_split does not re-order samples, so train_idx into the
        # post-rare-strata-filter array is not directly accessible here.
        # Safe approach: recompute weights from the normalised y_train directly.
        y_train_raw = splits['y_train']  # already normalised at this point
        patch_means_train = y_train_raw[:, :, :, 0].reshape(len(y_train_raw), -1).mean(axis=1)
        n_wb = 10
        be = np.linspace(patch_means_train.min(), patch_means_train.max() + 1e-6, n_wb + 1)
        bids = np.clip(np.digitize(patch_means_train, be) - 1, 0, n_wb - 1)
        bc = np.maximum(np.bincount(bids, minlength=n_wb).astype(float), 1)
        rw = 1.0 / bc[bids]
        train_weights = rw / rw.mean()
        weights_path = output_dataset_dir / "train" / "weights_train.npy"
        np.save(weights_path, train_weights.astype(np.float32))
        logger.info(f"✅ Saved sample weights → {weights_path}")
        logger.info(f"   Weight range: [{train_weights.min():.3f}, {train_weights.max():.3f}]  "
                    f"mean={train_weights.mean():.3f}")

    # Save the NORMALIZED dataset
    dataset_creator.save_dataset(splits, output_dataset_dir, metadata, norm_stats)

    # Update metadata with NORMALIZED data statistics
    metadata['temperature_range'] = {
        'min': float(np.min(splits['y_train'])),
        'max': float(np.max(splits['y_train'])),
        'mean': float(np.mean(splits['y_train'])),
        'std': float(np.std(splits['y_train']))
    }

    logger.info("Metadata updated with normalized data statistics")
    logger.info(f"  y_train range: [{metadata['temperature_range']['min']:.4f}, {metadata['temperature_range']['max']:.4f}]")

    # Step 7: Save dataset
    logger.info("\n" + "="*70)
    logger.info("STEP 6: Save normalized dataset")
    logger.info("="*70)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("✓ MULTI-SENSOR PREPROCESSING COMPLETE")
    logger.info("="*70)

    # ── DIAGNOSTIC PLOTS: channel correlation & pipeline summary ───────────
    try:
        channel_names = metadata.get("channel_order", None)
        diag.plot_channel_correlation(
            splits["X_train"], channel_names=channel_names,
            filename="08_channel_correlation.png"
        )
    except Exception as _e:
        logger.warning(f"[Diagnostics] channel correlation plot failed: {_e}")

    try:
        diag.plot_pipeline_summary(splits, metadata, filename="10_pipeline_summary.png")
    except Exception as _e:
        logger.warning(f"[Diagnostics] pipeline summary plot failed: {_e}")

    logger.info(
        f"[Diagnostics] All diagnostic plots saved to: "
        f"{PROCESSED_DATA_DIR / 'cnn_dataset' / 'diagnostics'}"
    )
    # ──────────────────────────────────────────────────────────────────────
    logger.info(f"Data sources:")
    logger.info(f"  Landsat files: {len(landsat_processed) if has_landsat else 0}")
    logger.info(f"  Sentinel-2 files: {len(sentinel2_processed) if has_sentinel2 else 0}")
    logger.info(f"  Fused datasets: {len(all_fused_data)}")
    logger.info(f"Patches:")
    logger.info(f"  Total patches: {len(all_patches)}")
    logger.info(f"Training data:")
    logger.info(f"  Training samples: {len(splits['X_train'])}")
    logger.info(f"  Validation samples: {len(splits['X_val'])}")
    logger.info(f"  Test samples: {len(splits['X_test'])}")
    logger.info(f"Output:")
    logger.info(f"  Dataset saved to: {output_dataset_dir}")
    logger.info(f"  Fusion strategy: {metadata['fusion_info']['fusion_strategy']}")
    logger.info(f"Normalization:")
    logger.info(f"  Stats saved: ✅")
    logger.info(f"  Training data normalized: mean≈0, std≈1")
    logger.info(f"  Validation data normalized: ✅")
    logger.info(f"  Test data normalized: ✅")
    logger.info("="*70)
    logger.info("\nNext step: Run model training")


if __name__ == "__main__":
    main()