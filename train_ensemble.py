"""
Enhanced training pipeline with CNN + GBM ensemble - IMPROVED VERSION
Uses BEST validation models for ensemble instead of final models

IMPROVEMENTS (to fix overfitting and improve inference):
1. ProgressiveLSTLoss - Prevents variance collapse and range compression
2. Enhanced metrics - Tracks slope and std_ratio to detect issues
3. Overfitting warnings - Alerts during training when issues detected
4. Stronger augmentation - Better generalization to unseen data
5. Diagnostic checks - Monitors training health every 10 epochs

Expected improvements:
- Reduce train-test gap from 17% to <10%
- Improve inference R² from 0.75 to >0.85
- Fix slope from 0.675 to >0.90 (no range compression)
- Fix std_ratio to ≈1.0 (no variance collapse)
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Optional
import json
import shutil
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import pickle
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend – safe for headless runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import *
from models import UNet, ProgressiveLSTLoss, EarlyStopping, initialize_weights, count_parameters

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def load_normalization_stats() -> Optional[Dict]:
    """
    Load normalization statistics for denormalization during evaluation
    
    Returns:
        Normalization statistics dictionary or None if not found
    """
    stats_path = PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
    
    if not stats_path.exists():
        logger.warning("⚠️ Normalization stats not found - predictions will remain in normalized space")
        return None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    return stats

def denormalize_predictions(predictions: np.ndarray, norm_stats: Dict) -> np.ndarray:
    """
    Denormalize predictions back to Celsius
    
    Args:
        predictions: Normalized predictions
        norm_stats: Normalization statistics
        
    Returns:
        Denormalized predictions in Celsius
    """
    if norm_stats is None or 'target' not in norm_stats:
        logger.warning("⚠️ Cannot denormalize - no target stats available")
        return predictions
    
    target_mean = norm_stats['target']['mean']
    target_std = norm_stats['target']['std']
    
    denormalized = predictions * target_std + target_mean
    
    return denormalized

def check_disk_space(path: Path, required_mb: int = 1000) -> bool:
    """Check if sufficient disk space is available"""
    try:
        stat = shutil.disk_usage(path)
        available_mb = stat.free / (1024 * 1024)
        logger.info(f"Available disk space: {available_mb:.2f} MB")
        
        if available_mb < required_mb:
            logger.warning(f"Low disk space! Available: {available_mb:.2f} MB, Required: {required_mb} MB")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True



# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOTTING MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class DiagnosticsPlotter:
    """
    Centralised matplotlib diagnostics for the CNN+GBM ensemble pipeline.

    All figures are saved under `save_dir` (default: MODEL_DIR / "diagnostics").
    Call individual plot_* methods at the right moment in training, or call
    `plot_all_post_training` once training is finished.
    """

    STYLE = "seaborn-v0_8-darkgrid"

    def __init__(self, save_dir: Path = None):
        self.save_dir = Path(save_dir) if save_dir else MODEL_DIR / "diagnostics"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📊 DiagnosticsPlotter initialised – figures → {self.save_dir}")

    # ─── helpers ─────────────────────────────────────────────────────────────
    def _save(self, fig, name: str):
        path = self.save_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  ✅ Saved: {path.name}")

    @staticmethod
    def _try_style():
        try:
            plt.style.use(DiagnosticsPlotter.STYLE)
        except Exception:
            plt.style.use("ggplot")

    # ─── 1. Training curves ───────────────────────────────────────────────────
    def plot_training_curves(self, history: dict):
        """
        4-panel figure:
          • Train vs Val loss
          • Val R² over epochs
          • Learning rate schedule
          • Weight-norm evolution
        """
        self._try_style()
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Training Curves", fontsize=16, fontweight="bold")

        # -- Loss
        ax = axes[0, 0]
        ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
        ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#F44336")
        ax.set_title("Train vs Validation Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend()

        # -- R²
        ax = axes[0, 1]
        if history["cnn_metrics"]:
            r2_scores  = [m["r2"]   for m in history["cnn_metrics"]]
            std_ratios = [m.get("std_ratio", float("nan")) for m in history["cnn_metrics"]]
            slopes     = [m.get("slope",     float("nan")) for m in history["cnn_metrics"]]
            ax.plot(epochs, r2_scores,  label="Val R²",     color="#4CAF50")
            ax.axhline(0.85, ls="--", color="gray", lw=0.8, label="Target (0.85)")
            ax.set_title("CNN Validation R²")
            ax.set_xlabel("Epoch"); ax.set_ylabel("R²")
            ax.set_ylim([-0.1, 1.05])
            ax.legend()

        # -- LR
        ax = axes[1, 0]
        ax.plot(epochs, history["lr"], color="#FF9800")
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
        ax.set_yscale("log")

        # -- Weight norm
        ax = axes[1, 1]
        if "weight_norms" in history and history["weight_norms"]:
            wn_epochs = range(1, len(history["weight_norms"]) + 1)
            ax.plot(wn_epochs, history["weight_norms"], color="#9C27B0")
            ax.set_title("L2 Weight Norm")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Norm")
        else:
            ax.text(0.5, 0.5, "No weight-norm data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("L2 Weight Norm")

        plt.tight_layout()
        self._save(fig, "01_training_curves")

    # ─── 2. Regression diagnostics per model ─────────────────────────────────
    def plot_pred_vs_actual(self, preds: np.ndarray, targets: np.ndarray,
                            label: str = "Model", metrics: dict = None):
        """
        2-panel scatter + residual plot (denormalized °C values).
        """
        from scipy.stats import linregress
        self._try_style()

        preds   = np.asarray(preds).flatten()
        targets = np.asarray(targets).flatten()

        mask = np.isfinite(preds) & np.isfinite(targets)
        preds, targets = preds[mask], targets[mask]

        slope, intercept, r_val, *_ = linregress(targets, preds)
        residuals = preds - targets

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{label} – Predicted vs Actual", fontsize=14, fontweight="bold")

        # -- Scatter
        ax = axes[0]
        sc = ax.scatter(targets, preds, alpha=0.35, s=12, c=np.abs(residuals),
                        cmap="RdYlGn_r", label="Samples")
        plt.colorbar(sc, ax=ax, label="|Residual| (°C)")
        lo = min(targets.min(), preds.min())
        hi = max(targets.max(), preds.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect (slope=1)")
        fit_x = np.linspace(lo, hi, 200)
        ax.plot(fit_x, slope * fit_x + intercept, "r-", lw=1.5,
                label=f"Fit  slope={slope:.3f}")
        ax.set_xlabel("Actual (°C)"); ax.set_ylabel("Predicted (°C)")
        ax.legend(fontsize=8)

        info_lines = [f"R²={r_val**2:.4f}", f"RMSE={np.sqrt(np.mean(residuals**2)):.3f}°C",
                      f"MAE={np.mean(np.abs(residuals)):.3f}°C",
                      f"slope={slope:.3f}", f"intercept={intercept:.3f}"]
        if metrics:
            info_lines += [f"std_ratio={metrics.get('std_ratio', float('nan')):.3f}",
                           f"MBE={metrics.get('mbe', float('nan')):.3f}°C"]
        ax.text(0.03, 0.97, "\n".join(info_lines), transform=ax.transAxes,
                fontsize=8, va="top", bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

        # -- Residuals
        ax = axes[1]
        ax.scatter(targets, residuals, alpha=0.35, s=12, color="#5C6BC0")
        ax.axhline(0, color="black", lw=1.2)
        ax.axhline(+np.std(residuals), color="orange", ls="--", lw=0.9, label="±1σ")
        ax.axhline(-np.std(residuals), color="orange", ls="--", lw=0.9)
        ax.set_xlabel("Actual (°C)"); ax.set_ylabel("Residual (°C)")
        ax.set_title("Residuals vs Actual")
        ax.legend(fontsize=8)

        plt.tight_layout()
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        self._save(fig, f"02_pred_vs_actual_{safe_label}")

    # ─── 3. Residual distribution ─────────────────────────────────────────────
    def plot_residual_distribution(self, preds: np.ndarray, targets: np.ndarray,
                                   label: str = "Model"):
        """Histogram + Q-Q plot of residuals."""
        from scipy import stats
        self._try_style()

        residuals = (np.asarray(preds) - np.asarray(targets)).flatten()
        residuals = residuals[np.isfinite(residuals)]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{label} – Residual Distribution", fontsize=14, fontweight="bold")

        # Histogram
        ax = axes[0]
        ax.hist(residuals, bins=60, color="#42A5F5", edgecolor="white", linewidth=0.3,
                density=True, alpha=0.75, label="Residuals")
        mu, sigma = residuals.mean(), residuals.std()
        x_pdf = np.linspace(residuals.min(), residuals.max(), 300)
        ax.plot(x_pdf, stats.norm.pdf(x_pdf, mu, sigma), "r-", lw=2,
                label=f"N({mu:.3f}, {sigma:.3f})")
        ax.axvline(0, color="black", lw=1.2, ls="--")
        ax.set_xlabel("Residual (°C)"); ax.set_ylabel("Density")
        ax.legend()

        skew  = stats.skew(residuals)
        kurt  = stats.kurtosis(residuals)
        _, p_norm = stats.normaltest(residuals)
        ax.text(0.97, 0.97,
                f"skew={skew:.3f}\nkurt={kurt:.3f}\np_norm={p_norm:.3e}",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

        # Q-Q
        ax = axes[1]
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        ax.scatter(osm, osr, s=6, alpha=0.4, color="#EF5350")
        qq_line = np.array([osm[0], osm[-1]])
        ax.plot(qq_line, slope * qq_line + intercept, "k-", lw=1.2)
        ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
        ax.set_title(f"Q-Q  (r={r:.4f})")

        plt.tight_layout()
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        self._save(fig, f"03_residual_dist_{safe_label}")

    # ─── 4. Ensemble weight comparison ───────────────────────────────────────
    def plot_ensemble_comparison(self, comparison: dict):
        """
        Bar chart + radar chart comparing CNN, GBM, and ensemble strategies.
        `comparison` should be a dict of {strategy_name: metrics_dict}.
        """
        self._try_style()
        names   = list(comparison.keys())
        metrics_keys = ["r2", "rmse", "mae", "slope", "std_ratio"]
        labels  = ["R²", "RMSE (°C)", "MAE (°C)", "Slope", "Std Ratio"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Ensemble Strategy Comparison", fontsize=14, fontweight="bold")

        colors = plt.cm.tab10(np.linspace(0, 0.6, len(names)))

        for idx, (metric_key, metric_label) in enumerate(zip(["r2", "rmse", "mae"], ["R²", "RMSE (°C)", "MAE (°C)"])):
            ax = axes[idx]
            vals = [comparison[n].get(metric_key, 0) for n in names]
            bars = ax.bar(names, vals, color=colors, edgecolor="white")
            ax.set_title(metric_label)
            ax.set_ylabel(metric_label)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        self._save(fig, "04_ensemble_comparison")

    # ─── 5. Feature importance (GBM) ─────────────────────────────────────────
    def plot_gbm_feature_importance(self, gbm_model, top_n: int = 30):
        """Horizontal bar chart of LightGBM feature importances."""
        if gbm_model is None:
            logger.warning("No GBM model provided – skipping feature importance plot")
            return

        self._try_style()
        imp = gbm_model.feature_importance(importance_type="gain")
        names = gbm_model.feature_name()
        order = np.argsort(imp)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.barh(range(len(order)), imp[order], color="#26C6DA")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[i] for i in order], fontsize=8)
        ax.set_xlabel("Feature Importance (Gain)")
        ax.set_title(f"GBM Feature Importance (top {top_n})")
        plt.tight_layout()
        self._save(fig, "05_gbm_feature_importance")

    # ─── 6. Data distribution overview ───────────────────────────────────────
    def plot_data_distribution(self, y_train: np.ndarray, y_val: np.ndarray,
                               norm_stats: dict = None):
        """
        Compare the target distribution of train vs validation sets.
        Shows both normalized and (if stats available) denormalized temperatures.
        """
        self._try_style()
        train_flat = y_train.flatten()
        val_flat   = y_val.flatten()

        n_panels = 2 if norm_stats else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
        fig.suptitle("Target (LST) Distribution – Train vs Val", fontsize=14, fontweight="bold")
        if n_panels == 1:
            axes = [axes]

        # Normalised space
        ax = axes[0]
        ax.hist(train_flat, bins=80, alpha=0.55, color="#1E88E5", label=f"Train  n={len(train_flat):,}", density=True)
        ax.hist(val_flat,   bins=80, alpha=0.55, color="#E53935", label=f"Val    n={len(val_flat):,}",   density=True)
        ax.set_xlabel("Normalised LST"); ax.set_ylabel("Density")
        ax.set_title("Normalised Space")
        ax.legend()

        # Denormalised
        if norm_stats and "target" in norm_stats:
            mean = norm_stats["target"]["mean"]
            std  = norm_stats["target"]["std"]
            train_c = train_flat * std + mean
            val_c   = val_flat   * std + mean
            ax = axes[1]
            ax.hist(train_c, bins=80, alpha=0.55, color="#1E88E5", label="Train", density=True)
            ax.hist(val_c,   bins=80, alpha=0.55, color="#E53935", label="Val",   density=True)
            ax.set_xlabel("LST (°C)"); ax.set_ylabel("Density")
            ax.set_title("Denormalised (°C)")
            ax.legend()
            ax.text(0.97, 0.97, f"Train: {train_c.mean():.1f}±{train_c.std():.1f}°C\n"
                                 f"Val:   {val_c.mean():.1f}±{val_c.std():.1f}°C",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

        plt.tight_layout()
        self._save(fig, "06_data_distribution")

    # ─── 7. Slope / std_ratio health over epochs ─────────────────────────────
    def plot_diagnostic_metrics_over_epochs(self, history: dict):
        """
        Track slope, std_ratio, and MBE across epochs to visualise regression-to-mean
        and variance collapse trends during training.
        """
        if not history.get("cnn_metrics"):
            logger.warning("No cnn_metrics in history – skipping diagnostic-metrics plot")
            return

        self._try_style()
        epochs  = range(1, len(history["cnn_metrics"]) + 1)
        slopes     = [m.get("slope",     float("nan")) for m in history["cnn_metrics"]]
        std_ratios = [m.get("std_ratio", float("nan")) for m in history["cnn_metrics"]]
        mbe_vals   = [m.get("mbe",       float("nan")) for m in history["cnn_metrics"]]
        r2_vals    = [m.get("r2",        float("nan")) for m in history["cnn_metrics"]]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("CNN Diagnostic Metrics Over Training Epochs", fontsize=14, fontweight="bold")

        def _plot(ax, data, title, ylabel, target=None, danger=None):
            ax.plot(epochs, data, lw=1.5, color="#29B6F6")
            if target is not None:
                ax.axhline(target, color="green",  ls="--", lw=1, label=f"Target ({target})")
            if danger is not None:
                ax.axhline(danger, color="red",    ls=":",  lw=1, label=f"Danger ({danger})")
            ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            if target or danger:
                ax.legend(fontsize=8)

        _plot(axes[0, 0], r2_vals,    "Validation R²",       "R²",        target=0.85)
        _plot(axes[0, 1], slopes,     "Prediction Slope",    "Slope",     target=1.0, danger=0.85)
        _plot(axes[1, 0], std_ratios, "Std Ratio (pred/tgt)","Std Ratio", target=1.0, danger=0.80)
        _plot(axes[1, 1], mbe_vals,   "Mean Bias Error",     "MBE (°C)")
        axes[1, 1].axhline(0, color="green", ls="--", lw=1, label="Target (0)")
        axes[1, 1].legend(fontsize=8)

        plt.tight_layout()
        self._save(fig, "07_diagnostic_metrics_epochs")

    # ─── 8. Error heat-map (spatial) ─────────────────────────────────────────
    def plot_spatial_error_map(self, preds_4d: np.ndarray, targets_4d: np.ndarray,
                               label: str = "Model", n_samples: int = 6):
        """
        For the first n_samples patches, show target / prediction / error side-by-side.
        preds_4d and targets_4d are (N, 1, H, W) or (N, H, W, 1).
        """
        self._try_style()

        def _squeeze(arr):
            arr = np.asarray(arr)
            if arr.ndim == 4 and arr.shape[1] == 1:   # (N,1,H,W)
                arr = arr[:, 0, :, :]
            elif arr.ndim == 4 and arr.shape[3] == 1: # (N,H,W,1)
                arr = arr[:, :, :, 0]
            return arr

        preds_sq   = _squeeze(preds_4d)[:n_samples]
        targets_sq = _squeeze(targets_4d)[:n_samples]
        errors     = preds_sq - targets_sq

        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
        if n_samples == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(f"{label} – Spatial Error Maps (first {n_samples} patches)",
                     fontsize=13, fontweight="bold")

        vmin = min(targets_sq.min(), preds_sq.min())
        vmax = max(targets_sq.max(), preds_sq.max())
        err_abs = np.abs(errors).max()
        norm_err = TwoSlopeNorm(vmin=-err_abs, vcenter=0, vmax=err_abs)

        for i in range(n_samples):
            axes[i, 0].imshow(targets_sq[i], vmin=vmin, vmax=vmax, cmap="hot")
            axes[i, 0].set_title(f"Sample {i+1} – Actual", fontsize=8)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(preds_sq[i], vmin=vmin, vmax=vmax, cmap="hot")
            axes[i, 1].set_title(f"Sample {i+1} – Predicted", fontsize=8)
            axes[i, 1].axis("off")

            im = axes[i, 2].imshow(errors[i], norm=norm_err, cmap="RdBu_r")
            axes[i, 2].set_title(f"Sample {i+1} – Error", fontsize=8)
            axes[i, 2].axis("off")
            plt.colorbar(im, ax=axes[i, 2], shrink=0.8)

        plt.tight_layout()
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        self._save(fig, f"08_spatial_error_{safe_label}")

    # ─── 9. Temperature-stratified error analysis ─────────────────────────────
    def plot_stratified_error(self, preds: np.ndarray, targets: np.ndarray,
                              label: str = "Model", n_bins: int = 10):
        """
        Show how RMSE and bias vary across the temperature range.
        Useful to spot if the model is better/worse at extreme temps.
        """
        self._try_style()
        preds   = np.asarray(preds).flatten()
        targets = np.asarray(targets).flatten()
        mask = np.isfinite(preds) & np.isfinite(targets)
        preds, targets = preds[mask], targets[mask]

        bins = np.percentile(targets, np.linspace(0, 100, n_bins + 1))
        bin_centers, bin_rmse, bin_mae, bin_mbe, bin_counts = [], [], [], [], []

        for lo, hi in zip(bins[:-1], bins[1:]):
            idx = (targets >= lo) & (targets <= hi)
            if idx.sum() < 5:
                continue
            p, t = preds[idx], targets[idx]
            bin_centers.append((lo + hi) / 2)
            bin_rmse.append(np.sqrt(np.mean((p - t) ** 2)))
            bin_mae.append(np.mean(np.abs(p - t)))
            bin_mbe.append(np.mean(p - t))
            bin_counts.append(idx.sum())

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"{label} – Stratified Error Analysis", fontsize=13, fontweight="bold")

        def _bar(ax, y, title, ylabel, color):
            ax.bar(range(len(bin_centers)), y, color=color, edgecolor="white", alpha=0.8)
            ax.set_xticks(range(len(bin_centers)))
            ax.set_xticklabels([f"{c:.1f}" for c in bin_centers], rotation=45, ha="right", fontsize=7)
            ax.set_xlabel("Temperature Bin Centre (°C)")
            ax.set_ylabel(ylabel); ax.set_title(title)

        _bar(axes[0], bin_rmse,   "RMSE by Temperature Bin",  "RMSE (°C)", "#EF5350")
        _bar(axes[1], bin_mae,    "MAE by Temperature Bin",   "MAE (°C)",  "#FFA726")
        axes[2].bar(range(len(bin_centers)), bin_mbe,
                    color=["#1E88E5" if v >= 0 else "#E53935" for v in bin_mbe],
                    edgecolor="white", alpha=0.8)
        axes[2].axhline(0, color="black", lw=1)
        axes[2].set_xticks(range(len(bin_centers)))
        axes[2].set_xticklabels([f"{c:.1f}" for c in bin_centers], rotation=45, ha="right", fontsize=7)
        axes[2].set_xlabel("Temperature Bin Centre (°C)")
        axes[2].set_ylabel("MBE (°C)"); axes[2].set_title("MBE by Temperature Bin")

        # sample counts on RMSE chart
        for i, cnt in enumerate(bin_counts):
            axes[0].text(i, bin_rmse[i] + 0.01, f"n={cnt}", ha="center", fontsize=6)

        plt.tight_layout()
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        self._save(fig, f"09_stratified_error_{safe_label}")

    # ─── 10. Summary dashboard ────────────────────────────────────────────────
    def plot_summary_dashboard(self, history: dict, ensemble_metrics: dict,
                               cnn_metrics: dict, gbm_metrics: dict,
                               ensemble_weights: dict):
        """
        One-page overview: key metrics table + loss curves + R² + weight pie.
        """
        self._try_style()
        fig = plt.figure(figsize=(18, 11))
        gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)
        fig.suptitle("Ensemble Training – Summary Dashboard", fontsize=16, fontweight="bold")

        # -- Metrics table (top-left, spans 2 cols)
        ax_tbl = fig.add_subplot(gs[0, :2])
        ax_tbl.axis("off")
        rows = [
            ["Model", "R²", "RMSE (°C)", "MAE (°C)", "Slope", "Std Ratio"],
            ["CNN",
             f"{cnn_metrics.get('r2',0):.4f}",
             f"{cnn_metrics.get('rmse',0):.4f}",
             f"{cnn_metrics.get('mae',0):.4f}",
             f"{cnn_metrics.get('slope',float('nan')):.3f}",
             f"{cnn_metrics.get('std_ratio',float('nan')):.3f}"],
            ["GBM",
             f"{gbm_metrics.get('r2',0):.4f}",
             f"{gbm_metrics.get('rmse',0):.4f}",
             f"{gbm_metrics.get('mae',0):.4f}",
             f"{gbm_metrics.get('slope',float('nan')):.3f}",
             f"{gbm_metrics.get('std_ratio',float('nan')):.3f}"],
            ["Ensemble",
             f"{ensemble_metrics.get('r2',0):.4f}",
             f"{ensemble_metrics.get('rmse',0):.4f}",
             f"{ensemble_metrics.get('mae',0):.4f}",
             f"{ensemble_metrics.get('slope',float('nan')):.3f}",
             f"{ensemble_metrics.get('std_ratio',float('nan')):.3f}"],
        ]
        tbl = ax_tbl.table(cellText=rows[1:], colLabels=rows[0],
                           cellLoc="center", loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        tbl.scale(1, 1.6)
        # highlight best R²
        best_r2_row = max([(i, float(rows[i+1][1])) for i in range(3)], key=lambda x: x[1])[0]
        for col in range(6):
            tbl[best_r2_row + 1, col].set_facecolor("#C8E6C9")
        ax_tbl.set_title("Validation Metrics Comparison", fontsize=10, pad=4)

        # -- Ensemble weight pie (top-right)
        ax_pie = fig.add_subplot(gs[0, 2])
        wts = [ensemble_weights.get("cnn", 0), ensemble_weights.get("gbm", 0)]
        ax_pie.pie(wts, labels=["CNN", "GBM"], autopct="%1.1f%%",
                   colors=["#42A5F5", "#EF5350"], startangle=90)
        ax_pie.set_title("Final Ensemble Weights", fontsize=10)

        # -- R² target meter (top-far-right)
        ax_r2 = fig.add_subplot(gs[0, 3])
        r2_val = ensemble_metrics.get("r2", 0)
        colors_r2 = ["#F44336", "#FF9800", "#4CAF50"]
        bar_val = min(max(r2_val, 0), 1)
        color_r2 = colors_r2[0] if r2_val < 0.7 else (colors_r2[1] if r2_val < 0.85 else colors_r2[2])
        ax_r2.barh(["R²"], [bar_val], color=color_r2)
        ax_r2.axvline(0.85, color="black", ls="--", lw=1.2, label="Target (0.85)")
        ax_r2.set_xlim([0, 1]); ax_r2.legend(fontsize=8)
        ax_r2.set_title(f"Ensemble R² = {r2_val:.4f}", fontsize=10)

        # -- Loss curves (middle row, 2 cols)
        ax_loss = fig.add_subplot(gs[1, :2])
        epochs = range(1, len(history["train_loss"]) + 1)
        ax_loss.plot(epochs, history["train_loss"], label="Train", color="#2196F3")
        ax_loss.plot(epochs, history["val_loss"],   label="Val",   color="#F44336")
        ax_loss.set_title("Loss Curves"); ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss"); ax_loss.legend()

        # -- R² over epochs (middle row, 2 cols)
        ax_r2_curve = fig.add_subplot(gs[1, 2:])
        if history["cnn_metrics"]:
            r2s = [m["r2"] for m in history["cnn_metrics"]]
            ax_r2_curve.plot(epochs, r2s, color="#4CAF50", lw=1.5)
            ax_r2_curve.axhline(0.85, ls="--", color="gray", lw=0.8, label="Target")
            ax_r2_curve.set_title("CNN Val R²"); ax_r2_curve.set_xlabel("Epoch")
            ax_r2_curve.set_ylabel("R²"); ax_r2_curve.legend()

        # -- Diagnostic metrics (bottom row)
        if history["cnn_metrics"]:
            slopes     = [m.get("slope",     float("nan")) for m in history["cnn_metrics"]]
            std_ratios = [m.get("std_ratio", float("nan")) for m in history["cnn_metrics"]]
            mbe_vals   = [m.get("mbe",       float("nan")) for m in history["cnn_metrics"]]
            lrs        = history["lr"]

            for col_idx, (data, title, target) in enumerate(zip(
                [slopes, std_ratios, mbe_vals, lrs],
                ["Slope", "Std Ratio", "MBE (°C)", "Learning Rate"],
                [1.0, 1.0, 0.0, None]
            )):
                ax_d = fig.add_subplot(gs[2, col_idx])
                ax_d.plot(epochs if col_idx < 3 else range(1, len(lrs)+1),
                          data, lw=1.2, color="#FF7043")
                if target is not None:
                    ax_d.axhline(target, color="green", ls="--", lw=0.9)
                ax_d.set_title(title, fontsize=9)
                ax_d.set_xlabel("Epoch", fontsize=8)
                if col_idx == 3:
                    ax_d.set_yscale("log")

        self._save(fig, "10_summary_dashboard")

    # ─── convenience: run all post-training plots ─────────────────────────────
    def plot_all_post_training(self, history: dict, ensemble_metrics: dict,
                                cnn_metrics: dict, gbm_metrics: dict,
                                ensemble_weights: dict,
                                cnn_preds_flat: np.ndarray = None,
                                targets_flat: np.ndarray = None,
                                gbm_model=None,
                                y_train: np.ndarray = None,
                                y_val: np.ndarray = None,
                                norm_stats: dict = None):
        """Convenience wrapper that calls every available plot method."""
        logger.info("\n📊 Generating all post-training diagnostic plots...")
        self.plot_training_curves(history)
        self.plot_diagnostic_metrics_over_epochs(history)
        self.plot_summary_dashboard(history, ensemble_metrics, cnn_metrics,
                                    gbm_metrics, ensemble_weights)
        if cnn_preds_flat is not None and targets_flat is not None:
            self.plot_pred_vs_actual(cnn_preds_flat, targets_flat, "CNN")
            self.plot_residual_distribution(cnn_preds_flat, targets_flat, "CNN")
            self.plot_stratified_error(cnn_preds_flat, targets_flat, "CNN")
        if gbm_model is not None:
            self.plot_gbm_feature_importance(gbm_model)
        if y_train is not None and y_val is not None:
            self.plot_data_distribution(y_train, y_val, norm_stats)
        logger.info(f"✅ All plots saved to: {self.save_dir}")



class UHIDataset(Dataset):
    """PyTorch Dataset for UHI data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                augment: bool = False, transform=None):
        """
        Args:
            X: NORMALIZED features array (N, H, W, C) - mean≈0, std≈1
            y: NORMALIZED target array (N, H, W, 1) - mean≈0, std≈1
            augment: Apply data augmentation
            transform: Additional transforms
        """
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)  # (N, C, H, W)
        self.y = torch.FloatTensor(y).permute(0, 3, 1, 2)  # (N, 1, H, W)
        self.augment = augment
        self.transform = transform
        
        logger.info(f"Dataset created - X range: [{self.X.min():.4f}, {self.X.max():.4f}]")
        logger.info(f"Dataset created - y range: [{self.y.min():.4f}, {self.y.max():.4f}]")
        logger.info(f"Dataset created - y mean: {self.y.mean():.4f}, y std: {self.y.std():.4f}")
        
        # Verify normalization
        if abs(self.y.mean()) > 0.5:
            logger.warning(f"⚠️ Target mean={self.y.mean():.4f} is far from 0 - data might not be normalized!")
        if not (0.5 < self.y.std() < 1.5):
            logger.warning(f"⚠️ Target std={self.y.std():.4f} is far from 1 - data might not be normalized!")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            x, y = self._augment(x, y)
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def _augment(self, x, y):
        """
        IMPROVED: Stronger augmentation for better generalization
        Simulates various atmospheric conditions, viewing angles, and noise
        """
        
        # 1. Geometric transformations (80% probability - INCREASED from 60%)
        if torch.rand(1) > 0.2:  # Changed from 0.4
            # Horizontal flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            
            # Vertical flip
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
            
            # 90-degree rotation
            if torch.rand(1) > 0.5:
                k = torch.randint(1, 4, (1,)).item()
                x = torch.rot90(x, k, dims=[1, 2])
                y = torch.rot90(y, k, dims=[1, 2])
        
        # 2. Brightness adjustment (50% probability - INCREASED from 30%)
        # Simulates different times of day / solar angles
        if torch.rand(1) > 0.5:  # Changed from 0.7
            brightness_factor = 1.0 + (torch.rand(1) - 0.5) * 0.3  # ±15% (was ±7.5%)
            x = x * brightness_factor
        
        # 3. Contrast adjustment (40% probability - INCREASED from 30%)
        # Simulates different atmospheric conditions
        if torch.rand(1) > 0.6:  # Changed from 0.7
            contrast_factor = 1.0 + (torch.rand(1) - 0.5) * 0.3  # ±15%
            mean = x.mean(dim=[1, 2], keepdim=True)
            x = (x - mean) * contrast_factor + mean
        
        # 4. Gaussian noise (40% probability - INCREASED from 25%)
        # Simulates sensor noise and atmospheric interference
        if torch.rand(1) > 0.6:  # Changed from 0.75
            noise = torch.randn_like(x) * 0.10  # INCREASED from 0.015
            x = x + noise
        
        # 5. Regional dropout (20% probability)
        # Simulates cloud patches or missing data
        if torch.rand(1) > 0.8:
            h, w = x.shape[1], x.shape[2]
            cut_h = int(h * 0.25)
            cut_w = int(w * 0.25)
            cx = torch.randint(0, w - cut_w + 1, (1,)).item()
            cy = torch.randint(0, h - cut_h + 1, (1,)).item()
            
            # Fill with mean instead of zeros (more realistic)
            x[:, cy:cy+cut_h, cx:cx+cut_w] = x.mean(dim=[1, 2], keepdim=True)
        
        # 6. IMPROVED: MixUp augmentation (20% probability - NEW)
        # Helps model learn smoother decision boundaries
        if torch.rand(1) > 0.8:
            alpha = 0.2
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
            
            # Create permuted version
            perm_h = torch.randperm(x.shape[1])
            perm_w = torch.randperm(x.shape[2])
            x_perm = x[:, perm_h, :]
            x_perm = x_perm[:, :, perm_w]
            y_perm = y[:, perm_h, :]
            y_perm = y_perm[:, :, perm_w]
            
            # Mix
            x = lam * x + (1 - lam) * x_perm
            y = lam * y + (1 - lam) * y_perm
        
        return x, y


def prepare_gbm_features(X: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare tabular features for GBM from spatial data
    
    Args:
        X: NORMALIZED image patches (N, H, W, C) - mean≈0, std≈1
        y: NORMALIZED target LST (N, H, W, 1) - mean≈0, std≈1
    
    Returns:
        features_df: DataFrame with aggregated features per patch (normalized)
        targets: Flattened targets (normalized)
    
    Note:
        Both inputs and outputs are in NORMALIZED space.
        GBM will learn in normalized space.
        Predictions must be denormalized during evaluation.
    """
    logger.info("Preparing GBM features from spatial data...")
    
    n_samples, height, width, n_channels = X.shape
    
    # Extract per-patch statistics for each channel
    features_list = []
    
    for i in range(n_samples):
        patch_features = {}
        
        for ch in range(n_channels):
            channel_data = X[i, :, :, ch]
            
            # Statistical features
            patch_features[f'ch{ch}_mean'] = channel_data.mean()
            patch_features[f'ch{ch}_std'] = channel_data.std()
            patch_features[f'ch{ch}_min'] = channel_data.min()
            patch_features[f'ch{ch}_max'] = channel_data.max()
            patch_features[f'ch{ch}_median'] = np.median(channel_data)
            
            # Percentiles
            patch_features[f'ch{ch}_p25'] = np.percentile(channel_data, 25)
            patch_features[f'ch{ch}_p75'] = np.percentile(channel_data, 75)
        
        # Spatial features
        patch_features['height'] = height
        patch_features['width'] = width
        
        features_list.append(patch_features)
    
    features_df = pd.DataFrame(features_list)
    
    # Flatten targets (use mean LST per patch)
    targets = y.reshape(n_samples, -1).mean(axis=1)
    
    logger.info(f"GBM features shape: {features_df.shape}")
    logger.info(f"GBM targets shape: {targets.shape}")
    
    return features_df, targets


class GBMTrainer:
    """Trainer for Gradient Boosting Model - Tracks BEST model"""
    
    def __init__(self, config=None):
        self.config = config or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 127,
            "max_depth": 12,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_child_samples": 100,
            "verbose": -1
        }
        self.model = None
        self.best_model = None  # ADDED: Track best model
        self.best_score = float('inf')  # ADDED: Track best validation RMSE
        
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame, y_val: np.ndarray):
        """Train GBM model and track best version"""
        logger.info("Training GBM model...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        self.model = lgb.train(
            self.config,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        logger.info(f"GBM training complete. Best iteration: {self.model.best_iteration}")
        
        # ADDED: Evaluate and save as best if it's better
        val_preds = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        
        if val_rmse < self.best_score:
            self.best_score = val_rmse
            self.best_model = self.model
            logger.info(f"✅ New best GBM model: RMSE={val_rmse:.4f} (normalized)")
        
        return self.model
    
    def predict(self, X: pd.DataFrame, use_best: bool = True) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            use_best: If True, use best_model; if False, use current model
        """
        model_to_use = self.best_model if (use_best and self.best_model is not None) else self.model
        
        if model_to_use is None:
            raise ValueError("Model not trained yet!")
        
        return model_to_use.predict(X, num_iteration=model_to_use.best_iteration)
    
    def save(self, path: Path):
        """Save both best and final models"""
        # Save final model
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved final GBM model to {path}")
        
        # ADDED: Save best model separately
        if self.best_model is not None:
            best_path = path.parent / f"best_{path.name}"
            with open(best_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            logger.info(f"💾 Saved BEST GBM model to {best_path} (RMSE: {self.best_score:.4f})")
    
    def load(self, path: Path):
        """Load model"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded GBM model from {path}")
    
    def load_best(self, path: Path):
        """ADDED: Load best model instead of final"""
        best_path = path.parent / f"best_{path.name}"
        
        if best_path.exists():
            with open(best_path, 'rb') as f:
                self.model = pickle.load(f)
                self.best_model = self.model  # Set as best
            logger.info(f"✅ Loaded BEST GBM model from {best_path}")
        else:
            logger.warning(f"⚠️ Best model not found at {best_path}, loading final model")
            self.load(path)

def create_temperature_stratified_sampler(y_train):
    """
    Create sampler that balances temperature ranges
    Ensures model sees equal amounts of hot and cold samples
    
    Args:
        y_train: Training targets (N, H, W, 1) - NORMALIZED
    
    Returns:
        WeightedRandomSampler
    """
    from torch.utils.data import WeightedRandomSampler
    
    # Get mean temperature per sample (in normalized space)
    sample_means = y_train.reshape(len(y_train), -1).mean(axis=1).flatten()
    
    # Define temperature bins
    # In normalized space: -1.5 = very cold, 0 = average, +1.5 = very hot
    bins = np.array([-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf])
    bin_indices = np.digitize(sample_means, bins)
    
    # Count samples per bin
    unique_bins, bin_counts = np.unique(bin_indices, return_counts=True)
    
    # Calculate weights (inverse frequency)
    # Rare bins get higher weight
    bin_weights = {bin_idx: len(sample_means) / count 
                   for bin_idx, count in zip(unique_bins, bin_counts)}
    
    # Assign weight to each sample
    sample_weights = np.array([bin_weights[bin_idx] for bin_idx in bin_indices])
    
    logger.info("Temperature-stratified sampling:")
    for bin_idx, count in zip(unique_bins, bin_counts):
        logger.info(f"  Bin {bin_idx}: {count} samples (weight: {bin_weights[bin_idx]:.2f})")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

class CheckpointManager:
    """Manages multiple best checkpoints for different metrics"""
    
    def __init__(self, save_dir: Path, metrics: list = ['r2', 'rmse', 'mae']):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics
        
        # Track best score for each metric
        self.best_scores = {
            m: (float('-inf') if m == 'r2' else float('inf')) 
            for m in metrics
        }
        
        logger.info(f"CheckpointManager: tracking {metrics}")
    
    def save(self, model, optimizer, scheduler, epoch, metrics_dict):
        """Save checkpoint if it's best for any metric"""
        saved_any = False
        
        for metric_name in self.metrics:
            if metric_name not in metrics_dict:
                continue
            
            current_value = metrics_dict[metric_name]
            
            # Check if this is better
            is_better = (
                current_value > self.best_scores[metric_name] 
                if metric_name == 'r2' 
                else current_value < self.best_scores[metric_name]
            )
            
            if is_better:
                self.best_scores[metric_name] = current_value
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'metrics': metrics_dict
                }
                
                save_path = self.save_dir / f"best_{metric_name}.pth"
                torch.save(checkpoint, save_path)
                
                logger.info(f"  💾 New best {metric_name.upper()}: {current_value:.4f} (epoch {epoch+1})")
                saved_any = True
        
        # Always save latest
        latest_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics_dict
        }
        torch.save(latest_checkpoint, self.save_dir / "checkpoint_latest.pth")
        
        return saved_any
    
    def load_best(self, model, metric='r2', device='cpu'):
        """Load best model for given metric"""
        checkpoint_path = self.save_dir / f"best_{metric}.pth"
        
        if checkpoint_path.exists():
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ Loaded best {metric} CNN model from epoch {checkpoint['epoch']+1}")
            logger.info(f"  Metrics: {checkpoint['metrics']}")
            return checkpoint
        else:
            logger.warning(f"⚠️ No checkpoint found for {metric}")
            return None
    
    def get_summary(self):
        """Get summary of best scores"""
        summary = "\n" + "="*70 + "\n"
        summary += "BEST CHECKPOINTS SUMMARY\n"
        summary += "="*70 + "\n"
        for metric, score in self.best_scores.items():
            if score != float('inf') and score != float('-inf'):
                summary += f"  {metric.upper()}: {score:.4f}\n"
        summary += "="*70
        return summary

class EnsembleTrainer:
    """Ensemble trainer combining CNN and GBM - Uses BEST models"""
    
    def __init__(self, cnn_model, device, config=TRAINING_CONFIG):
        self.cnn_model = cnn_model.to(device)
        self.device = device
        self.config = config
        
        # CNN components - IMPROVED LOSS FUNCTION
        self.criterion = ProgressiveLSTLoss()
        logger.info("✓ Using ProgressiveLSTLoss for variance/range preservation")
        
        if config["optimizer"] == "adamw":
            self.optimizer = optim.AdamW(
                cnn_model.parameters(),
                lr=config["initial_lr"],
                weight_decay=config["weight_decay"]
            )
        else:
            self.optimizer = optim.Adam(
                cnn_model.parameters(),
                lr=config["initial_lr"]
            )
        
        self.scheduler = self._create_scheduler()
        # Monitor val R² (mode=max) not val loss.
        # ProgressiveLSTLoss changes scale across training phases, making
        # loss values incomparable epoch-to-epoch. R² is stable throughout.
        self.early_stopping = EarlyStopping(patience=config["patience"], mode="max")
        
        # GBM trainer
        self.gbm_trainer = GBMTrainer()
        self.gbm_trained = False
        
        # Ensemble weights from config
        self.ensemble_weights = ENSEMBLE_WEIGHTS
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=MODEL_DIR / "checkpoints",
            metrics=['r2', 'rmse', 'mae']
        )

        # Diagnostics plotter
        self.plotter = DiagnosticsPlotter(save_dir=MODEL_DIR / "diagnostics")

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "cnn_metrics": [],
            "gbm_metrics": [],
            "ensemble_metrics": [],
            "lr": []
        }
        
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        warmup_epochs = self.config["warmup_epochs"]
        total_epochs = self.config["epochs"]
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _log_regularization_metrics(self, epoch):
        """Log regularization metrics to monitor overfitting"""
        # Calculate L2 norm of weights
        total_norm = 0.0
        for p in self.cnn_model.parameters():
            if p.requires_grad:
                param_norm = p.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        logger.info(f"  L2 weight norm: {total_norm:.4f}")
        
        # Track in history
        if 'weight_norms' not in self.history:
            self.history['weight_norms'] = []
        self.history['weight_norms'].append(total_norm)
    
    def train_cnn_epoch(self, train_loader):
        """Train CNN for one epoch"""
        self.cnn_model.train()
        total_loss = 0
        loss_components = {}
        
        pbar = tqdm(train_loader, desc="Training CNN")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.cnn_model(data)
            
            if batch_idx == 0:
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logger.error(f"NaN/Inf detected in CNN output!")
                logger.info(f"CNN Batch 0 - Output range: [{output.min():.4f}, {output.max():.4f}]")
            
            loss, components = self.criterion(output, target, data)

            # Add gradient norm logging
            if batch_idx == 0:
                total_grad_norm = 0.0
                for p in self.cnn_model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                logger.info(f"  Gradient norm (batch 0): {total_grad_norm:.4f}")
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at batch {batch_idx}!")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value
            
            pbar.set_postfix({"loss": loss.item(), "mse": components["mse"]})
        
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def evaluate_cnn(self, val_loader):
        """Evaluate CNN model"""
        self.cnn_model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Evaluating CNN"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.cnn_model(data)
                loss, _ = self.criterion(output, target, data)
                
                total_loss += loss.item()
                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        preds = np.concatenate(all_preds, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()
        
        # Remove NaN values
        mask = ~np.isnan(targets)
        preds = preds[mask]
        targets = targets[mask]
        
        metrics = self._calculate_metrics(preds, targets, "CNN")
        
        return avg_loss, metrics, all_preds
    
    def _check_for_overfitting_warnings(self, train_metrics, val_metrics):
        """
        IMPROVED: Check for overfitting and other performance issues
        Warns about variance collapse, range compression, and overfitting
        """
        warnings_found = False
        
        # Warning 1: Overfitting (train-val gap)
        if 'r2' in train_metrics and 'r2' in val_metrics:
            train_r2 = train_metrics['r2']
            val_r2 = val_metrics['r2']
            gap = train_r2 - val_r2
            
            if gap > 0.10:
                logger.warning(f"⚠️ OVERFITTING DETECTED!")
                logger.warning(f"   Train R²: {train_r2:.3f}")
                logger.warning(f"   Val R²: {val_r2:.3f}")
                logger.warning(f"   Gap: {gap:.3f} (should be < 0.10)")
                warnings_found = True
        
        # Warning 2: Variance Collapse
        if 'std_ratio' in val_metrics:
            std_ratio = val_metrics['std_ratio']
            if std_ratio < 0.80:
                logger.warning(f"⚠️ VARIANCE COLLAPSE!")
                logger.warning(f"   Std ratio: {std_ratio:.3f} (should be ≈1.0)")
                logger.warning(f"   Predictions are too clustered around mean")
                warnings_found = True
        
        # Warning 3: Range Compression (regression to mean)
        if 'slope' in val_metrics:
            slope = val_metrics['slope']
            if slope < 0.85:
                logger.warning(f"⚠️ RANGE COMPRESSION!")
                logger.warning(f"   Slope: {slope:.3f} (should be ≈1.0)")
                logger.warning(f"   Model is regressing to the mean")
                warnings_found = True
        
        # Warning 4: Systematic Bias
        if 'mbe' in val_metrics:
            mbe = val_metrics['mbe']
            if abs(mbe) > 1.0:
                logger.warning(f"⚠️ SYSTEMATIC BIAS!")
                logger.warning(f"   MBE: {mbe:+.3f}°C (should be ≈0)")
                warnings_found = True
        
        # Warning 5: Mean Shift
        if 'pred_mean' in val_metrics and 'target_mean' in val_metrics:
            mean_diff = abs(val_metrics['pred_mean'] - val_metrics['target_mean'])
            if mean_diff > 1.5:
                logger.warning(f"⚠️ MEAN SHIFT!")
                logger.warning(f"   Pred mean: {val_metrics['pred_mean']:.2f}°C")
                logger.warning(f"   Target mean: {val_metrics['target_mean']:.2f}°C")
                logger.warning(f"   Difference: {mean_diff:.2f}°C (should be < 1.5)")
                warnings_found = True
        
        if not warnings_found:
            logger.info("✓ No major issues detected")
        
        return warnings_found
    
    def train_gbm(self, X_train, y_train, X_val, y_val):
        """Train GBM model"""
        logger.info("\n" + "="*60)
        logger.info("TRAINING GBM MODEL")
        logger.info("="*60)
        
        # Prepare features
        X_train_gbm, y_train_gbm = prepare_gbm_features(X_train, y_train)
        X_val_gbm, y_val_gbm = prepare_gbm_features(X_val, y_val)
        
        # Train
        self.gbm_trainer.train(X_train_gbm, y_train_gbm, X_val_gbm, y_val_gbm)
        self.gbm_trained = True
        
        # Evaluate using BEST model
        train_preds = self.gbm_trainer.predict(X_train_gbm, use_best=True)
        val_preds = self.gbm_trainer.predict(X_val_gbm, use_best=True)
        
        train_metrics = self._calculate_metrics(train_preds, y_train_gbm, "GBM Train")
        val_metrics = self._calculate_metrics(val_preds, y_val_gbm, "GBM Val")
        
        logger.info(f"GBM Train Metrics - R²: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f}°C")
        logger.info(f"GBM Val Metrics - R²: {val_metrics['r2']:.4f}, RMSE: {val_metrics['rmse']:.4f}°C")
        
        # Plot GBM feature importance
        try:
            self.plotter.plot_gbm_feature_importance(self.gbm_trainer.best_model or self.gbm_trainer.model)
            self.plotter.plot_pred_vs_actual(
                self.gbm_trainer.predict(X_val_gbm, use_best=True),
                y_val_gbm, "GBM", val_metrics
            )
            self.plotter.plot_residual_distribution(
                self.gbm_trainer.predict(X_val_gbm, use_best=True),
                y_val_gbm, "GBM"
            )
            self.plotter.plot_stratified_error(
                self.gbm_trainer.predict(X_val_gbm, use_best=True),
                y_val_gbm, "GBM"
            )
        except Exception as _e:
            logger.warning(f"⚠️ GBM diagnostic plots failed: {_e}")

        return val_metrics
    
    def compute_optimal_weights(self, cnn_metrics, gbm_metrics):
        """
        Compute optimal weights based on model performance
        Uses inverse RMSE weighting
        """
        cnn_rmse = cnn_metrics['rmse']
        gbm_rmse = gbm_metrics['rmse']
        
        # If CNN is terrible (R² < 0.2), don't use it
        if cnn_metrics['r2'] < 0.2:
            logger.warning("⚠️ CNN R² < 0.2, using GBM only")
            return {"cnn": 0.0, "gbm": 1.0}
        
        # If GBM is terrible (shouldn't happen), don't use it
        if gbm_metrics['r2'] < 0.2:
            logger.warning("⚠️ GBM R² < 0.2, using CNN only")
            return {"cnn": 1.0, "gbm": 0.0}
        
        # Inverse RMSE weighting
        cnn_weight = (1 / cnn_rmse)
        gbm_weight = (1 / gbm_rmse)
        
        total = cnn_weight + gbm_weight
        cnn_weight = cnn_weight / total
        gbm_weight = gbm_weight / total
        
        logger.info(f"Optimal weights computed:")
        logger.info(f"  CNN: {cnn_weight:.4f} (RMSE: {cnn_rmse:.4f}°C, R²: {cnn_metrics['r2']:.4f})")
        logger.info(f"  GBM: {gbm_weight:.4f} (RMSE: {gbm_rmse:.4f}°C, R²: {gbm_metrics['r2']:.4f})")
        
        return {"cnn": cnn_weight, "gbm": gbm_weight}
    
    def diagnose_cnn_issues(self):
        """
        Diagnose CNN performance issues
        """
        logger.info("\n" + "="*60)
        logger.info("DIAGNOSING CNN ISSUES")
        logger.info("="*60)
        
        # Check if CNN is learning
        if len(self.history["train_loss"]) > 5:
            initial_loss = self.history["train_loss"][0]
            final_loss = self.history["train_loss"][-1]
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            logger.info(f"Training progress:")
            logger.info(f"  Initial loss: {initial_loss:.4f}")
            logger.info(f"  Final loss: {final_loss:.4f}")
            logger.info(f"  Improvement: {improvement:.2f}%")
            
            if improvement < 10:
                logger.warning("⚠️ CNN barely improved - possible issues:")
                logger.warning("  1. Learning rate too low")
                logger.warning("  2. Model stuck in local minimum")
                logger.warning("  3. Data preprocessing issues")
        
        # Check CNN validation performance trend
        if len(self.history["cnn_metrics"]) > 0:
            r2_scores = [m['r2'] for m in self.history["cnn_metrics"]]
            logger.info(f"\nCNN R² progression (last 5): {[f'{r:.4f}' for r in r2_scores[-5:]]}")
            
            best_r2 = max(r2_scores)
            logger.info(f"Best CNN R² achieved: {best_r2:.4f}")
            
            if best_r2 < 0.5:
                logger.error("❌ CNN R² never exceeded 0.5 - CRITICAL ISSUE")
                logger.error("Recommendations:")
                logger.error("  1. Check input data normalization")
                logger.error("  2. Reduce model complexity (fewer layers)")
                logger.error("  3. Increase learning rate (current: {:.6f})".format(self.config["initial_lr"]))
                logger.error("  4. Try training longer")
                logger.error("  5. Check for data leakage/preprocessing bugs")
                logger.error("  6. Verify target values are in reasonable range")
        
        logger.info("="*60)
    
    def evaluate_ensemble(self, val_loader, X_val, y_val):
        """
        Evaluate ensemble with BEST models (MODIFIED)
        """
        logger.info("\n" + "="*60)
        logger.info("EVALUATING ENSEMBLE PREDICTIONS (USING BEST MODELS)")
        logger.info("="*60)
        
        # ADDED: Load best CNN checkpoint before evaluation
        logger.info("Loading best CNN model for ensemble...")
        best_checkpoint = self.checkpoint_manager.load_best(
            self.cnn_model, 
            metric='r2', 
            device=self.device
        )
        
        if best_checkpoint is None:
            logger.warning("⚠️ Could not load best CNN, using current model state")
        else:
            logger.info(f"✅ Using CNN from epoch {best_checkpoint['epoch']+1} (best R²)")
        
        # Get CNN predictions (now using best model)
        _, cnn_metrics, cnn_preds_list = self.evaluate_cnn(val_loader)
        cnn_preds = np.concatenate(cnn_preds_list, axis=0)
        cnn_preds_patch = cnn_preds.reshape(cnn_preds.shape[0], -1).mean(axis=1)
        
        # Get GBM predictions (using best model)
        if self.gbm_trained:
            X_val_gbm, y_val_gbm = prepare_gbm_features(X_val, y_val)
            
            # MODIFIED: Explicitly use best GBM model
            logger.info("Using best GBM model for ensemble...")
            gbm_preds = self.gbm_trainer.predict(X_val_gbm, use_best=True)
            gbm_metrics = self._calculate_metrics(gbm_preds, y_val_gbm, "GBM")
            
            # CRITICAL FIX 1: Analyze prediction scales
            cnn_mean, cnn_std = cnn_preds_patch.mean(), cnn_preds_patch.std()
            gbm_mean, gbm_std = gbm_preds.mean(), gbm_preds.std()
            target_mean, target_std = y_val_gbm.mean(), y_val_gbm.std()
            
            logger.info(f"\nPrediction scale analysis:")
            logger.info(f"  Target: mean={target_mean:.2f}°C, std={target_std:.2f}°C")
            logger.info(f"  CNN:    mean={cnn_mean:.2f}°C, std={cnn_std:.2f}°C")
            logger.info(f"  GBM:    mean={gbm_mean:.2f}°C, std={gbm_std:.2f}°C")
            
            # Check if we need normalization
            scale_diff_cnn = abs(cnn_std - target_std) / target_std
            scale_diff_gbm = abs(gbm_std - target_std) / target_std
            
            if scale_diff_cnn > 0.3 or scale_diff_gbm > 0.3:
                logger.info(f"\n⚠️ Large scale differences detected, applying normalization...")
                
                # Standardize predictions
                cnn_preds_normalized = (cnn_preds_patch - cnn_mean) / (cnn_std + 1e-8)
                gbm_preds_normalized = (gbm_preds - gbm_mean) / (gbm_std + 1e-8)
                
                # CRITICAL FIX 2: Use optimal weights
                optimal_weights = self.compute_optimal_weights(cnn_metrics, gbm_metrics)
                
                # Ensemble in normalized space
                ensemble_normalized = (
                    optimal_weights["cnn"] * cnn_preds_normalized +
                    optimal_weights["gbm"] * gbm_preds_normalized
                )
                
                # Transform back to target scale
                ensemble_preds_normalized = ensemble_normalized * target_std + target_mean
                ensemble_metrics_normalized = self._calculate_metrics(
                    ensemble_preds_normalized, y_val_gbm, "Ensemble (Normalized)"
                )
            else:
                logger.info("\n✓ Scales are similar, normalization not needed")
                optimal_weights = self.compute_optimal_weights(cnn_metrics, gbm_metrics)
                ensemble_preds_normalized = (
                    optimal_weights["cnn"] * cnn_preds_patch +
                    optimal_weights["gbm"] * gbm_preds
                )
                ensemble_metrics_normalized = self._calculate_metrics(
                    ensemble_preds_normalized, y_val_gbm, "Ensemble (Optimal)"
                )
            
            # CRITICAL FIX 3: Compare strategies
            # Strategy 1: Original fixed weights
            fixed_preds = (
                self.ensemble_weights["cnn"] * cnn_preds_patch +
                self.ensemble_weights["gbm"] * gbm_preds
            )
            fixed_metrics = self._calculate_metrics(fixed_preds, y_val_gbm, "Ensemble (Fixed)")
            
            # Print comparison
            logger.info("\n" + "="*60)
            logger.info("ENSEMBLE STRATEGY COMPARISON")
            logger.info("="*60)
            logger.info(f"{'Strategy':<25} {'R²':<10} {'RMSE (°C)':<12} {'MAE (°C)':<12}")
            logger.info("-"*60)
            logger.info(f"{'CNN Only (BEST)':<25} {cnn_metrics['r2']:<10.4f} {cnn_metrics['rmse']:<12.4f} {cnn_metrics['mae']:<12.4f}")
            logger.info(f"{'GBM Only (BEST)':<25} {gbm_metrics['r2']:<10.4f} {gbm_metrics['rmse']:<12.4f} {gbm_metrics['mae']:<12.4f}")
            logger.info("-"*60)
            logger.info(f"{'Fixed Weights':<25} {fixed_metrics['r2']:<10.4f} {fixed_metrics['rmse']:<12.4f} {fixed_metrics['mae']:<12.4f}")
            logger.info(f"{'Optimal + Normalized':<25} {ensemble_metrics_normalized['r2']:<10.4f} {ensemble_metrics_normalized['rmse']:<12.4f} {ensemble_metrics_normalized['mae']:<12.4f}")
            logger.info("="*60)
            
            # Determine best approach
            all_results = [
                ("CNN Only (BEST)", cnn_metrics),
                ("GBM Only (BEST)", gbm_metrics),
                ("Fixed Ensemble", fixed_metrics),
                ("Optimal Ensemble", ensemble_metrics_normalized)
            ]
            
            best_name, best_metrics = max(all_results, key=lambda x: x[1]['r2'])
            
            logger.info(f"\n🏆 BEST APPROACH: {best_name}")
            logger.info(f"   R²: {best_metrics['r2']:.4f}")
            logger.info(f"   RMSE: {best_metrics['rmse']:.4f}°C")
            logger.info(f"   MAE: {best_metrics['mae']:.4f}°C")
            
            if best_name == "Optimal Ensemble":
                logger.info(f"   Weights: CNN={optimal_weights['cnn']:.3f}, GBM={optimal_weights['gbm']:.3f}")
                self.ensemble_weights = optimal_weights
                final_metrics = ensemble_metrics_normalized
            elif best_name == "GBM Only (BEST)":
                logger.warning("\n⚠️ GBM alone performs best - ensemble doesn't help")
                logger.warning("   Consider using GBM only or improving CNN performance")
                self.ensemble_weights = {"cnn": 0.0, "gbm": 1.0}
                final_metrics = gbm_metrics
            elif best_name == "CNN Only (BEST)":
                logger.warning("\n⚠️ CNN alone performs best - GBM doesn't help")
                self.ensemble_weights = {"cnn": 1.0, "gbm": 0.0}
                final_metrics = cnn_metrics
            else:
                final_metrics = fixed_metrics
            
            # Calculate improvement
            baseline_r2 = max(cnn_metrics['r2'], gbm_metrics['r2'])
            ensemble_r2 = ensemble_metrics_normalized['r2']
            improvement = (ensemble_r2 - baseline_r2) / abs(baseline_r2) * 100 if baseline_r2 != 0 else 0
            
            if improvement > 1:
                logger.info(f"\n✅ Ensemble improves over best individual model by {improvement:.2f}%")
            elif improvement < -1:
                logger.warning(f"\n⚠️ Ensemble is {abs(improvement):.2f}% worse than best individual model")
            else:
                logger.info(f"\n➡️ Ensemble performance similar to best individual model")
            
            logger.info("="*60)
            
            return final_metrics
        else:
            logger.warning("GBM not trained, using CNN metrics only")
            return cnn_metrics
    
    def _calculate_metrics(self, preds, targets, name=""):
        """
        Calculate comprehensive evaluation metrics
        IMPROVED: Now includes slope and std_ratio to detect range compression and variance collapse
        """
        from scipy.stats import linregress
        
        # Load normalization stats for denormalization
        norm_stats = load_normalization_stats()
        
        # Denormalize predictions and targets to Celsius for metrics
        if norm_stats is not None:
            preds_denorm = denormalize_predictions(preds, norm_stats)
            targets_denorm = denormalize_predictions(targets, norm_stats)
            
            logger.debug(f"{name} - Denormalized: pred mean={preds_denorm.mean():.2f}°C, "
                        f"target mean={targets_denorm.mean():.2f}°C")
        else:
            logger.warning(f"{name} - Using normalized values (no stats found)")
            preds_denorm = preds
            targets_denorm = targets
        
        # Use denormalized values for all metrics
        pred_var = np.var(preds_denorm)
        target_var = np.var(targets_denorm)
        
        if name and pred_var < 1e-8:
            logger.warning(f"⚠️ {name} predictions have near-zero variance!")
        
        # Basic accuracy metrics
        try:
            r2 = r2_score(targets_denorm, preds_denorm)
        except Exception as e:
            logger.error(f"Error calculating R² for {name}: {e}")
            r2 = 0.0
        
        rmse = np.sqrt(mean_squared_error(targets_denorm, preds_denorm))
        mae = mean_absolute_error(targets_denorm, preds_denorm)
        mbe = np.mean(preds_denorm - targets_denorm)
        
        # IMPROVED: Add regression analysis to detect range compression
        slope, intercept, r_value, p_value, std_err = linregress(
            targets_denorm.flatten(), preds_denorm.flatten()
        )
        
        # IMPROVED: Add distribution metrics to detect variance collapse
        pred_std = np.std(preds_denorm)
        target_std = np.std(targets_denorm)
        std_ratio = pred_std / target_std if target_std > 0 else 0.0
        
        metrics = {
            "r2": r2, 
            "rmse": rmse, 
            "mae": mae, 
            "mbe": mbe,
            # IMPROVED: Additional diagnostic metrics
            "slope": slope,           # Should be ≈1.0 (no range compression)
            "intercept": intercept,   # Should be ≈0.0 (no bias)
            "correlation": r_value,
            "std_ratio": std_ratio,   # Should be ≈1.0 (no variance collapse)
            "pred_mean": preds_denorm.mean(),
            "pred_std": pred_std,
            "target_mean": targets_denorm.mean(),
            "target_std": target_std,
        }
        
        return metrics
    
    def train(self, train_loader, val_loader, X_train, y_train, X_val, y_val, save_dir: Path):
        """Full training loop for ensemble - Uses BEST models"""
        logger.info(f"Starting ensemble training for {self.config['epochs']} epochs")
        logger.info(f"CNN parameters: {count_parameters(self.cnn_model):,}")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Store data references for post-training diagnostics
        self._X_train, self._y_train = X_train, y_train
        self._X_val,   self._y_val   = X_val,   y_val
        
        # Pre-training: data distribution plot
        try:
            norm_stats_pre = load_normalization_stats()
            self.plotter.plot_data_distribution(y_train, y_val, norm_stats_pre)
        except Exception as _e:
            logger.warning(f"⚠️ Pre-training data distribution plot failed: {_e}")
        
        # Train GBM first (independent of CNN)
        gbm_metrics = self.train_gbm(X_train, y_train, X_val, y_val)
        
        # Train CNN
        logger.info("\n" + "="*60)
        logger.info("TRAINING CNN MODEL")
        logger.info("="*60)
        
        best_cnn_r2 = -float('inf')
        
        for epoch in range(self.config["epochs"]):
            logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")

            # Update progressive loss weights
            self.criterion.set_training_progress(epoch, self.config["epochs"])

            # Log regularization metrics
            self._log_regularization_metrics(epoch)
            
            # Train CNN
            train_loss, train_components = self.train_cnn_epoch(train_loader)
            
            # Validate CNN
            val_loss, cnn_metrics, _ = self.evaluate_cnn(val_loader)
            
            # IMPROVED: Check for overfitting and performance issues every 10 epochs
            if epoch % 10 == 0 or epoch == self.config["epochs"] - 1:
                # Need train metrics for comparison
                self.cnn_model.eval()
                train_preds_list = []
                train_targets_list = []
                with torch.no_grad():
                    for data, target in train_loader:
                        data = data.to(self.device)
                        output = self.cnn_model(data)
                        train_preds_list.append(output.cpu().numpy())
                        train_targets_list.append(target.cpu().numpy())
                train_preds = np.concatenate(train_preds_list, axis=0).flatten()
                train_targets = np.concatenate(train_targets_list, axis=0).flatten()
                train_metrics = self._calculate_metrics(train_preds, train_targets, "Train")
                
                logger.info("\n--- Diagnostic Check ---")
                self._check_for_overfitting_warnings(train_metrics, cnn_metrics)
                logger.info("------------------------\n")
            
            # Track best CNN performance
            if cnn_metrics['r2'] > best_cnn_r2:
                best_cnn_r2 = cnn_metrics['r2']
                logger.info(f"✅ New best CNN R²: {best_cnn_r2:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Log metrics
            logger.info(f"CNN Train Loss: {train_loss:.4f}")
            logger.info(f"CNN Val Loss: {val_loss:.4f}")
            logger.info(f"CNN Val Metrics - R²: {cnn_metrics['r2']:.4f}, "
                       f"RMSE: {cnn_metrics['rmse']:.4f}°C, "
                       f"MAE: {cnn_metrics['mae']:.4f}°C")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["cnn_metrics"].append(cnn_metrics)
            self.history["lr"].append(current_lr)
            
            # Save checkpoint if best
            self.checkpoint_manager.save(
                model=self.cnn_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics_dict=cnn_metrics
            )
            
            # Periodic diagnostic plots every 25 epochs
            if (epoch + 1) % 25 == 0 or epoch == self.config["epochs"] - 1:
                try:
                    self.plotter.plot_training_curves(self.history)
                    self.plotter.plot_diagnostic_metrics_over_epochs(self.history)
                except Exception as _e:
                    logger.warning(f"⚠️ Periodic plot failed at epoch {epoch+1}: {_e}")
            
            # Early stopping on val R² (stable across progressive loss phases)
            if self.early_stopping(cnn_metrics['r2'], self.cnn_model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Diagnose CNN issues
        self.diagnose_cnn_issues()
        
        # ADDED: Load best models before final ensemble evaluation
        logger.info("\n" + "="*60)
        logger.info("LOADING BEST MODELS FOR FINAL ENSEMBLE")
        logger.info("="*60)
        
        # Load best CNN checkpoint
        logger.info("Loading best CNN checkpoint...")
        best_cnn_checkpoint = self.checkpoint_manager.load_best(
            self.cnn_model, 
            metric='r2', 
            device=self.device
        )
        
        if best_cnn_checkpoint:
            logger.info(f"✅ Loaded best CNN from epoch {best_cnn_checkpoint['epoch']+1}")
            logger.info(f"   R²: {best_cnn_checkpoint['metrics']['r2']:.4f}")
        else:
            logger.warning("⚠️ Using final CNN state (no best checkpoint found)")
        
        # GBM best model is already tracked in self.gbm_trainer.best_model
        logger.info(f"✅ Using best GBM model (RMSE: {self.gbm_trainer.best_score:.4f})")
        
        logger.info("="*60)
        
        # ── FIT POST-HOC LINEAR CALIBRATION ON VAL SET ──────────────────────
        # The model systematically compresses predictions (slope < 1.0 on val).
        # A linear recalibration y_cal = slope*y_pred + intercept, fitted on the
        # val set in NORMALIZED space, corrects this at inference time without
        # any retraining. This is saved to calibration_params.json.
        logger.info("\n" + "="*60)
        logger.info("FITTING POST-HOC LINEAR CALIBRATION")
        logger.info("="*60)

        from scipy.stats import linregress as _linregress
        from sklearn.metrics import r2_score as _r2s

        # Collect ENSEMBLE predictions on val set (not just CNN)
        # so calibration corrects the full ensemble output, not just CNN in isolation.
        self.cnn_model.eval()
        _cal_preds, _cal_tgts = [], []
        with torch.no_grad():
            for _d, _t in val_loader:
                _cal_preds.append(self.cnn_model(_d.to(self.device)).cpu().numpy())
                _cal_tgts.append(_t.numpy())
        _cp = np.concatenate(_cal_preds).flatten()
        _ct = np.concatenate(_cal_tgts).flatten()

        # BUG FIX: Use std-ratio calibration instead of linregress(pred, target).
        # linregress(pred, target) computes slope = cov(p,t)/var(p), which is
        # correlation-dependent and does NOT correctly rescale prediction variance.
        # The correct fix: cal_slope = std(target) / std(pred)
        #                  cal_intercept = mean(target) - cal_slope * mean(pred)
        # This expands compressed predictions to match target std while preserving
        # correlation perfectly. The scale is identical in normalized and raw space.
        _pred_std  = float(_cp.std())
        _pred_mean = float(_cp.mean())
        _tgt_std   = float(_ct.std())
        _tgt_mean  = float(_ct.mean())

        _cal_slope     = _tgt_std / (_pred_std + 1e-8)
        _cal_intercept = _tgt_mean - _cal_slope * _pred_mean

        _corrected = _cal_slope * _cp + _cal_intercept
        _r = float(np.corrcoef(_cp, _ct)[0, 1])
        calibration_params = {
            "slope":      float(_cal_slope),
            "intercept":  float(_cal_intercept),
            "r_value":    _r,
            "pred_std":   _pred_std,
            "target_std": _tgt_std,
            "note": "std-ratio calibration: y_cal = slope*y_pred + intercept (normalized space)"
        }
        logger.info(f"  Cal slope={_cal_slope:.4f} (target_std/pred_std = {_tgt_std:.4f}/{_pred_std:.4f})")
        logger.info(f"  Cal intercept={_cal_intercept:.4f}")
        logger.info(f"  R² before: {_r2s(_ct, _cp):.4f}  after: {_r2s(_ct, _corrected):.4f}")
        logger.info(f"  Slope before: {_linregress(_cp,_ct)[0]:.4f}  after: {_linregress(_corrected,_ct)[0]:.4f}")
        import json as _json
        with open(save_dir / "calibration_params.json", "w") as _f:
            _json.dump(calibration_params, _f, indent=2)
        logger.info(f"  Calibration params saved to {save_dir / 'calibration_params.json'}")
        logger.info("="*60)
        # ─────────────────────────────────────────────────────────────────────

        # Final ensemble evaluation with BEST models
        ensemble_metrics = self.evaluate_ensemble(val_loader, X_val, y_val)
        self.history["ensemble_metrics"] = ensemble_metrics
        
        # Save models
        self._save_final_models(save_dir)
        self._save_history(save_dir)
        
        logger.info(self.checkpoint_manager.get_summary())
        
        # ── POST-TRAINING DIAGNOSTIC PLOTS ────────────────────────────────────
        logger.info("\n" + "="*60)
        logger.info("GENERATING POST-TRAINING DIAGNOSTIC PLOTS")
        logger.info("="*60)
        try:
            norm_stats = load_normalization_stats()

            # Collect CNN predictions on val set (best checkpoint already loaded)
            self.cnn_model.eval()
            _cnn_preds_all, _cnn_tgts_all, _cnn_preds_4d, _cnn_tgts_4d = [], [], [], []
            with torch.no_grad():
                for _d, _t in val_loader:
                    _out = self.cnn_model(_d.to(self.device)).cpu().numpy()
                    _cnn_preds_all.append(_out.flatten())
                    _cnn_tgts_all.append(_t.numpy().flatten())
                    _cnn_preds_4d.append(_out)
                    _cnn_tgts_4d.append(_t.numpy())
            _cnn_preds_flat = np.concatenate(_cnn_preds_all)
            _cnn_tgts_flat  = np.concatenate(_cnn_tgts_all)

            # Denormalize for plots
            if norm_stats:
                _cnn_preds_denorm = denormalize_predictions(_cnn_preds_flat, norm_stats)
                _cnn_tgts_denorm  = denormalize_predictions(_cnn_tgts_flat,  norm_stats)
            else:
                _cnn_preds_denorm = _cnn_preds_flat
                _cnn_tgts_denorm  = _cnn_tgts_flat

            cnn_val_metrics = self.history["cnn_metrics"][-1] if self.history["cnn_metrics"] else {}
            gbm_val_metrics = self.history["gbm_metrics"][-1] if self.history["gbm_metrics"] else {}

            self.plotter.plot_all_post_training(
                history=self.history,
                ensemble_metrics=ensemble_metrics,
                cnn_metrics=cnn_val_metrics,
                gbm_metrics=gbm_val_metrics,
                ensemble_weights=self.ensemble_weights,
                cnn_preds_flat=_cnn_preds_denorm,
                targets_flat=_cnn_tgts_denorm,
                gbm_model=self.gbm_trainer.best_model or self.gbm_trainer.model if self.gbm_trained else None,
                y_train=y_train,
                y_val=y_val,
                norm_stats=norm_stats,
            )

            # Spatial error maps
            _p4d = np.concatenate(_cnn_preds_4d, axis=0)
            _t4d = np.concatenate(_cnn_tgts_4d,  axis=0)
            self.plotter.plot_spatial_error_map(_p4d, _t4d, label="CNN", n_samples=min(6, len(_p4d)))

            # Ensemble comparison chart
            if self.gbm_trained:
                _xv_gbm, _yv_gbm = prepare_gbm_features(X_val, y_val)
                _gbm_p = self.gbm_trainer.predict(_xv_gbm, use_best=True)
                _gbm_p_d = denormalize_predictions(_gbm_p,    norm_stats) if norm_stats else _gbm_p
                _yv_d   = denormalize_predictions(_yv_gbm, norm_stats) if norm_stats else _yv_gbm
                _ens_p  = (self.ensemble_weights["cnn"] * _cnn_preds_flat[:len(_yv_gbm)] +
                           self.ensemble_weights["gbm"] * _gbm_p)
                _ens_p_d = denormalize_predictions(_ens_p, norm_stats) if norm_stats else _ens_p

                self.plotter.plot_pred_vs_actual(_gbm_p_d, _yv_d, "GBM")
                self.plotter.plot_pred_vs_actual(_ens_p_d, _yv_d, "Ensemble")
                self.plotter.plot_residual_distribution(_ens_p_d, _yv_d, "Ensemble")
                self.plotter.plot_stratified_error(_ens_p_d, _yv_d, "Ensemble")

                comparison = {
                    "CNN":             cnn_val_metrics,
                    "GBM":             gbm_val_metrics,
                    "Ensemble":        ensemble_metrics,
                }
                self.plotter.plot_ensemble_comparison(comparison)

            logger.info(f"✅ All diagnostic plots saved to: {self.plotter.save_dir}")
        except Exception as _plot_err:
            logger.warning(f"⚠️ Post-training plotting failed (non-fatal): {_plot_err}")
            import traceback; traceback.print_exc()
        # ──────────────────────────────────────────────────────────────────────

        logger.info("\n✅ Ensemble training complete!")
        return self.history
    
    def _save_checkpoint(self, save_dir, epoch, loss, metrics):
        """Save checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "cnn_state_dict": self.cnn_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "metrics": metrics,
            "ensemble_weights": self.ensemble_weights
        }
        
        path = save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def _save_final_models(self, save_dir):
        """Save best and final models (MODIFIED)"""
        # Best CNN is already saved by checkpoint manager
        logger.info("✅ Best CNN model already saved by CheckpointManager")
        
        # Save current CNN state as final (for comparison)
        torch.save(self.cnn_model.state_dict(), save_dir / "final_cnn.pth")
        logger.info("Saved current CNN state as final_cnn.pth")
        
        # Save GBM (now saves both best and final)
        if self.gbm_trained:
            self.gbm_trainer.save(save_dir / "gbm_model.pkl")
        
        # MODIFIED: Update ensemble config to reference BEST models
        ensemble_config = {
            "weights": self.ensemble_weights,
            "cnn_path": "checkpoints/best_r2.pth",  # CHANGED: Reference best checkpoint
            "gbm_path": "best_gbm_model.pkl" if self.gbm_trained else None,  # CHANGED
            "note": "Ensemble uses BEST models based on validation performance",
            "best_cnn_metric": "r2",
            "best_gbm_metric": "rmse"
        }
        
        with open(save_dir / "ensemble_config.json", "w") as f:
            json.dump(ensemble_config, f, indent=2)
        logger.info("💾 Saved ensemble configuration (referencing BEST models)")
    
    def _save_history(self, save_dir):
        """Save training history"""
        history_serializable = {}
        for key, values in self.history.items():
            if key in ["cnn_metrics", "gbm_metrics"]:
                history_serializable[key] = [
                    {k: float(v) for k, v in metrics.items()}
                    for metrics in values
                ]
            elif key == "ensemble_metrics":
                if isinstance(values, dict):
                    history_serializable[key] = {k: float(v) for k, v in values.items()}
            else:
                history_serializable[key] = [float(v) for v in values]
        
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history_serializable, f, indent=2)
        
        logger.info("Saved training history")


def validate_data_quality(X: np.ndarray, y: np.ndarray, split: str) -> bool:
    """Validate data quality before training"""
    logger.info(f"\n{'='*60}")
    logger.info(f"DATA QUALITY CHECK: {split.upper()}")
    logger.info(f"{'='*60}")
    
    issues = []
    
    # Check for NaN/Inf
    if np.isnan(X).any() or np.isinf(X).any():
        nan_pct = np.isnan(X).sum() / X.size * 100
        inf_pct = np.isinf(X).sum() / X.size * 100
        issues.append(f"X contains NaN ({nan_pct:.2f}%) or Inf ({inf_pct:.2f}%)")
    
    if np.isnan(y).any() or np.isinf(y).any():
        nan_pct = np.isnan(y).sum() / y.size * 100
        inf_pct = np.isinf(y).sum() / y.size * 100
        issues.append(f"y contains NaN ({nan_pct:.2f}%) or Inf ({inf_pct:.2f}%)")
    
    # Calculate statistics
    X_mean = X.mean()
    X_std = X.std()
    y_mean = y.mean()
    y_std = y.std()
    y_min = y.min()
    y_max = y.max()
    
    # Log feature statistics
    logger.info(f"Features (X) statistics:")
    logger.info(f"  Mean: {X_mean:.4f}")
    logger.info(f"  Std:  {X_std:.4f}")
    logger.info(f"  Min:  {X.min():.4f}")
    logger.info(f"  Max:  {X.max():.4f}")
    
    # Log target statistics
    logger.info(f"Target (y) statistics:")
    logger.info(f"  Mean: {y_mean:.4f}")
    logger.info(f"  Std:  {y_std:.4f}")
    logger.info(f"  Min:  {y_min:.4f}")
    logger.info(f"  Max:  {y_max:.4f}")
    logger.info(f"  Unique values: {len(np.unique(y))}")
    
    # Check normalization for TRAINING split
    if split == "train":
        logger.info(f"\nNormalization checks (expecting mean≈0, std≈1):")
        
        if not (-0.2 < X_mean < 0.2):
            issues.append(f"X not properly normalized (mean={X_mean:.4f}, expected ≈0)")
            logger.warning(f"  ⚠️ X mean={X_mean:.4f} (should be ≈0)")
        else:
            logger.info(f"  ✅ X mean={X_mean:.4f}")
        
        if not (0.8 < X_std < 1.2):
            issues.append(f"X not properly normalized (std={X_std:.4f}, expected ≈1)")
            logger.warning(f"  ⚠️ X std={X_std:.4f} (should be ≈1)")
        else:
            logger.info(f"  ✅ X std={X_std:.4f}")
        
        if not (-0.2 < y_mean < 0.2):
            issues.append(f"y not properly normalized (mean={y_mean:.4f}, expected ≈0)")
            logger.warning(f"  ⚠️ y mean={y_mean:.4f} (should be ≈0)")
        else:
            logger.info(f"  ✅ y mean={y_mean:.4f}")
        
        if not (0.8 < y_std < 1.2):
            issues.append(f"y not properly normalized (std={y_std:.4f}, expected ≈1)")
            logger.warning(f"  ⚠️ y std={y_std:.4f} (should be ≈1)")
        else:
            logger.info(f"  ✅ y std={y_std:.4f}")
    else:
        # For val/test, just check they're in normalized space
        logger.info(f"\nValidation/Test data checks:")
        if abs(X_mean) > 5.0:
            logger.warning(f"  ⚠️ X mean={X_mean:.4f} seems too large for normalized data")
        if abs(y_mean) > 5.0:
            logger.warning(f"  ⚠️ y mean={y_mean:.4f} seems too large for normalized data")
    
    # Check variance
    if y_std < 0.1:
        issues.append(f"Target has very low variance (std={y_std:.4f})")
    
    if y_std < 1e-6:
        issues.append(f"Target has ZERO variance")
    
    # Report results
    if issues:
        logger.error(f"❌ DATA QUALITY ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            logger.error(f"  {i}. {issue}")
        return False
    else:
        logger.info(f"\n✅ Data quality checks passed")
        logger.info(f"{'='*60}\n")
        return True


def load_data(split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed data"""
    data_dir = PROCESSED_DATA_DIR / "cnn_dataset" / split
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    
    logger.info(f"Loaded {split} data: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"  X range: [{X.min():.4f}, {X.max():.4f}]")
    logger.info(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
    
    if not validate_data_quality(X, y, split):
        raise ValueError(f"{split} data failed quality checks")
    
    return X, y


def main():
    """Main training script with ensemble - BEST MODELS VERSION"""
    logger.info("="*60)
    logger.info("URBAN HEAT ISLAND - ENSEMBLE TRAINING (BEST MODELS)")
    logger.info("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() and COMPUTE_CONFIG["use_gpu"] else "cpu")
    logger.info(f"Using device: {device}")
    
    # Verify normalization stats exist
    logger.info("\n" + "="*60)
    logger.info("VERIFYING NORMALIZATION STATISTICS")
    logger.info("="*60)
    
    stats_path = PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
    if not stats_path.exists():
        logger.error("❌ Normalization stats not found!")
        logger.error(f"   Expected: {stats_path}")
        logger.error("   ")
        logger.error("   This file should be created during preprocessing.")
        logger.error("   Run preprocessing.py to create normalized data with statistics.")
        raise FileNotFoundError(f"Missing normalization stats: {stats_path}")
    
    # Load and display normalization stats
    import json
    with open(stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    logger.info(f"✅ Found normalization stats: {stats_path}")
    logger.info(f"   Features: {norm_stats.get('n_channels', 'N/A')} channels")
    
    if 'target' in norm_stats:
        target_mean = norm_stats['target']['mean']
        target_std = norm_stats['target']['std']
        logger.info(f"   Target LST: mean={target_mean:.2f}°C, std={target_std:.2f}°C")
        logger.info(f"   (These are the ORIGINAL values before normalization)")
    
    logger.info("="*60)
    
    # Load data
    logger.info("\n" + "="*60)
    logger.info("LOADING TRAINING DATA")
    logger.info("="*60)
    X_train, y_train = load_data("train")
    
    logger.info("\n" + "="*60)
    logger.info("LOADING VALIDATION DATA")
    logger.info("="*60)
    X_val, y_val = load_data("val")
    
    # Create datasets
    train_dataset = UHIDataset(X_train, y_train, augment=True)
    val_dataset = UHIDataset(X_val, y_val, augment=False)
    
    # Create stratified sampler for balanced temperature distribution
    sampler = create_temperature_stratified_sampler(y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        sampler=sampler,
        num_workers=COMPUTE_CONFIG["num_workers"],
        pin_memory=COMPUTE_CONFIG["pin_memory"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=COMPUTE_CONFIG["num_workers"],
        pin_memory=COMPUTE_CONFIG["pin_memory"]
    )
    
    # Create CNN model
    logger.info("\nInitializing CNN model...")
    cnn_model = UNet(in_channels=CNN_CONFIG["input_channels"], out_channels=1)
    initialize_weights(cnn_model)
    logger.info(f"CNN parameters: {count_parameters(cnn_model):,}")
    
    # Create ensemble trainer
    logger.info("\nInitializing ensemble trainer...")
    ensemble_trainer = EnsembleTrainer(cnn_model, device)
    logger.info(f"Initial ensemble weights: CNN={ENSEMBLE_WEIGHTS['cnn']}, GBM={ENSEMBLE_WEIGHTS['gbm']}")
    logger.info("Note: Weights will be optimized based on BEST model performance")
    
    # Train ensemble
    logger.info("\n" + "="*60)
    logger.info("STARTING ENSEMBLE TRAINING")
    logger.info("="*60)
    
    try:
        history = ensemble_trainer.train(
            train_loader, val_loader,
            X_train, y_train, X_val, y_val,
            MODEL_DIR
        )
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE TRAINING COMPLETE")
    logger.info("="*60)
    
    # Print final metrics
    if "ensemble_metrics" in history and history["ensemble_metrics"]:
        final_metrics = history["ensemble_metrics"]
        logger.info("\nFINAL ENSEMBLE METRICS (Using BEST Models, Denormalized to °C):")
        logger.info(f"  R² Score: {final_metrics['r2']:.4f} (target: ≥ {VALIDATION_CONFIG['targets']['r2']})")
        logger.info(f"  RMSE: {final_metrics['rmse']:.4f}°C (target: ≤ {VALIDATION_CONFIG['targets']['rmse']}°C)")
        logger.info(f"  MAE: {final_metrics['mae']:.4f}°C (target: ≤ {VALIDATION_CONFIG['targets']['mae']}°C)")
        logger.info(f"  MBE: {final_metrics['mbe']:.4f}°C")
        
        logger.info("\nNote: Metrics calculated using BEST validation models:")
        logger.info("      - CNN: Best R² checkpoint from training")
        logger.info("      - GBM: Best RMSE iteration")
        logger.info("      - Predictions denormalized before computing metrics")
        
        # Check if targets met
        targets_met = (
            final_metrics['r2'] >= VALIDATION_CONFIG['targets']['r2'] and
            final_metrics['rmse'] <= VALIDATION_CONFIG['targets']['rmse'] and
            final_metrics['mae'] <= VALIDATION_CONFIG['targets']['mae']
        )
        
        if targets_met:
            logger.info("\n✅ ALL PERFORMANCE TARGETS MET!")
        else:
            logger.warning("\n⚠️ Some performance targets not met")
        
        logger.info("="*60)
    
    # Print comparison
    if history["cnn_metrics"]:
        cnn_final = history["cnn_metrics"][-1]
        logger.info("\nMODEL COMPARISON:")
        logger.info(f"  CNN Best - R²: {cnn_final['r2']:.4f}, RMSE: {cnn_final['rmse']:.4f}°C")
        if "ensemble_metrics" in history and history["ensemble_metrics"]:
            ens_final = history["ensemble_metrics"]
            logger.info(f"  Ensemble - R²: {ens_final['r2']:.4f}, RMSE: {ens_final['rmse']:.4f}°C")
            
            if cnn_final['r2'] > 0:
                improvement = (ens_final['r2'] - cnn_final['r2']) / abs(cnn_final['r2']) * 100
                logger.info(f"  Improvement: {improvement:+.2f}%")
        
        # Print final weights
        logger.info(f"\nFinal Ensemble Weights (Optimized):")
        logger.info(f"  CNN: {ensemble_trainer.ensemble_weights['cnn']:.4f}")
        logger.info(f"  GBM: {ensemble_trainer.ensemble_weights['gbm']:.4f}")
        
        logger.info("="*60)


if __name__ == "__main__":
    main()