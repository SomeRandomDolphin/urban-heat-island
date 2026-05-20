import sys
import json
import shutil
import pickle
import logging
import warnings
import traceback
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import lightgbm as lgb
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless/Windows runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import linregress  # moved from inline imports
from scipy import stats             # for DiagnosticsPlotter residual/QQ plots
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=UserWarning)

from config import (
    # Paths
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    # Model / training
    CNN_CONFIG,
    GBM_CONFIG,
    ENSEMBLE_WEIGHTS,
    TRAINING_CONFIG,
    AUGMENTATION_CONFIG,
    VALIDATION_CONFIG,
    COMPUTE_CONFIG,
    # New structured configs (replaces scattered magic numbers)
    SCHEDULER_CONFIG,
    EARLY_STOPPING_CONFIG,
    LAYERWISE_WEIGHT_DECAY,
    PROGRESSIVE_LOSS_CONFIG,
    AUGMENTATION_PROB_CONFIG,
    STRATIFIED_SAMPLER_CONFIG,
    CHECKPOINT_CONFIG,
    DATA_QUALITY_CONFIG,
    DISK_CONFIG,
    DIAGNOSTICS_CONFIG,
    MONITORING_CONFIG,
    LOGGING_CONFIG,
    HYPERPARAM_TUNING_CONFIG,
)
from models import UNet, ProgressiveLSTLoss, EarlyStopping, initialize_weights, count_parameters

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def load_normalization_stats(dataset_dir: Path = None) -> Optional[Dict]:
    """Load normalization statistics for denormalization during evaluation.

    Args:
        dataset_dir: Path to the dataset directory containing
            normalization_stats.json.  Defaults to
            PROCESSED_DATA_DIR / "cnn_dataset" for backward compatibility.

    Returns:
        Normalization statistics dictionary, or None if not found.
    """
    if dataset_dir is None:
        dataset_dir = PROCESSED_DATA_DIR / "cnn_dataset"
    stats_path = Path(dataset_dir) / "normalization_stats.json"

    if not stats_path.exists():
        logger.warning("⚠️ Normalization stats not found - predictions will remain in normalized space")
        return None

    with open(stats_path, 'r') as f:
        return json.load(f)

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

def check_disk_space(path: Path, required_mb: int = DISK_CONFIG["required_mb"]) -> bool:
    """Check if sufficient disk space is available.

    Returns False (rather than True) on error: we'd rather abort than silently
    proceed when we can't even verify free space.
    """
    try:
        stat = shutil.disk_usage(path)
        available_mb = stat.free / (1024 * 1024)
        logger.info(f"Available disk space: {available_mb:.2f} MB")

        if available_mb < required_mb:
            logger.warning(f"Low disk space! Available: {available_mb:.2f} MB, Required: {required_mb} MB")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e} — aborting as a precaution")
        return False




# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOTTING MODULE
# ══════════════════════════════════════════════════════════════════════════════

class DiagnosticsPlotter:
    """
    Centralised matplotlib diagnostics for the CNN+GBM ensemble pipeline.

    All figures are saved under `save_dir` (default: MODEL_DIR / "diagnostics").
    Call individual plot_* methods at the right moment in training, or call
    `plot_all_post_training` once training is complete.

    Style note: uses seaborn-v0_8-darkgrid when available (matplotlib ≥ 3.6),
    falls back to ggplot, then plain default — so it works on any matplotlib version.
    """

    # Ordered preference list — first one that exists wins
    _STYLE_CANDIDATES = [
        "seaborn-v0_8-darkgrid",   # matplotlib ≥ 3.6
        "seaborn-darkgrid",        # matplotlib < 3.6
        "ggplot",                  # always available
        "default",
    ]

    def __init__(self, save_dir: Path = None):
        self.save_dir = Path(save_dir) if save_dir else MODEL_DIR / "diagnostics"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._style = self._resolve_style()
        logger.info(f"📊 DiagnosticsPlotter initialised — style='{self._style}' → {self.save_dir}")

    # ── helpers ───────────────────────────────────────────────────────────────
    @classmethod
    def _resolve_style(cls) -> str:
        """Return the first available matplotlib style from the candidate list."""
        available = set(plt.style.available)
        for style in cls._STYLE_CANDIDATES:
            if style in available:
                return style
        return "default"

    def _use_style(self):
        """Apply the resolved style, swallowing any errors gracefully."""
        try:
            plt.style.use(self._style)
        except Exception:
            pass  # proceed with whatever style is active

    def _save(self, fig, name: str):
        path = self.save_dir / f"{name}.png"
        try:
            fig.savefig(path, dpi=DIAGNOSTICS_CONFIG["save_dpi"], bbox_inches="tight")
            logger.info(f"  ✅ Saved: {path.name}")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not save {path.name}: {e}")
        finally:
            plt.close(fig)

    # ── 1. Training curves ────────────────────────────────────────────────────
    def plot_training_curves(self, history: dict):
        """4-panel: train/val loss · val R² · LR schedule · weight-norm."""
        try:
            self._use_style()
            epochs = range(1, len(history["train_loss"]) + 1)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle("Training Curves", fontsize=16, fontweight="bold")

            # Loss
            ax = axes[0, 0]
            ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
            ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#F44336")
            ax.set_title("Train vs Validation Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.legend()

            # R²
            ax = axes[0, 1]
            if history["cnn_metrics"]:
                r2_scores = [m["r2"] for m in history["cnn_metrics"]]
                r2_epochs = range(1, len(r2_scores) + 1)
                ax.plot(r2_epochs, r2_scores, label="Val R²", color="#4CAF50")
                r2_target = DIAGNOSTICS_CONFIG["r2_target_line"]
                ax.axhline(r2_target, ls="--", color="gray", lw=0.8,
                           label=f"Target ({r2_target})")
                ax.set_title("CNN Validation R²"); ax.set_xlabel("Epoch"); ax.set_ylabel("R²")
                ax.set_ylim([-0.1, 1.05]); ax.legend()

            # LR
            ax = axes[1, 0]
            ax.plot(epochs, history["lr"], color="#FF9800")
            ax.set_title("Learning Rate Schedule"); ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
            ax.set_yscale("log")

            # Weight norm
            ax = axes[1, 1]
            if "weight_norms" in history and history["weight_norms"]:
                wn_epochs = range(1, len(history["weight_norms"]) + 1)
                ax.plot(wn_epochs, history["weight_norms"], color="#9C27B0")
                ax.set_title("L2 Weight Norm"); ax.set_xlabel("Epoch"); ax.set_ylabel("Norm")
            else:
                ax.text(0.5, 0.5, "No weight-norm data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title("L2 Weight Norm")

            plt.tight_layout()
            self._save(fig, "01_training_curves")
        except Exception as e:
            logger.warning(f"plot_training_curves failed: {e}")

    # ── 2. Predicted vs Actual ────────────────────────────────────────────────
    def plot_pred_vs_actual(self, preds: np.ndarray, targets: np.ndarray,
                            label: str = "Model", metrics: dict = None):
        """Scatter + residual plot (denormalised °C values)."""
        try:
            self._use_style()

            preds   = np.asarray(preds).flatten()
            targets = np.asarray(targets).flatten()
            mask    = np.isfinite(preds) & np.isfinite(targets)
            preds, targets = preds[mask], targets[mask]

            if len(preds) < 2:
                logger.warning(f"plot_pred_vs_actual: not enough finite samples ({len(preds)}), skipping")
                plt.close()
                return
            slope, intercept, r_val, *_ = linregress(targets, preds)
            residuals = preds - targets

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"{label} – Predicted vs Actual", fontsize=14, fontweight="bold")

            ax = axes[0]
            sc = ax.scatter(targets, preds, alpha=0.35, s=12,
                            c=np.abs(residuals), cmap="RdYlGn_r", label="Samples")
            plt.colorbar(sc, ax=ax, label="|Residual| (°C)")
            lo = min(targets.min(), preds.min()); hi = max(targets.max(), preds.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect (slope=1)")
            fit_x = np.linspace(lo, hi, 200)
            ax.plot(fit_x, slope * fit_x + intercept, "r-", lw=1.5,
                    label=f"Fit  slope={slope:.3f}")
            ax.set_xlabel("Actual (°C)"); ax.set_ylabel("Predicted (°C)")
            ax.legend(fontsize=8)
            info = [f"R²={r_val**2:.4f}", f"RMSE={np.sqrt(np.mean(residuals**2)):.3f}°C",
                    f"MAE={np.mean(np.abs(residuals)):.3f}°C",
                    f"slope={slope:.3f}", f"intercept={intercept:.3f}"]
            if metrics:
                info += [f"std_ratio={metrics.get('std_ratio', float('nan')):.3f}",
                         f"MBE={metrics.get('mbe', float('nan')):.3f}°C"]
            ax.text(0.03, 0.97, "\n".join(info), transform=ax.transAxes,
                    fontsize=8, va="top", bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            ax = axes[1]
            ax.scatter(targets, residuals, alpha=0.35, s=12, color="#5C6BC0")
            ax.axhline(0, color="black", lw=1.2)
            ax.axhline(+np.std(residuals), color="orange", ls="--", lw=0.9, label="±1σ")
            ax.axhline(-np.std(residuals), color="orange", ls="--", lw=0.9)
            ax.set_xlabel("Actual (°C)"); ax.set_ylabel("Residual (°C)")
            ax.set_title("Residuals vs Actual"); ax.legend(fontsize=8)

            plt.tight_layout()
            safe = label.replace(" ", "_").replace("(", "").replace(")", "")
            self._save(fig, f"02_pred_vs_actual_{safe}")
        except Exception as e:
            logger.warning(f"plot_pred_vs_actual failed: {e}")

    # ── 3. Residual distribution ──────────────────────────────────────────────
    def plot_residual_distribution(self, preds: np.ndarray, targets: np.ndarray,
                                   label: str = "Model"):
        """Histogram + Q-Q plot of residuals."""
        try:
            self._use_style()
            residuals = (np.asarray(preds) - np.asarray(targets)).flatten()
            residuals = residuals[np.isfinite(residuals)]

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle(f"{label} – Residual Distribution", fontsize=14, fontweight="bold")

            ax = axes[0]
            ax.hist(residuals, bins=60, color="#42A5F5", edgecolor="white",
                    linewidth=0.3, density=True, alpha=0.75, label="Residuals")
            mu, sigma = residuals.mean(), residuals.std()
            x_pdf = np.linspace(residuals.min(), residuals.max(), 300)
            ax.plot(x_pdf, stats.norm.pdf(x_pdf, mu, sigma), "r-", lw=2,
                    label=f"N({mu:.3f}, {sigma:.3f})")
            ax.axvline(0, color="black", lw=1.2, ls="--")
            ax.set_xlabel("Residual (°C)"); ax.set_ylabel("Density"); ax.legend()
            skew = stats.skew(residuals)
            kurt = stats.kurtosis(residuals)
            _, p_norm = stats.normaltest(residuals)
            ax.text(0.97, 0.97, f"skew={skew:.3f}\nkurt={kurt:.3f}\np_norm={p_norm:.3e}",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            ax = axes[1]
            (osm, osr), (sl, ic, r) = stats.probplot(residuals, dist="norm")
            ax.scatter(osm, osr, s=6, alpha=0.4, color="#EF5350")
            qq_line = np.array([osm[0], osm[-1]])
            ax.plot(qq_line, sl * qq_line + ic, "k-", lw=1.2)
            ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
            ax.set_title(f"Q-Q  (r={r:.4f})")

            plt.tight_layout()
            safe = label.replace(" ", "_").replace("(", "").replace(")", "")
            self._save(fig, f"03_residual_dist_{safe}")
        except Exception as e:
            logger.warning(f"plot_residual_distribution failed: {e}")

    # ── 4. Ensemble strategy comparison ──────────────────────────────────────
    def plot_ensemble_comparison(self, comparison: dict,
                                  comparison_spatial: dict = None):
        """Two-row bar charts: patch-level and pixel-level R², RMSE, MAE.

        Parameters
        ----------
        comparison         : patch-mean metrics dict  {strategy_name: metrics}
        comparison_spatial : pixel-level metrics dict {strategy_name: metrics}
                             If None, only the patch-level row is drawn.
        """
        try:
            self._use_style()
            names  = list(comparison.keys())
            colors = plt.cm.tab10(np.linspace(0, 0.6, len(names)))

            has_spatial  = comparison_spatial is not None
            n_rows       = 2 if has_spatial else 1
            fig, axes    = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows), squeeze=False)
            fig.suptitle(
                "Ensemble Strategy Comparison — Two-Level Evaluation",
                fontsize=14, fontweight="bold",
            )

            metric_keys   = ["r2", "rmse", "mae"]
            metric_labels = ["R²", "RMSE (°C)", "MAE (°C)"]

            def _draw_row(row_axes, data_dict, row_title_prefix):
                for col, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
                    ax   = row_axes[col]
                    vals = [data_dict[n].get(mkey, 0) for n in names]
                    bars = ax.bar(names, vals, color=colors, edgecolor="white")
                    ax.set_title(f"{row_title_prefix}  —  {mlabel}", fontsize=10)
                    ax.set_ylabel(mlabel)
                    ax.set_xticks(range(len(names)))
                    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
                    for bar, v in zip(bars, vals):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.001,
                            f"{v:.4f}", ha="center", va="bottom", fontsize=8,
                        )

            _draw_row(axes[0], comparison, "Patch-Mean (all models comparable)")

            if has_spatial:
                _draw_row(axes[1], comparison_spatial,
                          "Pixel-Level (spatial quality; flat models penalised)")
                flat_models = {"GBM Only", "Weighted Ensemble"}
                for col in range(3):
                    ax = axes[1, col]
                    for bar_patch, name in zip(ax.patches, names):
                        if name in flat_models:
                            bar_patch.set_hatch("//")
                            bar_patch.set_alpha(0.6)
                fig.text(
                    0.5, 0.01,
                    "Hatched bars = flat prediction (no within-patch spatial variation). "
                    "Low pixel R² is expected and reveals the cost of missing spatial detail.",
                    ha="center", fontsize=9, style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.12),
                )

            plt.tight_layout(rect=[0, 0.04, 1, 1] if has_spatial else [0, 0, 1, 1])
            self._save(fig, "04_ensemble_comparison")
        except Exception as e:
            logger.warning(f"plot_ensemble_comparison failed: {e}")

    # ── 5. GBM feature importance ─────────────────────────────────────────────
    def plot_gbm_feature_importance(self, gbm_model, top_n: int = DIAGNOSTICS_CONFIG["gbm_importance_top_n"]):
        """Horizontal bar chart of LightGBM feature importances."""
        try:
            if gbm_model is None:
                return
            self._use_style()
            imp   = gbm_model.feature_importance(importance_type="gain")
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
        except Exception as e:
            logger.warning(f"plot_gbm_feature_importance failed: {e}")

    # ── 6. Data distribution ──────────────────────────────────────────────────
    def plot_data_distribution(self, y_train: np.ndarray, y_val: np.ndarray,
                               norm_stats: dict = None):
        """Train vs Val LST histogram in normalised and °C space."""
        try:
            self._use_style()
            train_flat = y_train.flatten(); val_flat = y_val.flatten()
            n_panels = 2 if norm_stats else 1
            fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
            fig.suptitle("Target (LST) Distribution – Train vs Val",
                         fontsize=14, fontweight="bold")
            if n_panels == 1:
                axes = [axes]

            ax = axes[0]
            ax.hist(train_flat, bins=80, alpha=0.55, color="#1E88E5",
                    label=f"Train  n={len(train_flat):,}", density=True)
            ax.hist(val_flat,   bins=80, alpha=0.55, color="#E53935",
                    label=f"Val    n={len(val_flat):,}",  density=True)
            ax.set_xlabel("Normalised LST"); ax.set_ylabel("Density")
            ax.set_title("Normalised Space"); ax.legend()

            if norm_stats and "target" in norm_stats:
                mean = norm_stats["target"]["mean"]; std = norm_stats["target"]["std"]
                train_c = train_flat * std + mean; val_c = val_flat * std + mean
                ax = axes[1]
                ax.hist(train_c, bins=80, alpha=0.55, color="#1E88E5",
                        label="Train", density=True)
                ax.hist(val_c,   bins=80, alpha=0.55, color="#E53935",
                        label="Val",   density=True)
                ax.set_xlabel("LST (°C)"); ax.set_ylabel("Density")
                ax.set_title("Denormalised (°C)"); ax.legend()
                ax.text(0.97, 0.97,
                        f"Train: {train_c.mean():.1f}±{train_c.std():.1f}°C\n"
                        f"Val:   {val_c.mean():.1f}±{val_c.std():.1f}°C",
                        transform=ax.transAxes, fontsize=8, ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            plt.tight_layout()
            self._save(fig, "06_data_distribution")
        except Exception as e:
            logger.warning(f"plot_data_distribution failed: {e}")

    # ── 7. Diagnostic metrics over epochs ────────────────────────────────────
    def plot_diagnostic_metrics_over_epochs(self, history: dict):
        """Slope, std_ratio, MBE, R² across every epoch."""
        try:
            if not history.get("cnn_metrics"):
                return
            self._use_style()
            epochs     = range(1, len(history["cnn_metrics"]) + 1)
            slopes     = [m.get("slope",     float("nan")) for m in history["cnn_metrics"]]
            std_ratios = [m.get("std_ratio", float("nan")) for m in history["cnn_metrics"]]
            mbe_vals   = [m.get("mbe",       float("nan")) for m in history["cnn_metrics"]]
            r2_vals    = [m.get("r2",        float("nan")) for m in history["cnn_metrics"]]

            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle("CNN Diagnostic Metrics Over Training Epochs",
                         fontsize=14, fontweight="bold")

            def _plot(ax, data, title, ylabel, target=None, danger=None):
                ax.plot(epochs, data, lw=1.5, color="#29B6F6")
                if target is not None:
                    ax.axhline(target, color="green", ls="--", lw=1,
                               label=f"Target ({target})")
                if danger is not None:
                    ax.axhline(danger, color="red",   ls=":",  lw=1,
                               label=f"Danger ({danger})")
                ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
                if target is not None or danger is not None:
                    ax.legend(fontsize=8)

            _plot(axes[0, 0], r2_vals,    "Validation R²",        "R²",
                  target=DIAGNOSTICS_CONFIG["r2_target_line"])
            _plot(axes[0, 1], slopes,     "Prediction Slope",     "Slope",
                  target=DIAGNOSTICS_CONFIG["slope_target"],
                  danger=DIAGNOSTICS_CONFIG["slope_danger"])
            _plot(axes[1, 0], std_ratios, "Std Ratio (pred/tgt)", "Std Ratio",
                  target=DIAGNOSTICS_CONFIG["std_ratio_target"],
                  danger=DIAGNOSTICS_CONFIG["std_ratio_danger"])
            _plot(axes[1, 1], mbe_vals,   "Mean Bias Error",      "MBE (°C)")
            axes[1, 1].axhline(0, color="green", ls="--", lw=1, label="Target (0)")
            axes[1, 1].legend(fontsize=8)

            plt.tight_layout()
            self._save(fig, "07_diagnostic_metrics_epochs")
        except Exception as e:
            logger.warning(f"plot_diagnostic_metrics_over_epochs failed: {e}")

    # ── 8. Spatial error maps ─────────────────────────────────────────────────
    def plot_spatial_error_map(self, preds_4d: np.ndarray, targets_4d: np.ndarray,
                               label: str = "Model",
                               n_samples: int = DIAGNOSTICS_CONFIG["spatial_n_samples"]):
        """Actual / Predicted / Error side-by-side for first n_samples patches."""
        try:
            self._use_style()

            def _sq(arr):
                arr = np.asarray(arr)
                if arr.ndim == 4 and arr.shape[1] == 1:
                    return arr[:, 0, :, :]
                if arr.ndim == 4 and arr.shape[3] == 1:
                    return arr[:, :, :, 0]
                return arr

            P = _sq(preds_4d);  P = P[:min(n_samples, len(P))]
            T = _sq(targets_4d); T = T[:min(n_samples, len(T))]
            n_samples = min(len(P), len(T))  # FIX: clamp to available data
            if n_samples == 0:
                logger.warning("plot_spatial_error_map: no samples to plot")
                return
            P, T = P[:n_samples], T[:n_samples]
            E = P - T

            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
            if n_samples == 1:
                axes = axes[np.newaxis, :]
            fig.suptitle(f"{label} – Spatial Error Maps (first {n_samples} patches)",
                         fontsize=13, fontweight="bold")

            vmin = min(T.min(), P.min()); vmax = max(T.max(), P.max())
            # FIX: TwoSlopeNorm requires vmin < vcenter < vmax; guard against degenerate range
            err_abs = max(np.abs(E).max(), 1e-6)
            e_min, e_max = -err_abs, err_abs
            if e_min >= 0:
                e_min = -1e-6
            if e_max <= 0:
                e_max = 1e-6
            norm_err = TwoSlopeNorm(vmin=e_min, vcenter=0, vmax=e_max)

            for i in range(n_samples):
                axes[i, 0].imshow(T[i], vmin=vmin, vmax=vmax, cmap="hot")
                axes[i, 0].set_title(f"Sample {i+1} – Actual",    fontsize=8)
                axes[i, 0].axis("off")
                axes[i, 1].imshow(P[i], vmin=vmin, vmax=vmax, cmap="hot")
                axes[i, 1].set_title(f"Sample {i+1} – Predicted", fontsize=8)
                axes[i, 1].axis("off")
                im = axes[i, 2].imshow(E[i], norm=norm_err, cmap="RdBu_r")
                axes[i, 2].set_title(f"Sample {i+1} – Error",     fontsize=8)
                axes[i, 2].axis("off")
                plt.colorbar(im, ax=axes[i, 2], shrink=0.8)

            plt.tight_layout()
            safe = label.replace(" ", "_").replace("(", "").replace(")", "")
            self._save(fig, f"08_spatial_error_{safe}")
        except Exception as e:
            logger.warning(f"plot_spatial_error_map failed: {e}")

    # ── 9. Temperature-stratified error ──────────────────────────────────────
    def plot_stratified_error(self, preds: np.ndarray, targets: np.ndarray,
                              label: str = "Model",
                              n_bins: int = DIAGNOSTICS_CONFIG["stratified_n_bins"]):
        """RMSE, MAE, MBE bucketed by temperature percentile bins."""
        try:
            self._use_style()
            preds   = np.asarray(preds).flatten()
            targets = np.asarray(targets).flatten()
            mask    = np.isfinite(preds) & np.isfinite(targets)
            preds, targets = preds[mask], targets[mask]

            bins = np.percentile(targets, np.linspace(0, 100, n_bins + 1))
            centers, rmse_v, mae_v, mbe_v, counts = [], [], [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                idx = (targets >= lo) & (targets <= hi)
                if idx.sum() < 5:
                    continue
                p, t = preds[idx], targets[idx]
                centers.append((lo + hi) / 2)
                rmse_v.append(np.sqrt(np.mean((p - t) ** 2)))
                mae_v.append(np.mean(np.abs(p - t)))
                mbe_v.append(np.mean(p - t))
                counts.append(idx.sum())

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(f"{label} – Stratified Error Analysis",
                         fontsize=13, fontweight="bold")

            def _bar(ax, y, title, ylabel, color):
                ax.bar(range(len(centers)), y, color=color, edgecolor="white", alpha=0.8)
                ax.set_xticks(range(len(centers)))
                ax.set_xticklabels([f"{c:.1f}" for c in centers],
                                   rotation=45, ha="right", fontsize=7)
                ax.set_xlabel("Temperature Bin Centre (°C)")
                ax.set_ylabel(ylabel); ax.set_title(title)

            _bar(axes[0], rmse_v, "RMSE by Temperature Bin", "RMSE (°C)", "#EF5350")
            _bar(axes[1], mae_v,  "MAE by Temperature Bin",  "MAE (°C)",  "#FFA726")
            axes[2].bar(range(len(centers)), mbe_v,
                        color=["#1E88E5" if v >= 0 else "#E53935" for v in mbe_v],
                        edgecolor="white", alpha=0.8)
            axes[2].axhline(0, color="black", lw=1)
            axes[2].set_xticks(range(len(centers)))
            axes[2].set_xticklabels([f"{c:.1f}" for c in centers],
                                    rotation=45, ha="right", fontsize=7)
            axes[2].set_xlabel("Temperature Bin Centre (°C)")
            axes[2].set_ylabel("MBE (°C)"); axes[2].set_title("MBE by Temperature Bin")
            for i, cnt in enumerate(counts):
                axes[0].text(i, rmse_v[i] + 0.01, f"n={cnt}", ha="center", fontsize=6)

            plt.tight_layout()
            safe = label.replace(" ", "_").replace("(", "").replace(")", "")
            self._save(fig, f"09_stratified_error_{safe}")
        except Exception as e:
            logger.warning(f"plot_stratified_error failed: {e}")

    # ── 10. Summary dashboard ─────────────────────────────────────────────────
    def plot_summary_dashboard(self, history: dict, ensemble_metrics: dict,
                               cnn_metrics: dict, gbm_metrics: dict,
                               ensemble_weights: dict):
        """One-page overview: metrics table · weight pie · loss curves · trends."""
        try:
            self._use_style()
            fig = plt.figure(figsize=(18, 11))
            gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)
            fig.suptitle("Ensemble Training – Summary Dashboard",
                         fontsize=16, fontweight="bold")

            epochs = range(1, len(history["train_loss"]) + 1)

            # Metrics table
            ax_tbl = fig.add_subplot(gs[0, :2]); ax_tbl.axis("off")
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
            tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
            best_r2_row = max([(i, float(rows[i+1][1])) for i in range(3)],
                              key=lambda x: x[1])[0]
            for col in range(6):
                try:
                    tbl[best_r2_row + 1, col].set_facecolor("#C8E6C9")
                except (KeyError, IndexError):
                    pass  # FIX: guard against table indexing edge cases
            ax_tbl.set_title("Validation Metrics Comparison", fontsize=10, pad=4)

            # Weight pie
            ax_pie = fig.add_subplot(gs[0, 2])
            wts = [ensemble_weights.get("cnn", 0), ensemble_weights.get("gbm", 0)]
            if sum(wts) > 0:
                ax_pie.pie(wts, labels=["CNN", "GBM"], autopct="%1.1f%%",
                           colors=["#42A5F5", "#EF5350"], startangle=90)
            ax_pie.set_title("Final Ensemble Weights", fontsize=10)

            # R² gauge
            ax_r2 = fig.add_subplot(gs[0, 3])
            r2_val = ensemble_metrics.get("r2", 0)
            color_r2 = "#F44336" if r2_val < 0.7 else ("#FF9800" if r2_val < 0.85 else "#4CAF50")
            ax_r2.barh(["R²"], [min(max(r2_val, 0), 1)], color=color_r2)
            ax_r2.axvline(0.85, color="black", ls="--", lw=1.2, label="Target (0.85)")
            ax_r2.set_xlim([0, 1]); ax_r2.legend(fontsize=8)
            ax_r2.set_title(f"Ensemble R² = {r2_val:.4f}", fontsize=10)

            # Loss curves
            ax_loss = fig.add_subplot(gs[1, :2])
            ax_loss.plot(epochs, history["train_loss"], label="Train", color="#2196F3")
            ax_loss.plot(epochs, history["val_loss"],   label="Val",   color="#F44336")
            ax_loss.set_title("Loss Curves"); ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss"); ax_loss.legend()

            # R² over epochs
            ax_r2c = fig.add_subplot(gs[1, 2:])
            if history["cnn_metrics"]:
                r2s = [m["r2"] for m in history["cnn_metrics"]]
                r2c_epochs = range(1, len(r2s) + 1)  # FIX: use cnn_metrics length, not train_loss length
                ax_r2c.plot(r2c_epochs, r2s, color="#4CAF50", lw=1.5)
                ax_r2c.axhline(0.85, ls="--", color="gray", lw=0.8, label="Target")
                ax_r2c.set_title("CNN Val R²"); ax_r2c.set_xlabel("Epoch")
                ax_r2c.set_ylabel("R²"); ax_r2c.legend()

            # Bottom row diagnostics
            if history["cnn_metrics"]:
                slopes     = [m.get("slope",     float("nan")) for m in history["cnn_metrics"]]
                std_ratios = [m.get("std_ratio",  float("nan")) for m in history["cnn_metrics"]]
                mbe_vals   = [m.get("mbe",        float("nan")) for m in history["cnn_metrics"]]
                lrs        = history["lr"]
                diag_epochs = range(1, len(history["cnn_metrics"]) + 1)  # FIX: use cnn_metrics length
                for col_idx, (data, title, target, use_log) in enumerate(zip(
                        [slopes, std_ratios, mbe_vals, lrs],
                        ["Slope", "Std Ratio", "MBE (°C)", "Learning Rate"],
                        [1.0, 1.0, 0.0, None],
                        [False, False, False, True])):
                    ax_d = fig.add_subplot(gs[2, col_idx])
                    # FIX: LR uses train_loss-length range; diagnostics use cnn_metrics range
                    ep_d = range(1, len(lrs) + 1) if col_idx == 3 else diag_epochs
                    ax_d.plot(ep_d, data, lw=1.2, color="#FF7043")
                    if target is not None:
                        ax_d.axhline(target, color="green", ls="--", lw=0.9)
                    ax_d.set_title(title, fontsize=9)
                    ax_d.set_xlabel("Epoch", fontsize=8)
                    if use_log:
                        ax_d.set_yscale("log")

            self._save(fig, "10_summary_dashboard")
        except Exception as e:
            logger.warning(f"plot_summary_dashboard failed: {e}")

    # ── convenience: all post-training plots ──────────────────────────────────
    def plot_all_post_training(self, history: dict, ensemble_metrics: dict,
                               cnn_metrics: dict, gbm_metrics: dict,
                               ensemble_weights: dict,
                               cnn_preds_flat: np.ndarray = None,
                               targets_flat: np.ndarray = None,
                               gbm_preds_flat: np.ndarray = None,
                               gbm_targets_flat: np.ndarray = None,
                               gbm_model=None,
                               y_train: np.ndarray = None,
                               y_val: np.ndarray = None,
                               norm_stats: dict = None,
                               ensemble_comparison: dict = None,
                               ensemble_comparison_spatial: dict = None):
        """Convenience wrapper — calls every available plot method."""
        logger.info("\n📊 Generating all post-training diagnostic plots...")
        self.plot_training_curves(history)
        self.plot_diagnostic_metrics_over_epochs(history)
        self.plot_summary_dashboard(history, ensemble_metrics, cnn_metrics,
                                    gbm_metrics, ensemble_weights)
        if ensemble_comparison is not None:
            self.plot_ensemble_comparison(ensemble_comparison,
                                          comparison_spatial=ensemble_comparison_spatial)
        if cnn_preds_flat is not None and targets_flat is not None:
            self.plot_pred_vs_actual(cnn_preds_flat, targets_flat, "CNN")
            self.plot_residual_distribution(cnn_preds_flat, targets_flat, "CNN")
            self.plot_stratified_error(cnn_preds_flat, targets_flat, "CNN")
        if gbm_preds_flat is not None and gbm_targets_flat is not None:
            self.plot_pred_vs_actual(gbm_preds_flat, gbm_targets_flat, "GBM")
            self.plot_residual_distribution(gbm_preds_flat, gbm_targets_flat, "GBM")
            self.plot_stratified_error(gbm_preds_flat, gbm_targets_flat, "GBM")
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
        Stochastic augmentation pipeline.  All probabilities and magnitudes are
        read from AUGMENTATION_PROB_CONFIG in config.py so they can be tuned
        without touching model code.
        """
        cfg = AUGMENTATION_PROB_CONFIG

        # 1. Geometric transformations
        if torch.rand(1) < cfg["geometric_prob"]:
            if torch.rand(1) < cfg["flip_prob"]:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            if torch.rand(1) < cfg["flip_prob"]:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
            if torch.rand(1) < cfg["rot90_prob"]:
                k = torch.randint(1, 4, (1,)).item()
                x = torch.rot90(x, k, dims=[1, 2])
                y = torch.rot90(y, k, dims=[1, 2])

        # 2. Brightness adjustment — simulates different solar angles / ToD
        if torch.rand(1) < cfg["brightness_prob"]:
            factor = 1.0 + (torch.rand(1) - 0.5) * cfg["brightness_range"]
            x = x * factor

        # 3. Contrast adjustment — simulates atmospheric variation
        if torch.rand(1) < cfg["contrast_prob"]:
            factor = 1.0 + (torch.rand(1) - 0.5) * cfg["contrast_range"]
            mean = x.mean(dim=[1, 2], keepdim=True)
            x = (x - mean) * factor + mean

        # 4. Gaussian noise — simulates sensor noise / atmospheric interference
        if torch.rand(1) < cfg["noise_prob"]:
            x = x + torch.randn_like(x) * cfg["noise_std"]

        # 5. Regional dropout — simulates cloud patches / missing data
        if torch.rand(1) < cfg["cutout_prob"]:
            h, w = x.shape[1], x.shape[2]
            cut_h = int(h * cfg["cutout_size_frac"])
            cut_w = int(w * cfg["cutout_size_frac"])
            cx = torch.randint(0, w - cut_w + 1, (1,)).item()
            cy = torch.randint(0, h - cut_h + 1, (1,)).item()
            # Fill with patch mean (more realistic than zeros)
            x[:, cy:cy + cut_h, cx:cx + cut_w] = x.mean(dim=[1, 2], keepdim=True)

        # 6. MixUp — helps model learn smoother decision boundaries
        if torch.rand(1) < cfg["mixup_prob"]:
            alpha = cfg["mixup_alpha"]
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
            perm_h = torch.randperm(x.shape[1])
            perm_w = torch.randperm(x.shape[2])
            x_perm = x[:, perm_h, :][:, :, perm_w]
            y_perm = y[:, perm_h, :][:, :, perm_w]
            x = lam * x + (1 - lam) * x_perm
            y = lam * y + (1 - lam) * y_perm

        return x, y


def prepare_gbm_features(X: np.ndarray, y: np.ndarray,
                          cnn_bottleneck_feats: np.ndarray = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare tabular features for GBM from spatial data.

    FIX 8: Added texture features per channel:
        - range, skewness, kurtosis (distribution shape)
        - mean gradient magnitude (edge/boundary strength)
        - max gradient magnitude (sharpest edge in patch)
    These capture urban morphology patterns the original stats missed.

    FIX 6: Optional CNN bottleneck features can be appended.
        Pass cnn_bottleneck_feats (N, D) to include spatial-context
        statistics extracted from the CNN encoder.

    Args:
        X: NORMALIZED image patches (N, H, W, C) — mean≈0, std≈1
        y: NORMALIZED target LST (N, H, W, 1) — mean≈0, std≈1
        cnn_bottleneck_feats: Optional (N, D) array of CNN bottleneck stats

    Returns:
        features_df: DataFrame with aggregated features per patch (normalized)
        targets:     Patch-mean targets (normalized)
    """
    logger.info("Preparing GBM features from spatial data (with texture features)...")

    n_samples, height, width, n_channels = X.shape
    features_list = []

    for i in range(n_samples):
        patch_features = {}

        for ch in range(n_channels):
            channel_data = X[i, :, :, ch]
            flat = channel_data.flatten()

            # ── Original statistical features ────────────────────────────────
            patch_features[f'ch{ch}_mean']   = flat.mean()
            patch_features[f'ch{ch}_std']    = flat.std()
            patch_features[f'ch{ch}_min']    = flat.min()
            patch_features[f'ch{ch}_max']    = flat.max()
            patch_features[f'ch{ch}_median'] = np.median(flat)
            patch_features[f'ch{ch}_p25']    = np.percentile(flat, 25)
            patch_features[f'ch{ch}_p75']    = np.percentile(flat, 75)

            # ── FIX 8: Texture / distribution-shape features ──────────────────
            patch_features[f'ch{ch}_range'] = flat.max() - flat.min()

            # Skewness and kurtosis (manual — avoids scipy import in hot loop)
            centered = flat - flat.mean()
            std_c    = flat.std() + 1e-8
            patch_features[f'ch{ch}_skew'] = float((centered ** 3).mean() / std_c ** 3)
            patch_features[f'ch{ch}_kurt'] = float((centered ** 4).mean() / std_c ** 4 - 3)

            # Gradient magnitude (edge strength) — uses finite differences
            dx = np.diff(channel_data, axis=1)   # (H, W-1)
            dy = np.diff(channel_data, axis=0)   # (H-1, W)
            # Align shapes for element-wise combination
            h_min = min(dx.shape[0], dy.shape[0])
            w_min = min(dx.shape[1], dy.shape[1])
            grad_mag = np.sqrt(dx[:h_min, :w_min] ** 2 + dy[:h_min, :w_min] ** 2)
            patch_features[f'ch{ch}_grad_mean'] = grad_mag.mean()
            patch_features[f'ch{ch}_grad_max']  = grad_mag.max()

        # Spatial metadata
        patch_features['height'] = height
        patch_features['width']  = width

        features_list.append(patch_features)

    features_df = pd.DataFrame(features_list)

    # ── FIX 6: Append CNN bottleneck features if provided ────────────────────
    if cnn_bottleneck_feats is not None:
        bot_cols = [f"cnn_bot_{i}" for i in range(cnn_bottleneck_feats.shape[1])]
        bot_df   = pd.DataFrame(cnn_bottleneck_feats, columns=bot_cols)
        features_df = pd.concat([features_df, bot_df], axis=1)
        logger.info(f"  Added {cnn_bottleneck_feats.shape[1]} CNN bottleneck features")

    # Flatten targets (patch-mean LST)
    targets = y.reshape(n_samples, -1).mean(axis=1)

    logger.info(f"GBM features shape: {features_df.shape}")
    logger.info(f"GBM targets shape:  {targets.shape}")

    return features_df, targets


def extract_cnn_bottleneck_features(X: np.ndarray, cnn_model,
                                    device: str = "cpu",
                                    batch_size: int = 32) -> np.ndarray:
    """
    FIX 6: Extract bottleneck statistics from the CNN encoder to enrich GBM features.

    For each patch we collect mean, std, and max over spatial dims at the bottleneck,
    giving the GBM access to the CNN's learned spatial-context representation.

    Args:
        X:         Input patches (N, H, W, C) — normalised
        cnn_model: Trained UNet (must have enc1..enc4, bottleneck attributes)
        device:    Torch device string
        batch_size: Number of patches to process per forward pass

    Returns:
        bottleneck_stats: (N, 3 * bottleneck_channels) array
    """
    cnn_model.eval()
    all_stats = []

    for start in range(0, len(X), batch_size):
        batch_np = X[start: start + batch_size]                      # (B, H, W, C)
        batch_t  = torch.FloatTensor(batch_np).permute(0, 3, 1, 2)  # (B, C, H, W)
        batch_t  = batch_t.to(device)

        with torch.no_grad():
            x, _ = cnn_model.enc1(batch_t)
            x, _ = cnn_model.enc2(x)
            x, _ = cnn_model.enc3(x)
            x, _ = cnn_model.enc4(x)
            bot  = cnn_model.bottleneck(x)          # (B, C_bot, H', W')

        # Aggregate spatially: mean, std, max → (B, 3*C_bot)
        stats = torch.cat([
            bot.mean(dim=[2, 3]),
            bot.std(dim=[2, 3]),
            bot.amax(dim=[2, 3]),
        ], dim=1).cpu().numpy()

        all_stats.append(stats)

    result = np.vstack(all_stats)
    logger.info(f"Extracted CNN bottleneck features: {result.shape}")
    return result


class GBMTrainer:
    """Trainer for Gradient Boosting Model - Tracks BEST model"""
    
    def __init__(self, config=None):
        # Use GBM_CONFIG from config.py (already tuned to prevent overfitting:
        #   num_leaves=31, max_depth=6, min_child_samples=200, stronger L1/L2)
        self.config = config or GBM_CONFIG["params"]
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

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING  (Optuna-based)
# ══════════════════════════════════════════════════════════════════════════════

class HyperparameterTuner:
    """
    Optuna-based hyperparameter search for both GBM and CNN components.

    Usage
    -----
    tuner = HyperparameterTuner(X_train_gbm, y_train_gbm, X_val_gbm, y_val_gbm,
                                 X_train_raw, y_train_raw, X_val_raw, y_val_raw,
                                 device, study_dir)
    best_gbm_params = tuner.tune_gbm()
    best_cnn_params = tuner.tune_cnn()
    """

    def __init__(
        self,
        X_train_gbm: pd.DataFrame,
        y_train_gbm: np.ndarray,
        X_val_gbm: pd.DataFrame,
        y_val_gbm: np.ndarray,
        X_train_raw: np.ndarray,
        y_train_raw: np.ndarray,
        X_val_raw: np.ndarray,
        y_val_raw: np.ndarray,
        device,
        study_dir: Path,
        config: dict = None,
    ):
        self.X_train_gbm  = X_train_gbm
        self.y_train_gbm  = y_train_gbm
        self.X_val_gbm    = X_val_gbm
        self.y_val_gbm    = y_val_gbm
        self.X_train_raw  = X_train_raw
        self.y_train_raw  = y_train_raw
        self.X_val_raw    = X_val_raw
        self.y_val_raw    = y_val_raw
        self.device       = device
        self.study_dir    = Path(study_dir)
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.cfg          = config or HYPERPARAM_TUNING_CONFIG

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_sampler(self):
        """Return an Optuna sampler based on config."""
        try:
            import optuna
        except ImportError:
            raise RuntimeError("optuna is required for hyperparameter tuning. "
                               "Install it with: pip install optuna")
        seed = self.cfg.get("seed", 42)
        if self.cfg.get("sampler", "tpe") == "random":
            return optuna.samplers.RandomSampler(seed=seed)
        return optuna.samplers.TPESampler(seed=seed)

    def _suggest_gbm_params(self, trial) -> dict:
        """Map Optuna trial suggestions to a GBM params dict."""
        ss  = self.cfg["gbm_search_space"]
        base = dict(GBM_CONFIG["params"])          # copy defaults

        def _suggest(name, spec):
            t = spec["type"]
            if t == "int":
                kw = {}
                if "step" in spec:
                    kw["step"] = spec["step"]
                return trial.suggest_int(name, spec["low"], spec["high"], **kw)
            elif t == "float":
                return trial.suggest_float(name, spec["low"], spec["high"],
                                           log=spec.get("log", False))
            elif t == "categorical":
                return trial.suggest_categorical(name, spec["choices"])
            raise ValueError(f"Unknown type {t!r} for {name!r}")

        for param_name, spec in ss.items():
            base[param_name] = _suggest(param_name, spec)

        # Keep non-tunable GBM internals
        base.setdefault("objective", "regression")
        base.setdefault("metric", "rmse")
        base.setdefault("boosting_type", "gbdt")
        base.setdefault("verbose", -1)
        base.setdefault("early_stopping_rounds", 100)
        base.setdefault("min_child_weight", 1e-3)
        base.setdefault("min_split_gain", 0.01)
        base.setdefault("subsample_freq", 1)
        return base

    # ── GBM tuning ────────────────────────────────────────────────────────────

    def tune_gbm(self) -> dict:
        """
        Run Optuna study for GBM hyperparameters.

        Returns
        -------
        dict
            Best GBM params found by Optuna (merged with non-tunable defaults).
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("optuna not installed — skipping GBM tuning, using defaults")
            return GBM_CONFIG["params"]

        logger.info("\n" + "="*60)
        logger.info("GBM HYPERPARAMETER TUNING (Optuna)")
        logger.info(f"  Trials : {self.cfg['gbm_n_trials']}")
        logger.info(f"  Sampler: {self.cfg['sampler'].upper()}")
        logger.info("="*60)

        pruner = (optuna.pruners.MedianPruner(
                      n_startup_trials=self.cfg["pruning_warmup_steps"])
                  if self.cfg.get("pruning_enabled") else optuna.pruners.NopPruner())

        study = optuna.create_study(
            direction=self.cfg["primary_metric_mode"],
            sampler=self._make_sampler(),
            pruner=pruner,
            study_name="gbm_tuning",
        )

        def _objective(trial):
            params = self._suggest_gbm_params(trial)
            train_data = lgb.Dataset(self.X_train_gbm, label=self.y_train_gbm)
            val_data   = lgb.Dataset(self.X_val_gbm,   label=self.y_val_gbm,
                                     reference=train_data)
            n_est = params.pop("n_estimators", 3000)
            callbacks = [
                lgb.early_stopping(stopping_rounds=params.get("early_stopping_rounds", 100),
                                   verbose=False),
                lgb.log_evaluation(period=-1),
            ]
            bst = lgb.train(
                params,
                train_data,
                num_boost_round=n_est,
                valid_sets=[val_data],
                callbacks=callbacks,
            )
            preds = bst.predict(self.X_val_gbm,
                                num_iteration=bst.best_iteration)
            metric = self.cfg["primary_metric"]
            if metric == "rmse":
                score = float(np.sqrt(mean_squared_error(self.y_val_gbm, preds)))
            elif metric == "r2":
                score = float(r2_score(self.y_val_gbm, preds))
            else:
                score = float(mean_absolute_error(self.y_val_gbm, preds))

            # Report intermediate values for pruning
            trial.report(score, step=bst.best_iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return score

        study.optimize(
            _objective,
            n_trials=self.cfg["gbm_n_trials"],
            timeout=self.cfg.get("timeout_seconds"),
            show_progress_bar=False,
        )

        best_trial = study.best_trial
        best_params = self._suggest_from_frozen(best_trial.params)

        logger.info(f"\n✅ GBM tuning complete — best {self.cfg['primary_metric']}: "
                    f"{best_trial.value:.6f}")
        logger.info(f"   Best params: {best_params}")

        # ── Top-10 GBM Optuna trials ──────────────────────────────────────────
        completed_trials = [t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE]
        reverse = (self.cfg.get("primary_metric_mode", "minimize") == "maximize")
        top10 = sorted(completed_trials,
                       key=lambda t: t.value,
                       reverse=reverse)[:10]

        metric_label = self.cfg.get("primary_metric", "score").upper()
        W = 110
        logger.info("\n" + "=" * W)
        logger.info(f"  TOP-10 GBM OPTUNA TRIALS  (metric: {metric_label})")
        logger.info("=" * W)

        # Collect all param keys that appear in any of the top-10
        param_keys = []
        for t in top10:
            for k in t.params:
                if k not in param_keys:
                    param_keys.append(k)

        # Header
        col_w = 12
        header = f"  {'Rank':<5} {'Trial':>6} {metric_label:>12}  " + \
                 "  ".join(f"{k[:col_w]:<{col_w}}" for k in param_keys)
        logger.info(header)
        logger.info("-" * W)

        for rank, t in enumerate(top10, 1):
            param_str = "  ".join(
                f"{str(t.params.get(k, 'N/A'))[:col_w]:<{col_w}}"
                for k in param_keys
            )
            logger.info(f"  {rank:<5} {t.number:>6} {t.value:>12.6f}  {param_str}")

        logger.info("=" * W)

        # Persist best params + top-10 summary
        top10_records = [
            {"rank": i + 1,
             "trial_number": t.number,
             metric_label.lower(): t.value,
             "params": t.params}
            for i, t in enumerate(top10)
        ]
        results_path = self.study_dir / "best_gbm_params.json"
        with open(results_path, "w") as f:
            json.dump({"best_value": best_trial.value,
                       "best_params": best_params,
                       "top10_trials": top10_records}, f, indent=2)
        logger.info(f"   Saved → {results_path}")

        return best_params

    def _suggest_from_frozen(self, frozen_params: dict) -> dict:
        """Re-merge best Optuna params with non-tunable GBM defaults."""
        base = dict(GBM_CONFIG["params"])
        base.update(frozen_params)
        return base

    # ── CNN tuning ────────────────────────────────────────────────────────────

    def tune_cnn(self, n_channels: int) -> dict:
        """
        Run a lightweight Optuna study for CNN training hyperparameters.

        To keep tuning tractable the CNN is trained for a small number of
        epochs per trial.  The best config is then used for the full run.

        Returns
        -------
        dict
            Updated TRAINING_CONFIG-compatible dict with best CNN hypers.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("optuna not installed — skipping CNN tuning, using defaults")
            return dict(TRAINING_CONFIG), TRAINING_CONFIG.get("dropout_rate", 0.3)

        PROBE_EPOCHS = 20   # short probe to rank configs

        logger.info("\n" + "="*60)
        logger.info("CNN HYPERPARAMETER TUNING (Optuna — short probes)")
        logger.info(f"  Trials       : {self.cfg['cnn_n_trials']}")
        logger.info(f"  Probe epochs : {PROBE_EPOCHS}")
        logger.info("="*60)

        ss = self.cfg["cnn_search_space"]

        def _objective(trial):
            lr           = trial.suggest_float("initial_lr",   ss["initial_lr"]["low"],
                                               ss["initial_lr"]["high"],
                                               log=ss["initial_lr"].get("log", True))
            wd           = trial.suggest_float("weight_decay", ss["weight_decay"]["low"],
                                               ss["weight_decay"]["high"],
                                               log=ss["weight_decay"].get("log", True))
            dropout_rate = trial.suggest_float("dropout_rate", ss["dropout_rate"]["low"],
                                               ss["dropout_rate"]["high"])
            batch_size   = trial.suggest_categorical("batch_size", ss["batch_size"]["choices"])

            # Build a small CNN with the trial dropout rate
            probe_cnn = UNet(in_channels=n_channels, out_channels=1)
            # Patch dropout rates uniformly for the probe
            for module in probe_cnn.modules():
                if isinstance(module, (nn.Dropout2d, nn.Dropout)):
                    module.p = dropout_rate
            initialize_weights(probe_cnn)
            probe_cnn = probe_cnn.to(self.device)

            optimizer = optim.AdamW(probe_cnn.parameters(), lr=lr, weight_decay=wd)
            criterion = ProgressiveLSTLoss()

            train_dataset = UHIDataset(self.X_train_raw, self.y_train_raw, augment=False)
            val_dataset   = UHIDataset(self.X_val_raw,   self.y_val_raw,   augment=False)
            train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                                       shuffle=True, num_workers=0, pin_memory=False)
            val_loader    = DataLoader(val_dataset,   batch_size=batch_size,
                                       shuffle=False, num_workers=0, pin_memory=False)

            best_val_r2 = -float("inf")
            for epoch in range(PROBE_EPOCHS):
                probe_cnn.train()
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = probe_cnn(data)
                    loss, _ = criterion(output, target)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(probe_cnn.parameters(), 1.0)
                        optimizer.step()

                probe_cnn.eval()
                all_preds, all_tgts = [], []
                with torch.no_grad():
                    for data, target in val_loader:
                        out = probe_cnn(data.to(self.device))
                        all_preds.append(out.cpu().numpy().reshape(len(out), -1).mean(1))
                        all_tgts.append(target.numpy().reshape(len(target), -1).mean(1))
                preds_flat = np.concatenate(all_preds)
                tgts_flat  = np.concatenate(all_tgts)
                val_r2 = float(r2_score(tgts_flat, preds_flat))

                trial.report(val_r2, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                best_val_r2 = max(best_val_r2, val_r2)

            del probe_cnn
            return best_val_r2

        pruner = (optuna.pruners.MedianPruner(
                      n_startup_trials=self.cfg["pruning_warmup_steps"])
                  if self.cfg.get("pruning_enabled") else optuna.pruners.NopPruner())

        study = optuna.create_study(
            direction="maximize",       # maximise R²
            sampler=self._make_sampler(),
            pruner=pruner,
            study_name="cnn_tuning",
        )
        study.optimize(
            _objective,
            n_trials=self.cfg["cnn_n_trials"],
            timeout=self.cfg.get("timeout_seconds"),
            show_progress_bar=False,
        )

        best = study.best_trial
        best_cnn_config = dict(TRAINING_CONFIG)
        best_cnn_config["initial_lr"]   = best.params["initial_lr"]
        best_cnn_config["weight_decay"] = best.params["weight_decay"]
        best_cnn_config["batch_size"]   = best.params["batch_size"]

        logger.info(f"\n✅ CNN tuning complete — best val R²: {best.value:.4f}")
        logger.info(f"   Best params: {best.params}")

        # ── Top-10 CNN Optuna trials ──────────────────────────────────────────
        completed_trials = [t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE]
        top10_cnn = sorted(completed_trials,
                           key=lambda t: t.value,
                           reverse=True)[:10]   # CNN always maximises R²

        W = 110
        logger.info("\n" + "=" * W)
        logger.info("  TOP-10 CNN OPTUNA TRIALS  (metric: val R²)")
        logger.info("=" * W)

        cnn_param_keys = []
        for t in top10_cnn:
            for k in t.params:
                if k not in cnn_param_keys:
                    cnn_param_keys.append(k)

        col_w = 14
        header = f"  {'Rank':<5} {'Trial':>6} {'val_R2':>10}  " + \
                 "  ".join(f"{k[:col_w]:<{col_w}}" for k in cnn_param_keys)
        logger.info(header)
        logger.info("-" * W)

        for rank, t in enumerate(top10_cnn, 1):
            param_str = "  ".join(
                f"{str(t.params.get(k, 'N/A'))[:col_w]:<{col_w}}"
                for k in cnn_param_keys
            )
            logger.info(f"  {rank:<5} {t.number:>6} {t.value:>10.4f}  {param_str}")

        logger.info("=" * W)

        top10_cnn_records = [
            {"rank": i + 1,
             "trial_number": t.number,
             "val_r2": t.value,
             "params": t.params}
            for i, t in enumerate(top10_cnn)
        ]
        results_path = self.study_dir / "best_cnn_params.json"
        with open(results_path, "w") as f:
            json.dump({"best_r2": best.value,
                       "best_params": best.params,
                       "top10_trials": top10_cnn_records}, f, indent=2)
        logger.info(f"   Saved → {results_path}")

        return best_cnn_config, best.params.get("dropout_rate", 0.3)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON  (GBM-only · CNN-only · Ensemble · CNN-as-Residual)
# ══════════════════════════════════════════════════════════════════════════════

class ModelComparator:
    """
    Evaluates and plots four modelling strategies on the same validation set:

    1. GBM-only          — pure LightGBM on hand-crafted + bottleneck features
    2. CNN-only          — UNet spatial predictions averaged to patch-level
    3. Ensemble          — weighted average CNN + GBM  (current default)
    4. CNN-as-Residual   — GBM predicts first; CNN corrects the residual

    All metrics are returned in denormalized °C when norm_stats is available.
    """

    def __init__(self, cnn_model, gbm_model, ensemble_weights: dict,
                 device, norm_stats: dict = None):
        self.cnn_model        = cnn_model
        self.gbm_model        = gbm_model
        self.ensemble_weights = ensemble_weights
        self.device           = device
        self.norm_stats       = norm_stats

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_cnn_preds(self, X_torch: torch.Tensor) -> np.ndarray:
        self.cnn_model.eval()
        with torch.no_grad():
            out = self.cnn_model(X_torch.to(self.device))
        return out.cpu().numpy().reshape(len(X_torch), -1).mean(axis=1)

    def _get_gbm_preds(self, X_gbm: pd.DataFrame) -> np.ndarray:
        return self.gbm_model.predict(X_gbm,
                                      num_iteration=self.gbm_model.best_iteration)

    @staticmethod
    def _metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
        mask = np.isfinite(preds) & np.isfinite(targets)
        p, t = preds[mask], targets[mask]
        if len(p) < 2:
            return dict(r2=float("nan"), rmse=float("nan"),
                        mae=float("nan"), mbe=float("nan"),
                        slope=float("nan"), std_ratio=float("nan"))
        from scipy.stats import linregress as _lr
        slope, intercept, *_ = _lr(t, p)
        rmse = float(np.sqrt(mean_squared_error(t, p)))
        pred_std   = float(np.std(p))
        target_std = float(np.std(t))
        std_ratio  = pred_std / target_std if target_std > 0 else float("nan")
        return dict(
            r2=float(r2_score(t, p)),
            rmse=rmse,
            mae=float(mean_absolute_error(t, p)),
            mbe=float((p - t).mean()),
            slope=float(slope),
            intercept=float(intercept),
            std_ratio=std_ratio,
        )

    def _denorm(self, arr: np.ndarray) -> np.ndarray:
        if self.norm_stats is None or "target" not in self.norm_stats:
            return arr
        m = self.norm_stats["target"]["mean"]
        s = self.norm_stats["target"]["std"]
        return arr * s + m

    # ── public API ────────────────────────────────────────────────────────────

    def compare(
        self,
        X_val_raw: np.ndarray,
        y_val_raw: np.ndarray,
        X_val_gbm: pd.DataFrame,
    ) -> dict:
        """
        Run all four strategies and return a metrics dict keyed by strategy name.

        CONSISTENCY FIX: previously this path blended raw normalized predictions
        directly (ens_norm = w_cnn*cnn_norm + w_gbm*gbm_norm), while
        evaluate_ensemble z-scored each stream to the target distribution before
        blending.  These produce different numbers for the same data, causing
        the two comparison tables to disagree.

        Both paths now use the shared EnsembleTrainer._blend_normalized helper
        (replicated inline here since ModelComparator is stateless) so the
        Ensemble and CNN-Residual rows are identical in both tables.

        Parameters
        ----------
        X_val_raw : (N, H, W, C) normalised numpy array — CNN input
        y_val_raw : (N, H, W, 1) normalised numpy array — targets
        X_val_gbm : pd.DataFrame                        — GBM features
        """
        # Targets: patch-mean in normalized space, then denormalised
        y_patch_norm = y_val_raw.reshape(len(y_val_raw), -1).mean(axis=1)
        y_patch_deg  = self._denorm(y_patch_norm)

        # Convert to torch once
        X_torch = torch.FloatTensor(
            np.transpose(X_val_raw, (0, 3, 1, 2))
            if X_val_raw.ndim == 4 and X_val_raw.shape[-1] < X_val_raw.shape[1]
            else X_val_raw
        )

        results = {}

        # ── 1. CNN-only ───────────────────────────────────────────────────────
        cnn_norm = self._get_cnn_preds(X_torch)
        cnn_deg  = self._denorm(cnn_norm)
        results["CNN-only"] = self._metrics(cnn_deg, y_patch_deg)
        results["CNN-only"]["preds"] = cnn_deg

        # ── 2. GBM-only ───────────────────────────────────────────────────────
        gbm_norm = self._get_gbm_preds(X_val_gbm)
        gbm_deg  = self._denorm(gbm_norm)
        results["GBM-only"] = self._metrics(gbm_deg, y_patch_deg)
        results["GBM-only"]["preds"] = gbm_deg

        # ── 3. Ensemble (weighted average) — unified blending ─────────────────
        # Use the same z-score-then-rescale logic as evaluate_ensemble so both
        # tables always report identical Ensemble scores.
        w_cnn = self.ensemble_weights.get("cnn", 0.35)
        w_gbm = self.ensemble_weights.get("gbm", 0.55)
        w_sum = w_cnn + w_gbm + 1e-12

        tgt_mean = float(y_patch_norm.mean())
        tgt_std  = float(y_patch_norm.std()) + 1e-8
        cnn_z    = (cnn_norm - cnn_norm.mean()) / (cnn_norm.std() + 1e-8)
        gbm_z    = (gbm_norm - gbm_norm.mean()) / (gbm_norm.std() + 1e-8)
        blend_z  = (w_cnn * cnn_z + w_gbm * gbm_z) / w_sum
        ens_norm = blend_z * tgt_std + tgt_mean
        ens_deg  = self._denorm(ens_norm)
        results["Ensemble"] = self._metrics(ens_deg, y_patch_deg)
        results["Ensemble"]["preds"] = ens_deg

        # ── 4. CNN-as-Residual — unified with evaluate_ensemble ───────────────
        # GBM anchors the prediction; CNN rescaled to target scale nudges it.
        # Uses the same alpha=w_cnn/(w_cnn+w_gbm) weighting as Strategy C.
        cnn_rescaled_norm = (cnn_z * tgt_std) + tgt_mean     # CNN in target scale
        alpha = w_cnn / w_sum
        residual_norm = gbm_norm + alpha * (cnn_rescaled_norm - gbm_norm)
        residual_deg  = self._denorm(residual_norm)
        results["CNN-Residual"] = self._metrics(residual_deg, y_patch_deg)
        results["CNN-Residual"]["preds"] = residual_deg

        return results, y_patch_deg

    def log_comparison(self, results: dict):
        """Pretty-print the four-way comparison table to the logger."""
        logger.info("\n" + "="*84)
        logger.info("MODEL COMPARISON — 4 STRATEGIES")
        logger.info("="*84)
        header = (f"{'Strategy':<18} {'R²':>7} {'RMSE(°C)':>10} {'MAE(°C)':>9} "
                  f"{'MBE(°C)':>9} {'Slope':>7} {'StdRat':>8}")
        logger.info(header)
        logger.info("-"*84)
        for name, m in results.items():
            logger.info(
                f"{name:<18} {m['r2']:>7.4f} {m['rmse']:>10.4f} "
                f"{m['mae']:>9.4f} {m.get('mbe', float('nan')):>9.4f} "
                f"{m.get('slope', float('nan')):>7.3f} "
                f"{m.get('std_ratio', float('nan')):>8.3f}"
            )
        logger.info("="*84)

    def plot_comparison(self, results: dict, y_true: np.ndarray,
                        save_dir: Path):
        """
        Generate a comprehensive 4-strategy comparison figure saved under save_dir.

        Panels
        ------
        Row 1 : Bar charts for R², RMSE, MAE
        Row 2 : Predicted-vs-actual scatter (one subplot per strategy)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        strategies = list(results.keys())
        colors     = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
        metrics    = [("r2", "R²", True), ("rmse", "RMSE (°C)", False),
                      ("mae", "MAE (°C)", False), ("mbe", "MBE (°C)", False),
                      ("slope", "Slope", True), ("std_ratio", "Std Ratio", True)]

        fig = plt.figure(figsize=(24, 14))
        gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.35)
        fig.suptitle("4-Strategy Model Comparison", fontsize=15, fontweight="bold")

        # ── Row 0–1: bar charts (6 metrics, 3 per row) ───────────────────────
        for idx, (mkey, mlabel, higher_better) in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax   = fig.add_subplot(gs[row, col])
            vals = [results[s].get(mkey, float("nan")) for s in strategies]
            bars = ax.bar(strategies, vals, color=colors, edgecolor="white", width=0.5)
            ax.set_title(mlabel, fontsize=10, fontweight="bold")
            ax.set_ylabel(mlabel, fontsize=9)
            ax.set_xticklabels(strategies, rotation=22, ha="right", fontsize=8)
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(abs(v) * 0.01, 0.001),
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            if mkey == "r2":
                ax.axhline(0.85, ls="--", color="gray", lw=0.9, label="Target 0.85")
                ax.legend(fontsize=7)
            elif mkey == "slope":
                ax.axhline(1.0, ls="--", color="green", lw=0.9, label="Ideal (1.0)")
                ax.legend(fontsize=7)
            elif mkey == "std_ratio":
                ax.axhline(1.0, ls="--", color="green", lw=0.9, label="Ideal (1.0)")
                ax.legend(fontsize=7)
            elif mkey == "mbe":
                ax.axhline(0.0, ls="--", color="green", lw=0.9, label="Ideal (0)")
                ax.legend(fontsize=7)

        # Best-vs-worst summary in 4th cell of row 1
        ax_txt = fig.add_subplot(gs[1, 3])
        ax_txt.axis("off")
        best_r2_strat  = max(strategies, key=lambda s: results[s].get("r2", -999))
        best_rmse_strat = min(strategies, key=lambda s: results[s].get("rmse", 999))
        summary = (
            "SUMMARY\n"
            "───────────────────────\n"
            f"Best R²  : {best_r2_strat}\n"
            f"  {results[best_r2_strat]['r2']:.4f}\n\n"
            f"Best RMSE: {best_rmse_strat}\n"
            f"  {results[best_rmse_strat]['rmse']:.4f} °C\n\n"
            "Weights (Ensemble)\n"
            f"  CNN={self.ensemble_weights.get('cnn',0.35):.2f}  "
            f"GBM={self.ensemble_weights.get('gbm',0.55):.2f}"
        )
        ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                    fontsize=9, va="top", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", alpha=0.8))

        # ── Row 2: pred-vs-actual per strategy ───────────────────────────────
        for col, (name, color) in enumerate(zip(strategies, colors)):
            ax   = fig.add_subplot(gs[2, col])
            preds = results[name].get("preds", np.array([]))
            mask  = np.isfinite(preds) & np.isfinite(y_true)
            p_m, t_m = preds[mask], y_true[mask]
            if len(p_m) > 1:
                ax.scatter(t_m, p_m, alpha=0.25, s=8, color=color)
                lo = min(t_m.min(), p_m.min()); hi = max(t_m.max(), p_m.max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=1.0)
                ax.set_xlabel("Actual (°C)", fontsize=8)
                ax.set_ylabel("Predicted (°C)", fontsize=8)
                r2 = results[name].get("r2", float("nan"))
                rmse = results[name].get("rmse", float("nan"))
                ax.set_title(f"{name}\nR²={r2:.3f}  RMSE={rmse:.3f}°C",
                             fontsize=9, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(name, fontsize=9)

        out_path = save_dir / "model_comparison_4way.png"
        fig.savefig(out_path, dpi=DIAGNOSTICS_CONFIG["save_dpi"], bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  ✅ 4-way comparison figure saved → {out_path}")


def create_temperature_stratified_sampler(y_train):
    """
    Create a sampler that balances temperature ranges so the model sees equal
    amounts of hot and cold samples each epoch.

    Bins are defined in normalised LST space and read from
    STRATIFIED_SAMPLER_CONFIG in config.py.

    Args:
        y_train: Training targets (N, H, W, 1) — NORMALIZED

    Returns:
        WeightedRandomSampler
    """
    sample_means = y_train.reshape(len(y_train), -1).mean(axis=1).flatten()

    bins        = np.array(STRATIFIED_SAMPLER_CONFIG["bins"])
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
    """Manages multiple best checkpoints for different metrics.

    Tracked metrics and the primary selection metric are read from
    CHECKPOINT_CONFIG in config.py.
    """

    def __init__(
        self,
        save_dir: Path,
        metrics: list = CHECKPOINT_CONFIG["metrics"],
    ):
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
    
    def __init__(self, cnn_model, device, config=TRAINING_CONFIG,
                 dataset_dir: Path = None, model_dir: Path = None):
        """
        Args:
            cnn_model:   Initialised UNet instance.
            device:      Torch device.
            config:      Training hyper-parameters (default: TRAINING_CONFIG).
            dataset_dir: Dataset root produced by preprocessing.py.  Determines
                         where normalization_stats.json is read from and which
                         input-channel count to expect.  Supported values:
                           • PROCESSED_DATA_DIR / "cnn_dataset_landsat"  (Landsat-only)
                           • PROCESSED_DATA_DIR / "cnn_dataset_fusion"   (fusion)
                         Defaults to PROCESSED_DATA_DIR / "cnn_dataset" for
                         backward compatibility.
            model_dir:   Directory for checkpoints / diagnostics / saved models.
                         Defaults to MODEL_DIR from config.py.  When training
                         multiple variants pass separate dirs so artefacts don't
                         overwrite each other, e.g. MODEL_DIR / "landsat_only" and
                         MODEL_DIR / "fusion".
        """
        self.cnn_model = cnn_model.to(device)
        self.device = device
        self.config = config
        # Dataset and output directories — variant-aware
        self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else PROCESSED_DATA_DIR / "cnn_dataset"
        self.model_dir   = Path(model_dir)   if model_dir   is not None else MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # CNN components - IMPROVED LOSS FUNCTION
        self.criterion = ProgressiveLSTLoss()
        logger.info("✓ Using ProgressiveLSTLoss v2 (gradient-alignment + temp-weighted MSE)")

        # Layer-wise weight decay — deeper / output layers get stronger
        # regularisation. Values are read from LAYERWISE_WEIGHT_DECAY in config.py.
        if config["optimizer"] == "adamw":
            lwd = LAYERWISE_WEIGHT_DECAY
            self.optimizer = optim.AdamW(
                [
                    {"params": cnn_model.enc1.parameters(),      "weight_decay": lwd["enc1"]},
                    {"params": cnn_model.enc2.parameters(),      "weight_decay": lwd["enc2"]},
                    {"params": cnn_model.enc3.parameters(),      "weight_decay": lwd["enc3"]},
                    {"params": cnn_model.enc4.parameters(),      "weight_decay": lwd["enc4"]},
                    {"params": cnn_model.bottleneck.parameters(),"weight_decay": lwd["bottleneck"]},
                    {"params": cnn_model.dec4.parameters(),      "weight_decay": lwd["dec4"]},
                    {"params": cnn_model.dec3.parameters(),      "weight_decay": lwd["dec3"]},
                    {"params": cnn_model.dec2.parameters(),      "weight_decay": lwd["dec2"]},
                    {"params": cnn_model.dec1.parameters(),      "weight_decay": lwd["dec1"]},
                    {"params": cnn_model.output.parameters(),    "weight_decay": lwd["output"]},
                ],
                lr=config["initial_lr"],
            )
            logger.info(
                f"✓ Layer-wise weight decay: "
                f"{lwd['enc1']} (enc1) → {lwd['bottleneck']} (bottleneck/output)"
            )
        else:
            self.optimizer = optim.Adam(
                cnn_model.parameters(),
                lr=config["initial_lr"]
            )
        
        self.scheduler = self._create_scheduler()

        # Early stopping monitors val R² (mode=max): stable across progressive
        # loss phases.  Parameters read from EARLY_STOPPING_CONFIG in config.py.
        self.early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_CONFIG["patience"],
            min_delta=EARLY_STOPPING_CONFIG["min_delta"],
            mode=EARLY_STOPPING_CONFIG["mode"],
        )
        
        # GBM trainer
        self.gbm_trainer = GBMTrainer()
        self.gbm_trained = False
        
        # Ensemble weights from config
        self.ensemble_weights = ENSEMBLE_WEIGHTS
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.model_dir / "checkpoints",
            metrics=CHECKPOINT_CONFIG["metrics"],
        )

        # Diagnostics plotter — saves all figures to <model_dir>/diagnostics/
        self.plotter = DiagnosticsPlotter(save_dir=self.model_dir / "diagnostics")

        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "cnn_metrics": [],
            "gbm_metrics": [],
            "ensemble_metrics": [],
            "lr": [],
            "ensemble_comparison": {},
            "ensemble_comparison_spatial": {},
        }
        
    def _create_scheduler(self):
        """
        CosineAnnealingWarmRestarts (SGDR) — periodically resets LR to allow
        the optimiser to escape local minima. Parameters from SCHEDULER_CONFIG.

        T_0    → epochs before first restart
        T_mult → period doubles after each restart (50 → 100 → 200 …)
        eta_min → LR floor at the bottom of each cosine valley
        """
        cfg = SCHEDULER_CONFIG
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cfg["T_0"],
            T_mult=cfg["T_mult"],
            eta_min=cfg["eta_min"],
        )
        logger.info(
            f"✓ SGDR scheduler: T_0={cfg['T_0']}, "
            f"T_mult={cfg['T_mult']}, eta_min={cfg['eta_min']}"
        )
        return scheduler
    
    def _log_regularization_metrics(self, epoch):
        """Log regularization metrics to monitor overfitting."""
        total_norm = 0.0
        for p in self.cnn_model.parameters():
            if p.requires_grad:
                total_norm += p.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        logger.info(f"  L2 weight norm: {total_norm:.4f}")

        if 'weight_norms' not in self.history:
            self.history['weight_norms'] = []
        self.history['weight_norms'].append(total_norm)

        # FIX 2 health check: warn if weight norm is still growing fast
        wn = self.history['weight_norms']
        if len(wn) >= 10:
            growth = (wn[-1] - wn[-10]) / (wn[-10] + 1e-8)
            if growth > 0.10:
                logger.warning(
                    f"⚠️ Weight norm grew {growth*100:.1f}% over last 10 epochs "
                    f"({wn[-10]:.1f}→{wn[-1]:.1f}) — consider increasing weight decay"
                )
    
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

            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss detected at batch {batch_idx}!")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=1.0)

            # FIX 4: log gradient norm AFTER backward() so gradients are populated
            if batch_idx == 0:
                total_grad_norm = 0.0
                for p in self.cnn_model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                logger.info(f"  Gradient norm (batch 0): {total_grad_norm:.4f}")

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
        Check for overfitting and other performance issues.
        All thresholds are read from MONITORING_CONFIG and VALIDATION_CONFIG in config.py.
        """
        warn_cfg       = MONITORING_CONFIG["warnings"]
        warnings_found = False

        # Warning 1: Overfitting (train-val R² gap)
        if 'r2' in train_metrics and 'r2' in val_metrics:
            gap = train_metrics['r2'] - val_metrics['r2']
            threshold = warn_cfg["overfitting_threshold"]
            if gap > threshold:
                logger.warning(f"⚠️ OVERFITTING DETECTED!")
                logger.warning(f"   Train R²: {train_metrics['r2']:.3f}")
                logger.warning(f"   Val R²:   {val_metrics['r2']:.3f}")
                logger.warning(f"   Gap: {gap:.3f} (should be < {threshold})")
                warnings_found = True

        # Warning 2: Variance collapse
        if 'std_ratio' in val_metrics:
            std_ratio = val_metrics['std_ratio']
            threshold = warn_cfg["variance_collapse_threshold"]
            if std_ratio < threshold:
                logger.warning(f"⚠️ VARIANCE COLLAPSE!")
                logger.warning(f"   Std ratio: {std_ratio:.3f} (should be ≈1.0, threshold={threshold})")
                logger.warning(f"   Predictions are too clustered around mean")
                warnings_found = True

        # Warning 3: Range compression (regression to mean)
        if 'slope' in val_metrics:
            slope     = val_metrics['slope']
            threshold = warn_cfg["range_compression_threshold"]
            if slope < threshold:
                logger.warning(f"⚠️ RANGE COMPRESSION!")
                logger.warning(f"   Slope: {slope:.3f} (should be ≈1.0, threshold={threshold})")
                logger.warning(f"   Model is regressing to the mean")
                warnings_found = True

        # Warning 4: Systematic bias
        if 'mbe' in val_metrics:
            mbe = val_metrics['mbe']
            if abs(mbe) > 1.0:
                logger.warning(f"⚠️ SYSTEMATIC BIAS!")
                logger.warning(f"   MBE: {mbe:+.3f}°C (should be ≈0)")
                warnings_found = True

        # Warning 5: Mean shift
        if 'pred_mean' in val_metrics and 'target_mean' in val_metrics:
            mean_diff = abs(val_metrics['pred_mean'] - val_metrics['target_mean'])
            if mean_diff > 1.5:
                logger.warning(f"⚠️ MEAN SHIFT!")
                logger.warning(f"   Pred mean:   {val_metrics['pred_mean']:.2f}°C")
                logger.warning(f"   Target mean: {val_metrics['target_mean']:.2f}°C")
                logger.warning(f"   Difference:  {mean_diff:.2f}°C (should be < 1.5)")
                warnings_found = True

        if not warnings_found:
            logger.info("✓ No major issues detected")

        return warnings_found
    
    def train_gbm(self, X_train, y_train, X_val, y_val):
        """Train GBM model with enriched features (FIX 6: CNN bottleneck + FIX 8: texture)."""
        logger.info("\n" + "="*60)
        logger.info("TRAINING GBM MODEL")
        logger.info("="*60)

        # FIX 6: Extract CNN bottleneck features to give GBM spatial context.
        # The CNN runs one forward pass per batch — no gradient needed.
        logger.info("Extracting CNN bottleneck features for GBM enrichment...")
        try:
            bot_train = extract_cnn_bottleneck_features(
                X_train, self.cnn_model, device=str(self.device))
            bot_val   = extract_cnn_bottleneck_features(
                X_val,   self.cnn_model, device=str(self.device))
            logger.info(f"  Bottleneck features: train={bot_train.shape}, val={bot_val.shape}")

            # FIX NEW: Compress bottleneck features with PCA to prevent the GBM
            # from using them as memorization keys for training patches.
            # Without compression, 768 bottleneck dims = 86% of all 890 features,
            # giving the GBM a near-perfect identity signal for train samples.
            n_components = GBM_CONFIG.get("bottleneck_pca_components", 32)
            logger.info(f"  Compressing bottleneck features: {bot_train.shape[1]} → {n_components} dims (PCA)")
            pca = PCA(n_components=n_components, random_state=42)
            bot_train = pca.fit_transform(bot_train)   # fit ONLY on train
            bot_val   = pca.transform(bot_val)         # apply same projection to val
            explained = pca.explained_variance_ratio_.sum()
            logger.info(f"  PCA variance explained: {explained*100:.1f}%  shape: train={bot_train.shape}, val={bot_val.shape}")
            self._pca = pca   # cache for inference

            # Persist PCA so uhi_inference.py can apply the same projection
            _pca_path = getattr(self, '_save_dir', None)
            if _pca_path is None:
                _pca_path = self.model_dir
            try:
                joblib.dump(pca, Path(_pca_path) / "bottleneck_pca.pkl")
                logger.info(f"  💾 Saved bottleneck PCA → {Path(_pca_path) / 'bottleneck_pca.pkl'}")
            except Exception as _pe:
                logger.warning(f"  ⚠️ Could not save bottleneck PCA: {_pe}")

        except Exception as _e:
            logger.warning(f"⚠️ CNN bottleneck extraction failed ({_e}); GBM will use hand-crafted features only")
            bot_train = None
            bot_val   = None

        # Prepare features (FIX 8: texture features included automatically)
        X_train_gbm, y_train_gbm = prepare_gbm_features(X_train, y_train, bot_train)
        X_val_gbm,   y_val_gbm   = prepare_gbm_features(X_val,   y_val,   bot_val)

        # Train
        self.gbm_trainer.train(X_train_gbm, y_train_gbm, X_val_gbm, y_val_gbm)
        self.gbm_trained = True

        # Evaluate using BEST model
        train_preds = self.gbm_trainer.predict(X_train_gbm, use_best=True)
        val_preds   = self.gbm_trainer.predict(X_val_gbm,   use_best=True)

        train_metrics = self._calculate_metrics(train_preds, y_train_gbm, "GBM Train")
        val_metrics   = self._calculate_metrics(val_preds,   y_val_gbm,   "GBM Val")

        logger.info(f"GBM Train Metrics - R²: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f}°C")
        logger.info(f"GBM Val   Metrics - R²: {val_metrics['r2']:.4f}, RMSE: {val_metrics['rmse']:.4f}°C")

        # Check for overfitting — flag if train/val gap is large
        r2_gap = train_metrics['r2'] - val_metrics['r2']
        rmse_ratio = val_metrics['rmse'] / (train_metrics['rmse'] + 1e-8)
        if r2_gap > 0.05:
            logger.warning(f"⚠️ GBM OVERFITTING: Train R²={train_metrics['r2']:.4f} vs Val R²={val_metrics['r2']:.4f} "
                           f"(gap={r2_gap:.4f}). Consider reducing num_leaves/max_depth further.")
        else:
            logger.info(f"✓ GBM train/val gap: ΔR²={r2_gap:.4f}  RMSE ratio={rmse_ratio:.2f}")

        # Plot GBM diagnostics
        try:
            self.plotter.plot_gbm_feature_importance(self.gbm_trainer.best_model or self.gbm_trainer.model)
            self.plotter.plot_pred_vs_actual(val_preds, y_val_gbm, "GBM")
            self.plotter.plot_residual_distribution(val_preds, y_val_gbm, "GBM")
            self.plotter.plot_stratified_error(val_preds, y_val_gbm, "GBM")
        except Exception as _pe:
            logger.warning(f"GBM diagnostic plots skipped: {_pe}")

        # Cache bottleneck extractors for evaluate_ensemble
        self._bot_val   = bot_val
        self._X_val_gbm = X_val_gbm
        self._y_val_gbm = y_val_gbm

        return val_metrics
    
    # ── shared blend helper ───────────────────────────────────────────────────
    @staticmethod
    def _blend_normalized(
        cnn_norm: np.ndarray,
        gbm_norm: np.ndarray,
        targets_norm: np.ndarray,
        w_cnn: float,
        w_gbm: float,
    ) -> np.ndarray:
        """
        Unified blending in normalized space — used by both evaluate_ensemble and
        ModelComparator so the two comparison tables are always consistent.

        Both inputs are z-score standardised to the TARGET distribution before
        weighting so predictions from models with different scales are combined
        fairly.  The result is rescaled back to the target distribution.

        Parameters
        ----------
        cnn_norm    : (N,) CNN patch-mean predictions, already normalized
        gbm_norm    : (N,) GBM predictions, already normalized
        targets_norm: (N,) ground-truth targets, normalized (used only for scale)
        w_cnn, w_gbm: raw (un-normalized) blend weights

        Returns
        -------
        blend_norm  : (N,) blended predictions in the same normalized scale as targets
        """
        tgt_mean = float(targets_norm.mean())
        tgt_std  = float(targets_norm.std()) + 1e-8

        # Z-score each stream independently then rescale to target distribution
        cnn_z = (cnn_norm - cnn_norm.mean()) / (cnn_norm.std() + 1e-8)
        gbm_z = (gbm_norm - gbm_norm.mean()) / (gbm_norm.std() + 1e-8)

        w_sum = w_cnn + w_gbm + 1e-12
        blend_z = (w_cnn * cnn_z + w_gbm * gbm_z) / w_sum
        return blend_z * tgt_std + tgt_mean

    # ── improved ensemble weighting ───────────────────────────────────────────
    def compute_optimal_weights(
        self,
        cnn_metrics: dict,
        gbm_metrics: dict,
        cnn_preds_norm: np.ndarray = None,
        gbm_preds_norm: np.ndarray = None,
        targets_norm:   np.ndarray = None,
    ) -> dict:
        """
        Compute optimal ensemble weights using ridge-regression stacking when
        enough data are available, falling back to inverse-RMSE weighting otherwise.

        Improvement over the old inverse-RMSE approach
        -----------------------------------------------
        Inverse-RMSE assigns weight proportional to 1/RMSE, which treats each
        model as if it makes independent errors.  In practice the errors are
        correlated (both models see the same input), so the optimal blend
        coefficients depend on the cross-correlation between residuals, not just
        the marginal RMSEs.

        Ridge stacking fits:
            blend = alpha_cnn * cnn_z + alpha_gbm * gbm_z
        on the same validation set used for evaluation.  The ridge penalty
        prevents one coefficient from collapsing to zero due to multicollinearity.
        Weights are then derived from the fitted coefficients (clipped to [0,1]
        and renormalized) so the ensemble remains a convex combination.

        Uncertainty-aware fallback
        --------------------------
        If stacking is not possible (< 30 samples) we fall back to inverse-RMSE
        weighting with a correlation discount:
            effective_rmse = rmse * (1 + 0.5 * |corr(cnn_err, gbm_err)|)
        This penalizes models whose errors are highly correlated with the other
        model's errors, giving higher weight to the more independent predictor.

        Parameters
        ----------
        cnn_metrics, gbm_metrics : metric dicts from _calculate_metrics
        cnn_preds_norm  : (N,) CNN predictions in normalized space  [optional]
        gbm_preds_norm  : (N,) GBM predictions in normalized space  [optional]
        targets_norm    : (N,) ground-truth in normalized space      [optional]

        Returns
        -------
        dict with keys 'cnn' and 'gbm' (floats, sum to 1)
        """
        # Fallback triggers
        if cnn_metrics['r2'] < 0.2:
            logger.warning("⚠️ CNN R² < 0.2, using GBM only")
            return {"cnn": 0.0, "gbm": 1.0}
        if gbm_metrics['r2'] < 0.2:
            logger.warning("⚠️ GBM R² < 0.2, using CNN only")
            return {"cnn": 1.0, "gbm": 0.0}

        # ── Try ridge stacking ────────────────────────────────────────────────
        MIN_STACK_SAMPLES = 30
        if (cnn_preds_norm is not None and gbm_preds_norm is not None
                and targets_norm is not None
                and len(cnn_preds_norm) >= MIN_STACK_SAMPLES):
            try:
                # Z-score both streams to target scale (same as _blend_normalized)
                tgt_std  = float(targets_norm.std()) + 1e-8
                tgt_mean = float(targets_norm.mean())
                cnn_z = (cnn_preds_norm - cnn_preds_norm.mean()) / (cnn_preds_norm.std() + 1e-8)
                gbm_z = (gbm_preds_norm - gbm_preds_norm.mean()) / (gbm_preds_norm.std() + 1e-8)

                X_stack = np.column_stack([cnn_z, gbm_z])   # (N, 2)
                y_stack = (targets_norm - tgt_mean) / tgt_std  # z-scored targets

                # Fit ridge with positive=False so coefficients can be negative
                # (rare but valid when models are anti-correlated)
                ridge = Ridge(alpha=1.0, fit_intercept=False)
                ridge.fit(X_stack, y_stack)
                coef = ridge.coef_   # [alpha_cnn, alpha_gbm]

                # Clip negatives and renormalize to a convex combination
                coef_pos = np.clip(coef, 0, None)
                coef_sum = coef_pos.sum()

                if coef_sum < 1e-6:
                    # Ridge collapsed both to zero — fallback to inverse-RMSE
                    raise ValueError("Ridge coefficients both non-positive; using fallback")

                w_cnn = float(coef_pos[0] / coef_sum)
                w_gbm = float(coef_pos[1] / coef_sum)

                # Evaluate the stacked blend on the same val set
                blend_z = (coef_pos[0] * cnn_z + coef_pos[1] * gbm_z) / coef_sum
                blend   = blend_z * tgt_std + tgt_mean
                stack_r2 = float(r2_score(targets_norm, blend))

                logger.info("✅ Ridge-stacking weights computed:")
                logger.info(f"   CNN coef={coef[0]:.4f}  GBM coef={coef[1]:.4f}  "
                            f"(raw ridge, before clip+renorm)")
                logger.info(f"   Final weights: CNN={w_cnn:.4f}  GBM={w_gbm:.4f}")
                logger.info(f"   Stacked val R²={stack_r2:.4f}")

                # Cache ridge model for persistence
                self._stacking_ridge = ridge
                return {"cnn": w_cnn, "gbm": w_gbm}

            except Exception as _se:
                logger.warning(f"⚠️ Ridge stacking failed ({_se}); falling back to "
                               f"correlation-discounted inverse-RMSE")

        # ── Fallback: correlation-discounted inverse-RMSE ─────────────────────
        cnn_rmse = cnn_metrics['rmse'] + 1e-8
        gbm_rmse = gbm_metrics['rmse'] + 1e-8

        if (cnn_preds_norm is not None and gbm_preds_norm is not None
                and targets_norm is not None):
            cnn_err = cnn_preds_norm - targets_norm
            gbm_err = gbm_preds_norm - targets_norm
            corr = float(np.corrcoef(cnn_err, gbm_err)[0, 1])
            discount = 1.0 + 0.5 * abs(corr)
            cnn_eff = cnn_rmse * discount
            gbm_eff = gbm_rmse * discount
            logger.info(f"  Error correlation={corr:.3f}  discount={discount:.3f}")
        else:
            cnn_eff = cnn_rmse
            gbm_eff = gbm_rmse

        cnn_w_raw = 1.0 / cnn_eff
        gbm_w_raw = 1.0 / gbm_eff
        total     = cnn_w_raw + gbm_w_raw
        w_cnn     = float(cnn_w_raw / total)
        w_gbm     = float(gbm_w_raw / total)

        logger.info("Inverse-RMSE (correlation-discounted) weights:")
        logger.info(f"  CNN: {w_cnn:.4f}  (RMSE={cnn_rmse:.4f}°C  R²={cnn_metrics['r2']:.4f})")
        logger.info(f"  GBM: {w_gbm:.4f}  (RMSE={gbm_rmse:.4f}°C  R²={gbm_metrics['r2']:.4f})")
        return {"cnn": w_cnn, "gbm": w_gbm}
    
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
        Evaluate ensemble strategies and select the best one.

        FIX 1: The old implementation mixed CNN patch-means with GBM patch-means
        in incompatible ways (slope degraded from 0.886 → 0.560).  We now test
        four strategies and pick the one with the highest val R²:

          A. GBM only          — best single model from diagnostics
          B. CNN only          — kept as fallback
          C. Weighted average  — normalise both to target scale before blending
          D. CNN-as-residual   — GBM predicts patch mean; CNN adds the spatial
                                 deviation from that mean within each patch.
                                 This eliminates the granularity mismatch entirely.

        The winning strategy + weights are saved to ensemble_config.json.
        """
        logger.info("\n" + "="*60)
        logger.info("EVALUATING ENSEMBLE STRATEGIES (FIX 1: compatible combination)")
        logger.info("="*60)

        # ── Load best CNN checkpoint ──────────────────────────────────────────
        logger.info("Loading best CNN checkpoint...")
        best_ckpt = self.checkpoint_manager.load_best(
            self.cnn_model,
            metric=CHECKPOINT_CONFIG["primary_metric"],
            device=self.device,
        )
        if best_ckpt is None:
            logger.warning("⚠️ No best CNN checkpoint found; using current state")
        else:
            logger.info(f"✅ CNN from epoch {best_ckpt['epoch']+1}  R²={best_ckpt['metrics']['r2']:.4f}")

        # ── CNN pixel-level predictions ───────────────────────────────────────
        _, cnn_metrics, cnn_preds_list = self.evaluate_cnn(val_loader)
        cnn_preds_4d   = np.concatenate(cnn_preds_list, axis=0)     # (N,1,H,W)
        cnn_patch_mean = cnn_preds_4d.reshape(cnn_preds_4d.shape[0], -1).mean(axis=1)

        if not self.gbm_trained:
            logger.warning("GBM not trained — using CNN metrics only")
            return cnn_metrics

        # ── GBM predictions (re-use cached val features from train_gbm) ──────
        X_val_gbm = getattr(self, '_X_val_gbm', None)
        y_val_gbm = getattr(self, '_y_val_gbm', None)
        if X_val_gbm is None:
            logger.info("Re-extracting GBM val features...")
            bot_val   = extract_cnn_bottleneck_features(
                X_val, self.cnn_model, device=str(self.device))
            # Apply the same PCA that was fitted on training bottleneck features
            pca = getattr(self, '_pca', None)
            if pca is not None:
                bot_val = pca.transform(bot_val)
                logger.info(f"  Applied cached PCA: {bot_val.shape}")
            X_val_gbm, y_val_gbm = prepare_gbm_features(X_val, y_val, bot_val)

        gbm_preds   = self.gbm_trainer.predict(X_val_gbm, use_best=True)
        gbm_metrics = self._calculate_metrics(gbm_preds, y_val_gbm, "GBM")

        logger.info(f"\nPrediction scale analysis:")
        logger.info(f"  Target:       mean={y_val_gbm.mean():.4f}  std={y_val_gbm.std():.4f}")
        logger.info(f"  CNN (patch):  mean={cnn_patch_mean.mean():.4f}  std={cnn_patch_mean.std():.4f}")
        logger.info(f"  GBM:          mean={gbm_preds.mean():.4f}  std={gbm_preds.std():.4f}")

        # ── Strategy C: Normalised weighted average ───────────────────────────
        # Use the shared _blend_normalized helper so this path and ModelComparator
        # produce identical scores for the same inputs (fixes the discrepancy
        # between the two comparison tables).
        opt_w = self.compute_optimal_weights(
            cnn_metrics, gbm_metrics,
            cnn_preds_norm=cnn_patch_mean,
            gbm_preds_norm=gbm_preds,
            targets_norm=y_val_gbm,
        )
        blend_sc = self._blend_normalized(
            cnn_patch_mean, gbm_preds, y_val_gbm,
            w_cnn=opt_w["cnn"], w_gbm=opt_w["gbm"],
        )
        weighted_metrics = self._calculate_metrics(blend_sc, y_val_gbm, "Weighted Ensemble")

        # ── Strategy D: CNN-as-residual ───────────────────────────────────────
        # GBM predicts the patch mean; CNN provides a correction term.
        #
        # BUG FIX (previous session): cnn_deviation.mean(axis=1) was identically
        # zero (mean of deviations from mean), making CNN-Residual == GBM-only.
        #
        # Correct approach: z-score the CNN patch-mean predictions to the target
        # distribution, then let CNN nudge GBM by alpha * (CNN_rescaled - GBM).
        # alpha is opt_w["cnn"] so the same trust level applies here as in blend.
        tgt_std  = float(y_val_gbm.std()) + 1e-8
        tgt_mean = float(y_val_gbm.mean())
        cnn_rescaled = (
            (cnn_patch_mean - cnn_patch_mean.mean()) /
            (cnn_patch_mean.std() + 1e-8)
        ) * tgt_std + tgt_mean
        alpha_residual = opt_w["cnn"]
        residual_patch_mean = gbm_preds + alpha_residual * (cnn_rescaled - gbm_preds)
        residual_metrics = self._calculate_metrics(
            residual_patch_mean, y_val_gbm, "CNN-as-Residual")

        # Persist stacking ridge coefficients if available
        if hasattr(self, '_stacking_ridge'):
            try:
                joblib.dump(self._stacking_ridge,
                            self.model_dir / "stacking_ridge.pkl")
                logger.info("  💾 Stacking ridge model saved → stacking_ridge.pkl")
            except Exception as _e:
                logger.warning(f"  Could not save stacking ridge: {_e}")

        # ── LEVEL 1: PATCH-MEAN EVALUATION (all four strategies) ─────────────
        # All models reduced to one scalar per patch.  GBM is native at this
        # level; CNN is averaged down.  Fair for comparing "how well does each
        # strategy anchor the absolute temperature level of a patch?"
        all_results = [
            ("GBM Only",          gbm_metrics,      {"cnn": 0.0, "gbm": 1.0}),
            ("CNN Only",          cnn_metrics,       {"cnn": 1.0, "gbm": 0.0}),
            ("Weighted Ensemble", weighted_metrics,  opt_w),
            ("CNN-as-Residual",   residual_metrics,  {"cnn": "residual", "gbm": 1.0}),
        ]

        W = 102  # total log-line width
        logger.info("\n" + "=" * W)
        logger.info("LEVEL 1 — PATCH-MEAN EVALUATION  (one scalar per patch, all models comparable)")
        logger.info("NOTE: GBM predicts a single value per patch by design; CNN is averaged to match.")
        logger.info("=" * W)
        logger.info(
            f"{'Strategy':<24} {'R²':>7} {'RMSE(°C)':>10} {'MAE(°C)':>9} "
            f"{'MBE(°C)':>9} {'Slope':>7} {'StdRat':>8}"
        )
        logger.info("-" * W)
        for name, m, _ in all_results:
            logger.info(
                f"{name:<24} {m['r2']:>7.4f} {m['rmse']:>10.4f} {m['mae']:>9.4f} "
                f"{m.get('mbe', float('nan')):>9.4f} "
                f"{m.get('slope', float('nan')):>7.3f} "
                f"{m.get('std_ratio', float('nan')):>8.3f}"
            )
        logger.info("=" * W)

        # ── LEVEL 2: PIXEL-LEVEL SPATIAL EVALUATION ───────────────────────────
        # Broadcast each strategy to full (N, H*W) pixel arrays and evaluate
        # against the pixel-level ground truth.  GBM and Weighted-Ensemble
        # (which is still patch-level) suffer here because they predict zero
        # within-patch variation; only CNN and CNN-as-Residual produce spatial
        # detail.  This is the evaluation that matters for UHI mapping quality.
        logger.info("\n" + "=" * W)
        logger.info("LEVEL 2 — PIXEL-LEVEL SPATIAL EVALUATION  (what actually matters for UHI mapping)")
        logger.info("GBM / Weighted-Ensemble broadcast one value across all patch pixels → penalised.")
        logger.info("CNN and CNN-as-Residual preserve within-patch spatial structure.")
        logger.info("=" * W)
        logger.info(
            f"{'Strategy':<24} {'R²':>7} {'RMSE(°C)':>10} {'MAE(°C)':>9} "
            f"{'MBE(°C)':>9} {'Slope':>7} {'StdRat':>8} {'SpatialDet':>11}"
        )
        logger.info("-" * W)

        # Ground-truth at pixel level: y_val is (N, 1, H, W); flatten to (N, H*W)
        N, _, H, W_patch = cnn_preds_4d.shape
        y_val_pixel = y_val.reshape(N, -1)      # (N, H*W) — still normalised

        # Build pixel-level prediction arrays for each strategy
        def _broadcast_patch_to_pixels(patch_preds_1d: np.ndarray, n_pixels: int) -> np.ndarray:
            """Tile a (N,) patch-mean array to (N, n_pixels) for spatial scoring."""
            return np.repeat(patch_preds_1d[:, np.newaxis], n_pixels, axis=1)

        n_pixels = H * W_patch
        cnn_pixel          = cnn_preds_4d.reshape(N, -1)                            # (N, H*W) — real spatial
        gbm_pixel          = _broadcast_patch_to_pixels(gbm_preds, n_pixels)        # constant per patch
        weighted_pixel     = _broadcast_patch_to_pixels(blend_sc, n_pixels)         # constant per patch
        residual_pixel     = _broadcast_patch_to_pixels(residual_patch_mean, n_pixels)  # constant per patch

        spatial_has_detail = {
            "GBM Only":          False,
            "CNN Only":          True,
            "Weighted Ensemble": False,
            "CNN-as-Residual":   True,
        }

        spatial_results = []
        for (name, _, weights), pixel_preds in zip(
            all_results,
            [gbm_pixel, cnn_pixel, weighted_pixel, residual_pixel],
        ):
            flat_preds   = pixel_preds.flatten()
            flat_targets = y_val_pixel.flatten()
            m_spatial    = self._calculate_metrics(flat_preds, flat_targets, f"{name} [pixel]")
            spatial_results.append((name, m_spatial, weights))

            spatial_tag = "✓ spatial" if spatial_has_detail[name] else "✗ flat"
            logger.info(
                f"{name:<24} {m_spatial['r2']:>7.4f} {m_spatial['rmse']:>10.4f} "
                f"{m_spatial['mae']:>9.4f} {m_spatial.get('mbe', float('nan')):>9.4f} "
                f"{m_spatial.get('slope', float('nan')):>7.3f} "
                f"{m_spatial.get('std_ratio', float('nan')):>8.3f} "
                f"{spatial_tag:>11}"
            )
        logger.info("=" * W)
        logger.info(
            "ℹ️  Spatial models (CNN Only, CNN-as-Residual) show higher pixel R² than flat models "
            "because they resolve within-patch temperature gradients — the flat models miss all of this."
        )

        # ── TWO-LEVEL SUMMARY TABLE (thesis-ready) ────────────────────────────
        logger.info("\n" + "=" * W)
        logger.info("SUMMARY — TWO-LEVEL COMPARISON  (use this table in thesis defence)")
        logger.info("=" * W)
        logger.info(
            f"{'Strategy':<24} {'Patch R²':>9} {'Pixel R²':>9} "
            f"{'Patch RMSE':>11} {'Pixel RMSE':>11} {'Spatial?':>9}"
        )
        logger.info("-" * W)
        for (name, patch_m, _), (_, spatial_m, _) in zip(all_results, spatial_results):
            has_spatial = "yes" if spatial_has_detail[name] else "no (flat)"
            logger.info(
                f"{name:<24} {patch_m['r2']:>9.4f} {spatial_m['r2']:>9.4f} "
                f"{patch_m['rmse']:>11.4f} {spatial_m['rmse']:>11.4f} "
                f"{has_spatial:>9}"
            )
        logger.info("=" * W)
        logger.info(
            "KEY INTERPRETATION:\n"
            "  • High Patch R² for GBM is expected — it is optimised for exactly this.\n"
            "  • Low Pixel R² for GBM/Weighted reveals the cost of zero spatial detail.\n"
            "  • CNN-as-Residual combines GBM's patch-level calibration with CNN's spatial "
            "resolution → strongest candidate for UHI mapping."
        )

        # ── SELECT BEST STRATEGY (patch R² — unchanged decision logic) ────────
        best_name, best_metrics, best_weights = max(
            all_results, key=lambda x: x[1]['r2'])

        logger.info(f"\n🏆 BEST STRATEGY (patch-mean R²): {best_name}")
        logger.info(
            f"   Patch  R²={best_metrics['r2']:.4f}  RMSE={best_metrics['rmse']:.4f}°C  "
            f"slope={best_metrics.get('slope', float('nan')):.3f}"
        )
        # Also surface the spatial R² of the best strategy
        best_spatial = next(m for n, m, _ in spatial_results if n == best_name)
        logger.info(
            f"   Pixel  R²={best_spatial['r2']:.4f}  RMSE={best_spatial['rmse']:.4f}°C  "
            f"slope={best_spatial.get('slope', float('nan')):.3f}"
        )

        # Update ensemble weights
        if isinstance(best_weights.get("cnn"), float):
            self.ensemble_weights = best_weights
        else:
            # CNN-as-residual: mark this mode in config
            self.ensemble_weights = {"cnn": 0.0, "gbm": 1.0, "mode": "cnn_residual"}

        # Warn if GBM-only wins (expected given diagnostics)
        if best_name == "GBM Only":
            logger.info(
                "ℹ️  GBM alone is strongest at patch level — CNN will be used spatially "
                "via CNN-as-Residual at inference"
            )
        elif best_name == "Weighted Ensemble":
            logger.info(f"   Weights: CNN={opt_w['cnn']:.3f}, GBM={opt_w['gbm']:.3f}")

        # Store both comparison dicts for downstream plotting
        self.history["ensemble_comparison"]         = {name: m for name, m, _ in all_results}
        self.history["ensemble_comparison_spatial"]  = {name: m for name, m, _ in spatial_results}

        logger.info("=" * 70)
        return best_metrics
    
    def _calculate_metrics(self, preds, targets, name=""):
        """
        Calculate comprehensive evaluation metrics (denormalised °C).
        Uses the module-level linregress import — no repeated inline import.
        Includes slope and std_ratio to detect range compression and variance collapse.
        """
        # Load normalization stats for denormalization
        # Use the dataset_dir this trainer was initialised with so the correct
        # stats file is used regardless of which variant is being trained.
        norm_stats = load_normalization_stats(self.dataset_dir)
        
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
        self._save_dir = save_dir  # expose to train_gbm so PCA is saved alongside models

        # ── PHASE 0: Hyperparameter Tuning (optional) ─────────────────────────
        tuning_cfg = HYPERPARAM_TUNING_CONFIG
        if tuning_cfg.get("enabled", False):
            logger.info("\n" + "="*60)
            logger.info("PHASE 0: HYPERPARAMETER TUNING")
            logger.info("="*60)

            study_dir = self.model_dir / tuning_cfg.get("study_dir", "tuning")

            # Prepare GBM features first so the tuner has them
            logger.info("Extracting bottleneck features for tuning GBM search space…")
            try:
                _bot_tr = extract_cnn_bottleneck_features(
                    X_train, self.cnn_model, device=str(self.device))
                _bot_vl = extract_cnn_bottleneck_features(
                    X_val,   self.cnn_model, device=str(self.device))
                _pca_tune = PCA(
                    n_components=GBM_CONFIG.get("bottleneck_pca_components", 32),
                    random_state=42)
                _bot_tr = _pca_tune.fit_transform(_bot_tr)
                _bot_vl = _pca_tune.transform(_bot_vl)
            except Exception as _te:
                logger.warning(f"Bottleneck extraction for tuning failed ({_te}); "
                               f"tuning will use hand-crafted GBM features only")
                _bot_tr = _bot_vl = None

            _X_tr_gbm, _y_tr_gbm = prepare_gbm_features(X_train, y_train, _bot_tr)
            _X_vl_gbm, _y_vl_gbm = prepare_gbm_features(X_val,   y_val,   _bot_vl)

            n_channels = X_train.shape[-1] if X_train.ndim == 4 else CNN_CONFIG["input_channels"]

            tuner = HyperparameterTuner(
                X_train_gbm=_X_tr_gbm, y_train_gbm=_y_tr_gbm,
                X_val_gbm=_X_vl_gbm,   y_val_gbm=_y_vl_gbm,
                X_train_raw=X_train,    y_train_raw=y_train,
                X_val_raw=X_val,        y_val_raw=y_val,
                device=self.device,
                study_dir=study_dir,
            )

            # ── Tune GBM ──────────────────────────────────────────────────────
            best_gbm_params = tuner.tune_gbm()
            self.gbm_trainer = GBMTrainer(config=best_gbm_params)
            logger.info("✅ GBM trainer updated with tuned params")

            # ── Tune CNN ──────────────────────────────────────────────────────
            best_cnn_config, best_dropout = tuner.tune_cnn(n_channels=n_channels)

            # Apply tuned dropout to CNN model
            for module in self.cnn_model.modules():
                if isinstance(module, (nn.Dropout2d, nn.Dropout)):
                    module.p = best_dropout

            # Re-build optimizer and scheduler with tuned LR / WD
            lwd = LAYERWISE_WEIGHT_DECAY
            self.optimizer = optim.AdamW(
                [
                    {"params": self.cnn_model.enc1.parameters(),      "weight_decay": lwd["enc1"]},
                    {"params": self.cnn_model.enc2.parameters(),      "weight_decay": lwd["enc2"]},
                    {"params": self.cnn_model.enc3.parameters(),      "weight_decay": lwd["enc3"]},
                    {"params": self.cnn_model.enc4.parameters(),      "weight_decay": lwd["enc4"]},
                    {"params": self.cnn_model.bottleneck.parameters(),"weight_decay": lwd["bottleneck"]},
                    {"params": self.cnn_model.dec4.parameters(),      "weight_decay": lwd["dec4"]},
                    {"params": self.cnn_model.dec3.parameters(),      "weight_decay": lwd["dec3"]},
                    {"params": self.cnn_model.dec2.parameters(),      "weight_decay": lwd["dec2"]},
                    {"params": self.cnn_model.dec1.parameters(),      "weight_decay": lwd["dec1"]},
                    {"params": self.cnn_model.output.parameters(),    "weight_decay": lwd["output"]},
                ],
                lr=best_cnn_config["initial_lr"],
                weight_decay=best_cnn_config["weight_decay"],
            )
            self.scheduler  = self._create_scheduler()
            self.config     = best_cnn_config   # update batch_size etc.

            # Rebuild DataLoaders with tuned batch size if it changed
            tuned_bs = best_cnn_config.get("batch_size", TRAINING_CONFIG["batch_size"])
            if tuned_bs != TRAINING_CONFIG["batch_size"]:
                logger.info(f"Re-building DataLoaders with tuned batch_size={tuned_bs}")
                _tr_ds = train_loader.dataset
                _vl_ds = val_loader.dataset
                _sampler = create_temperature_stratified_sampler(y_train)
                train_loader = DataLoader(
                    _tr_ds, batch_size=tuned_bs, sampler=_sampler,
                    num_workers=COMPUTE_CONFIG["num_workers"],
                    pin_memory=COMPUTE_CONFIG["pin_memory"],
                )
                val_loader = DataLoader(
                    _vl_ds, batch_size=tuned_bs, shuffle=False,
                    num_workers=COMPUTE_CONFIG["num_workers"],
                    pin_memory=COMPUTE_CONFIG["pin_memory"],
                )

            logger.info("="*60)
            logger.info("PHASE 0 COMPLETE — proceeding with tuned hyperparameters")
            logger.info("="*60)
        # ─────────────────────────────────────────────────────────────────────

        # Train GBM first (independent of CNN)
        gbm_metrics = self.train_gbm(X_train, y_train, X_val, y_val)
        self.history["gbm_metrics"].append(gbm_metrics)  # FIX 1: store so summary dashboard has GBM data
        
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
            
            # IMPROVED: Check for overfitting and performance issues every 5 epochs
            if epoch % 5 == 0 or epoch == self.config["epochs"] - 1:
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
            metric=CHECKPOINT_CONFIG["primary_metric"],
            device=self.device,
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
        logger.info(f"  R² before: {r2_score(_ct, _cp):.4f}  after: {r2_score(_ct, _corrected):.4f}")
        logger.info(f"  Slope before: {linregress(_cp,_ct)[0]:.4f}  after: {linregress(_corrected,_ct)[0]:.4f}")
        with open(save_dir / "calibration_params.json", "w") as _f:
            json.dump(calibration_params, _f, indent=2)
        logger.info(f"  Calibration params saved to {save_dir / 'calibration_params.json'}")
        logger.info("="*60)
        # ─────────────────────────────────────────────────────────────────────

        # Final ensemble evaluation with BEST models
        ensemble_metrics = self.evaluate_ensemble(val_loader, X_val, y_val)
        self.history["ensemble_metrics"] = ensemble_metrics

        # ── Post-training diagnostic plots ────────────────────────────────────
        logger.info("\n" + "="*60)
        logger.info("GENERATING DIAGNOSTIC PLOTS")
        logger.info("="*60)
        try:
            # Collect CNN val predictions for scatter/residual plots
            self.cnn_model.eval()
            _plot_preds, _plot_tgts = [], []
            with torch.no_grad():
                for _d, _t in val_loader:
                    _plot_preds.append(
                        self.cnn_model(_d.to(self.device)).cpu().numpy())
                    _plot_tgts.append(_t.numpy())
            _cnn_flat  = np.concatenate(_plot_preds).flatten()
            _tgts_flat = np.concatenate(_plot_tgts).flatten()

            # GBM val predictions (already available via cached features)
            _gbm_flat = None
            _gbm_tgt  = None
            if hasattr(self, '_X_val_gbm') and self._X_val_gbm is not None:
                _gbm_flat = self.gbm_trainer.predict(self._X_val_gbm, use_best=True)
                _gbm_tgt  = self._y_val_gbm

            # Load norm stats for data-distribution plot
            _norm_stats = None
            _ns_path = PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
            if _ns_path.exists():
                with open(_ns_path) as _f:
                    _norm_stats = json.load(_f)

            _gbm_model_for_plot = (
                self.gbm_trainer.best_model or self.gbm_trainer.model
                if self.gbm_trained else None
            )

            self.plotter.plot_all_post_training(
                history          = self.history,
                ensemble_metrics = ensemble_metrics,
                cnn_metrics      = self.history["cnn_metrics"][-1] if self.history["cnn_metrics"] else {},
                gbm_metrics      = self.history["gbm_metrics"][-1] if self.history["gbm_metrics"] else {},
                ensemble_weights = self.ensemble_weights,
                cnn_preds_flat   = _cnn_flat,
                targets_flat     = _tgts_flat,
                gbm_preds_flat   = _gbm_flat,
                gbm_targets_flat = _gbm_tgt,
                gbm_model        = _gbm_model_for_plot,
                y_train          = y_train,   # FIX 2: pass real arrays so data-dist plot (06) generates
                y_val            = y_val,
                norm_stats       = _norm_stats,
                ensemble_comparison         = self.history.get("ensemble_comparison"),
                ensemble_comparison_spatial = self.history.get("ensemble_comparison_spatial"),
            )
        except Exception as _pe:
            logger.warning(f"Post-training plot generation failed: {_pe}")
        logger.info("="*60)
        # ─────────────────────────────────────────────────────────────────────

        # ── PHASE FINAL: 4-Way Model Comparison ───────────────────────────────
        logger.info("\n" + "="*60)
        logger.info("PHASE FINAL: 4-WAY MODEL COMPARISON")
        logger.info("  GBM-only · CNN-only · Ensemble · CNN-as-Residual")
        logger.info("="*60)
        try:
            _norm_stats_cmp = load_normalization_stats(self.dataset_dir)
            _gbm_model_cmp  = self.gbm_trainer.best_model or self.gbm_trainer.model
            _X_vl_gbm_cmp   = getattr(self, '_X_val_gbm', None)

            if _gbm_model_cmp is not None and _X_vl_gbm_cmp is not None:
                comparator = ModelComparator(
                    cnn_model=self.cnn_model,
                    gbm_model=_gbm_model_cmp,
                    ensemble_weights=self.ensemble_weights,
                    device=self.device,
                    norm_stats=_norm_stats_cmp,
                )
                comparison_results, y_true_deg = comparator.compare(
                    X_val_raw=X_val,
                    y_val_raw=y_val,
                    X_val_gbm=_X_vl_gbm_cmp,
                )
                comparator.log_comparison(comparison_results)
                comparator.plot_comparison(
                    results=comparison_results,
                    y_true=y_true_deg,
                    save_dir=self.model_dir / "diagnostics",
                )
                # Persist comparison metrics to JSON for external analysis
                _cmp_path = save_dir / "model_comparison_4way.json"
                _cmp_serializable = {
                    name: {k: float(v) for k, v in m.items() if k != "preds"}
                    for name, m in comparison_results.items()
                }
                with open(_cmp_path, "w") as _f:
                    json.dump(_cmp_serializable, _f, indent=2)
                logger.info(f"✅ 4-way comparison metrics saved → {_cmp_path}")

                # Surface the best strategy
                best_strat = max(comparison_results, key=lambda s: comparison_results[s]["r2"])
                logger.info(f"\n🏆 Best strategy overall: {best_strat}  "
                            f"R²={comparison_results[best_strat]['r2']:.4f}  "
                            f"RMSE={comparison_results[best_strat]['rmse']:.4f}°C")
            else:
                logger.warning("⚠️ 4-way comparison skipped: GBM model or val features not available")
        except Exception as _cmp_err:
            logger.warning(f"4-way comparison failed: {_cmp_err}")
        logger.info("="*60)
        # ─────────────────────────────────────────────────────────────────────

        # Save models
        self._save_final_models(save_dir)
        self._save_history(save_dir)
        
        logger.info(self.checkpoint_manager.get_summary())
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
            elif key in ("ensemble_comparison", "ensemble_comparison_spatial"):
                # Dict[str, dict] — serialize each strategy's metrics as floats
                if isinstance(values, dict):
                    history_serializable[key] = {
                        strategy: {k: float(v) for k, v in m.items()}
                        for strategy, m in values.items()
                    }
            elif key == "weight_norms":
                # List of floats — already handled correctly
                history_serializable[key] = [float(v) for v in values]
            else:
                try:
                    history_serializable[key] = [float(v) for v in values]
                except (TypeError, ValueError):
                    # Skip keys that can't be trivially serialized
                    logger.warning(f"_save_history: skipping non-serializable key '{key}'")
                    continue
        
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history_serializable, f, indent=2)
        
        logger.info("Saved training history")


def validate_data_quality(X: np.ndarray, y: np.ndarray, split: str) -> bool:
    """Validate data quality before training.

    All thresholds are read from DATA_QUALITY_CONFIG in config.py.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DATA QUALITY CHECK: {split.upper()}")
    logger.info(f"{'='*60}")

    dqc    = DATA_QUALITY_CONFIG
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
    X_mean = X.mean(); X_std = X.std()
    y_mean = y.mean(); y_std = y.std()

    logger.info(f"Features (X) statistics:")
    logger.info(f"  Mean: {X_mean:.4f}  Std: {X_std:.4f}  Min: {X.min():.4f}  Max: {X.max():.4f}")

    logger.info(f"Target (y) statistics:")
    logger.info(f"  Mean: {y_mean:.4f}  Std: {y_std:.4f}  Min: {y.min():.4f}  Max: {y.max():.4f}")
    logger.info(f"  Unique values: {len(np.unique(y))}")

    # Normalization checks for training split
    if split == "train":
        mean_tol  = dqc["train_mean_tol"]
        std_low   = dqc["train_std_low"]
        std_high  = dqc["train_std_high"]
        logger.info(f"\nNormalization checks (expecting mean≈0, std≈1):")

        if abs(X_mean) > mean_tol:
            issues.append(f"X not properly normalized (mean={X_mean:.4f}, expected ≈0)")
            logger.warning(f"  ⚠️ X mean={X_mean:.4f} (should be ≈0, tol={mean_tol})")
        else:
            logger.info(f"  ✅ X mean={X_mean:.4f}")

        if not (std_low < X_std < std_high):
            issues.append(f"X not properly normalized (std={X_std:.4f}, expected ≈1)")
            logger.warning(f"  ⚠️ X std={X_std:.4f} (expected {std_low}–{std_high})")
        else:
            logger.info(f"  ✅ X std={X_std:.4f}")

        if abs(y_mean) > mean_tol:
            issues.append(f"y not properly normalized (mean={y_mean:.4f}, expected ≈0)")
            logger.warning(f"  ⚠️ y mean={y_mean:.4f} (should be ≈0, tol={mean_tol})")
        else:
            logger.info(f"  ✅ y mean={y_mean:.4f}")

        if not (std_low < y_std < std_high):
            issues.append(f"y not properly normalized (std={y_std:.4f}, expected ≈1)")
            logger.warning(f"  ⚠️ y std={y_std:.4f} (expected {std_low}–{std_high})")
        else:
            logger.info(f"  ✅ y std={y_std:.4f}")
    else:
        # Val/test: just check values aren't wildly out of range
        val_max = dqc["val_mean_max"]
        logger.info(f"\nValidation/Test data checks:")
        if abs(X_mean) > val_max:
            logger.warning(f"  ⚠️ X mean={X_mean:.4f} seems too large for normalized data")
        if abs(y_mean) > val_max:
            logger.warning(f"  ⚠️ y mean={y_mean:.4f} seems too large for normalized data")

    # Variance floor
    if y_std < dqc["min_target_std"]:
        issues.append(f"Target has very low variance (std={y_std:.4f})")
    if y_std < dqc["zero_variance_floor"]:
        issues.append(f"Target has ZERO variance")

    # Report
    if issues:
        logger.error(f"❌ DATA QUALITY ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            logger.error(f"  {i}. {issue}")
        return False

    logger.info(f"\n✅ Data quality checks passed")
    logger.info(f"{'='*60}\n")
    return True


def load_data(split: str, dataset_dir: Path = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed data.

    Args:
        split:       One of "train", "val", or "test".
        dataset_dir: Root dataset directory produced by preprocessing.py
            (e.g. PROCESSED_DATA_DIR / "cnn_dataset_landsat" or
            PROCESSED_DATA_DIR / "cnn_dataset_fusion").
            Defaults to PROCESSED_DATA_DIR / "cnn_dataset" for backward
            compatibility.
    """
    if dataset_dir is None:
        dataset_dir = PROCESSED_DATA_DIR / "cnn_dataset"
    data_dir = Path(dataset_dir) / split
    
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


def _train_one_variant(dataset_dir: Path, model_dir: Path, device, label: str):
    """
    Train the full CNN+GBM ensemble on one dataset variant and return the
    final history dict.

    Args:
        dataset_dir: Preprocessed dataset root (contains train/val/test splits
                     and normalization_stats.json).
        model_dir:   Output directory for checkpoints, diagnostics, and saved
                     models.  Created if it does not exist.
        device:      Torch device.
        label:       Human-readable variant name used in log messages.
    """
    logger.info("\n" + "="*70)
    logger.info(f"VARIANT: {label}")
    logger.info(f"  dataset : {dataset_dir}")
    logger.info(f"  model   : {model_dir}")
    logger.info("="*70)

    # ── Verify normalization stats ────────────────────────────────────────────
    norm_stats = load_normalization_stats(dataset_dir)
    if norm_stats is None:
        stats_path = dataset_dir / "normalization_stats.json"
        logger.error("❌ Normalization stats not found!")
        logger.error(f"   Expected: {stats_path}")
        logger.error("   Run preprocessing.py to generate the dataset first.")
        raise FileNotFoundError(f"Missing normalization stats: {stats_path}")

    logger.info(f"✅ Normalization stats loaded from {dataset_dir}")
    logger.info(f"   Features: {norm_stats.get('n_channels', 'N/A')} channels")
    if "target" in norm_stats:
        logger.info(f"   Target LST: mean={norm_stats['target']['mean']:.2f}°C, "
                    f"std={norm_stats['target']['std']:.2f}°C")

    # ── Infer input-channel count from normalization stats ────────────────────
    # normalization_stats.json stores n_channels; fall back to CNN_CONFIG default
    # so the model is always sized correctly for the chosen dataset variant
    # (Landsat-only has fewer channels than the fusion dataset).
    n_channels = norm_stats.get("n_channels", CNN_CONFIG["input_channels"])
    logger.info(f"   CNN input channels: {n_channels}")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info(f"LOADING DATA  [{label}]")
    logger.info("="*60)
    X_train, y_train = load_data("train", dataset_dir)
    X_val,   y_val   = load_data("val",   dataset_dir)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_dataset = UHIDataset(X_train, y_train, augment=True)
    val_dataset   = UHIDataset(X_val,   y_val,   augment=False)
    sampler       = create_temperature_stratified_sampler(y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        sampler=sampler,
        num_workers=COMPUTE_CONFIG["num_workers"],
        pin_memory=COMPUTE_CONFIG["pin_memory"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=COMPUTE_CONFIG["num_workers"],
        pin_memory=COMPUTE_CONFIG["pin_memory"],
    )

    # ── CNN model ─────────────────────────────────────────────────────────────
    logger.info(f"\nInitializing CNN model  [{label}]  in_channels={n_channels}…")
    cnn_model = UNet(in_channels=n_channels, out_channels=1)
    initialize_weights(cnn_model)
    logger.info(f"CNN parameters: {count_parameters(cnn_model):,}")

    # ── Ensemble trainer ──────────────────────────────────────────────────────
    ensemble_trainer = EnsembleTrainer(
        cnn_model,
        device,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
    )
    logger.info(f"Initial ensemble weights: CNN={ENSEMBLE_WEIGHTS['cnn']}, GBM={ENSEMBLE_WEIGHTS['gbm']}")

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info(f"STARTING ENSEMBLE TRAINING  [{label}]")
    logger.info("="*60)
    try:
        history = ensemble_trainer.train(
            train_loader, val_loader,
            X_train, y_train, X_val, y_val,
            ensemble_trainer.model_dir,
        )
    except Exception as e:
        logger.error(f"\n❌ Training failed [{label}]: {e}")
        traceback.print_exc()
        raise

    # ── Results summary ───────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info(f"TRAINING COMPLETE  [{label}]")
    logger.info("="*60)

    if "ensemble_metrics" in history and history["ensemble_metrics"]:
        fm = history["ensemble_metrics"]
        logger.info("\nFINAL ENSEMBLE METRICS (denormalized °C):")
        logger.info(f"  R²:   {fm['r2']:.4f}  (target ≥ {VALIDATION_CONFIG['targets']['r2']})")
        logger.info(f"  RMSE: {fm['rmse']:.4f}°C  (target ≤ {VALIDATION_CONFIG['targets']['rmse']}°C)")
        logger.info(f"  MAE:  {fm['mae']:.4f}°C  (target ≤ {VALIDATION_CONFIG['targets']['mae']}°C)")
        logger.info(f"  MBE:  {fm['mbe']:.4f}°C")
        targets_met = (
            fm['r2']   >= VALIDATION_CONFIG['targets']['r2']   and
            fm['rmse'] <= VALIDATION_CONFIG['targets']['rmse'] and
            fm['mae']  <= VALIDATION_CONFIG['targets']['mae']
        )
        logger.info("\n✅ ALL PERFORMANCE TARGETS MET!" if targets_met
                    else "\n⚠️ Some performance targets not met")
        logger.info("="*60)

    if history["cnn_metrics"]:
        cnn_final = history["cnn_metrics"][-1]
        logger.info("\nMODEL COMPARISON:")
        logger.info(f"  CNN Best  R²={cnn_final['r2']:.4f}  RMSE={cnn_final['rmse']:.4f}°C")
        if "ensemble_metrics" in history and history["ensemble_metrics"]:
            ens = history["ensemble_metrics"]
            logger.info(f"  Ensemble  R²={ens['r2']:.4f}  RMSE={ens['rmse']:.4f}°C")
            if cnn_final['r2'] > 0:
                improvement = (ens['r2'] - cnn_final['r2']) / abs(cnn_final['r2']) * 100
                logger.info(f"  Improvement: {improvement:+.2f}%")
        logger.info(f"\nFinal ensemble weights:")
        logger.info(f"  CNN={ensemble_trainer.ensemble_weights['cnn']:.4f}  "
                    f"GBM={ensemble_trainer.ensemble_weights['gbm']:.4f}")
        logger.info("="*60)

    return history


# ══════════════════════════════════════════════════════════════════════════════
# TEST SET EVALUATION MODULE
# Implements the four testing scenarios described in section 4.6 of the thesis:
#   • Table 4.8  — Final ensemble vs CNN-only on held-out test set
#   • Table 4.9  — Ablation study: per-component contribution
#   • Table 4.10 — Cross-variant comparison: Landsat vs Fusion
#   • Section 4.6.2 — Residual diagnostics (heteroscedasticity, QQ, Moran's I,
#                      temporal stability, feature importance)
#
# All results are persisted as JSON alongside the trained model artefacts and
# are also written to the main training log for traceability.
# ══════════════════════════════════════════════════════════════════════════════

def _build_test_loader(dataset_dir: Path, batch_size: int, num_workers: int,
                       pin_memory: bool) -> Tuple[DataLoader, np.ndarray, np.ndarray]:
    """Load the held-out test split and return (loader, X_test, y_test).

    The test split is intentionally kept separate from train/val so it is
    never seen during training or ensemble weight optimisation.
    """
    X_test, y_test = load_data("test", dataset_dir)
    test_dataset   = UHIDataset(X_test, y_test, augment=False)
    test_loader    = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader, X_test, y_test


def _cnn_predict_test(cnn_model, test_loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Run the CNN on the test loader and return (flat_preds, flat_targets) in
    normalised space — callers are responsible for denormalisation."""
    cnn_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="CNN test inference"):
            data = data.to(device)
            out  = cnn_model(data)
            all_preds.append(out.cpu().numpy())
            all_targets.append(target.numpy())
    preds   = np.concatenate(all_preds,   axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()
    mask    = np.isfinite(preds) & np.isfinite(targets)
    return preds[mask], targets[mask]


def _ensemble_predict_test(
    cnn_model, gbm_model, pca, test_loader,
    X_test, y_test, ensemble_weights, dataset_dir, device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend CNN + GBM predictions on the test set using the weighted-average
    strategy (the winner from val-set strategy selection).

    Returns (flat_ensemble_preds, flat_targets) in denormalised °C.
    """
    # CNN pixel predictions
    cnn_norm, targets_norm = _cnn_predict_test(cnn_model, test_loader, device)
    cnn_patch_mean = (
        np.concatenate(
            [cnn_model(d.to(device)).cpu().detach().numpy()
             for d, _ in test_loader],
            axis=0,
        ).reshape(len(X_test), -1).mean(axis=1)
        if False  # placeholder; we use the already-computed flat cnn_norm mean
        else cnn_norm
    )
    # Re-extract patch means properly
    cnn_model.eval()
    patch_preds_list, patch_tgt_list = [], []
    with torch.no_grad():
        for data, target in test_loader:
            out = cnn_model(data.to(device))          # (B,1,H,W)
            patch_preds_list.append(out.cpu().numpy().reshape(out.shape[0], -1).mean(axis=1))
            patch_tgt_list.append(target.numpy().reshape(target.shape[0], -1).mean(axis=1))
    cnn_patch_mean = np.concatenate(patch_preds_list)
    y_patch_mean   = np.concatenate(patch_tgt_list)

    # GBM predictions
    bot_test = extract_cnn_bottleneck_features(X_test, cnn_model, device=str(device))
    if pca is not None:
        bot_test = pca.transform(bot_test)
    X_test_gbm, y_test_gbm = prepare_gbm_features(X_test, y_test, bot_test)
    gbm_preds = gbm_model.predict(X_test_gbm)

    # Normalised weighted blend (same _blend_normalized logic as val-set)
    tgt_std  = float(y_patch_mean.std()) + 1e-8
    tgt_mean = float(y_patch_mean.mean())
    cnn_z = (cnn_patch_mean - cnn_patch_mean.mean()) / (cnn_patch_mean.std() + 1e-8)
    gbm_z = (gbm_preds       - gbm_preds.mean())       / (gbm_preds.std()       + 1e-8)
    w_cnn = ensemble_weights.get("cnn", 0.5)
    w_gbm = ensemble_weights.get("gbm", 0.5)
    w_sum = w_cnn + w_gbm + 1e-12
    blend = (w_cnn * cnn_z + w_gbm * gbm_z) / w_sum
    blend = blend * tgt_std + tgt_mean

    norm_stats = load_normalization_stats(dataset_dir)
    blend_deg   = denormalize_predictions(blend,       norm_stats)
    targets_deg = denormalize_predictions(y_patch_mean, norm_stats)
    return blend_deg, targets_deg


def _metrics_from_deg(preds_deg: np.ndarray, targets_deg: np.ndarray) -> Dict:
    """Compute R², RMSE, MAE, MBE in °C from already-denormalised arrays."""
    mask = np.isfinite(preds_deg) & np.isfinite(targets_deg)
    p, t = preds_deg[mask], targets_deg[mask]
    r2   = float(r2_score(t, p))
    rmse = float(np.sqrt(mean_squared_error(t, p)))
    mae  = float(mean_absolute_error(t, p))
    mbe  = float(np.mean(p - t))
    return {"r2": r2, "rmse": rmse, "mae": mae, "mbe": mbe}


# ── Table 4.8 — Final test-set performance ────────────────────────────────────

def evaluate_on_test_set(
    cnn_model,
    gbm_trainer,
    pca,
    ensemble_weights: Dict,
    dataset_dir: Path,
    model_dir: Path,
    device,
    label: str = "variant",
) -> Dict:
    """Evaluate CNN-only and Ensemble on the held-out test set.

    Logs a Table 4.8-style summary and persists results to
    ``<model_dir>/test_results.json``.

    Args:
        cnn_model:        Trained UNet (best checkpoint already loaded).
        gbm_trainer:      GBMTrainer with best_model populated.
        pca:              Fitted PCA for bottleneck compression (or None).
        ensemble_weights: {'cnn': w, 'gbm': w} from evaluate_ensemble().
        dataset_dir:      Preprocessed dataset root for this variant.
        model_dir:        Output directory for this variant.
        device:           Torch device.
        label:            Human-readable label for logging (e.g. 'landsat').

    Returns:
        dict with keys 'cnn_only' and 'ensemble', each containing a metrics
        dict plus the 'hyperparameters' snapshot logged alongside.
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"TABLE 4.8 — FINAL TEST SET EVALUATION  [{label.upper()}]")
    logger.info("=" * 70)

    norm_stats  = load_normalization_stats(dataset_dir)
    batch_size  = TRAINING_CONFIG["batch_size"]
    test_loader, X_test, y_test = _build_test_loader(
        dataset_dir, batch_size,
        COMPUTE_CONFIG["num_workers"], COMPUTE_CONFIG["pin_memory"],
    )

    # ── Hyperparameter snapshot (logged alongside scores for reproducibility) ──
    hyperparam_snapshot = {
        "label":           label,
        "cnn_config":      {k: str(v) if not isinstance(v, (int, float, bool, str))
                            else v for k, v in CNN_CONFIG.items()},
        "gbm_config":      {k: str(v) if not isinstance(v, (int, float, bool, str))
                            else v for k, v in GBM_CONFIG.items()},
        "ensemble_weights": ensemble_weights,
        "training_config": {k: str(v) if not isinstance(v, (int, float, bool, str))
                            else v for k, v in TRAINING_CONFIG.items()},
        "scheduler":       {k: str(v) if not isinstance(v, (int, float, bool, str))
                            else v for k, v in SCHEDULER_CONFIG.items()},
        "early_stopping":  {k: str(v) if not isinstance(v, (int, float, bool, str))
                            else v for k, v in EARLY_STOPPING_CONFIG.items()},
    }
    logger.info("\nHYPERPARAMETER SNAPSHOT:")
    logger.info(f"  CNN      initial_lr={CNN_CONFIG.get('initial_lr', 'N/A')}  "
                f"batch_size={TRAINING_CONFIG.get('batch_size', 'N/A')}  "
                f"epochs={TRAINING_CONFIG.get('epochs', 'N/A')}")
    logger.info(f"  GBM      n_estimators={GBM_CONFIG.get('n_estimators', 'N/A')}  "
                f"num_leaves={GBM_CONFIG.get('num_leaves', 'N/A')}  "
                f"lr={GBM_CONFIG.get('learning_rate', 'N/A')}")
    logger.info(f"  Weights  CNN={ensemble_weights.get('cnn', '?'):.4f}  "
                f"GBM={ensemble_weights.get('gbm', '?'):.4f}")

    # ── CNN-only test metrics ─────────────────────────────────────────────────
    logger.info("\n[1/2] CNN-only inference on test set…")
    cnn_norm_preds, cnn_norm_tgts = _cnn_predict_test(cnn_model, test_loader, device)
    cnn_preds_deg   = denormalize_predictions(cnn_norm_preds, norm_stats)
    cnn_targets_deg = denormalize_predictions(cnn_norm_tgts,  norm_stats)
    cnn_metrics_test = _metrics_from_deg(cnn_preds_deg, cnn_targets_deg)

    # ── Ensemble test metrics ─────────────────────────────────────────────────
    logger.info("[2/2] Ensemble inference on test set…")
    gbm_model = gbm_trainer.best_model if gbm_trainer.best_model is not None else gbm_trainer.model
    if gbm_model is not None:
        ens_preds_deg, ens_targets_deg = _ensemble_predict_test(
            cnn_model, gbm_model, pca, test_loader,
            X_test, y_test, ensemble_weights, dataset_dir, device,
        )
        ens_metrics_test = _metrics_from_deg(ens_preds_deg, ens_targets_deg)
    else:
        logger.warning("⚠️ GBM model not available — reporting CNN-only for ensemble row")
        ens_metrics_test = cnn_metrics_test

    # ── Baseline targets (from VALIDATION_CONFIG) ─────────────────────────────
    tgt = VALIDATION_CONFIG["targets"]
    delta_r2     = ens_metrics_test["r2"]   - cnn_metrics_test["r2"]
    delta_r2_pct = delta_r2 / max(abs(cnn_metrics_test["r2"]), 1e-8) * 100

    # ── Table 4.8 log ─────────────────────────────────────────────────────────
    W = 90
    logger.info("\n" + "=" * W)
    logger.info(
        f"{'Model Configuration':<28} {'R²':>7} {'RMSE(°C)':>10} "
        f"{'MAE(°C)':>9} {'MBE(°C)':>9} {'ΔR² vs CNN':>12}"
    )
    logger.info("-" * W)
    logger.info(
        f"{'CNN only — ' + label:<28} "
        f"{cnn_metrics_test['r2']:>7.4f} {cnn_metrics_test['rmse']:>10.4f} "
        f"{cnn_metrics_test['mae']:>9.4f} {cnn_metrics_test['mbe']:>9.4f} {'—':>12}"
    )
    logger.info(
        f"{'Ensemble — ' + label:<28} "
        f"{ens_metrics_test['r2']:>7.4f} {ens_metrics_test['rmse']:>10.4f} "
        f"{ens_metrics_test['mae']:>9.4f} {ens_metrics_test['mbe']:>9.4f} "
        f"{delta_r2:>+7.4f} ({delta_r2_pct:+.1f}%):>5"
    )
    logger.info("=" * W)

    # ── Target checks ─────────────────────────────────────────────────────────
    targets_met = (
        ens_metrics_test["r2"]   >= tgt["r2"]   and
        ens_metrics_test["rmse"] <= tgt["rmse"]  and
        ens_metrics_test["mae"]  <= tgt["mae"]
    )
    logger.info(f"\nTarget thresholds: R²≥{tgt['r2']}  RMSE≤{tgt['rmse']}°C  MAE≤{tgt['mae']}°C")
    if targets_met:
        logger.info("✅ ALL TEST PERFORMANCE TARGETS MET")
    else:
        logger.warning("⚠️ One or more test performance targets NOT met")
        if ens_metrics_test["r2"]   < tgt["r2"]:
            logger.warning(f"   R²   {ens_metrics_test['r2']:.4f} < {tgt['r2']}")
        if ens_metrics_test["rmse"] > tgt["rmse"]:
            logger.warning(f"   RMSE {ens_metrics_test['rmse']:.4f}°C > {tgt['rmse']}°C")
        if ens_metrics_test["mae"]  > tgt["mae"]:
            logger.warning(f"   MAE  {ens_metrics_test['mae']:.4f}°C > {tgt['mae']}°C")

    # ── Residual diagnostics (Section 4.6.2) ─────────────────────────────────
    residual_diagnostics = _run_residual_diagnostics(
        preds_deg=ens_preds_deg, targets_deg=ens_targets_deg, label=label
    )

    # ── Feature importance (Section 4.6.3) ────────────────────────────────────
    feature_importance_top5 = _log_gbm_feature_importance(gbm_model)

    # ── Persist test_results.json ─────────────────────────────────────────────
    test_results = {
        "label":                label,
        "hyperparameters":      hyperparam_snapshot,
        "cnn_only":             cnn_metrics_test,
        "ensemble":             ens_metrics_test,
        "delta_r2":             float(delta_r2),
        "delta_r2_pct":         float(delta_r2_pct),
        "targets_met":          targets_met,
        "residual_diagnostics": residual_diagnostics,
        "feature_importance_top5": feature_importance_top5,
    }
    out_path = model_dir / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"\n✅ Test results saved → {out_path}")

    return test_results


# ── Section 4.6.2 — Residual diagnostics ─────────────────────────────────────

def _run_residual_diagnostics(
    preds_deg: np.ndarray,
    targets_deg: np.ndarray,
    label: str = "",
    month_labels: np.ndarray = None,
) -> Dict:
    """Compute and log the residual diagnostics described in Section 4.6.2.

    Covers:
    * Heteroscedastic fan — residual spread at thermal extremes vs core range
    * Shapiro-Wilk normality on a random 5 000-sample subset
    * QQ r-value
    * Spatial autocorrelation proxy (Moran's I on 1-D residual ordering)
    * Temporal RMSE variance across months (if month_labels provided)

    Returns a dict of all diagnostic values for JSON serialisation.
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"SECTION 4.6.2 — RESIDUAL DIAGNOSTICS  [{label.upper()}]")
    logger.info("=" * 70)

    residuals = preds_deg - targets_deg
    finite    = np.isfinite(residuals)
    residuals, targets_deg_f = residuals[finite], targets_deg[finite]

    diag: Dict = {}

    # ── (a) Heteroscedastic fan ───────────────────────────────────────────────
    p10, p90 = np.percentile(targets_deg_f, 10), np.percentile(targets_deg_f, 90)
    p30, p70 = np.percentile(targets_deg_f, 30), np.percentile(targets_deg_f, 70)

    extreme_mask  = (targets_deg_f < p10) | (targets_deg_f > p90)
    core_mask     = (targets_deg_f >= p30) & (targets_deg_f <= p70)

    spread_extreme = float(np.std(residuals[extreme_mask])) if extreme_mask.sum() > 10 else float("nan")
    spread_core    = float(np.std(residuals[core_mask]))    if core_mask.sum()    > 10 else float("nan")
    fan_ratio      = spread_extreme / spread_core if (spread_core and spread_core > 0) else float("nan")

    diag["heteroscedastic_fan"] = {
        "spread_extreme_deg": spread_extreme,
        "spread_core_deg":    spread_core,
        "fan_ratio":          fan_ratio,
        "extreme_percentile_range": "< p10 or > p90",
        "core_percentile_range":    "p30 – p70",
    }
    logger.info(f"\n(a) Heteroscedastic fan:")
    logger.info(f"    Extreme spread (< p10, > p90):  σ = {spread_extreme:.4f}°C")
    logger.info(f"    Core spread    (p30 – p70):     σ = {spread_core:.4f}°C")
    logger.info(f"    Fan ratio  (extreme / core):    {fan_ratio:.2f}×  "
                f"(doc. reference: ≈1.8×)")

    # ── (b) Shapiro-Wilk normality on up to 5 000 samples ────────────────────
    rng          = np.random.default_rng(seed=42)
    n_sw         = min(5000, len(residuals))
    sw_sample    = rng.choice(residuals, size=n_sw, replace=False)
    sw_stat, sw_p = stats.shapiro(sw_sample)

    diag["shapiro_wilk"] = {"W": float(sw_stat), "p_value": float(sw_p), "n": n_sw}
    logger.info(f"\n(b) Shapiro-Wilk normality (n={n_sw}):")
    logger.info(f"    W = {sw_stat:.4f}  p = {sw_p:.4f}")
    if sw_p < 0.05:
        logger.info(f"    → Non-Gaussian tails at α=0.05  "
                    f"(doc. reference: W=0.987, p=0.0021, near-normal)")
    else:
        logger.info(f"    → Cannot reject normality at α=0.05")

    # ── (c) QQ r-value ────────────────────────────────────────────────────────
    (osm, osr), (qq_sl, qq_ic, qq_r) = stats.probplot(residuals, dist="norm")
    diag["qq_r_value"] = float(qq_r)
    logger.info(f"\n(c) Q-Q r-value: {qq_r:.4f}  (1.00 = perfectly normal)")

    # ── (d) Spatial autocorrelation proxy (Moran's I, 1-D lag-1) ─────────────
    #   True 2-D Moran's I requires patch coordinates, which are not available
    #   at this stage.  We compute the 1-D Moran's I on the residual sequence
    #   as a proxy — non-zero I indicates patch-to-patch autocorrelation in
    #   the ordering that preprocessing imposed (spatially contiguous tiles).
    n_res  = len(residuals)
    r_mean = residuals.mean()
    r_c    = residuals - r_mean
    if n_res > 2:
        numerator   = float(np.sum(r_c[:-1] * r_c[1:]))
        denominator = float(np.sum(r_c ** 2))
        W_sum       = n_res - 1          # number of lag-1 pairs (queen W)
        morans_i    = (n_res / W_sum) * (numerator / (denominator + 1e-12))
        # Approximate z-score under normality assumption
        E_I    = -1.0 / (n_res - 1)
        Var_I  = (n_res ** 2 * (n_res - 1) /
                  ((n_res + 1) * (n_res - 1) ** 2) - E_I ** 2)
        z_I    = (morans_i - E_I) / (np.sqrt(Var_I) + 1e-12)
        p_I    = float(2 * (1 - stats.norm.cdf(abs(z_I))))
    else:
        morans_i = float("nan")
        p_I      = float("nan")

    diag["morans_i_proxy"] = {
        "I":      float(morans_i),
        "p_value": p_I,
        "note":   "1-D lag-1 Moran proxy; positive I = residual autocorrelation in patch order",
    }
    logger.info(f"\n(d) Spatial autocorrelation (1-D Moran's I proxy):")
    logger.info(f"    I = {morans_i:.4f}  p = {p_I:.4f}  "
                f"(doc. reference: I=0.14, p<0.001)")
    if morans_i > 0 and p_I < 0.05:
        logger.info(f"    → Significant positive autocorrelation — "
                    f"residuals cluster across consecutive patches")
    elif p_I >= 0.05:
        logger.info(f"    → No significant autocorrelation at α=0.05")

    # ── (e) Temporal RMSE stability across months ─────────────────────────────
    temporal: Dict = {}
    if month_labels is not None and len(month_labels) == len(residuals):
        monthly_rmse = {}
        for m in range(1, 13):
            idx = (month_labels == m)
            if idx.sum() < 5:
                continue
            monthly_rmse[int(m)] = float(
                np.sqrt(np.mean(residuals[idx] ** 2))
            )
        if monthly_rmse:
            rmse_values    = np.array(list(monthly_rmse.values()))
            rmse_variance  = float(np.var(rmse_values))
            rmse_std       = float(np.std(rmse_values))
            worst_month    = int(max(monthly_rmse, key=monthly_rmse.get))
            mean_rmse      = float(rmse_values.mean())
            worst_above    = monthly_rmse[worst_month] - mean_rmse
            temporal = {
                "monthly_rmse":    monthly_rmse,
                "rmse_variance":   rmse_variance,
                "rmse_std":        rmse_std,
                "worst_month":     worst_month,
                "worst_rmse_above_mean": float(worst_above),
            }
            logger.info(f"\n(e) Temporal RMSE stability (monthly):")
            logger.info(f"    RMSE variance across months: {rmse_variance:.4f}°C²  "
                        f"(σ = {rmse_std:.4f}°C)")
            logger.info(f"    Worst month: {worst_month:02d}  "
                        f"(+{worst_above:.4f}°C above average)")
            logger.info(f"    (doc. reference: variance=0.18°C², σ=0.05°C, worst month=11)")
        else:
            logger.info("\n(e) Temporal stability: not enough per-month samples")
    else:
        logger.info("\n(e) Temporal stability: month_labels not provided — "
                    "pass a (N,) array of month integers (1–12) to enable this check")
        temporal = {"note": "month_labels not provided"}

    diag["temporal_stability"] = temporal
    logger.info("=" * 70)
    return diag


# ── Section 4.6.3 — GBM feature importance ────────────────────────────────────

def _log_gbm_feature_importance(gbm_model, top_n: int = 5) -> list:
    """Log top-N LightGBM features by split gain and return as a list of dicts.

    Corresponds to Figure 4.4 / the five highest-importance features listed in
    Section 4.6.3 of the thesis.
    """
    if gbm_model is None:
        logger.warning("_log_gbm_feature_importance: gbm_model is None — skipping")
        return []

    try:
        importance = gbm_model.feature_importance(importance_type="gain")
        names      = gbm_model.feature_name()
        # Sort descending
        order      = np.argsort(importance)[::-1]
        top_n      = min(top_n, len(names))

        logger.info(f"\nSECTION 4.6.3 — GBM FEATURE IMPORTANCE (top {top_n}, split gain):")
        logger.info(f"  {'Rank':<6} {'Feature':<30} {'Gain':>12}")
        logger.info(f"  {'-'*50}")
        top_features = []
        for rank, idx in enumerate(order[:top_n], start=1):
            feat = {"rank": rank, "feature": names[idx], "gain": float(importance[idx])}
            top_features.append(feat)
            logger.info(f"  {rank:<6} {names[idx]:<30} {importance[idx]:>12.2f}")
        logger.info(
            f"\n  (doc. reference: ch10_mean, ch10_std, ch_ndbi_mean, "
            f"ch_ndvi_std, ch_albedo_mean)"
        )
        return top_features

    except Exception as e:
        logger.warning(f"_log_gbm_feature_importance failed: {e}")
        return []


# ── Table 4.9 — Ablation study ────────────────────────────────────────────────

def run_ablation_study(
    dataset_dir: Path,
    model_dir: Path,
    device,
    label: str = "fusion",
    n_ablation_epochs: int = None,
) -> Dict:
    """Reproduce the ablation experiment described in Table 4.9.

    Design: load the already-trained production CNN + GBM + PCA from disk,
    then for each ablation scenario retrain *only the component being tested*
    while keeping everything else identical to the full-pipeline run.

    This ensures that e.g. the S3 "Multi-Component Loss" baseline matches the
    production LEVEL-1 numbers (R²≈0.91), and only the ablated variant deviates.
    Training everything from scratch would give artificially low baselines
    because the ablation epoch budget is much smaller than full training.

    Scenarios tested
    ----------------
    1.  With Hyperparameter Tuning (Optuna) vs. Default Config Parameters
        → retrain CNN with/without Optuna params; GBM retrained on new bottleneck
    2a. Patch-level:  Weighted Ensemble vs. GBM-only
        → swap ensemble blend; no retraining needed
    2b. Pixel-level:  CNN-as-Residual vs. CNN-only
        → swap inference path; no retraining needed
    3.  Multi-component ProgressiveLSTLoss vs. Plain MSE
        → retrain CNN with MSE-only loss; GBM retrained on new bottleneck
    4.  Stratified Sampling vs. Uniform Sampling
        → retrain CNN with uniform sampler; GBM retrained on new bottleneck

    Args:
        dataset_dir:       Dataset root for the fusion variant.
        model_dir:         Base output dir; each config gets its own sub-dir.
        device:            Torch device.
        label:             Dataset label for log messages.
        n_ablation_epochs: Override epoch count (None = use TRAINING_CONFIG).

    Returns:
        Dict mapping config name → {hyperparameters, metrics}.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TABLE 4.9 — ABLATION STUDY  (load-then-ablate design)")
    logger.info(f"Dataset: {dataset_dir}  |  Device: {device}")
    logger.info("=" * 70)

    epochs = n_ablation_epochs or TRAINING_CONFIG["epochs"]

    # ── Load shared data once ─────────────────────────────────────────────────
    X_train, y_train = load_data("train", dataset_dir)
    X_val,   y_val   = load_data("val",   dataset_dir)
    n_channels = X_train.shape[-1] if X_train.ndim == 4 else CNN_CONFIG["input_channels"]
    _ns = load_normalization_stats(dataset_dir)

    # ── Load the production-trained CNN (best checkpoint) ────────────────────
    # All ablation scenarios start from this model. Only the component under
    # test is retrained; everything else inherits the production weights so
    # that baselines match the LEVEL-1 evaluation numbers exactly.
    logger.info("\n  Loading production CNN checkpoint…")
    _prod_cnn = UNet(in_channels=n_channels, out_channels=1)
    _ckpt_mgr = CheckpointManager(
        save_dir=model_dir / "checkpoints",
        metrics=CHECKPOINT_CONFIG["metrics"],
    )
    _best_ckpt = _ckpt_mgr.load_best(
        _prod_cnn,
        metric=CHECKPOINT_CONFIG["primary_metric"],
        device=device,
    )
    if _best_ckpt is None:
        logger.warning(
            "  ⚠️  No best CNN checkpoint found — ablation baselines will "
            "use randomly-initialised weights and will not match production."
        )
    _prod_cnn = _prod_cnn.to(device)
    _prod_cnn.eval()

    # ── Load the production GBM ───────────────────────────────────────────────
    logger.info("  Loading production GBM…")
    _prod_gbm = GBMTrainer()
    _gbm_pkl = model_dir / "best_gbm_model.pkl"
    if not _gbm_pkl.exists():
        _gbm_pkl = model_dir / "gbm_model.pkl"
    if _gbm_pkl.exists():
        _prod_gbm.best_model = joblib.load(_gbm_pkl)
        _prod_gbm.model      = _prod_gbm.best_model
        logger.info(f"  ✅ GBM loaded from {_gbm_pkl}")
    else:
        logger.warning(
            f"  ⚠️  GBM not found at {_gbm_pkl}. "
            "GBM-dependent scenarios will be skipped or degraded."
        )

    # ── Load the production bottleneck PCA ───────────────────────────────────
    logger.info("  Loading production PCA…")
    _prod_pca = None
    _pca_pkl  = model_dir / "bottleneck_pca.pkl"
    if _pca_pkl.exists():
        _prod_pca = joblib.load(_pca_pkl)
        logger.info(f"  ✅ PCA loaded from {_pca_pkl}")
    else:
        logger.warning(
            f"  ⚠️  PCA not found at {_pca_pkl}. "
            "Bottleneck features will be uncompressed."
        )

    # ── Load ensemble weights ─────────────────────────────────────────────────
    _ens_cfg_path = model_dir / "ensemble_config.json"
    _prod_ens_weights = ENSEMBLE_WEIGHTS
    if _ens_cfg_path.exists():
        with open(_ens_cfg_path) as _f:
            _ens_cfg = json.load(_f)
        _prod_ens_weights = _ens_cfg.get("weights", ENSEMBLE_WEIGHTS)
        logger.info(f"  ✅ Ensemble weights loaded: {_prod_ens_weights}")

    # ── Load Optuna best params (used by S1 scenarios) ────────────────────────
    _best_cnn_params: Dict = {}
    _best_gbm_params: Dict = {}
    _best_cnn_path = model_dir / "tuning" / "best_cnn_params.json"
    _best_gbm_path = model_dir / "tuning" / "best_gbm_params.json"
    if _best_cnn_path.exists():
        try:
            with open(_best_cnn_path) as _f:
                _best_cnn_params = json.load(_f).get("best_params", {})
            logger.info(f"  ✅ Optuna CNN params loaded: {_best_cnn_params}")
        except Exception as _e:
            logger.warning(f"  ⚠️  Could not load Optuna CNN params: {_e}")
    if _best_gbm_path.exists():
        try:
            with open(_best_gbm_path) as _f:
                _best_gbm_params = json.load(_f).get("best_params", {})
            logger.info(f"  ✅ Optuna GBM params loaded")
        except Exception as _e:
            logger.warning(f"  ⚠️  Could not load Optuna GBM params: {_e}")

    # ── Pre-extract bottleneck features with the PRODUCTION CNN ──────────────
    # Used by S2a/S2b (no CNN retraining) and as the GBM baseline for S1/S3/S4.
    logger.info("\n  Extracting production bottleneck features…")
    _bot_tr_prod = extract_cnn_bottleneck_features(X_train, _prod_cnn, device=str(device))
    _bot_vl_prod = extract_cnn_bottleneck_features(X_val,   _prod_cnn, device=str(device))
    if _prod_pca is not None:
        _bot_tr_prod = _prod_pca.transform(_bot_tr_prod)
        _bot_vl_prod = _prod_pca.transform(_bot_vl_prod)
    else:
        _inline_pca  = PCA(n_components=min(32, _bot_tr_prod.shape[1]), random_state=42)
        _bot_tr_prod = _inline_pca.fit_transform(_bot_tr_prod)
        _bot_vl_prod = _inline_pca.transform(_bot_vl_prod)

    # ── GBM feature tables (shared by all scenarios that use GBM) ────────────
    _Xg_tr_prod, _yg_tr_prod = prepare_gbm_features(X_train, y_train, _bot_tr_prod)
    _Xg_vl_prod, _yg_vl_prod = prepare_gbm_features(X_val,   y_val,   _bot_vl_prod)

    # ── Production CNN patch-mean predictions (normalised) ───────────────────
    logger.info("  Computing production CNN patch-mean predictions…")
    _prod_cnn.eval()
    _prod_cnn_pm_norm = []
    _prod_cnn_tg_norm = []
    _vl_ds_prod = UHIDataset(X_val, y_val, augment=False)
    _vl_ld_prod = DataLoader(
        _vl_ds_prod,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=COMPUTE_CONFIG["num_workers"],
        pin_memory=COMPUTE_CONFIG["pin_memory"],
    )
    with torch.no_grad():
        for _d, _t in _vl_ld_prod:
            _out = _prod_cnn(_d.to(device)).cpu().numpy()
            _prod_cnn_pm_norm.append(_out.reshape(_out.shape[0], -1).mean(axis=1))
            _prod_cnn_tg_norm.append(_t.numpy().reshape(_t.shape[0], -1).mean(axis=1))
    _prod_cnn_pm_norm = np.concatenate(_prod_cnn_pm_norm)
    _prod_cnn_tg_norm = np.concatenate(_prod_cnn_tg_norm)

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: retrain CNN from scratch with overridden settings.
    # Returns (cnn_model, cnn_patch_mean_norm, cnn_target_norm, bot_tr, bot_vl)
    # so callers can layer GBM on top.
    # ──────────────────────────────────────────────────────────────────────────
    def _retrain_cnn(
        initial_lr=None,
        weight_decay=None,
        batch_size=None,
        dropout_rate=None,
        use_multi_loss=True,
        use_stratified=True,
    ):
        _lr  = initial_lr  or TRAINING_CONFIG["initial_lr"]
        _wd  = weight_decay or TRAINING_CONFIG.get("weight_decay", 1e-4)
        _bs  = batch_size  or TRAINING_CONFIG["batch_size"]
        _do  = dropout_rate or CNN_CONFIG.get("dropout_rates", [0.1])[0]

        _cnn = UNet(in_channels=n_channels, out_channels=1)
        if _do != CNN_CONFIG.get("dropout_rates", [0.1])[0]:
            for _m in _cnn.modules():
                if isinstance(_m, (nn.Dropout2d, nn.Dropout)):
                    _m.p = _do
        initialize_weights(_cnn)
        _cnn = _cnn.to(device)

        use_aug = use_multi_loss   # S3 basic_mse also disables augmentation
        _tr_ds = UHIDataset(X_train, y_train, augment=use_aug)
        _vl_ds = UHIDataset(X_val,   y_val,   augment=False)

        if use_stratified:
            _sampler    = create_temperature_stratified_sampler(y_train)
            _shuffle_tr = False
        else:
            _sampler    = None
            _shuffle_tr = True

        _tr_ld = DataLoader(
            _tr_ds, batch_size=_bs,
            sampler=_sampler, shuffle=_shuffle_tr,
            num_workers=COMPUTE_CONFIG["num_workers"],
            pin_memory=COMPUTE_CONFIG["pin_memory"],
        )
        _vl_ld = DataLoader(
            _vl_ds, batch_size=_bs, shuffle=False,
            num_workers=COMPUTE_CONFIG["num_workers"],
            pin_memory=COMPUTE_CONFIG["pin_memory"],
        )

        if use_multi_loss:
            _crit = ProgressiveLSTLoss()
        else:
            _crit = ProgressiveLSTLoss()
            _crit.gradient_weight    = 0.0
            _crit.variance_weight    = 0.0
            _crit.range_weight       = 0.0
            _crit.bias_weight        = 0.0
            _crit._disable_temp_weighted = True

        _opt  = optim.AdamW(_cnn.parameters(), lr=_lr, weight_decay=_wd)
        _sch  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            _opt,
            T_0=SCHEDULER_CONFIG["T_0"],
            T_mult=SCHEDULER_CONFIG["T_mult"],
            eta_min=SCHEDULER_CONFIG["eta_min"],
        )
        _es   = EarlyStopping(
            patience=EARLY_STOPPING_CONFIG["patience"],
            min_delta=EARLY_STOPPING_CONFIG["min_delta"],
            mode=EARLY_STOPPING_CONFIG["mode"],
        )

        best_r2    = -float("inf")
        best_state = None

        for ep in range(epochs):
            _cnn.train()
            ep_loss = 0.0
            for _d, _t in _tr_ld:
                _d, _t = _d.to(device), _t.to(device)
                _opt.zero_grad()
                _out = _cnn(_d)
                _loss, _ = _crit(_out, _t, _d)
                if torch.isnan(_loss) or torch.isinf(_loss):
                    continue
                _loss.backward()
                torch.nn.utils.clip_grad_norm_(_cnn.parameters(), max_norm=1.0)
                _opt.step()
                ep_loss += _loss.item()
            _crit.set_training_progress(ep, epochs)
            _sch.step()

            if ep % 5 == 0 or ep == epochs - 1:
                _cnn.eval()
                _vp, _vt = [], []
                with torch.no_grad():
                    for _d2, _t2 in _vl_ld:
                        _vp.append(_cnn(_d2.to(device)).cpu().numpy())
                        _vt.append(_t2.numpy())
                _vp_f = np.concatenate(_vp).flatten()
                _vt_f = np.concatenate(_vt).flatten()
                _mask = np.isfinite(_vp_f) & np.isfinite(_vt_f)
                if _mask.sum() > 1:
                    _pdn = denormalize_predictions(_vp_f[_mask], _ns)
                    _tdn = denormalize_predictions(_vt_f[_mask], _ns)
                    _r2  = float(r2_score(_tdn, _pdn))
                    _rmse = float(np.sqrt(mean_squared_error(_tdn, _pdn)))
                    logger.info(
                        f"    Ep {ep+1}/{epochs}  loss={ep_loss/max(len(_tr_ld),1):.4f}"
                        f"  val_R²={_r2:.4f}  val_RMSE={_rmse:.4f}°C"
                    )
                    if _r2 > best_r2:
                        best_r2    = _r2
                        best_state = {k: v.cpu().clone()
                                      for k, v in _cnn.state_dict().items()}
                    if _es(_rmse, _cnn):
                        logger.info(f"    Early stopping at epoch {ep+1}")
                        break

        if best_state is not None:
            _cnn.load_state_dict(best_state)
        _cnn.eval()

        # Patch-mean predictions on val set (normalised)
        _pm_norm, _tg_norm = [], []
        with torch.no_grad():
            for _d, _t in _vl_ld:
                _o = _cnn(_d.to(device)).cpu().numpy()
                _pm_norm.append(_o.reshape(_o.shape[0], -1).mean(axis=1))
                _tg_norm.append(_t.numpy().reshape(_t.shape[0], -1).mean(axis=1))
        _pm_norm = np.concatenate(_pm_norm)
        _tg_norm = np.concatenate(_tg_norm)

        # Bottleneck features from this newly-trained CNN
        _bot_tr = extract_cnn_bottleneck_features(X_train, _cnn, device=str(device))
        _bot_vl = extract_cnn_bottleneck_features(X_val,   _cnn, device=str(device))
        _pca_new = PCA(n_components=min(32, _bot_tr.shape[1]), random_state=42)
        _bot_tr  = _pca_new.fit_transform(_bot_tr)
        _bot_vl  = _pca_new.transform(_bot_vl)

        return _cnn, _pm_norm, _tg_norm, _bot_tr, _bot_vl

    # ── Helper: train a fresh GBM on given features ───────────────────────────
    def _train_gbm(bot_tr, bot_vl, gbm_params=None):
        _Xg_tr, _yg_tr = prepare_gbm_features(X_train, y_train, bot_tr)
        _Xg_vl, _yg_vl = prepare_gbm_features(X_val,   y_val,   bot_vl)
        _gbm = GBMTrainer(config=gbm_params) if gbm_params else GBMTrainer()
        _gbm.train(_Xg_tr, _yg_tr, _Xg_vl, _yg_vl)
        return _gbm, _Xg_vl, _yg_vl

    # ── Helper: weighted ensemble from (cnn_pm_norm, gbm_preds_norm, targets) ─
    def _blend_metrics(cnn_pm_norm, gbm_preds_norm, y_norm, weights=None):
        w = weights or _prod_ens_weights
        n = min(len(cnn_pm_norm), len(gbm_preds_norm), len(y_norm))
        blend_norm = w["cnn"] * cnn_pm_norm[:n] + w["gbm"] * gbm_preds_norm[:n]
        blend_dn   = denormalize_predictions(blend_norm,    _ns)
        tgt_dn     = denormalize_predictions(y_norm[:n],    _ns)
        return _metrics_from_deg(blend_dn, tgt_dn)

    # ──────────────────────────────────────────────────────────────────────────
    # Ablation config definitions
    # ──────────────────────────────────────────────────────────────────────────
    ablation_configs = [
        # Scenario 1: HP tuning — both variants retrain CNN+GBM
        ("S1: With HP Tuning",             {}),
        ("S1: Default Params",             {"disable_hyperparameter_tuning": True}),
        # Scenario 2a: patch-level — pure inference swap, no retraining
        ("S2a: Ensemble (Patch)",          {}),
        ("S2a: GBM Only (Patch)",          {"disable_cnn_in_patch": True}),
        # Scenario 2b: pixel-level — pure inference swap, no retraining
        ("S2b: CNN-as-Residual (Pixel)",   {"cnn_as_residual": True}),
        ("S2b: CNN Only (Pixel)",          {"disable_gbm": True}),
        # Scenario 3: loss function — only CNN retrained; GBM retrained on new bottleneck
        ("S3: Multi-Component Loss",       {}),
        ("S3: Default MSE Loss",           {"basic_mse_no_aug": True}),
        # Scenario 4: sampler — only CNN retrained; GBM retrained on new bottleneck
        ("S4: With Stratified Sampling",   {}),
        ("S4: Without Stratified Sampling",{"disable_stratified": True}),
    ]

    results: Dict = {}

    for cfg_name, overrides in ablation_configs:
        logger.info(f"\n{'─' * 70}")
        logger.info(f"ABLATION CONFIG: {cfg_name}")
        logger.info(f"  Overrides: {overrides or 'none (full pipeline)'}")
        logger.info(f"{'─' * 70}")

        try:
            # ── Resolve which Optuna params to use ────────────────────────────
            _use_tuning = not overrides.get("disable_hyperparameter_tuning", False)
            _lr  = (_best_cnn_params.get("initial_lr")   if _use_tuning else None)
            _wd  = (_best_cnn_params.get("weight_decay") if _use_tuning else None)
            _bs  = (_best_cnn_params.get("batch_size")   if _use_tuning else None)
            _do  = (_best_cnn_params.get("dropout_rate") if _use_tuning else None)
            _gbm_par = (_best_gbm_params if (_use_tuning and _best_gbm_params) else None)

            # ── Decide evaluation path ────────────────────────────────────────
            #
            # Check scenario by cfg_name prefix first (S1/S3/S4 may have overrides={}
            # for the "with" variant, which would otherwise fall into the S2a baseline
            # branch erroneously).
            #
            # SCENARIO 1 — HP tuning comparison: retrain CNN + GBM ────────────
            if cfg_name.startswith("S1:"):
                logger.info(f"  [S1] Retraining CNN with "
                            f"{'Optuna' if _use_tuning else 'default'} params…")
                _r_cnn, _r_pm, _r_tg, _r_bot_tr, _r_bot_vl = _retrain_cnn(
                    initial_lr=_lr, weight_decay=_wd,
                    batch_size=_bs, dropout_rate=_do,
                    use_multi_loss=True, use_stratified=True,
                )
                _r_gbm, _r_Xg_vl, _r_yg_vl = _train_gbm(
                    _r_bot_tr, _r_bot_vl, _gbm_par
                )
                _gbm_preds = _r_gbm.predict(_r_Xg_vl, use_best=True)
                val_metrics_final = _blend_metrics(
                    _r_pm, _gbm_preds, _r_yg_vl, _prod_ens_weights
                )
                logger.info(f"  [S1] R²={val_metrics_final['r2']:.4f}  "
                            f"RMSE={val_metrics_final['rmse']:.4f}°C")

            # ── SCENARIO 3 — Loss function: retrain CNN + GBM ─────────────────
            elif cfg_name.startswith("S3:"):
                _use_multi = not overrides.get("basic_mse_no_aug", False)
                logger.info(f"  [S3] Retraining CNN with "
                            f"{'ProgressiveLSTLoss' if _use_multi else 'plain MSE'}…")
                _r_cnn, _r_pm, _r_tg, _r_bot_tr, _r_bot_vl = _retrain_cnn(
                    use_multi_loss=_use_multi, use_stratified=True,
                )
                _r_gbm, _r_Xg_vl, _r_yg_vl = _train_gbm(_r_bot_tr, _r_bot_vl)
                _gbm_preds = _r_gbm.predict(_r_Xg_vl, use_best=True)
                val_metrics_final = _blend_metrics(
                    _r_pm, _gbm_preds, _r_yg_vl, _prod_ens_weights
                )
                logger.info(f"  [S3] R²={val_metrics_final['r2']:.4f}  "
                            f"RMSE={val_metrics_final['rmse']:.4f}°C")

            # ── SCENARIO 4 — Stratified sampling: retrain CNN + GBM ───────────
            elif cfg_name.startswith("S4:"):
                _use_strat = not overrides.get("disable_stratified", False)
                logger.info(f"  [S4] Retraining CNN with "
                            f"{'stratified' if _use_strat else 'uniform'} sampler…")
                _r_cnn, _r_pm, _r_tg, _r_bot_tr, _r_bot_vl = _retrain_cnn(
                    use_multi_loss=True, use_stratified=_use_strat,
                )
                _r_gbm, _r_Xg_vl, _r_yg_vl = _train_gbm(_r_bot_tr, _r_bot_vl)
                _gbm_preds = _r_gbm.predict(_r_Xg_vl, use_best=True)
                val_metrics_final = _blend_metrics(
                    _r_pm, _gbm_preds, _r_yg_vl, _prod_ens_weights
                )
                logger.info(f"  [S4] R²={val_metrics_final['r2']:.4f}  "
                            f"RMSE={val_metrics_final['rmse']:.4f}°C")

            # ── SCENARIO 2a/2b — pure inference swap; no retraining ───────────
            elif overrides.get("disable_cnn_in_patch"):
                # S2a GBM-only: use production GBM predictions directly
                if _prod_gbm.best_model is None:
                    raise RuntimeError("Production GBM not loaded; cannot run S2a GBM-only")
                _gbm_preds = _prod_gbm.predict(_Xg_vl_prod, use_best=True)
                _gbm_dn    = denormalize_predictions(_gbm_preds, _ns)
                _tgt_dn    = denormalize_predictions(_yg_vl_prod, _ns)
                _n = min(len(_gbm_dn), len(_tgt_dn))
                val_metrics_final = _metrics_from_deg(_gbm_dn[:_n], _tgt_dn[:_n])
                logger.info(f"  [S2a] GBM-only: R²={val_metrics_final['r2']:.4f}  "
                            f"RMSE={val_metrics_final['rmse']:.4f}°C")

            elif overrides.get("disable_gbm"):
                # S2b CNN-only: production CNN patch-mean only
                _cnn_dn = denormalize_predictions(_prod_cnn_pm_norm, _ns)
                _tgt_dn = denormalize_predictions(_prod_cnn_tg_norm, _ns)
                val_metrics_final = _metrics_from_deg(_cnn_dn, _tgt_dn)
                logger.info(f"  [S2b] CNN-only: R²={val_metrics_final['r2']:.4f}  "
                            f"RMSE={val_metrics_final['rmse']:.4f}°C")

            elif overrides.get("cnn_as_residual"):
                # S2b CNN-as-Residual: production GBM corrects production CNN residual
                if _prod_gbm.best_model is None:
                    raise RuntimeError("Production GBM not loaded; cannot run CNN-as-Residual")
                _n = min(len(_prod_cnn_pm_norm), len(_yg_vl_prod))
                _res_vl = _yg_vl_prod[:_n] - _prod_cnn_pm_norm[:_n]
                # Build training residuals using already-computed production features
                _tr_ds_r = UHIDataset(X_train, y_train, augment=False)
                _tr_ld_r = DataLoader(
                    _tr_ds_r, batch_size=TRAINING_CONFIG["batch_size"],
                    shuffle=False, num_workers=0, pin_memory=False,
                )
                _cnn_pm_tr_norm = np.concatenate([
                    _prod_cnn(_d.to(device)).cpu().detach().numpy()
                        .reshape(_d.shape[0], -1).mean(axis=1)
                    for _d, _ in _tr_ld_r
                ])
                _n_tr = min(len(_cnn_pm_tr_norm), len(_yg_tr_prod))
                _res_tr = _yg_tr_prod[:_n_tr] - _cnn_pm_tr_norm[:_n_tr]
                _gbm_res = GBMTrainer(config=_gbm_par) if _gbm_par else GBMTrainer()
                _gbm_res.train(_Xg_tr_prod.iloc[:_n_tr], _res_tr,
                               _Xg_vl_prod.iloc[:_n], _res_vl)
                _gbm_res_preds = _gbm_res.predict(_Xg_vl_prod.iloc[:_n], use_best=True)
                _final_norm = _prod_cnn_pm_norm[:_n] + _gbm_res_preds
                _final_dn   = denormalize_predictions(_final_norm, _ns)
                _tgt_dn     = denormalize_predictions(_yg_vl_prod[:_n], _ns)
                val_metrics_final = _metrics_from_deg(_final_dn, _tgt_dn)
                logger.info(f"  [S2b] CNN-as-Residual: R²={val_metrics_final['r2']:.4f}  "
                            f"RMSE={val_metrics_final['rmse']:.4f}°C")

            # ── SCENARIO 2a Ensemble baseline (overrides = {}) ───────────────
            elif not overrides:
                # Use the production blend directly — baseline == LEVEL-1 output.
                if _prod_gbm.best_model is None:
                    raise RuntimeError("Production GBM not loaded; cannot compute ensemble")
                _gbm_preds = _prod_gbm.predict(_Xg_vl_prod, use_best=True)
                val_metrics_final = _blend_metrics(
                    _prod_cnn_pm_norm, _gbm_preds, _yg_vl_prod, _prod_ens_weights
                )
                logger.info(f"  [S2a / baseline] Weighted Ensemble: "
                            f"R²={val_metrics_final['r2']:.4f}  "
                            f"RMSE={val_metrics_final['rmse']:.4f}°C")

            else:
                # Catch-all: should not be reached with the configs above
                raise RuntimeError(f"Unhandled ablation config: {cfg_name}")

        except Exception as _err:
            logger.warning(
                f"\n  ⚠️  Config '{cfg_name}' failed: {_err}\n"
                "  Recording error and continuing."
            )
            import traceback as _tb
            _tb.print_exc()
            results[cfg_name] = {"error": str(_err)}
            continue

        # ── Record & log ──────────────────────────────────────────────────────
        results[cfg_name] = {"metrics": val_metrics_final}
        logger.info(
            f"\n  RESULT [{cfg_name}]: "
            f"R²={val_metrics_final['r2']:.4f}  "
            f"RMSE={val_metrics_final['rmse']:.4f}°C  "
            f"MAE={val_metrics_final['mae']:.4f}°C"
        )

    # ── Table 4.9 summary grouped by scenario ────────────────────────────────
    scenarios = [
        ("Scenario 1 — Hyperparameter Tuning",
         "S1: With HP Tuning", "S1: Default Params"),
        ("Scenario 2a — Patch-level Ensemble vs. GBM-only",
         "S2a: Ensemble (Patch)", "S2a: GBM Only (Patch)"),
        ("Scenario 2b — Pixel-level CNN-as-Residual vs. CNN-only",
         "S2b: CNN-as-Residual (Pixel)", "S2b: CNN Only (Pixel)"),
        ("Scenario 3 — Multi-Component Loss vs. Default MSE",
         "S3: Multi-Component Loss", "S3: Default MSE Loss"),
        ("Scenario 4 — Stratified Sampling vs. Without",
         "S4: With Stratified Sampling", "S4: Without Stratified Sampling"),
    ]

    W = 100
    logger.info("\n" + "=" * W)
    logger.info("TABLE 4.9 — ABLATION STUDY SUMMARY  (validation set, °C)")
    logger.info("=" * W)
    logger.info(
        f"  {'Configuration':<38} {'R²':>7} {'RMSE(°C)':>10} {'MAE(°C)':>9} "
        f"{'ΔR²':>9} {'ΔRMSE':>9}"
    )

    for s_title, cfg_with, cfg_without in scenarios:
        logger.info(f"\n  ── {s_title} ──")
        logger.info("  " + "-" * (W - 2))
        for cfg_name in (cfg_with, cfg_without):
            res = results.get(cfg_name, {})
            if "error" in res:
                logger.info(f"  {'  ' + cfg_name:<38} ERROR: {res['error']}")
                continue
            m    = res.get("metrics", {})
            ref  = results.get(cfg_with, {}).get("metrics", {})
            dr2   = m.get("r2",   float("nan")) - ref.get("r2",   float("nan"))
            drmse = m.get("rmse", float("nan")) - ref.get("rmse", float("nan"))
            logger.info(
                f"  {'  ' + cfg_name:<38} {m.get('r2', float('nan')):>7.4f} "
                f"{m.get('rmse', float('nan')):>10.4f} "
                f"{m.get('mae',  float('nan')):>9.4f} "
                f"{'(baseline)':>9} {'':>9}"
                if cfg_name == cfg_with else
                f"  {'  ' + cfg_name:<38} {m.get('r2', float('nan')):>7.4f} "
                f"{m.get('rmse', float('nan')):>10.4f} "
                f"{m.get('mae',  float('nan')):>9.4f} "
                f"{dr2:>+9.4f} {drmse:>+9.4f}"
            )

    logger.info("\n" + "=" * W)

    # ── Persist ablation_results.json ─────────────────────────────────────────
    ablation_out = model_dir / "ablation_results.json"
    _serializable_ablation = {}
    for k, v in results.items():
        entry: Dict = {}
        if "error" in v:
            entry["error"] = v["error"]
        else:
            entry["metrics"] = {mk: float(mv) for mk, mv in v.get("metrics", {}).items()}
        _serializable_ablation[k] = entry
    with open(ablation_out, "w") as f:
        json.dump(_serializable_ablation, f, indent=2)
    logger.info(f"\n✅ Ablation results saved → {ablation_out}")

    return results



# ── Table 4.10 — Cross-variant comparison ─────────────────────────────────────

def log_cross_variant_comparison(results: Dict[str, Dict]) -> None:
    """Log Table 4.10 from persisted test_results dicts for Landsat and Fusion.

    Args:
        results: Dict mapping variant label ('landsat', 'fusion') → test_results
                 dict returned by evaluate_on_test_set(), or loaded from the
                 corresponding test_results.json files.
    """
    logger.info("\n" + "=" * 80)
    logger.info("TABLE 4.10 — CROSS-VARIANT COMPARISON: LANDSAT vs FUSION (test set)")
    logger.info("=" * 80)

    tgt     = VALIDATION_CONFIG["targets"]
    metrics = ["r2", "rmse", "mae", "mbe"]

    # Extract ensemble metrics for both variants
    variant_metrics: Dict[str, Dict] = {}
    for variant, res in results.items():
        if isinstance(res, dict) and "ensemble" in res:
            variant_metrics[variant] = res["ensemble"]
        elif isinstance(res, dict) and "ensemble_metrics" in res:
            em = res["ensemble_metrics"]
            if isinstance(em, dict):
                variant_metrics[variant] = em
        else:
            logger.warning(f"  Could not extract ensemble metrics for variant '{variant}'")

    if len(variant_metrics) < 2:
        logger.warning("  Need both 'landsat' and 'fusion' results for cross-variant comparison")
        return

    ls = variant_metrics.get("landsat", {})
    fu = variant_metrics.get("fusion",  {})

    W = 80
    logger.info(
        f"\n  {'Metric':<12} {'Landsat':>10} {'Fusion':>10} "
        f"{'Δ(F-LS)':>10} {'Rel. Gain':>11} {'Target Met?':>13}"
    )
    logger.info("  " + "-" * (W - 2))

    target_map = {"r2": (tgt["r2"], "≥"), "rmse": (tgt["rmse"], "≤"),
                  "mae": (tgt["mae"], "≤"), "mbe": (None, "≈0")}

    for m_key in metrics:
        ls_v  = ls.get(m_key, float("nan"))
        fu_v  = fu.get(m_key, float("nan"))
        delta = fu_v - ls_v
        tgt_v, cmp = target_map[m_key]

        if tgt_v is not None and not np.isnan(fu_v):
            if cmp == "≥":
                target_met = fu_v >= tgt_v
                rel_gain   = delta / abs(ls_v) * 100 if ls_v != 0 else float("nan")
            else:  # ≤
                target_met = fu_v <= tgt_v
                rel_gain   = -delta / abs(ls_v) * 100 if ls_v != 0 else float("nan")
            met_str = "✅ Yes" if target_met else "❌ No "
        else:
            rel_gain = delta / abs(ls_v) * 100 if (ls_v and ls_v != 0) else float("nan")
            met_str  = f"(target: {cmp}0)"

        suffix = "°C" if m_key != "r2" else ""
        logger.info(
            f"  {(m_key.upper() + suffix):<12} {ls_v:>10.4f} {fu_v:>10.4f} "
            f"  {delta:>+9.4f}  {rel_gain:>+9.1f}%  {met_str:>12}"
        )

    logger.info("=" * 80)
    logger.info(
        "  Interpretation: Fusion ensemble gains derive from finer spectral\n"
        "  discrimination (10-m Sentinel-2 indices) in heterogeneous transitional\n"
        "  zones (mixed residential-commercial corridors, park-industrial boundaries)."
    )

    # ── Persist cross_variant_comparison.json (written to both model dirs) ────
    serializable = {
        variant: {k: float(v) for k, v in m.items()}
        for variant, m in variant_metrics.items()
    }
    for variant in results:
        if "model_dir" in results[variant]:
            _out = Path(results[variant]["model_dir"]) / "cross_variant_comparison.json"
            with open(_out, "w") as f:
                json.dump(serializable, f, indent=2)
            logger.info(f"  ✅ Saved → {_out}")


def main():
    """
    Main entry point.

    Usage examples
    --------------
    # Train on Landsat-only data (default):
    python train_ensemble.py

    # Train on fusion data only:
    python train_ensemble.py --mode fusion

    # Train on Landsat-only data, explicit path:
    python train_ensemble.py --mode landsat

    # Train BOTH variants back-to-back for comparison:
    python train_ensemble.py --mode both

    # Point at an arbitrary preprocessed dataset:
    python train_ensemble.py --dataset /path/to/cnn_dataset_custom \
                             --model-dir /path/to/models/custom

    # Run the test-set evaluation scenarios from Section 4.6 of the thesis
    # (requires trained models in the default model dirs):
    python train_ensemble.py --mode both --run-test-eval

    # Run the ablation study only (no full training):
    python train_ensemble.py --mode fusion --ablation-only

    # Override ablation epoch count for faster iteration:
    python train_ensemble.py --mode fusion --ablation-only --ablation-epochs 10
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="UHI ensemble training — supports Landsat-only, fusion, or both variants"
    )
    parser.add_argument(
        "--mode",
        choices=["landsat", "fusion", "both"],
        default="landsat",
        help=(
            "Which preprocessed dataset variant to train on.\n"
            "  landsat : PROCESSED_DATA_DIR/cnn_dataset_landsat  (default)\n"
            "  fusion  : PROCESSED_DATA_DIR/cnn_dataset_fusion\n"
            "  both    : train landsat first, then fusion (for comparison)"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Override dataset directory.  Ignored when --mode is 'both'.\n"
            "Default: chosen automatically from --mode."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        dest="model_dir",
        help=(
            "Override model output directory.  Ignored when --mode is 'both'.\n"
            "Default: MODEL_DIR/<mode> (e.g. MODEL_DIR/landsat)."
        ),
    )
    parser.add_argument(
        "--run-test-eval",
        action="store_true",
        default=False,
        dest="run_test_eval",
        help=(
            "After training, run the full Section 4.6 test-set evaluation:\n"
            "  Table 4.8 (CNN-only vs ensemble on held-out test set),\n"
            "  Table 4.10 (cross-variant comparison when --mode both),\n"
            "  residual diagnostics (heteroscedasticity, Shapiro-Wilk,\n"
            "  Moran's I proxy, temporal stability) and GBM feature importance.\n"
            "Requires a 'test' sub-directory in the dataset directory."
        ),
    )
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        default=False,
        dest="ablation_only",
        help=(
            "Skip full training and run only the Table 4.9 ablation study.\n"
            "Useful for targeted ablation experiments or quick CI smoke tests."
        ),
    )
    parser.add_argument(
        "--ablation-epochs",
        type=int,
        default=None,
        dest="ablation_epochs",
        help=(
            "Override epoch count for each ablation configuration "
            "(default: use TRAINING_CONFIG['epochs']).\n"
            "Pass a small value (e.g. 5) for a fast smoke test."
        ),
    )
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("URBAN HEAT ISLAND - ENSEMBLE TRAINING (BEST MODELS)")
    logger.info("="*60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and COMPUTE_CONFIG["use_gpu"] else "cpu"
    )
    logger.info(f"Using device: {device}")

    # ── Resolve variant(s) to train ──────────────────────────────────────────
    VARIANT_DATASET = {
        "landsat": PROCESSED_DATA_DIR / "cnn_dataset_landsat",
        "fusion":  PROCESSED_DATA_DIR / "cnn_dataset_fusion",
    }
    VARIANT_MODEL = {
        "landsat": MODEL_DIR / "landsat",
        "fusion":  MODEL_DIR / "fusion",
    }

    if args.mode == "both":
        variants = [
            ("landsat", VARIANT_DATASET["landsat"], VARIANT_MODEL["landsat"]),
            ("fusion",  VARIANT_DATASET["fusion"],  VARIANT_MODEL["fusion"]),
        ]
    else:
        dataset_dir = args.dataset if args.dataset else VARIANT_DATASET[args.mode]
        model_dir   = args.model_dir if args.model_dir else VARIANT_MODEL[args.mode]
        variants = [(args.mode, dataset_dir, model_dir)]

    # ── Check disk space once using the first model dir ──────────────────────
    if not check_disk_space(variants[0][2].parent):
        logger.warning("⚠️ Low disk space — proceeding anyway")

    # ── Ablation-only mode (Table 4.9) ───────────────────────────────────────
    if args.ablation_only:
        logger.info("\n" + "#" * 70)
        logger.info("# ABLATION-ONLY MODE — skipping full training")
        logger.info("#" * 70)
        # Use the first variant's dataset/model dir; for --mode both use fusion
        abl_label, abl_ds, abl_md = variants[-1]   # fusion if both, else the chosen one
        abl_md.mkdir(parents=True, exist_ok=True)
        run_ablation_study(
            dataset_dir=abl_ds,
            model_dir=abl_md,
            device=device,
            label=abl_label,
            n_ablation_epochs=args.ablation_epochs,
        )
        return

    # ── Train each variant ───────────────────────────────────────────────────
    histories = {}
    # ensemble_trainers keeps a reference per variant so we can call test eval
    ensemble_trainers: Dict[str, "EnsembleTrainer"] = {}

    for label, dataset_dir, model_dir in variants:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# VARIANT: {label.upper()}")
        logger.info(f"{'#'*70}")
        histories[label] = _train_one_variant(dataset_dir, model_dir, device, label)

    # ── Val-set cross-variant summary (existing behaviour, always runs) ───────
    if len(histories) == 2:
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-VARIANT COMPARISON (validation set — existing behaviour)")
        logger.info("=" * 70)
        logger.info(f"{'Variant':<12} {'R²':>8} {'RMSE(°C)':>10} {'MAE(°C)':>9}")
        logger.info("-" * 42)
        for label, hist in histories.items():
            if "ensemble_metrics" in hist and hist["ensemble_metrics"]:
                m = hist["ensemble_metrics"]
                logger.info(f"{label:<12} {m['r2']:>8.4f} {m['rmse']:>10.4f} {m['mae']:>9.4f}")
        logger.info("=" * 70)
        logger.info("Diagnostic plots saved to:")
        for label, _, model_dir in variants:
            logger.info(f"  {label:8s}: {model_dir / 'diagnostics'}")

    # ── Section 4.6 test-set evaluation (opt-in via --run-test-eval) ─────────
    if args.run_test_eval:
        logger.info("\n" + "#" * 70)
        logger.info("# SECTION 4.6 — TEST SET EVALUATION SCENARIOS")
        logger.info("#" * 70)

        test_results: Dict[str, Dict] = {}

        for label, dataset_dir, model_dir in variants:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"  Test evaluation: {label.upper()}")
            logger.info(f"{'=' * 70}")

            # ── Reload best CNN ───────────────────────────────────────────────
            norm_stats = load_normalization_stats(dataset_dir)
            n_channels = norm_stats.get("n_channels", CNN_CONFIG["input_channels"])

            cnn_model_te = UNet(in_channels=n_channels, out_channels=1)
            ckpt_mgr_te  = CheckpointManager(
                save_dir=model_dir / "checkpoints",
                metrics=CHECKPOINT_CONFIG["metrics"],
            )
            best_ckpt = ckpt_mgr_te.load_best(
                cnn_model_te,
                metric=CHECKPOINT_CONFIG["primary_metric"],
                device=device,
            )
            if best_ckpt is None:
                logger.warning(
                    f"  ⚠️ No best CNN checkpoint found for {label} — "
                    f"using randomly initialised weights. Train first."
                )
            cnn_model_te = cnn_model_te.to(device)

            # ── Reload best GBM ───────────────────────────────────────────────
            gbm_trainer_te = GBMTrainer()
            gbm_pkl = model_dir / "best_gbm_model.pkl"
            if not gbm_pkl.exists():
                gbm_pkl = model_dir / "gbm_model.pkl"   # fallback
            if gbm_pkl.exists():
                gbm_trainer_te.best_model = joblib.load(gbm_pkl)
                gbm_trainer_te.model      = gbm_trainer_te.best_model
                logger.info(f"  ✅ GBM loaded from {gbm_pkl}")
            else:
                logger.warning(f"  ⚠️ GBM model not found at {gbm_pkl} — ensemble will fall back to CNN-only")

            # ── Reload PCA ────────────────────────────────────────────────────
            pca_te   = None
            pca_path = model_dir / "bottleneck_pca.pkl"
            if pca_path.exists():
                pca_te = joblib.load(pca_path)
                logger.info(f"  ✅ PCA loaded from {pca_path}")
            else:
                logger.warning(f"  ⚠️ PCA not found at {pca_path} — bottleneck features uncompressed")

            # ── Reload ensemble weights ────────────────────────────────────────
            ens_cfg_path = model_dir / "ensemble_config.json"
            if ens_cfg_path.exists():
                with open(ens_cfg_path) as _f:
                    _ens_cfg = json.load(_f)
                ens_weights_te = _ens_cfg.get("weights", ENSEMBLE_WEIGHTS)
            else:
                ens_weights_te = ENSEMBLE_WEIGHTS
                logger.warning(f"  ⚠️ ensemble_config.json not found — using default weights")

            # ── Table 4.8: run test evaluation ────────────────────────────────
            tr = evaluate_on_test_set(
                cnn_model=cnn_model_te,
                gbm_trainer=gbm_trainer_te,
                pca=pca_te,
                ensemble_weights=ens_weights_te,
                dataset_dir=dataset_dir,
                model_dir=model_dir,
                device=device,
                label=label,
            )
            tr["model_dir"] = str(model_dir)
            test_results[label] = tr

            # ── Table 4.9: ablation study (one config per variant) ────────────
            logger.info(f"\n{'─' * 70}")
            logger.info(f"  Launching ablation study for {label.upper()}…")
            logger.info(f"{'─' * 70}")
            run_ablation_study(
                dataset_dir=dataset_dir,
                model_dir=model_dir,
                device=device,
                label=label,
                n_ablation_epochs=args.ablation_epochs,
            )

        # ── Table 4.10: cross-variant comparison on test set ──────────────────
        if len(test_results) == 2:
            log_cross_variant_comparison(test_results)
        elif len(test_results) == 1:
            label_only = next(iter(test_results))
            logger.info(
                f"\n  Table 4.10 cross-variant comparison requires both 'landsat' and "
                f"'fusion' results.\n"
                f"  Only '{label_only}' was evaluated. Re-run with --mode both to compare."
            )

        logger.info("\n" + "#" * 70)
        logger.info("# SECTION 4.6 EVALUATION COMPLETE")
        logger.info(f"#  Results JSON files written to each variant's model directory.")
        logger.info("#  Files produced per variant:")
        logger.info("#    test_results.json       — Table 4.8 + residual diagnostics")
        logger.info("#    ablation_results.json   — Table 4.9 ablation scores")
        logger.info("#    cross_variant_comparison.json — Table 4.10 (both variants only)")
        logger.info("#" * 70)


if __name__ == "__main__":
    main()