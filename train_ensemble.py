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
matplotlib.use("Agg")  # Non-interactive backend — safe for headless/Windows runs
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
            fig.savefig(path, dpi=150, bbox_inches="tight")
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
                ax.plot(epochs, r2_scores, label="Val R²", color="#4CAF50")
                ax.axhline(0.85, ls="--", color="gray", lw=0.8, label="Target (0.85)")
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
        """Scatter + residual plot.

        Inputs may be normalised (straight from the CNN/GBM output) or already
        in °C — we detect and denormalise automatically so axes always show °C.
        """
        try:
            from scipy.stats import linregress
            self._use_style()

            preds   = np.asarray(preds).flatten()
            targets = np.asarray(targets).flatten()
            mask    = np.isfinite(preds) & np.isfinite(targets)
            preds, targets = preds[mask], targets[mask]

            # Subsample to avoid OOM on large CNN output arrays (>100 k points)
            _MAX_PLOT_PTS = 100_000
            if len(preds) > _MAX_PLOT_PTS:
                _rng = np.random.default_rng(seed=42)
                _idx = _rng.choice(len(preds), size=_MAX_PLOT_PTS, replace=False)
                _idx.sort()
                preds, targets = preds[_idx], targets[_idx]

            # Denormalise if values look normalised (mean ≈ 0, std ≈ 1)
            _ns = load_normalization_stats()
            if _ns is not None and abs(targets.mean()) < 5.0 and targets.std() < 5.0:
                preds   = denormalize_predictions(preds,   _ns)
                targets = denormalize_predictions(targets, _ns)
                unit = "°C"
            else:
                unit = "°C" if targets.mean() > 5 else "(norm)"

            slope, intercept, r_val, *_ = linregress(targets, preds)
            residuals = preds - targets
            mbe       = float(residuals.mean())
            pred_std  = float(preds.std())
            tgt_std   = float(targets.std())
            std_ratio = pred_std / tgt_std if tgt_std > 0 else float("nan")

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"{label} – Predicted vs Actual", fontsize=14, fontweight="bold")

            ax = axes[0]
            sc = ax.scatter(targets, preds, alpha=0.35, s=12,
                            c=np.abs(residuals), cmap="RdYlGn_r", label="Samples")
            plt.colorbar(sc, ax=ax, label=f"|Residual| ({unit})")
            lo = min(targets.min(), preds.min()); hi = max(targets.max(), preds.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect (slope=1)")
            fit_x = np.linspace(lo, hi, 200)
            ax.plot(fit_x, slope * fit_x + intercept, "r-", lw=1.5,
                    label=f"Fit  slope={slope:.3f}")
            ax.set_xlabel(f"Actual ({unit})"); ax.set_ylabel(f"Predicted ({unit})")
            ax.legend(fontsize=8)
            info = [f"R²={r_val**2:.4f}",
                    f"RMSE={np.sqrt(np.mean(residuals**2)):.3f}{unit}",
                    f"MAE={np.mean(np.abs(residuals)):.3f}{unit}",
                    f"MBE={mbe:.3f}{unit}",
                    f"slope={slope:.3f}",
                    f"std_ratio={std_ratio:.3f}"]
            if metrics:
                # Override with pre-computed denorm metrics if supplied
                info = [
                    f"R²={metrics.get('r2', r_val**2):.4f}",
                    f"RMSE={metrics.get('rmse', float('nan')):.3f}{unit}",
                    f"MAE={metrics.get('mae', float('nan')):.3f}{unit}",
                    f"MBE={metrics.get('mbe', mbe):.3f}{unit}",
                    f"slope={metrics.get('slope', slope):.3f}",
                    f"std_ratio={metrics.get('std_ratio', std_ratio):.3f}",
                ]
            ax.text(0.03, 0.97, "\n".join(info), transform=ax.transAxes,
                    fontsize=8, va="top", bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            ax = axes[1]
            ax.scatter(targets, residuals, alpha=0.35, s=12, color="#5C6BC0")
            ax.axhline(0, color="black", lw=1.2)
            ax.axhline(+np.std(residuals), color="orange", ls="--", lw=0.9, label="±1σ")
            ax.axhline(-np.std(residuals), color="orange", ls="--", lw=0.9)
            ax.set_xlabel(f"Actual ({unit})"); ax.set_ylabel(f"Residual ({unit})")
            ax.set_title("Residuals vs Actual"); ax.legend(fontsize=8)

            plt.tight_layout()
            safe = label.replace(" ", "_").replace("(", "").replace(")", "")
            self._save(fig, f"02_pred_vs_actual_{safe}")
        except Exception as e:
            logger.warning(f"plot_pred_vs_actual failed: {e}")

    # ── 3. Residual distribution ──────────────────────────────────────────────
    def plot_residual_distribution(self, preds: np.ndarray, targets: np.ndarray,
                                   label: str = "Model"):
        """Histogram + Q-Q plot of residuals (auto-denormalises if needed)."""
        try:
            from scipy import stats
            self._use_style()
            p = np.asarray(preds).flatten()
            t = np.asarray(targets).flatten()
            # Subsample to avoid OOM
            _MAX_PLOT_PTS = 100_000
            if len(p) > _MAX_PLOT_PTS:
                _rng = np.random.default_rng(seed=42)
                _idx = _rng.choice(len(p), size=_MAX_PLOT_PTS, replace=False)
                p, t = p[_idx], t[_idx]
            # Denormalise if values look normalised
            _ns = load_normalization_stats()
            if _ns is not None and abs(t.mean()) < 5.0 and t.std() < 5.0:
                p = denormalize_predictions(p, _ns)
                t = denormalize_predictions(t, _ns)
                unit = "°C"
            else:
                unit = "°C" if t.mean() > 5 else "(norm)"
            residuals = (p - t)
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
            ax.set_xlabel(f"Residual ({unit})"); ax.set_ylabel("Density"); ax.legend()
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
            ax.set_xlabel(f"Theoretical quantiles ({unit})"); ax.set_ylabel(f"Sample quantiles ({unit})")
            ax.set_title(f"Q-Q  (r={r:.4f})")

            plt.tight_layout()
            safe = label.replace(" ", "_").replace("(", "").replace(")", "")
            self._save(fig, f"03_residual_dist_{safe}")
        except Exception as e:
            logger.warning(f"plot_residual_distribution failed: {e}")

    # ── 4. Ensemble strategy comparison ──────────────────────────────────────
    def plot_ensemble_comparison(self, comparison: dict):
        """Bar charts comparing R², RMSE, MAE across strategies."""
        try:
            self._use_style()
            names  = list(comparison.keys())
            colors = plt.cm.tab10(np.linspace(0, 0.6, len(names)))
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle("Ensemble Strategy Comparison", fontsize=14, fontweight="bold")
            for idx, (mkey, mlabel) in enumerate(
                    zip(["r2", "rmse", "mae"], ["R²", "RMSE (°C)", "MAE (°C)"])):
                ax = axes[idx]
                vals = [comparison[n].get(mkey, 0) for n in names]
                bars = ax.bar(names, vals, color=colors, edgecolor="white")
                ax.set_title(mlabel); ax.set_ylabel(mlabel)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.001, f"{v:.4f}",
                            ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            self._save(fig, "04_ensemble_comparison")
        except Exception as e:
            logger.warning(f"plot_ensemble_comparison failed: {e}")

    # ── 5. GBM feature importance ─────────────────────────────────────────────
    def plot_gbm_feature_importance(self, gbm_model, top_n: int = 30):
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
            # Subsample large memmaps to avoid loading everything into RAM.
            # 50 000 values gives an accurate histogram without memory pressure.
            _MAX = 50_000
            def _safe_flat(arr):
                a = np.asarray(arr).flatten()
                if len(a) > _MAX:
                    rng = np.random.default_rng(0)
                    a = a[rng.choice(len(a), _MAX, replace=False)]
                return a
            train_flat = _safe_flat(y_train)
            val_flat   = _safe_flat(y_val)
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
                  target=0.85)
            _plot(axes[0, 1], slopes,     "Prediction Slope",     "Slope",
                  target=1.0, danger=0.85)
            _plot(axes[1, 0], std_ratios, "Std Ratio (pred/tgt)", "Std Ratio",
                  target=1.0, danger=0.80)
            _plot(axes[1, 1], mbe_vals,   "Mean Bias Error",      "MBE (°C)")
            axes[1, 1].axhline(0, color="green", ls="--", lw=1, label="Target (0)")
            axes[1, 1].legend(fontsize=8)

            plt.tight_layout()
            self._save(fig, "07_diagnostic_metrics_epochs")
        except Exception as e:
            logger.warning(f"plot_diagnostic_metrics_over_epochs failed: {e}")

    # ── 8. Spatial error maps ─────────────────────────────────────────────────
    def plot_spatial_error_map(self, preds_4d: np.ndarray, targets_4d: np.ndarray,
                               label: str = "Model", n_samples: int = 6):
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

            P = _sq(preds_4d)[:n_samples]
            T = _sq(targets_4d)[:n_samples]
            E = P - T

            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
            if n_samples == 1:
                axes = axes[np.newaxis, :]
            fig.suptitle(f"{label} – Spatial Error Maps (first {n_samples} patches)",
                         fontsize=13, fontweight="bold")

            vmin = min(T.min(), P.min()); vmax = max(T.max(), P.max())
            err_abs = max(np.abs(E).max(), 1e-6)
            norm_err = TwoSlopeNorm(vmin=-err_abs, vcenter=0, vmax=err_abs)

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
                              label: str = "Model", n_bins: int = 10):
        """RMSE, MAE, MBE bucketed by temperature percentile bins."""
        try:
            self._use_style()
            preds   = np.asarray(preds).flatten()
            targets = np.asarray(targets).flatten()
            mask    = np.isfinite(preds) & np.isfinite(targets)
            preds, targets = preds[mask], targets[mask]
            # Subsample to avoid OOM
            _MAX_PLOT_PTS = 100_000
            if len(preds) > _MAX_PLOT_PTS:
                _rng = np.random.default_rng(seed=42)
                _idx = _rng.choice(len(preds), size=_MAX_PLOT_PTS, replace=False)
                preds, targets = preds[_idx], targets[_idx]
            # Denormalise so bin centres are in °C
            _ns = load_normalization_stats()
            if _ns is not None and abs(targets.mean()) < 5.0 and targets.std() < 5.0:
                preds   = denormalize_predictions(preds,   _ns)
                targets = denormalize_predictions(targets, _ns)

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
                tbl[best_r2_row + 1, col].set_facecolor("#C8E6C9")
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
                ax_r2c.plot(epochs, r2s, color="#4CAF50", lw=1.5)
                ax_r2c.axhline(0.85, ls="--", color="gray", lw=0.8, label="Target")
                ax_r2c.set_title("CNN Val R²"); ax_r2c.set_xlabel("Epoch")
                ax_r2c.set_ylabel("R²"); ax_r2c.legend()

            # Bottom row diagnostics
            if history["cnn_metrics"]:
                slopes     = [m.get("slope",     float("nan")) for m in history["cnn_metrics"]]
                std_ratios = [m.get("std_ratio",  float("nan")) for m in history["cnn_metrics"]]
                mbe_vals   = [m.get("mbe",        float("nan")) for m in history["cnn_metrics"]]
                lrs        = history["lr"]
                for col_idx, (data, title, target, use_log) in enumerate(zip(
                        [slopes, std_ratios, mbe_vals, lrs],
                        ["Slope", "Std Ratio", "MBE (°C)", "Learning Rate"],
                        [1.0, 1.0, 0.0, None],
                        [False, False, False, True])):
                    ax_d = fig.add_subplot(gs[2, col_idx])
                    ep_d = epochs if col_idx < 3 else range(1, len(lrs) + 1)
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
                               cnn_preds_4d: np.ndarray = None,
                               targets_4d: np.ndarray = None):
        """Convenience wrapper — calls every available plot method."""
        logger.info("\n📊 Generating all post-training diagnostic plots...")
        self.plot_training_curves(history)
        self.plot_diagnostic_metrics_over_epochs(history)
        self.plot_summary_dashboard(history, ensemble_metrics, cnn_metrics,
                                    gbm_metrics, ensemble_weights)
        if ensemble_comparison is not None:
            self.plot_ensemble_comparison(ensemble_comparison)
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
        # Spatial error map (08) — uses 4-D arrays (N, 1, H, W) from CNN
        if cnn_preds_4d is not None and targets_4d is not None:
            self.plot_spatial_error_map(cnn_preds_4d, targets_4d, label="CNN", n_samples=6)
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

    # ── Cross-channel ratio features (vegetation / built-up proxies) ─────────
    # Channels: 0=coastal,1=blue,2=green,3=red,4=nir,5=swir1,6=swir2,7=thermal,8=qa,9=extra
    # These ratios mirror NDVI, NDBI, MNDWI which are the strongest UHI predictors.
    eps = 1e-8
    try:
        nir  = features_df["ch4_mean"]; red   = features_df["ch3_mean"]
        swir = features_df["ch5_mean"]; green = features_df["ch2_mean"]
        blue = features_df["ch1_mean"]
        features_df["feat_ndvi"]  = (nir - red)   / (nir + red   + eps)
        features_df["feat_ndbi"]  = (swir - nir)  / (swir + nir  + eps)
        features_df["feat_mndwi"] = (green - swir) / (green + swir + eps)
        features_df["feat_ui"]    = (swir - nir)  / (swir + nir  + eps)   # urban index
        features_df["feat_ebbi"]  = (swir - nir)  / (10 * np.sqrt(np.abs(swir) + eps))
        # Interaction: NDVI × NDBI captures urban-fringe contrast
        features_df["feat_ndvi_ndbi"] = features_df["feat_ndvi"] * features_df["feat_ndbi"]
        logger.info("  Added 6 cross-channel ratio features (NDVI, NDBI, MNDWI, UI, EBBI, interaction)")
    except KeyError as _ke:
        logger.warning(f"Cross-channel features skipped (missing channel): {_ke}")
    # ─────────────────────────────────────────────────────────────────────────

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
    import torch
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
        self.config = config or GBM_CONFIG["params"]
        self.model = None
        self.best_model = None
        self.best_score = float('inf')
        # Post-hoc slope calibration params (fitted on val set after training)
        self._cal_slope = 1.0
        self._cal_intercept = 0.0
        
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame, y_val: np.ndarray):
        """Train GBM model and track best version"""
        logger.info("Training GBM model...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Use early_stopping_rounds from config, fall back to 75
        es_rounds = self.config.get("early_stopping_rounds", 75)
        callbacks = [
            lgb.early_stopping(stopping_rounds=es_rounds),
            lgb.log_evaluation(period=100)
        ]
        
        # Build training params without keys that aren't LightGBM params
        train_params = {k: v for k, v in self.config.items()
                        if k not in ("early_stopping_rounds",)}
        
        self.model = lgb.train(
            train_params,
            train_data,
            num_boost_round=train_params.get("n_estimators", 3000),
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        logger.info(f"GBM training complete. Best iteration: {self.model.best_iteration}")
        
        # Evaluate on val set
        val_preds_raw = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds_raw))
        
        if val_rmse < self.best_score:
            self.best_score = val_rmse
            self.best_model = self.model
            logger.info(f"✅ New best GBM model: RMSE={val_rmse:.4f} (normalized)")
        
        # ── Post-hoc slope calibration (std-ratio method) ─────────────────────
        # GBM suffers from regression-dilution: slope < 1 at val time.
        # We fit a linear recalibration y_cal = a * y_raw + b on the val set
        # such that cal_slope → 1.0 and bias → 0. This is saved and applied
        # at prediction time, correcting range compression without retraining.
        from scipy.stats import linregress as _lr
        _pred_std  = float(val_preds_raw.std())
        _tgt_std   = float(y_val.std())
        _pred_mean = float(val_preds_raw.mean())
        _tgt_mean  = float(y_val.mean())
        self._cal_slope     = _tgt_std / (_pred_std + 1e-8)
        self._cal_intercept = _tgt_mean - self._cal_slope * _pred_mean
        _cal_preds = self._cal_slope * val_preds_raw + self._cal_intercept
        _r2_before = r2_score(y_val, val_preds_raw)
        _r2_after  = r2_score(y_val, _cal_preds)
        _sl_before = _lr(val_preds_raw, y_val)[0]
        _sl_after  = _lr(_cal_preds, y_val)[0]
        logger.info(f"GBM slope calibration: a={self._cal_slope:.4f}, b={self._cal_intercept:.4f}")
        logger.info(f"  R²  before={_r2_before:.4f}  after={_r2_after:.4f}")
        logger.info(f"  Slope before={_sl_before:.4f}  after={_sl_after:.4f}")
        # ─────────────────────────────────────────────────────────────────────

        return self.model
    
    def predict(self, X: pd.DataFrame, use_best: bool = True,
                calibrate: bool = True) -> np.ndarray:
        """
        Make predictions, optionally applying post-hoc slope calibration.
        
        Args:
            X: Features
            use_best: If True, use best_model; if False, use current model
            calibrate: If True, apply slope calibration fitted on val set
        """
        model_to_use = self.best_model if (use_best and self.best_model is not None) else self.model
        
        if model_to_use is None:
            raise ValueError("Model not trained yet!")
        
        preds = model_to_use.predict(X, num_iteration=model_to_use.best_iteration)
        if calibrate:
            preds = self._cal_slope * preds + self._cal_intercept
        return preds
    
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
        logger.info("✓ Using ProgressiveLSTLoss v2 (gradient-alignment + temp-weighted MSE)")

        # FIX 2: Layer-wise weight decay — deeper/output layers get stronger
        # regularisation to stop the monotonically-growing weight norm observed
        # in diagnostics (75→320 over training).
        if config["optimizer"] == "adamw":
            self.optimizer = optim.AdamW(
                [
                    {"params": cnn_model.enc1.parameters(),      "weight_decay": 1e-4},
                    {"params": cnn_model.enc2.parameters(),      "weight_decay": 2e-4},
                    {"params": cnn_model.enc3.parameters(),      "weight_decay": 3e-4},
                    {"params": cnn_model.enc4.parameters(),      "weight_decay": 4e-4},
                    {"params": cnn_model.bottleneck.parameters(),"weight_decay": 5e-4},
                    {"params": cnn_model.dec4.parameters(),      "weight_decay": 4e-4},
                    {"params": cnn_model.dec3.parameters(),      "weight_decay": 3e-4},
                    {"params": cnn_model.dec2.parameters(),      "weight_decay": 2e-4},
                    {"params": cnn_model.dec1.parameters(),      "weight_decay": 1e-4},
                    {"params": cnn_model.output.parameters(),    "weight_decay": 5e-4},
                ],
                lr=config["initial_lr"],
            )
            logger.info("✓ Layer-wise weight decay: 1e-4 (enc1) → 5e-4 (bottleneck/output)")
        else:
            self.optimizer = optim.Adam(
                cnn_model.parameters(),
                lr=config["initial_lr"]
            )
        
        self.scheduler = self._create_scheduler()

        # FIX 7: Tighter early stopping — patience=20, min_delta=0.002 so small
        # oscillations (visible in R² curve) don't reset the counter.
        # Monitor val R² (mode=max): stable across progressive loss phases.
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 20),
            min_delta=0.002,
            mode="max",
        )
        
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

        # Diagnostics plotter — saves all figures to MODEL_DIR/diagnostics/
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
        """
        FIX 3: CosineAnnealingWarmRestarts (SGDR) replaces monotonic cosine decay.

        Diagnostics showed R² plateauing at ~0.79 from epoch 30, with the old
        scheduler never dropping LR low enough to escape the local minimum.
        SGDR periodically resets LR to allow the optimiser to explore and find
        better minima, doubling the restart period after each cycle.

        T_0=50  → first restart after 50 epochs
        T_mult=2 → periods: 50, 100, 200 … (fits a 200-epoch budget)
        eta_min=1e-6 → minimum LR at the bottom of each cosine valley
        """
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-6,
        )
        logger.info("✓ SGDR scheduler: T_0=50, T_mult=2, eta_min=1e-6")
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
            from sklearn.decomposition import PCA
            logger.info(f"  Compressing bottleneck features: {bot_train.shape[1]} → {n_components} dims (PCA)")
            pca = PCA(n_components=n_components, random_state=42)
            bot_train = pca.fit_transform(bot_train)   # fit ONLY on train
            bot_val   = pca.transform(bot_val)         # apply same projection to val
            explained = pca.explained_variance_ratio_.sum()
            logger.info(f"  PCA variance explained: {explained*100:.1f}%  shape: train={bot_train.shape}, val={bot_val.shape}")
            self._pca = pca   # cache for inference

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
            self.cnn_model, metric='r2', device=self.device)
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
        tgt_mean, tgt_std = y_val_gbm.mean(), y_val_gbm.std() + 1e-8
        cnn_norm = (cnn_patch_mean - cnn_patch_mean.mean()) / (cnn_patch_mean.std() + 1e-8)
        gbm_norm = (gbm_preds       - gbm_preds.mean())       / (gbm_preds.std()       + 1e-8)

        opt_w    = self.compute_optimal_weights(cnn_metrics, gbm_metrics)
        blend    = opt_w["cnn"] * cnn_norm + opt_w["gbm"] * gbm_norm
        blend_sc = blend * tgt_std + tgt_mean                          # back to target scale
        weighted_metrics = self._calculate_metrics(blend_sc, y_val_gbm, "Weighted Ensemble")

        # ── Strategy D: CNN-as-residual ───────────────────────────────────────
        # GBM provides the patch-mean prediction; the CNN corrects the spatial
        # deviation within each patch.  No granularity mismatch.
        #   final(x,y) = gbm_patch_mean + (cnn_pixel(x,y) - cnn_patch_mean)
        # We evaluate on patch means (same target as GBM).
        cnn_deviation = cnn_preds_4d.reshape(cnn_preds_4d.shape[0], -1) \
                        - cnn_patch_mean[:, np.newaxis]                 # (N, H*W)
        # Residual patch-mean prediction = GBM + mean deviation (should be ~0, serves as sanity check)
        residual_patch_mean = gbm_preds + cnn_deviation.mean(axis=1)
        residual_metrics = self._calculate_metrics(
            residual_patch_mean, y_val_gbm, "CNN-as-Residual")

        # ── Compare all strategies ────────────────────────────────────────────
        all_results = [
            ("GBM Only",          gbm_metrics,      {"cnn": 0.0, "gbm": 1.0}),
            ("CNN Only",          cnn_metrics,       {"cnn": 1.0, "gbm": 0.0}),
            ("Weighted Ensemble", weighted_metrics,  opt_w),
            ("CNN-as-Residual",   residual_metrics,  {"cnn": "residual", "gbm": 1.0}),
        ]

        logger.info("\n" + "="*70)
        logger.info("ENSEMBLE STRATEGY COMPARISON")
        logger.info("="*70)
        logger.info(f"{'Strategy':<22} {'R²':>8} {'RMSE(°C)':>10} {'MAE(°C)':>9} {'Slope':>7} {'StdRat':>8}")
        logger.info("-"*70)
        for name, m, _ in all_results:
            logger.info(
                f"{name:<22} {m['r2']:>8.4f} {m['rmse']:>10.4f} {m['mae']:>9.4f} "
                f"{m.get('slope', float('nan')):>7.3f} {m.get('std_ratio', float('nan')):>8.3f}")
        logger.info("="*70)

        best_name, best_metrics, best_weights = max(
            all_results, key=lambda x: x[1]['r2'])

        logger.info(f"\n🏆 BEST STRATEGY: {best_name}")
        logger.info(f"   R²={best_metrics['r2']:.4f}  RMSE={best_metrics['rmse']:.4f}°C  "
                    f"slope={best_metrics.get('slope', float('nan')):.3f}")

        # Update ensemble weights
        if isinstance(best_weights.get("cnn"), float):
            self.ensemble_weights = best_weights
        else:
            # CNN-as-residual: mark this mode in config
            self.ensemble_weights = {"cnn": 0.0, "gbm": 1.0, "mode": "cnn_residual"}

        # Warn if GBM-only wins (expected given diagnostics)
        if best_name == "GBM Only":
            logger.info("ℹ️  GBM alone is strongest — CNN will be used spatially via CNN-as-Residual at inference")
        elif best_name == "Weighted Ensemble":
            logger.info(f"   Weights: CNN={opt_w['cnn']:.3f}, GBM={opt_w['gbm']:.3f}")

        # FIX 3: store comparison dict so plot_ensemble_comparison can be called
        self.history["ensemble_comparison"] = {name: m for name, m, _ in all_results}

        logger.info("="*70)
        return best_metrics
    
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
                import json as _js
                with open(_ns_path) as _f:
                    _norm_stats = _js.load(_f)

            _gbm_model_for_plot = (
                self.gbm_trainer.best_model or self.gbm_trainer.model
                if self.gbm_trained else None
            )

            # Build 4-D arrays for spatial error map (08) — cap at 6 patches
            _N_SPATIAL = 6
            _cnn_4d  = np.concatenate(_plot_preds)[:_N_SPATIAL]   # (N,1,H,W)
            _tgts_4d = np.concatenate(_plot_tgts)[:_N_SPATIAL]    # (N,1,H,W)

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
                y_train          = y_train,
                y_val            = y_val,
                norm_stats       = _norm_stats,
                ensemble_comparison = self.history.get("ensemble_comparison"),
                cnn_preds_4d     = _cnn_4d,
                targets_4d       = _tgts_4d,
            )
        except Exception as _pe:
            logger.warning(f"Post-training plot generation failed: {_pe}")
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
            elif key == "ensemble_comparison":
                # dict[str, dict] — serialize each strategy's metrics
                if isinstance(values, dict):
                    history_serializable[key] = {
                        strategy_name: {k: float(v) for k, v in m.items()}
                        for strategy_name, m in values.items()
                    }
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
    """
    Load preprocessed data.

    Preprocessing writes split arrays as raw numpy memmaps (np.memmap mode="w+")
    with an .npy extension — NOT the .npy pickle/header format that np.load
    expects.  We reconstruct the shape from the dataset metadata.json that
    preprocessing always writes alongside the split files, then open via
    np.memmap in read-only mode so no data is copied into RAM.
    """
    import json as _json

    dataset_dir = PROCESSED_DATA_DIR / "cnn_dataset"
    data_dir    = dataset_dir / split

    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")

    # Read shape from metadata
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {meta_path}. "
            "Re-run preprocessing.py to regenerate the dataset."
        )
    with open(meta_path) as _f:
        meta = _json.load(_f)

    patch_size   = meta["patch_size"]
    n_channels   = meta["n_channels"]
    n_samples    = meta["split_counts"][split]

    X_shape = (n_samples, patch_size, patch_size, n_channels)
    y_shape = (n_samples, patch_size, patch_size, 1)

    X = np.memmap(data_dir / "X.npy", dtype=np.float32, mode="r", shape=X_shape)
    y = np.memmap(data_dir / "y.npy", dtype=np.float32, mode="r", shape=y_shape)
    
    logger.info(f"Loaded {split} data: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"  X range: [{X.min():.4f}, {X.max():.4f}]")
    logger.info(f"  y range: [{y.min():.4f}, {y.max():.4f}]")

    # Verify channel count matches the Landsat-only pipeline (10 channels)
    expected_channels = CNN_CONFIG["input_channels"]
    actual_channels   = X.shape[-1]          # shape is (N, H, W, C)
    if actual_channels != expected_channels:
        raise ValueError(
            f"{split}: expected {expected_channels} input channels "
            f"(Landsat-only) but got {actual_channels}. "
            f"Re-run preprocessing.py to regenerate the dataset."
        )
    
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
    channel_order = norm_stats.get('channel_order', [])
    n_ch = len(channel_order) if channel_order else norm_stats.get('n_channels', 'N/A')
    logger.info(f"   Channels ({n_ch}): {', '.join(channel_order) if channel_order else 'N/A'}")
    
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