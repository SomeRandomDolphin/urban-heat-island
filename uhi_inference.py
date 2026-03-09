"""
FIXED: UHI Inference Pipeline - Expects NORMALIZED data (same as training)
"""
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Optional, List
import json
import pickle
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless/Windows runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import *
from models import UNet

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE DIAGNOSTIC PLOTTING MODULE
# ══════════════════════════════════════════════════════════════════════════════

class InferenceDiagnosticsPlotter:
    """
    Centralised matplotlib diagnostics for the UHI inference pipeline.

    Mirrors the DiagnosticsPlotter style from train_ensemble.py, but is focused
    on inference-time concerns: prediction distributions, spatial output maps,
    uncertainty visualisation, calibration analysis, TTA variance, and
    model agreement between CNN and GBM branches.

    All figures are saved under `save_dir`
    (default: MODEL_DIR / "inference_diagnostics").

    Usage::

        plotter = InferenceDiagnosticsPlotter()
        plotter.plot_all(results, X, targets=y_test)   # full suite
        # — or call individual plot_* methods as needed —
    """

    _STYLE_CANDIDATES = [
        "seaborn-v0_8-darkgrid",   # matplotlib ≥ 3.6
        "seaborn-darkgrid",        # matplotlib < 3.6
        "ggplot",
        "default",
    ]

    def __init__(self, save_dir: Path = None):
        self.save_dir = Path(save_dir) if save_dir else MODEL_DIR / "inference_diagnostics"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._style = self._resolve_style()
        logger.info(
            f"📊 InferenceDiagnosticsPlotter initialised — "
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

    def _save(self, fig, name: str):
        path = self.save_dir / f"{name}.png"
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"  ✅ Saved: {path.name}")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not save {path.name}: {e}")
        finally:
            plt.close(fig)

    # ── 1. Prediction distribution ────────────────────────────────────────────

    def plot_prediction_distribution(self, results: Dict[str, np.ndarray],
                                     targets: np.ndarray = None):
        """
        Histogram of CNN, GBM and ensemble predictions (°C), optionally
        overlaid with ground-truth targets.

        Args:
            results:  Output dict from EnsemblePredictor.predict_ensemble().
            targets:  Optional ground-truth array for comparison overlay.
        """
        try:
            self._use_style()
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle("Inference – Prediction Distributions (°C)",
                         fontsize=14, fontweight="bold")

            series = [
                ("CNN",      results.get("cnn_patch",      results.get("cnn")),      "#42A5F5"),
                ("GBM",      results.get("gbm"),                                       "#EF5350"),
                ("Ensemble", results.get("ensemble_patch", results.get("ensemble")), "#66BB6A"),
            ]

            for ax, (label, preds, color) in zip(axes, series):
                if preds is None:
                    ax.set_title(f"{label} (no data)"); continue
                flat = np.asarray(preds).flatten()
                flat = flat[np.isfinite(flat)]
                ax.hist(flat, bins=60, color=color, alpha=0.75,
                        density=True, edgecolor="white", linewidth=0.3,
                        label=label)
                if targets is not None:
                    tgt = np.asarray(targets).flatten()
                    tgt = tgt[np.isfinite(tgt)]
                    ax.hist(tgt, bins=60, color="gray", alpha=0.45,
                            density=True, edgecolor="white", linewidth=0.3,
                            label="Target")
                ax.axvline(float(np.nanmean(flat)), color=color,
                           ls="--", lw=1.2, label=f"Mean {np.nanmean(flat):.1f}°C")
                ax.set_xlabel("LST (°C)"); ax.set_ylabel("Density")
                ax.set_title(label); ax.legend(fontsize=8)
                ax.text(0.97, 0.97,
                        f"mean={np.nanmean(flat):.2f}°C\nstd={np.nanstd(flat):.2f}°C\n"
                        f"[{flat.min():.1f}, {flat.max():.1f}]",
                        transform=ax.transAxes, fontsize=8,
                        ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            plt.tight_layout()
            self._save(fig, "01_prediction_distribution")
        except Exception as e:
            logger.warning(f"plot_prediction_distribution failed: {e}")

    # ── 2. Predicted vs Actual (when ground truth is available) ──────────────

    def plot_pred_vs_actual(self, preds: np.ndarray, targets: np.ndarray,
                            label: str = "Ensemble", metrics: dict = None):
        """
        Scatter + residual plot — identical contract to DiagnosticsPlotter.plot_pred_vs_actual
        so the same call-site pattern works for both training and inference evaluation.
        """
        try:
            from scipy.stats import linregress
            self._use_style()

            preds   = np.asarray(preds).flatten()
            targets = np.asarray(targets).flatten()
            mask    = np.isfinite(preds) & np.isfinite(targets)
            preds, targets = preds[mask], targets[mask]

            if len(preds) < 2:
                logger.warning(f"plot_pred_vs_actual: only {len(preds)} finite samples, skipping")
                return

            slope, intercept, r_val, *_ = linregress(targets, preds)
            residuals = preds - targets

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"Inference – {label}: Predicted vs Actual",
                         fontsize=14, fontweight="bold")

            ax = axes[0]
            sc = ax.scatter(targets, preds, alpha=0.35, s=12,
                            c=np.abs(residuals), cmap="RdYlGn_r", label="Samples")
            plt.colorbar(sc, ax=ax, label="|Residual| (°C)")
            lo = min(targets.min(), preds.min()); hi = max(targets.max(), preds.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect (slope=1)")
            fit_x = np.linspace(lo, hi, 200)
            ax.plot(fit_x, slope * fit_x + intercept, "r-", lw=1.5,
                    label=f"Fit slope={slope:.3f}")
            ax.set_xlabel("Actual (°C)"); ax.set_ylabel("Predicted (°C)")
            ax.legend(fontsize=8)
            info = [f"R²={r_val**2:.4f}",
                    f"RMSE={np.sqrt(np.mean(residuals**2)):.3f}°C",
                    f"MAE={np.mean(np.abs(residuals)):.3f}°C",
                    f"slope={slope:.3f}", f"intercept={intercept:.3f}"]
            if metrics:
                info += [f"std_ratio={metrics.get('std_ratio', float('nan')):.3f}",
                         f"MBE={metrics.get('mbe', float('nan')):.3f}°C"]
            ax.text(0.03, 0.97, "\n".join(info), transform=ax.transAxes,
                    fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

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

    # ── 3. Spatial output maps ────────────────────────────────────────────────

    def plot_spatial_predictions(self, results: Dict[str, np.ndarray],
                                 targets: np.ndarray = None,
                                 n_samples: int = 4):
        """
        Side-by-side spatial maps: CNN · GBM (broadcast) · Ensemble · Error (if targets given).

        An additional figure (03b) shows the full mosaicked ensemble prediction
        across all patches, giving a whole-area view of the inference result.

        Args:
            results:   Output dict from predict_ensemble().
            targets:   Optional (N, H, W) ground-truth for error column.
            n_samples: How many patches to display (rows) in the per-patch figure.
        """
        try:
            self._use_style()

            ensemble = np.asarray(results.get("ensemble", results.get("ensemble_patch")))
            cnn      = np.asarray(results.get("cnn"))
            gbm      = np.asarray(results.get("gbm"))

            # Ensure spatial shape (N, H, W)
            def _to_spatial(arr):
                arr = np.asarray(arr)
                if arr.ndim == 4 and arr.shape[1] == 1:
                    return arr[:, 0]
                if arr.ndim == 4 and arr.shape[3] == 1:
                    return arr[:, :, :, 0]
                if arr.ndim == 3:
                    return arr
                return arr

            ensemble = _to_spatial(ensemble)
            cnn      = _to_spatial(cnn)

            n = min(n_samples, len(ensemble))
            if n == 0:
                logger.warning("plot_spatial_predictions: no samples, skipping"); return

            has_targets = targets is not None
            n_cols = 4 if has_targets else 3
            col_titles = ["CNN", "GBM (patch avg)", "Ensemble",
                          "Error (Ensemble−Target)"][:n_cols]

            fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 3 * n))
            if n == 1:
                axes = axes[np.newaxis, :]
            fig.suptitle(f"Inference – Spatial Predictions (first {n} patches)",
                         fontsize=13, fontweight="bold")

            vmin = ensemble[:n].min(); vmax = ensemble[:n].max()

            for i in range(n):
                # CNN
                axes[i, 0].imshow(cnn[i], vmin=vmin, vmax=vmax, cmap="hot")
                axes[i, 0].set_title(f"S{i+1} – {col_titles[0]}", fontsize=8)
                axes[i, 0].axis("off")

                # GBM broadcast
                gbm_val = float(gbm[i]) if gbm.ndim == 1 else float(gbm[i].mean())
                gbm_patch = np.full_like(cnn[i], gbm_val)
                axes[i, 1].imshow(gbm_patch, vmin=vmin, vmax=vmax, cmap="hot")
                axes[i, 1].set_title(f"S{i+1} – {col_titles[1]}\n{gbm_val:.1f}°C",
                                     fontsize=8)
                axes[i, 1].axis("off")

                # Ensemble
                axes[i, 2].imshow(ensemble[i], vmin=vmin, vmax=vmax, cmap="hot")
                axes[i, 2].set_title(f"S{i+1} – {col_titles[2]}", fontsize=8)
                axes[i, 2].axis("off")

                # Error
                if has_targets:
                    tgt = _to_spatial(np.asarray(targets))
                    err = ensemble[i] - tgt[i]
                    err_abs = max(float(np.abs(err).max()), 1e-6)
                    norm_err = TwoSlopeNorm(vmin=-err_abs, vcenter=0, vmax=err_abs)
                    im = axes[i, 3].imshow(err, norm=norm_err, cmap="RdBu_r")
                    axes[i, 3].set_title(
                        f"S{i+1} – {col_titles[3]}\nRMSE={np.sqrt(np.mean(err**2)):.2f}°C",
                        fontsize=8)
                    axes[i, 3].axis("off")
                    plt.colorbar(im, ax=axes[i, 3], shrink=0.8)

            plt.tight_layout()
            self._save(fig, "03_spatial_predictions")

            # ── 03b: Full mosaicked map across ALL patches ────────────────────
            self._plot_full_mosaic(ensemble, cnn, gbm, targets)

        except Exception as e:
            logger.warning(f"plot_spatial_predictions failed: {e}")

    def _mosaic_patches(self, patches: np.ndarray) -> np.ndarray:
        """Tile (N, H, W) patches into a single (rows*H, cols*W) mosaic."""
        N, H, W = patches.shape
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))
        padded = np.full((rows * cols, H, W), np.nan, dtype=patches.dtype)
        padded[:N] = patches
        return padded.reshape(rows, cols, H, W).transpose(0, 2, 1, 3).reshape(rows * H, cols * W)

    def _plot_full_mosaic(self, ensemble: np.ndarray, cnn: np.ndarray,
                          gbm: np.ndarray, targets: np.ndarray = None):
        """
        Figure 03b — full mosaicked inference result covering the entire study area.

        Shows: CNN mosaic · GBM mosaic · Ensemble mosaic · Error mosaic (if targets given).
        """
        try:
            N_total = len(ensemble)
            logger.info(f"  Mosaicking {N_total} patches for full-area map …")

            ens_mosaic = self._mosaic_patches(ensemble)
            cnn_mosaic = self._mosaic_patches(cnn)

            # GBM: broadcast per-patch scalar or per-patch spatial map
            H, W = ensemble.shape[1], ensemble.shape[2]
            if gbm.ndim == 1:
                gbm_tiles = np.stack([np.full((H, W), float(v)) for v in gbm])
            else:
                gbm_tiles = gbm
            gbm_mosaic = self._mosaic_patches(gbm_tiles)

            has_targets = targets is not None
            n_cols = 4 if has_targets else 3
            col_titles = ["CNN", "GBM", "Ensemble", "Error (Ens − Target)"][:n_cols]

            fig_w = max(10, 5 * n_cols)
            fig_h = max(6, int(fig_w * ens_mosaic.shape[0] / (ens_mosaic.shape[1] * n_cols)) + 2)
            fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
            if n_cols == 1:
                axes = [axes]

            vmin = float(np.nanmin(ens_mosaic)); vmax = float(np.nanmax(ens_mosaic))

            for ax, data, title in zip(axes[:3], [cnn_mosaic, gbm_mosaic, ens_mosaic],
                                       col_titles[:3]):
                im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="hot",
                               interpolation="bilinear")
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.axis("off")
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label("LST (°C)", fontsize=8)
                # Annotate stats
                ax.text(0.02, 0.02,
                        f"mean={np.nanmean(data):.1f}°C\nstd={np.nanstd(data):.2f}°C\n"
                        f"[{np.nanmin(data):.1f}, {np.nanmax(data):.1f}]",
                        transform=ax.transAxes, fontsize=7,
                        color="white", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

            if has_targets:
                tgt_arr = np.asarray(targets)
                if tgt_arr.ndim == 4 and tgt_arr.shape[1] == 1:
                    tgt_arr = tgt_arr[:, 0]
                tgt_mosaic = self._mosaic_patches(tgt_arr)
                err_mosaic = ens_mosaic - tgt_mosaic
                err_abs = max(float(np.nanmax(np.abs(err_mosaic))), 1e-6)
                norm_err = TwoSlopeNorm(vmin=-err_abs, vcenter=0, vmax=err_abs)
                im_err = axes[3].imshow(err_mosaic, norm=norm_err, cmap="RdBu_r",
                                        interpolation="bilinear")
                axes[3].set_title(col_titles[3], fontsize=10, fontweight="bold")
                axes[3].axis("off")
                cb_err = plt.colorbar(im_err, ax=axes[3], fraction=0.046, pad=0.04)
                cb_err.set_label("Error (°C)", fontsize=8)
                rmse_all = float(np.sqrt(np.nanmean(err_mosaic ** 2)))
                axes[3].text(0.02, 0.02, f"RMSE={rmse_all:.2f}°C",
                             transform=axes[3].transAxes, fontsize=7,
                             color="white", va="bottom",
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

            cols_grid = int(np.ceil(np.sqrt(N_total)))
            rows_grid = int(np.ceil(N_total / cols_grid))
            fig.suptitle(
                f"Inference – Full Study-Area Mosaic  "
                f"({N_total} patches → {rows_grid}×{cols_grid} grid, "
                f"{ens_mosaic.shape[0]}×{ens_mosaic.shape[1]} px)",
                fontsize=12, fontweight="bold")
            plt.tight_layout()
            self._save(fig, "03b_full_mosaic_predictions")

        except Exception as e:
            logger.warning(f"_plot_full_mosaic failed: {e}")

    # ── 4. Uncertainty maps ───────────────────────────────────────────────────

    def plot_uncertainty_maps(self, results: Dict[str, np.ndarray],
                              n_samples: int = 4):
        """
        Visualise MC Dropout and/or TTA uncertainty alongside the ensemble prediction.

        Args:
            results:   Output dict containing 'ensemble', 'mc_uncertainty',
                       'tta_uncertainty', or 'combined_uncertainty'.
            n_samples: Patches to show.
        """
        try:
            unc_keys = [k for k in
                        ["mc_uncertainty", "tta_uncertainty", "combined_uncertainty"]
                        if k in results]
            if not unc_keys:
                logger.info("plot_uncertainty_maps: no uncertainty arrays found, skipping")
                return

            self._use_style()
            # Always prefer the full spatial array (N,H,W); fall back to ensemble_patch only
            # as a last resort — a 1D patch array produces a solid-colour imshow tile.
            ensemble_raw = results.get("ensemble")
            if ensemble_raw is None:
                ensemble_raw = results.get("ensemble_patch")

            def _to_spatial(arr):
                arr = np.asarray(arr)
                if arr.ndim == 4 and arr.shape[1] == 1:
                    return arr[:, 0]
                if arr.ndim == 4 and arr.shape[3] == 1:
                    return arr[:, :, :, 0]
                if arr.ndim == 3:
                    return arr
                # 1-D patch-level array — cannot display spatially
                return None

            ensemble = _to_spatial(ensemble_raw)
            if ensemble is None:
                logger.warning("plot_uncertainty_maps: no spatial ensemble array available; "
                               "skipping ensemble column")
                # Still show uncertainty maps without the ensemble column
                unc_keys_use = unc_keys
                n_cols = len(unc_keys_use)
                show_ensemble_col = False
            else:
                show_ensemble_col = True
                n_cols = 1 + len(unc_keys)

            n = min(n_samples, len(ensemble) if ensemble is not None else n_samples)
            fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 3 * n))
            if n == 0:
                return

            fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 3 * n))
            if n == 1:
                axes = axes[np.newaxis, :]
            fig.suptitle(f"Inference – Prediction Uncertainty (first {n} patches)",
                         fontsize=13, fontweight="bold")

            for i in range(n):
                unc_start_col = 0  # column offset for uncertainty panels
                if show_ensemble_col:
                    vmin = ensemble[:n].min(); vmax = ensemble[:n].max()
                    axes[i, 0].imshow(ensemble[i], vmin=vmin, vmax=vmax, cmap="hot")
                    axes[i, 0].set_title(f"S{i+1} – Ensemble", fontsize=8)
                    axes[i, 0].axis("off")
                    unc_start_col = 1

                for col_idx, key in enumerate(unc_keys, start=unc_start_col):
                    unc_arr = _to_spatial(np.asarray(results[key]))
                    if unc_arr is None:
                        axes[i, col_idx].set_title(f"S{i+1} – {key}\n(no spatial data)", fontsize=8)
                        axes[i, col_idx].axis("off")
                        continue
                    if unc_arr.ndim == 2:   # single-sample flat map; pad
                        unc_arr = unc_arr[np.newaxis, :]
                    unc_i = unc_arr[i] if i < len(unc_arr) else unc_arr[0]
                    im = axes[i, col_idx].imshow(unc_i, cmap="viridis")
                    plt.colorbar(im, ax=axes[i, col_idx], shrink=0.8, label="σ (°C)")
                    title = key.replace("_", " ").title()
                    axes[i, col_idx].set_title(
                        f"S{i+1} – {title}\nμ={unc_i.mean():.2f}°C", fontsize=8)
                    axes[i, col_idx].axis("off")

            plt.tight_layout()
            self._save(fig, "04_uncertainty_maps")
        except Exception as e:
            logger.warning(f"plot_uncertainty_maps failed: {e}")

    # ── 5. CNN vs GBM agreement ───────────────────────────────────────────────

    def plot_model_agreement(self, results: Dict[str, np.ndarray]):
        """
        Scatter of CNN patch-mean vs GBM predictions, coloured by ensemble output,
        plus a bias histogram showing (CNN − GBM) disagreement.

        Args:
            results: Output dict from predict_ensemble().
        """
        try:
            self._use_style()

            cnn_patch = results.get("cnn_patch")
            gbm       = results.get("gbm")
            ensemble  = results.get("ensemble_patch", results.get("ensemble"))

            if cnn_patch is None or gbm is None:
                logger.warning("plot_model_agreement: cnn_patch or gbm missing, skipping")
                return

            cnn_flat = np.asarray(cnn_patch).flatten()
            gbm_flat = np.asarray(gbm).flatten()
            ens_flat = np.asarray(ensemble).flatten() if ensemble is not None else None

            mask = np.isfinite(cnn_flat) & np.isfinite(gbm_flat)
            cnn_flat = cnn_flat[mask]; gbm_flat = gbm_flat[mask]
            if ens_flat is not None:
                ens_flat = ens_flat[mask[:len(ens_flat)]]

            diff = cnn_flat - gbm_flat

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("Inference – CNN vs GBM Agreement",
                         fontsize=14, fontweight="bold")

            ax = axes[0]
            c_vals = ens_flat if ens_flat is not None else diff
            sc = ax.scatter(gbm_flat, cnn_flat, c=c_vals, cmap="plasma",
                            alpha=0.45, s=14)
            plt.colorbar(sc, ax=ax,
                         label="Ensemble (°C)" if ens_flat is not None else "CNN−GBM (°C)")
            lo = min(gbm_flat.min(), cnn_flat.min())
            hi = max(gbm_flat.max(), cnn_flat.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect agreement")
            ax.set_xlabel("GBM prediction (°C)"); ax.set_ylabel("CNN prediction (°C)")
            ax.set_title("CNN vs GBM (patch averages)"); ax.legend(fontsize=8)
            ax.text(0.03, 0.97,
                    f"mean diff={diff.mean():.3f}°C\nstd diff={diff.std():.3f}°C\n"
                    f"n={len(cnn_flat):,}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            ax = axes[1]
            ax.hist(diff, bins=60, color="#AB47BC", alpha=0.75,
                    density=True, edgecolor="white", linewidth=0.3)
            ax.axvline(0, color="black", lw=1.2, ls="--", label="Zero bias")
            ax.axvline(diff.mean(), color="#EF5350", lw=1.5,
                       label=f"Mean={diff.mean():.3f}°C")
            ax.set_xlabel("CNN − GBM (°C)"); ax.set_ylabel("Density")
            ax.set_title("Model Disagreement Distribution"); ax.legend(fontsize=8)

            plt.tight_layout()
            self._save(fig, "05_model_agreement")
        except Exception as e:
            logger.warning(f"plot_model_agreement failed: {e}")

    # ── 6. Calibration analysis ───────────────────────────────────────────────

    def plot_calibration_analysis(self, results: Dict[str, np.ndarray],
                                  targets: np.ndarray,
                                  cal_slope: float = 1.0,
                                  cal_intercept: float = 0.0):
        """
        Show the effect of post-hoc linear calibration: scatter of raw CNN vs
        calibrated CNN predictions against targets, plus a bias-vs-temperature plot.

        Args:
            results:       predict_ensemble() output dict.
            targets:       Ground-truth LST patches (°C, already denormalised).
            cal_slope:     Calibration slope applied at inference time.
            cal_intercept: Calibration intercept applied at inference time.
        """
        try:
            self._use_style()

            cnn = results.get("cnn")
            ens = results.get("ensemble_patch", results.get("ensemble"))
            if cnn is None:
                logger.warning("plot_calibration_analysis: CNN predictions missing, skipping")
                return

            cnn_flat = np.asarray(cnn).flatten()
            ens_flat = np.asarray(ens).flatten() if ens is not None else cnn_flat
            tgt_flat = np.asarray(targets).flatten()

            n = min(len(cnn_flat), len(tgt_flat))
            cnn_flat = cnn_flat[:n]; ens_flat = ens_flat[:n]; tgt_flat = tgt_flat[:n]
            mask = np.isfinite(cnn_flat) & np.isfinite(tgt_flat)
            cnn_flat = cnn_flat[mask]; ens_flat = ens_flat[mask]; tgt_flat = tgt_flat[mask]

            # Reconstruct raw (pre-calibration) CNN values in °C
            # raw = (calibrated − intercept) / slope  [both in normalised space,
            # but denorm is linear so the ratio still holds]
            raw_flat = (cnn_flat - cal_intercept) / (cal_slope + 1e-8)

            from scipy.stats import linregress
            slope_raw, _, _, *_ = linregress(tgt_flat, raw_flat)
            slope_cal, _, _, *_ = linregress(tgt_flat, cnn_flat)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("Inference – Calibration Effect Analysis",
                         fontsize=14, fontweight="bold")

            ax = axes[0]
            lo = min(tgt_flat.min(), cnn_flat.min()); hi = max(tgt_flat.max(), cnn_flat.max())
            ax.scatter(tgt_flat, raw_flat, alpha=0.3, s=10,
                       color="#90CAF9", label=f"Pre-cal (slope={slope_raw:.3f})")
            ax.scatter(tgt_flat, cnn_flat, alpha=0.3, s=10,
                       color="#EF5350", label=f"Post-cal (slope={slope_cal:.3f})")
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect (slope=1)")
            ax.set_xlabel("Actual (°C)"); ax.set_ylabel("CNN Predicted (°C)")
            ax.set_title("Pre vs Post Calibration Scatter"); ax.legend(fontsize=8)

            ax.text(0.03, 0.97,
                    f"cal_slope={cal_slope:.4f}\ncal_intercept={cal_intercept:.4f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            ax = axes[1]
            residuals_raw = raw_flat - tgt_flat
            residuals_cal = cnn_flat - tgt_flat
            ax.scatter(tgt_flat, residuals_raw, alpha=0.25, s=10,
                       color="#90CAF9", label="Pre-cal residual")
            ax.scatter(tgt_flat, residuals_cal, alpha=0.25, s=10,
                       color="#EF5350", label="Post-cal residual")
            ax.axhline(0, color="black", lw=1.2)
            ax.set_xlabel("Actual (°C)"); ax.set_ylabel("Residual (°C)")
            ax.set_title("Residuals Before and After Calibration"); ax.legend(fontsize=8)

            plt.tight_layout()
            self._save(fig, "06_calibration_analysis")
        except Exception as e:
            logger.warning(f"plot_calibration_analysis failed: {e}")

    # ── 7. Stratified error (inference) ──────────────────────────────────────

    def plot_stratified_error(self, preds: np.ndarray, targets: np.ndarray,
                              label: str = "Ensemble", n_bins: int = 10):
        """
        RMSE, MAE, MBE bucketed by temperature-percentile bins — identical
        contract to DiagnosticsPlotter.plot_stratified_error.
        """
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

            if not centers:
                logger.warning("plot_stratified_error: no bins with enough samples, skipping")
                return

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(f"Inference – {label}: Stratified Error by Temperature",
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
            self._save(fig, f"07_stratified_error_{safe}")
        except Exception as e:
            logger.warning(f"plot_stratified_error failed: {e}")

    # ── 8. TTA variance analysis ──────────────────────────────────────────────

    def plot_tta_variance(self, results: Dict[str, np.ndarray]):
        """
        Visualise TTA variance: histogram of per-pixel TTA std (°C),
        scatter of uncertainty vs ensemble prediction, and a simple reliability
        diagram (uncertainty magnitude vs actual error, if targets available).

        Args:
            results: Output dict containing at least 'tta_uncertainty' and 'ensemble'.
        """
        try:
            tta_unc = results.get("tta_uncertainty")
            if tta_unc is None:
                logger.info("plot_tta_variance: no tta_uncertainty in results, skipping")
                return

            self._use_style()
            unc_flat = np.asarray(tta_unc).flatten()
            unc_flat = unc_flat[np.isfinite(unc_flat)]

            ens_flat = np.asarray(
                results.get("ensemble", results.get("ensemble_patch"))
            ).flatten()
            n = min(len(unc_flat), len(ens_flat))
            unc_flat = unc_flat[:n]; ens_flat = ens_flat[:n]
            mask = np.isfinite(unc_flat) & np.isfinite(ens_flat)
            unc_flat = unc_flat[mask]; ens_flat = ens_flat[mask]

            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle("Inference – TTA Variance Analysis",
                         fontsize=14, fontweight="bold")

            ax = axes[0]
            ax.hist(unc_flat, bins=60, color="#26C6DA", alpha=0.75,
                    density=True, edgecolor="white", linewidth=0.3)
            ax.axvline(unc_flat.mean(), color="#EF5350", lw=1.5, ls="--",
                       label=f"Mean={unc_flat.mean():.3f}°C")
            ax.set_xlabel("TTA Std (°C)"); ax.set_ylabel("Density")
            ax.set_title("TTA Uncertainty Distribution"); ax.legend(fontsize=8)
            ax.text(0.97, 0.97,
                    f"median={np.median(unc_flat):.3f}°C\n"
                    f"p95={np.percentile(unc_flat, 95):.3f}°C",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))

            ax = axes[1]
            ax.hexbin(ens_flat, unc_flat, gridsize=40, cmap="YlOrRd",
                      mincnt=1, linewidths=0.2)
            ax.set_xlabel("Ensemble Prediction (°C)")
            ax.set_ylabel("TTA Uncertainty (°C)")
            ax.set_title("Uncertainty vs Prediction Temperature")

            plt.tight_layout()
            self._save(fig, "08_tta_variance")
        except Exception as e:
            logger.warning(f"plot_tta_variance failed: {e}")

    # ── 9. Summary dashboard ──────────────────────────────────────────────────

    def plot_summary_dashboard(self, results: Dict[str, np.ndarray],
                               targets: np.ndarray = None,
                               cal_slope: float = 1.0,
                               cal_intercept: float = 0.0,
                               weights: Dict = None):
        """
        One-page overview: prediction stats table · ensemble weight pie ·
        CNN/GBM/Ensemble distribution · agreement scatter · uncertainty overview.

        Args:
            results:       predict_ensemble() output dict (Celsius).
            targets:       Optional ground-truth patches (°C).
            cal_slope:     Calibration slope loaded by EnsemblePredictor.
            cal_intercept: Calibration intercept loaded by EnsemblePredictor.
            weights:       {"cnn": float, "gbm": float} ensemble weights.
        """
        try:
            self._use_style()
            fig = plt.figure(figsize=(18, 12))
            gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)
            fig.suptitle("Inference – Summary Dashboard",
                         fontsize=16, fontweight="bold")

            # ── Row 0: metrics table + weights pie + calibration info ─────────
            ax_tbl = fig.add_subplot(gs[0, :2]); ax_tbl.axis("off")

            def _stats(arr):
                a = np.asarray(arr).flatten()
                a = a[np.isfinite(a)]
                return (f"{a.mean():.2f}", f"{a.std():.2f}",
                        f"{a.min():.2f}", f"{a.max():.2f}") if len(a) else ("—",)*4

            series_names = ["CNN (patch)", "GBM", "Ensemble (patch)"]
            series_data  = [
                results.get("cnn_patch", results.get("cnn")),
                results.get("gbm"),
                results.get("ensemble_patch", results.get("ensemble")),
            ]
            rows = [["Model", "Mean (°C)", "Std (°C)", "Min (°C)", "Max (°C)"]]
            for name, data in zip(series_names, series_data):
                if data is not None:
                    rows.append([name] + list(_stats(data)))
                else:
                    rows.append([name, "—", "—", "—", "—"])

            # Optional: add accuracy row if targets available
            if targets is not None:
                ens = results.get("ensemble_patch", results.get("ensemble"))
                if ens is not None:
                    e = np.asarray(ens).flatten()
                    t = np.asarray(targets).flatten()
                    n = min(len(e), len(t))
                    e, t = e[:n], t[:n]
                    mask = np.isfinite(e) & np.isfinite(t)
                    if mask.sum() >= 2:
                        from sklearn.metrics import r2_score as _r2
                        rmse = float(np.sqrt(np.mean((e[mask] - t[mask])**2)))
                        r2   = float(_r2(t[mask], e[mask]))
                        rows.append(["Ensemble Accuracy",
                                     f"R²={r2:.4f}", f"RMSE={rmse:.3f}°C",
                                     "—", "—"])

            tbl = ax_tbl.table(cellText=rows[1:], colLabels=rows[0],
                               cellLoc="center", loc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
            ax_tbl.set_title("Inference Prediction Statistics", fontsize=10, pad=4)

            # Weights pie
            ax_pie = fig.add_subplot(gs[0, 2])
            if weights:
                wts = [weights.get("cnn", 0), weights.get("gbm", 0)]
                if sum(wts) > 0:
                    ax_pie.pie(wts, labels=["CNN", "GBM"], autopct="%1.1f%%",
                               colors=["#42A5F5", "#EF5350"], startangle=90)
            ax_pie.set_title("Ensemble Weights", fontsize=10)

            # Calibration info box
            ax_cal = fig.add_subplot(gs[0, 3]); ax_cal.axis("off")
            cal_text = (
                "Post-hoc Calibration\n"
                "─────────────────\n"
                f"slope      = {cal_slope:.4f}\n"
                f"intercept = {cal_intercept:.4f}\n\n"
                f"{'✅ Applied' if (cal_slope != 1.0 or cal_intercept != 0.0) else '⚠️ Identity (not applied)'}"
            )
            ax_cal.text(0.5, 0.5, cal_text, transform=ax_cal.transAxes,
                        fontsize=9, va="center", ha="center",
                        fontfamily="monospace",
                        bbox=dict(boxstyle="round,pad=0.5", alpha=0.12))
            ax_cal.set_title("Calibration", fontsize=10)

            # ── Row 1: overlaid distribution + CNN vs GBM scatter ─────────────
            ax_dist = fig.add_subplot(gs[1, :2])
            colours = {"CNN (patch)": "#42A5F5", "GBM": "#EF5350",
                       "Ensemble": "#66BB6A", "Target": "gray"}
            plot_pairs = [
                ("CNN (patch)",  results.get("cnn_patch", results.get("cnn")),  "#42A5F5"),
                ("GBM",          results.get("gbm"),                             "#EF5350"),
                ("Ensemble",     results.get("ensemble_patch",
                                             results.get("ensemble")),           "#66BB6A"),
            ]
            if targets is not None:
                plot_pairs.append(("Target", targets, "gray"))
            for (lbl, arr, col) in plot_pairs:
                if arr is None: continue
                flat = np.asarray(arr).flatten()
                flat = flat[np.isfinite(flat)]
                ax_dist.hist(flat, bins=60, color=col, alpha=0.45,
                             density=True, label=lbl, edgecolor="white", linewidth=0.2)
            ax_dist.set_xlabel("LST (°C)"); ax_dist.set_ylabel("Density")
            ax_dist.set_title("Overlaid Prediction Distributions")
            ax_dist.legend(fontsize=8)

            # CNN vs GBM agreement scatter
            ax_ag = fig.add_subplot(gs[1, 2:])
            cnn_p = results.get("cnn_patch", results.get("cnn"))
            gbm_p = results.get("gbm")
            if cnn_p is not None and gbm_p is not None:
                cf = np.asarray(cnn_p).flatten()
                gf = np.asarray(gbm_p).flatten()
                n  = min(len(cf), len(gf))
                cf, gf = cf[:n], gf[:n]
                mask = np.isfinite(cf) & np.isfinite(gf)
                cf, gf = cf[mask], gf[mask]
                diff = cf - gf
                ax_ag.scatter(gf, cf, c=diff, cmap="RdBu_r", alpha=0.35, s=10)
                lo = min(gf.min(), cf.min()); hi = max(gf.max(), cf.max())
                ax_ag.plot([lo, hi], [lo, hi], "k--", lw=1)
                ax_ag.set_xlabel("GBM (°C)"); ax_ag.set_ylabel("CNN (°C)")
                ax_ag.set_title(f"CNN vs GBM  (mean diff={diff.mean():.3f}°C)")

            # ── Row 2: uncertainty overview + spatial sample ──────────────────
            unc_keys = [k for k in ["mc_uncertainty", "tta_uncertainty",
                                    "combined_uncertainty"] if k in results]
            if unc_keys:
                ax_unc = fig.add_subplot(gs[2, :2])
                for key in unc_keys:
                    uf = np.asarray(results[key]).flatten()
                    uf = uf[np.isfinite(uf)]
                    ax_unc.hist(uf, bins=60, alpha=0.55, density=True,
                                label=key.replace("_", " ").title(),
                                edgecolor="white", linewidth=0.2)
                ax_unc.set_xlabel("Uncertainty (°C)"); ax_unc.set_ylabel("Density")
                ax_unc.set_title("Uncertainty Distribution"); ax_unc.legend(fontsize=8)

            # Spatial sample of first patch
            ens_arr = results.get("ensemble")
            if ens_arr is not None:
                ens_arr = np.asarray(ens_arr)
                if ens_arr.ndim == 4 and ens_arr.shape[1] == 1:
                    ens_arr = ens_arr[:, 0]
                if ens_arr.ndim >= 3:
                    ax_sp = fig.add_subplot(gs[2, 2])
                    im = ax_sp.imshow(ens_arr[0], cmap="hot")
                    plt.colorbar(im, ax=ax_sp, label="°C", shrink=0.8)
                    ax_sp.set_title("Patch 0 – Ensemble LST", fontsize=9)
                    ax_sp.axis("off")

            if unc_keys:
                first_unc = np.asarray(results[unc_keys[0]])
                if first_unc.ndim == 4 and first_unc.shape[1] == 1:
                    first_unc = first_unc[:, 0]
                if first_unc.ndim >= 3:
                    ax_up = fig.add_subplot(gs[2, 3])
                    im2 = ax_up.imshow(first_unc[0], cmap="viridis")
                    plt.colorbar(im2, ax=ax_up, label="σ (°C)", shrink=0.8)
                    title = unc_keys[0].replace("_", " ").title()
                    ax_up.set_title(f"Patch 0 – {title}", fontsize=9)
                    ax_up.axis("off")

            self._save(fig, "09_summary_dashboard")
        except Exception as e:
            logger.warning(f"plot_summary_dashboard failed: {e}")

    # ── convenience: full suite ───────────────────────────────────────────────

    def plot_all(self, results: Dict[str, np.ndarray],
                 X: np.ndarray = None,
                 targets: np.ndarray = None,
                 cal_slope: float = 1.0,
                 cal_intercept: float = 0.0,
                 weights: Dict = None,
                 label: str = "Ensemble"):
        """
        Convenience wrapper — calls every available plot method.

        Args:
            results:       Output from EnsemblePredictor.predict_ensemble() or
                           predict_ensemble_with_tta() — values in Celsius.
            X:             Raw input patches (N, H, W, C) — used only for
                           context; not strictly required.
            targets:       Optional ground-truth LST patches (°C, denormalised).
            cal_slope:     Calibration slope (from EnsemblePredictor.cal_slope).
            cal_intercept: Calibration intercept.
            weights:       {"cnn": float, "gbm": float} from EnsemblePredictor.weights.
            label:         Label for scatter / residual plot titles.
        """
        logger.info("\n📊 Generating all inference diagnostic plots...")

        self.plot_prediction_distribution(results, targets=targets)
        self.plot_spatial_predictions(results, targets=targets)
        self.plot_model_agreement(results)
        self.plot_uncertainty_maps(results)
        self.plot_tta_variance(results)
        self.plot_summary_dashboard(results, targets=targets,
                                    cal_slope=cal_slope,
                                    cal_intercept=cal_intercept,
                                    weights=weights)

        if targets is not None:
            ens = results.get("ensemble_patch", results.get("ensemble"))
            if ens is not None:
                self.plot_pred_vs_actual(ens, targets, label=label)
                self.plot_stratified_error(ens, targets, label=label)
            if cal_slope != 1.0 or cal_intercept != 0.0:
                self.plot_calibration_analysis(results, targets,
                                               cal_slope=cal_slope,
                                               cal_intercept=cal_intercept)

        logger.info(f"✅ All inference plots saved to: {self.save_dir}")


class MCDropoutUNet(nn.Module):
    """UNet wrapper with properly implemented MC Dropout layers"""
    
    def __init__(self, base_model: UNet, dropout_rate: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self._add_dropout_layers()
        logger.info(f"✓ Added MC Dropout layers with rate={dropout_rate}")
        
    def _add_dropout_layers(self):
        """Recursively add dropout layers to the model"""
        def add_dropout_to_module(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                    setattr(module, name, nn.Sequential(
                        child,
                        nn.Dropout2d(p=self.dropout_rate)
                    ))
                else:
                    add_dropout_to_module(child)
        
        add_dropout_to_module(self.base_model)
    
    def forward(self, x):
        return self.base_model(x)
    
    def enable_mc_dropout(self):
        """Enable dropout for inference (MC Dropout)"""
        def enable_dropout(module):
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        
        self.eval()
        self.apply(enable_dropout)
    
    def disable_mc_dropout(self):
        """Disable dropout (standard inference)"""
        self.eval()


class DataNormalizer:
    """
    CRITICAL: Handles denormalization of predictions back to Celsius
    Inference now expects NORMALIZED inputs (same as training)
    """
    
    def __init__(self, stats_path: Path):
        """
        Load normalization statistics from training
        
        Args:
            stats_path: Path to normalization_stats.json
        """
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {stats_path}\n"
                f"These should be created during preprocessing."
            )
        
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        
        logger.info("✓ Loaded normalization statistics")
        logger.info(f"  Channels: {self.stats.get('n_channels', 'N/A')}")
        
        if 'target' in self.stats:
            logger.info(f"  Target LST: mean={self.stats['target']['mean']:.2f}°C, "
                       f"std={self.stats['target']['std']:.2f}°C")
    
    def denormalize_predictions(self, predictions: np.ndarray, 
                                prediction_type: str = "target") -> np.ndarray:
        """
        Denormalize predictions back to Celsius
        
        Args:
            predictions: Normalized predictions (mean≈0, std≈1)
            prediction_type: 'target' for CNN or 'gbm_target' for GBM
            
        Returns:
            Denormalized predictions in Celsius
        """
        if prediction_type not in self.stats:
            logger.warning(f"⚠️ '{prediction_type}' not in normalization stats, "
                          f"returning predictions as-is")
            return predictions
        
        stats = self.stats[prediction_type]
        mean = stats['mean']
        std = stats['std']
        
        denormalized = predictions * std + mean
        
        logger.debug(f"Denormalized {prediction_type}: "
                    f"mean={denormalized.mean():.2f}°C, "
                    f"std={denormalized.std():.2f}°C")
        
        return denormalized


class EnsemblePredictor:
    """
    FIXED: Ensemble predictor - Expects NORMALIZED input (same as training)
    Denormalization happens ONCE at the end, not multiple times
    """
    
    def __init__(self, 
                 cnn_model_path: Path, 
                 gbm_model_path: Path, 
                 ensemble_config_path: Path, 
                 normalization_stats_path: Path,
                 device: str = "cuda",
                 mc_dropout_rate: float = 0.1):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load normalization stats for denormalization
        self.normalizer = DataNormalizer(normalization_stats_path)
        
        # Load ensemble config
        with open(ensemble_config_path, 'r') as f:
            self.ensemble_config = json.load(f)
        
        self.weights = self.ensemble_config.get("weights", {"cnn": 0.5, "gbm": 0.5})
        
        logger.info(f"Ensemble weights - CNN: {self.weights['cnn']:.3f}, "
                   f"GBM: {self.weights['gbm']:.3f}")
        
        # Load base CNN model
        # Handle both checkpoint dicts (CheckpointManager) and plain state dicts (final_cnn.pth)
        logger.info("Loading CNN model...")
        base_model = UNet(in_channels=CNN_CONFIG["input_channels"], out_channels=1)
        _raw = torch.load(cnn_model_path, map_location=self.device, weights_only=False)
        if isinstance(_raw, dict) and "model_state_dict" in _raw:
            base_model.load_state_dict(_raw["model_state_dict"])
            logger.info(f"  Loaded checkpoint from epoch {_raw.get('epoch','?')}")
        else:
            base_model.load_state_dict(_raw)
            logger.info("  Loaded plain state dict")
        
        # Wrap with MC Dropout
        self.cnn_model = MCDropoutUNet(base_model, dropout_rate=mc_dropout_rate)
        self.cnn_model.to(self.device)
        logger.info("✅ CNN model loaded with MC Dropout support")
        
        # Load GBM model
        logger.info("Loading GBM model...")
        with open(gbm_model_path, 'rb') as f:
            self.gbm_model = pickle.load(f)
        logger.info("GBM model loaded")

        # Load post-hoc linear calibration (fitted on val set during training).
        # BUG FIX: train_ensemble.py saves calibration_params.json to save_dir (model_dir root),
        # but cnn_model_path may point to model_dir/checkpoints/best_r2.pth, making
        # Path(cnn_model_path).parent = model_dir/checkpoints — the wrong directory.
        # Fix: search the immediate parent first, then walk up one level.
        _cnn_parent = Path(cnn_model_path).parent
        _cal_candidates = [
            _cnn_parent / "calibration_params.json",          # same dir as model file
            _cnn_parent.parent / "calibration_params.json",   # one level up (model_dir root)
        ]
        _cal_path = next((p for p in _cal_candidates if p.exists()), None)

        if _cal_path is not None:
            with open(_cal_path, 'r') as _f:
                _cal = json.load(_f)
            self.cal_slope     = float(_cal.get("slope",     1.0))
            self.cal_intercept = float(_cal.get("intercept", 0.0))
            logger.info(f"Loaded calibration from {_cal_path}:")
            logger.info(f"  slope={self.cal_slope:.4f}, intercept={self.cal_intercept:.4f}")
        else:
            self.cal_slope, self.cal_intercept = 1.0, 0.0
            logger.warning(f"calibration_params.json not found (searched: "
                           f"{[str(p) for p in _cal_candidates]}). "
                           "Predictions will NOT be calibrated — re-run training to generate it.")

        # Load bottleneck PCA (fitted during training to compress 3*C_bot dims → 32).
        # Without this, _extract_cnn_bottleneck_features returns 3*256=768 columns,
        # which causes the "inference=890, training=154" feature-mismatch warning.
        _model_root = Path(cnn_model_path).parent
        _pca_candidates = [
            _model_root / "bottleneck_pca.pkl",
            _model_root.parent / "bottleneck_pca.pkl",
        ]
        _pca_path = next((p for p in _pca_candidates if p.exists()), None)
        if _pca_path is not None:
            try:
                import joblib
                self._bottleneck_pca = joblib.load(_pca_path)
                logger.info(f"✅ Loaded bottleneck PCA ({self._bottleneck_pca.n_components_} "
                            f"components) from {_pca_path}")
            except Exception as _pe:
                self._bottleneck_pca = None
                logger.warning(f"⚠️ Could not load bottleneck PCA ({_pe}); "
                               "bottleneck features will be passed raw and may cause a feature mismatch")
        else:
            self._bottleneck_pca = None
            logger.warning(
                "⚠️ bottleneck_pca.pkl not found — GBM feature count may not match training. "
                "To fix: re-run train_ensemble.py (the updated version now saves the PCA). "
                f"Searched: {[str(p) for p in _pca_candidates]}"
            )

        # Diagnostics plotter — saves all inference figures to MODEL_DIR/inference_diagnostics/
        self.plotter = InferenceDiagnosticsPlotter(
            save_dir=Path(cnn_model_path).parent.parent / "inference_diagnostics"
        )
    
    def validate_input(self, X: np.ndarray) -> bool:
        """Validate that input data is properly normalized"""
        logger.info("\n" + "="*70)
        logger.info("VALIDATING INPUT DATA")
        logger.info("="*70)
        
        X_mean = X.mean()
        X_std = X.std()
        
        logger.info(f"Input statistics:")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Mean: {X_mean:.4f}")
        logger.info(f"  Std:  {X_std:.4f}")
        logger.info(f"  Range: [{X.min():.4f}, {X.max():.4f}]")
        
        # Check if normalized
        is_normalized = (-0.5 < X_mean < 0.5) and (0.5 < X_std < 1.5)
        
        if is_normalized:
            logger.info("✅ Input data appears to be NORMALIZED (mean≈0, std≈1)")
            logger.info("  This is correct! Inference expects normalized data.")
            return True
        else:
            logger.warning("⚠️ Input data does NOT appear to be normalized!")
            logger.warning(f"  Mean={X_mean:.4f} (expected ≈0)")
            logger.warning(f"  Std={X_std:.4f} (expected ≈1)")
            return False
        
        logger.info("="*70 + "\n")
    
    def _predict_cnn_normalized(self, X: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """
        CNN predictions in NORMALIZED space
        
        Args:
            X: NORMALIZED features (N, H, W, C)
            
        Returns:
            Predictions in NORMALIZED space (DO NOT denormalize here)
        """
        logger.info("Running CNN predictions...")
        
        n_samples = X.shape[0]
        predictions = []
        
        self.cnn_model.disable_mc_dropout()
        
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), desc="CNN Inference"):
                batch = X[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).permute(0, 3, 1, 2).to(self.device)
                
                output = self.cnn_model(batch_tensor)
                predictions.append(output.cpu().numpy())
        
        # Concatenate predictions (KEEP NORMALIZED)
        predictions = np.concatenate(predictions, axis=0).squeeze(1)

        logger.info(f"  CNN raw predictions (normalized): "
                   f"mean={predictions.mean():.4f}, std={predictions.std():.4f}")

        # Apply post-hoc linear calibration in normalized space.
        # Corrects range compression (slope < 1) without retraining.
        # calibration_params.json is generated at the end of train_ensemble.py.
        if self.cal_slope != 1.0 or self.cal_intercept != 0.0:
            predictions = self.cal_slope * predictions + self.cal_intercept
            logger.info(f"  CNN calibrated (normalized): "
                       f"mean={predictions.mean():.4f}, std={predictions.std():.4f}")

        return predictions  # Return NORMALIZED (calibrated) predictions
    
    def _predict_gbm_normalized(self, X: np.ndarray) -> np.ndarray:
        """
        GBM predictions in NORMALIZED space.

        Updated to match the enriched feature schema from train_ensemble.py FIX 6+8:
          Original 7 stats:  mean, std, min, max, median, p25, p75
          FIX 8 texture:     range, skew, kurt, grad_mean, grad_max  (5 new per channel)
          FIX 6 bottleneck:  CNN encoder mean/std/max per bottleneck channel
          Plus:              height, width

        The GBM model's feature_name() list is used as the ground-truth column order,
        so inference is always aligned to whatever was present at training time.
        """
        logger.info("Running GBM predictions (enriched feature schema)...")

        n_samples, height, width, n_channels = X.shape
        features_list = []

        for i in tqdm(range(n_samples), desc="Extracting GBM features"):
            patch_features = {}

            for ch in range(n_channels):
                channel_data = X[i, :, :, ch]
                flat = channel_data.flatten()

                # ── Original 7 stats ───────────────────────────────────────────
                patch_features[f'ch{ch}_mean']   = flat.mean()
                patch_features[f'ch{ch}_std']    = flat.std()
                patch_features[f'ch{ch}_min']    = flat.min()
                patch_features[f'ch{ch}_max']    = flat.max()
                patch_features[f'ch{ch}_median'] = np.median(flat)
                patch_features[f'ch{ch}_p25']    = np.percentile(flat, 25)
                patch_features[f'ch{ch}_p75']    = np.percentile(flat, 75)

                # ── FIX 8: Texture features ────────────────────────────────────
                patch_features[f'ch{ch}_range'] = flat.max() - flat.min()

                centered = flat - flat.mean()
                std_c    = flat.std() + 1e-8
                patch_features[f'ch{ch}_skew'] = float((centered ** 3).mean() / std_c ** 3)
                patch_features[f'ch{ch}_kurt'] = float((centered ** 4).mean() / std_c ** 4 - 3)

                # Gradient magnitude (edge strength)
                dx = np.diff(channel_data, axis=1)
                dy = np.diff(channel_data, axis=0)
                h_min = min(dx.shape[0], dy.shape[0])
                w_min = min(dx.shape[1], dy.shape[1])
                grad_mag = np.sqrt(dx[:h_min, :w_min] ** 2 + dy[:h_min, :w_min] ** 2)
                patch_features[f'ch{ch}_grad_mean'] = grad_mag.mean()
                patch_features[f'ch{ch}_grad_max']  = grad_mag.max()

            patch_features['height'] = height
            patch_features['width']  = width
            features_list.append(patch_features)

        features_df = pd.DataFrame(features_list)

        # ── FIX 6: Append CNN bottleneck features ──────────────────────────────
        try:
            bot_feats = self._extract_cnn_bottleneck_features(X)
            bot_feats = self._apply_bottleneck_pca(bot_feats)   # compress to training dims
            bot_cols  = [f"cnn_bot_{i}" for i in range(bot_feats.shape[1])]
            bot_df    = pd.DataFrame(bot_feats, columns=bot_cols)
            features_df = pd.concat([features_df, bot_df], axis=1)
            logger.info(f"  + {bot_feats.shape[1]} CNN bottleneck features "
                        f"→ {features_df.shape[1]} total")
        except Exception as _e:
            logger.warning(f"  ⚠️ CNN bottleneck extraction skipped ({_e}) — "
                           "proceeding without bottleneck features")

        # ── Align columns to what the GBM was trained on ───────────────────────
        train_cols = self.gbm_model.feature_name()
        infer_n    = features_df.shape[1]
        train_n    = len(train_cols)

        if infer_n != train_n or list(features_df.columns) != train_cols:
            logger.warning(f"  ⚠️ Feature mismatch: inference={infer_n}, "
                           f"training={train_n}. Aligning to training column list...")
            # Add any missing columns as zeros
            missing = [c for c in train_cols if c not in features_df.columns]
            if missing:
                logger.warning(f"    Adding {len(missing)} missing columns as 0 "
                               f"(e.g. {missing[:3]})")
                for col in missing:
                    features_df[col] = 0.0
            # Drop extra columns and reorder
            extra = [c for c in features_df.columns if c not in train_cols]
            if extra:
                logger.warning(f"    Dropping {len(extra)} extra columns "
                               f"(e.g. {extra[:3]})")
            features_df = features_df[train_cols]
            logger.info(f"  ✅ Aligned to {len(train_cols)} training features")

        # GBM prediction (KEEP NORMALIZED)
        predictions = self.gbm_model.predict(
            features_df,
            num_iteration=self.gbm_model.best_iteration
        )

        logger.info(f"  GBM raw predictions (normalized): "
                    f"mean={predictions.mean():.4f}, std={predictions.std():.4f}")

        return predictions

    def _extract_cnn_bottleneck_features(self, X: np.ndarray,
                                          batch_size: int = 32) -> np.ndarray:
        """
        Extract CNN encoder bottleneck statistics to enrich GBM features (FIX 6).
        Mirrors extract_cnn_bottleneck_features() from train_ensemble.py exactly.

        Returns (N, 3 * bottleneck_channels) — mean/std/max per channel.
        """
        import torch
        base = self.cnn_model.base_model   # unwrap MCDropoutUNet
        base.eval()
        all_stats = []

        for start in range(0, len(X), batch_size):
            batch_np = X[start: start + batch_size]
            batch_t  = torch.FloatTensor(batch_np).permute(0, 3, 1, 2).to(self.device)

            with torch.no_grad():
                x, _ = base.enc1(batch_t)
                x, _ = base.enc2(x)
                x, _ = base.enc3(x)
                x, _ = base.enc4(x)
                bot  = base.bottleneck(x)          # (B, C_bot, H', W')

            stats = torch.cat([
                bot.mean(dim=[2, 3]),
                bot.std(dim=[2, 3]),
                bot.amax(dim=[2, 3]),
            ], dim=1).cpu().numpy()

            all_stats.append(stats)

        return np.vstack(all_stats)

    def _apply_bottleneck_pca(self, bot_feats: np.ndarray) -> np.ndarray:
        """
        Apply the PCA that was fitted during training to compress raw bottleneck
        statistics.  If no PCA was loaded (e.g. old checkpoint), returns raw feats
        and emits a warning so the caller can fall back to column-alignment padding.
        """
        if self._bottleneck_pca is not None:
            try:
                compressed = self._bottleneck_pca.transform(bot_feats)
                logger.info(f"  PCA: {bot_feats.shape[1]} → {compressed.shape[1]} bottleneck dims")
                return compressed
            except Exception as _e:
                logger.warning(f"  ⚠️ PCA transform failed ({_e}); using raw bottleneck features")
        return bot_feats
    
    def predict_ensemble(self, X: np.ndarray, batch_size: int = 8, 
                        return_uncertainty: bool = False,
                        use_spatial_ensemble: bool = True,
                        skip_validation: bool = False,
                        targets: np.ndarray = None,
                        save_diagnostics: bool = True) -> Dict[str, np.ndarray]:
        """
        Make ensemble predictions
        
        Args:
            X: NORMALIZED input features (N, H, W, C)
            batch_size: Batch size for inference
            return_uncertainty: Whether to compute MC Dropout uncertainty
            use_spatial_ensemble: Use spatial-level ensemble vs patch-level
            skip_validation: Skip input validation
            targets: Optional ground-truth LST patches (°C, denormalised) for
                     diagnostic plots.  If provided and save_diagnostics=True,
                     accuracy metrics will be included in the dashboard.
            save_diagnostics: Generate and save diagnostic plots via
                              InferenceDiagnosticsPlotter.  Default True.
            
        Returns:
            Dictionary with ensemble predictions in CELSIUS
        """
        logger.info("\n" + "="*70)
        logger.info("ENSEMBLE PREDICTION PIPELINE")
        logger.info("="*70)
        
        # Validate input
        if not skip_validation:
            if not self.validate_input(X):
                logger.error("❌ Input validation failed!")
        
        # Get predictions in NORMALIZED space
        cnn_preds_norm = self._predict_cnn_normalized(X, batch_size)
        gbm_preds_norm = self._predict_gbm_normalized(X)
        
        # Calculate patch-level averages (STILL NORMALIZED)
        cnn_preds_patch_norm = cnn_preds_norm.reshape(cnn_preds_norm.shape[0], -1).mean(axis=1)
        
        logger.info("\n📊 Prediction Statistics (NORMALIZED space):")
        logger.info(f"  CNN spatial: mean={cnn_preds_norm.mean():.4f}, std={cnn_preds_norm.std():.4f}")
        logger.info(f"  CNN patch avg: mean={cnn_preds_patch_norm.mean():.4f}, std={cnn_preds_patch_norm.std():.4f}")
        logger.info(f"  GBM patch: mean={gbm_preds_norm.mean():.4f}, std={gbm_preds_norm.std():.4f}")
        
        # Ensemble combination in NORMALIZED space
        if use_spatial_ensemble and self.weights["cnn"] > 0:
            logger.info("\n🔧 Using SPATIAL ensemble (normalized space)")

            # Broadcast GBM patch predictions to spatial dimensions
            gbm_spatial_norm = np.zeros_like(cnn_preds_norm)
            for i in range(len(gbm_preds_norm)):
                gbm_spatial_norm[i] = gbm_preds_norm[i]

            # Weighted combination — preserves CNN spatial texture
            ensemble_preds_norm = (
                self.weights["cnn"] * cnn_preds_norm +
                self.weights["gbm"] * gbm_spatial_norm
            )

            ensemble_preds_patch_norm = ensemble_preds_norm.reshape(
                ensemble_preds_norm.shape[0], -1
            ).mean(axis=1)

        else:
            # CNN weight is 0 (GBM-only mode) — but broadcasting GBM scalar to every
            # pixel produces a solid-colour spatial map.  Instead, use CNN-as-residual:
            # shift CNN's spatial pattern so its patch mean matches the GBM prediction,
            # preserving all spatial detail while trusting GBM for the absolute level.
            logger.info("\n🔧 Using CNN-as-residual spatial mode (GBM mean + CNN spatial deviation)")

            # Patch-level ensemble prediction comes entirely from GBM
            ensemble_preds_patch_norm = (
                self.weights["cnn"] * cnn_preds_patch_norm +
                self.weights["gbm"] * gbm_preds_norm
            )

            # Spatial map: GBM patch mean + CNN spatial deviation from its own patch mean
            # This gives full spatial resolution without a solid-colour artefact.
            cnn_deviation = cnn_preds_norm - cnn_preds_patch_norm[:, np.newaxis, np.newaxis]
            ensemble_preds_norm = (
                gbm_preds_norm[:, np.newaxis, np.newaxis] + cnn_deviation
            )
            logger.info(f"  Spatial std (CNN deviation): {cnn_deviation.std():.4f}")
        
        logger.info(f"\n📊 Ensemble (NORMALIZED): mean={ensemble_preds_norm.mean():.4f}, "
                   f"std={ensemble_preds_norm.std():.4f}")
        
        # DENORMALIZE ONCE - at the very end
        logger.info("\n🔄 Denormalizing predictions to Celsius...")
        
        cnn_preds = self.normalizer.denormalize_predictions(cnn_preds_norm, "target")
        gbm_preds = self.normalizer.denormalize_predictions(gbm_preds_norm, "target")
        cnn_preds_patch = self.normalizer.denormalize_predictions(cnn_preds_patch_norm, "target")
        ensemble_preds = self.normalizer.denormalize_predictions(ensemble_preds_norm, "target")
        ensemble_preds_patch = self.normalizer.denormalize_predictions(ensemble_preds_patch_norm, "target")
        
        logger.info(f"\n📈 Final Ensemble Statistics (CELSIUS):")
        logger.info(f"  Mean: {ensemble_preds.mean():.2f}°C")
        logger.info(f"  Std:  {ensemble_preds.std():.2f}°C")
        logger.info(f"  Range: [{ensemble_preds.min():.2f}, {ensemble_preds.max():.2f}]°C")
        
        results = {
            "ensemble": ensemble_preds,
            "cnn": cnn_preds,
            "gbm": gbm_preds,
            "ensemble_patch": ensemble_preds_patch,
            "cnn_patch": cnn_preds_patch
        }
        
        # Uncertainty estimation
        if return_uncertainty:
            logger.info("\n🎲 Computing MC Dropout uncertainty...")
            uncertainty = self._compute_mc_dropout_uncertainty(X, n_samples=50, batch_size=batch_size)
            results["uncertainty"] = uncertainty
        
        logger.info(f"\n✅ Ensemble predictions complete")
        logger.info("="*70 + "\n")

        # ── Diagnostic plots ──────────────────────────────────────────────────
        if save_diagnostics:
            try:
                self.plotter.plot_all(
                    results,
                    X=X,
                    targets=targets,
                    cal_slope=self.cal_slope,
                    cal_intercept=self.cal_intercept,
                    weights=self.weights,
                )
            except Exception as _pe:
                logger.warning(f"Inference diagnostic plots failed: {_pe}")
        # ─────────────────────────────────────────────────────────────────────

        return results
    
    def _compute_mc_dropout_uncertainty(self, X: np.ndarray, n_samples: int = 50, 
                                       batch_size: int = 8) -> np.ndarray:
        """Compute uncertainty using MC Dropout"""
        logger.info(f"  Running {n_samples} MC Dropout forward passes...")
        
        self.cnn_model.enable_mc_dropout()
        
        n_inputs = X.shape[0]
        mc_predictions = []
        
        for mc_iter in tqdm(range(n_samples), desc="MC Dropout samples"):
            predictions = []
            
            with torch.no_grad():
                for i in range(0, n_inputs, batch_size):
                    batch = X[i:i+batch_size]
                    batch_tensor = torch.FloatTensor(batch).permute(0, 3, 1, 2).to(self.device)
                    
                    output = self.cnn_model(batch_tensor)
                    predictions.append(output.cpu().numpy())
            
            preds = np.concatenate(predictions, axis=0).squeeze(1)
            
            # Denormalize each MC sample
            preds = self.normalizer.denormalize_predictions(preds, "target")
            mc_predictions.append(preds)
        
        self.cnn_model.disable_mc_dropout()
        
        # Calculate statistics
        mc_predictions = np.array(mc_predictions)
        mean_pred = np.mean(mc_predictions, axis=0)
        epistemic_uncertainty = np.std(mc_predictions, axis=0)
        
        logger.info(f"\n  MC Dropout Statistics:")
        logger.info(f"    Mean prediction: {mean_pred.mean():.2f}±{mean_pred.std():.2f}°C")
        logger.info(f"    Epistemic uncertainty: {epistemic_uncertainty.mean():.3f}°C")
        
        return epistemic_uncertainty
    
    def _apply_tta_augmentation(self, X: np.ndarray, aug_idx: int) -> np.ndarray:
        """
        Apply test-time augmentation
        
        Args:
            X: Input array (N, H, W, C)
            aug_idx: Augmentation index (determines which augmentation to apply)
            
        Returns:
            Augmented array
        """
        X_aug = X.copy()
        
        # Define 8 augmentation combinations
        # 0: No augmentation (original)
        # 1: Horizontal flip
        # 2: Vertical flip
        # 3: Both flips
        # 4: Rotate 90°
        # 5: Rotate 180°
        # 6: Rotate 270°
        # 7: Transpose
        
        if aug_idx == 0:
            return X_aug
        
        elif aug_idx == 1:
            # Horizontal flip
            X_aug = np.flip(X_aug, axis=2)
        
        elif aug_idx == 2:
            # Vertical flip
            X_aug = np.flip(X_aug, axis=1)
        
        elif aug_idx == 3:
            # Both flips
            X_aug = np.flip(X_aug, axis=2)
            X_aug = np.flip(X_aug, axis=1)
        
        elif aug_idx == 4:
            # Rotate 90° clockwise
            X_aug = np.rot90(X_aug, k=-1, axes=(1, 2))
        
        elif aug_idx == 5:
            # Rotate 180°
            X_aug = np.rot90(X_aug, k=2, axes=(1, 2))
        
        elif aug_idx == 6:
            # Rotate 270° clockwise (90° counter-clockwise)
            X_aug = np.rot90(X_aug, k=1, axes=(1, 2))
        
        elif aug_idx == 7:
            # Transpose
            X_aug = np.transpose(X_aug, (0, 2, 1, 3))
        
        return X_aug.copy()
    
    def _reverse_tta_augmentation(self, preds: np.ndarray, aug_idx: int) -> np.ndarray:
        """
        Reverse augmentation applied to predictions
        
        Args:
            preds: Predictions (N, H, W) or (N,)
            aug_idx: Augmentation index
            
        Returns:
            Predictions with augmentation reversed
        """
        if aug_idx == 0:
            return preds
        
        # Only reverse if predictions are spatial (3D)
        if preds.ndim == 3:
            if aug_idx == 1:
                preds = np.flip(preds, axis=2)
            elif aug_idx == 2:
                preds = np.flip(preds, axis=1)
            elif aug_idx == 3:
                preds = np.flip(preds, axis=2)
                preds = np.flip(preds, axis=1)
            elif aug_idx == 4:
                preds = np.rot90(preds, k=1, axes=(1, 2))
            elif aug_idx == 5:
                preds = np.rot90(preds, k=2, axes=(1, 2))
            elif aug_idx == 6:
                preds = np.rot90(preds, k=-1, axes=(1, 2))
            elif aug_idx == 7:
                preds = np.transpose(preds, (0, 2, 1))
        
        return preds.copy()
    
    def predict_ensemble_with_tta(self, X: np.ndarray, 
                                  batch_size: int = 8,
                                  n_augmentations: int = 8,
                                  return_uncertainty: bool = True,
                                  use_spatial_ensemble: bool = True,
                                  targets: np.ndarray = None,
                                  save_diagnostics: bool = True) -> Dict[str, np.ndarray]:
        """
        Make ensemble predictions with Test-Time Augmentation
        
        Args:
            X: NORMALIZED input features (N, H, W, C)
            batch_size: Batch size for inference
            n_augmentations: Number of augmented predictions to average
            return_uncertainty: Whether to compute uncertainty
            use_spatial_ensemble: Use spatial-level ensemble
            
        Returns:
            Dictionary with ensemble predictions and TTA uncertainty
        """
        logger.info("\n" + "="*70)
        logger.info(f"ENSEMBLE PREDICTION WITH TTA ({n_augmentations} augmentations)")
        logger.info("="*70)
        
        # Validate input
        if not self.validate_input(X):
            logger.warning("⚠️ Input validation failed, but continuing...")
        
        # Store predictions from all augmentations
        cnn_preds_spatial_tta = []
        cnn_preds_patch_tta = []
        gbm_preds_tta = []
        
        for aug_idx in tqdm(range(n_augmentations), desc="TTA iterations"):
            # Apply augmentation
            X_aug = self._apply_tta_augmentation(X, aug_idx)
            
            # Get predictions (in NORMALIZED space)
            cnn_preds_norm = self._predict_cnn_normalized(X_aug, batch_size)
            gbm_preds_norm = self._predict_gbm_normalized(X_aug)
            
            # Reverse augmentation for spatial predictions
            cnn_preds_norm = self._reverse_tta_augmentation(cnn_preds_norm, aug_idx)
            
            # Calculate patch-level average
            cnn_preds_patch_norm = cnn_preds_norm.reshape(
                cnn_preds_norm.shape[0], -1
            ).mean(axis=1)
            
            cnn_preds_spatial_tta.append(cnn_preds_norm)
            cnn_preds_patch_tta.append(cnn_preds_patch_norm)
            gbm_preds_tta.append(gbm_preds_norm)
        
        # Average predictions across augmentations (STILL NORMALIZED)
        cnn_preds_spatial_norm = np.mean(cnn_preds_spatial_tta, axis=0)
        cnn_preds_patch_norm = np.mean(cnn_preds_patch_tta, axis=0)
        gbm_preds_norm = np.mean(gbm_preds_tta, axis=0)
        
        # Calculate TTA uncertainty (std across augmentations)
        cnn_tta_uncertainty = np.std(cnn_preds_spatial_tta, axis=0)
        
        logger.info(f"\n📊 TTA Statistics (NORMALIZED space):")
        logger.info(f"  CNN spatial: mean={cnn_preds_spatial_norm.mean():.4f}, "
                   f"TTA std={cnn_tta_uncertainty.mean():.4f}")
        logger.info(f"  CNN patch: mean={cnn_preds_patch_norm.mean():.4f}")
        logger.info(f"  GBM: mean={gbm_preds_norm.mean():.4f}")
        
        # Ensemble in NORMALIZED space
        if use_spatial_ensemble and self.weights["cnn"] > 0:
            logger.info("\n🔧 Using SPATIAL ensemble with TTA")
            
            # Broadcast GBM to spatial dimensions
            gbm_spatial_norm = np.zeros_like(cnn_preds_spatial_norm)
            for i in range(len(gbm_preds_norm)):
                gbm_spatial_norm[i] = gbm_preds_norm[i]
            
            # Weighted combination
            ensemble_preds_norm = (
                self.weights["cnn"] * cnn_preds_spatial_norm +
                self.weights["gbm"] * gbm_spatial_norm
            )
            
            ensemble_preds_patch_norm = ensemble_preds_norm.reshape(
                ensemble_preds_norm.shape[0], -1
            ).mean(axis=1)
            
        else:
            logger.info("\n🔧 Using CNN-as-residual spatial mode with TTA (GBM mean + CNN spatial deviation)")

            ensemble_preds_patch_norm = (
                self.weights["cnn"] * cnn_preds_patch_norm +
                self.weights["gbm"] * gbm_preds_norm
            )

            # Preserve spatial texture: GBM sets the patch-level mean,
            # CNN contributes the within-patch deviation
            cnn_deviation = cnn_preds_spatial_norm - cnn_preds_patch_norm[:, np.newaxis, np.newaxis]
            ensemble_preds_norm = (
                gbm_preds_norm[:, np.newaxis, np.newaxis] + cnn_deviation
            )
        
        # DENORMALIZE to Celsius
        logger.info("\n🔄 Denormalizing predictions to Celsius...")
        
        cnn_preds = self.normalizer.denormalize_predictions(
            cnn_preds_spatial_norm, "target"
        )
        gbm_preds = self.normalizer.denormalize_predictions(
            gbm_preds_norm, "target"
        )
        ensemble_preds = self.normalizer.denormalize_predictions(
            ensemble_preds_norm, "target"
        )
        cnn_preds_patch = self.normalizer.denormalize_predictions(
            cnn_preds_patch_norm, "target"
        )
        ensemble_preds_patch = self.normalizer.denormalize_predictions(
            ensemble_preds_patch_norm, "target"
        )
        
        # Denormalize TTA uncertainty
        # Uncertainty is in terms of std, so only multiply by target_std
        tta_uncertainty = cnn_tta_uncertainty * self.normalizer.stats['target']['std']
        
        logger.info(f"\n📈 Final Ensemble Statistics (CELSIUS):")
        logger.info(f"  Mean: {ensemble_preds.mean():.2f}°C")
        logger.info(f"  Std: {ensemble_preds.std():.2f}°C")
        logger.info(f"  Range: [{ensemble_preds.min():.2f}, {ensemble_preds.max():.2f}]°C")
        logger.info(f"  TTA Uncertainty: {tta_uncertainty.mean():.2f}°C")
        
        results = {
            "ensemble": ensemble_preds,
            "cnn": cnn_preds,
            "gbm": gbm_preds,
            "ensemble_patch": ensemble_preds_patch,
            "cnn_patch": cnn_preds_patch,
            "tta_uncertainty": tta_uncertainty
        }
        
        # Optional: MC Dropout uncertainty (in addition to TTA)
        if return_uncertainty:
            logger.info("\n🎲 Computing MC Dropout uncertainty (in addition to TTA)...")
            mc_uncertainty = self._compute_mc_dropout_uncertainty(
                X, n_samples=30, batch_size=batch_size
            )
            results["mc_uncertainty"] = mc_uncertainty
            
            # Combined uncertainty (TTA + MC Dropout)
            # Use root sum of squares
            combined_uncertainty = np.sqrt(
                tta_uncertainty**2 + mc_uncertainty**2
            )
            results["combined_uncertainty"] = combined_uncertainty
            
            logger.info(f"  TTA uncertainty: {tta_uncertainty.mean():.2f}°C")
            logger.info(f"  MC uncertainty: {mc_uncertainty.mean():.2f}°C")
            logger.info(f"  Combined uncertainty: {combined_uncertainty.mean():.2f}°C")
        
        logger.info(f"\n✅ TTA ensemble predictions complete")
        logger.info("="*70 + "\n")

        # ── Diagnostic plots ──────────────────────────────────────────────────
        if save_diagnostics:
            try:
                self.plotter.plot_all(
                    results,
                    X=X,
                    targets=targets,
                    cal_slope=self.cal_slope,
                    cal_intercept=self.cal_intercept,
                    weights=self.weights,
                    label="Ensemble+TTA",
                )
            except Exception as _pe:
                logger.warning(f"Inference TTA diagnostic plots failed: {_pe}")
        # ─────────────────────────────────────────────────────────────────────
        
        return results


class PostProcessor:
    """Post-processing for LST predictions"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "bilateral_sigma_spatial": 5,
            "bilateral_sigma_range": 2.0,
            "temp_min": 20.0,
            "temp_max": 50.0
        }
    
    def apply_bilateral_filter(self, lst_map: np.ndarray) -> np.ndarray:
        """Apply bilateral filter to smooth while preserving edges"""
        import cv2
        
        logger.info("Applying bilateral filter...")
        
        if lst_map.std() < 0.5:
            logger.warning("⚠️ Input has low variance, skipping filtering")
            return lst_map
        
        lst_min, lst_max = lst_map.min(), lst_map.max()
        if lst_max - lst_min < 1.0:
            logger.warning("⚠️ Input range too small for filtering")
            return lst_map
        
        lst_normalized = ((lst_map - lst_min) / (lst_max - lst_min) * 255).astype(np.uint8)
        
        filtered = cv2.bilateralFilter(
            lst_normalized, 
            d=9,
            sigmaColor=self.config["bilateral_sigma_range"],
            sigmaSpace=self.config["bilateral_sigma_spatial"]
        )
        
        filtered = filtered.astype(np.float32) / 255.0 * (lst_max - lst_min) + lst_min
        
        logger.info(f"  Filtered: std={filtered.std():.2f}°C")
        return filtered
    
    def fill_nodata(self, lst_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Fill no-data pixels using spatial interpolation"""
        from scipy.interpolate import griddata
        
        if mask is None:
            mask = ~np.isnan(lst_map)
        
        if mask.all():
            return lst_map
        
        n_missing = (~mask).sum()
        logger.info(f"Filling {n_missing} no-data pixels ({n_missing/mask.size*100:.1f}%)...")
        
        y, x = np.indices(lst_map.shape)
        valid_coords = np.column_stack([x[mask], y[mask]])
        valid_values = lst_map[mask]
        invalid_coords = np.column_stack([x[~mask], y[~mask]])
        
        filled_values = griddata(
            valid_coords, valid_values, invalid_coords,
            method='cubic', fill_value=np.nanmean(lst_map)
        )
        
        filled_map = lst_map.copy()
        filled_map[~mask] = filled_values
        
        return filled_map
    
    def clip_values(self, lst_map: np.ndarray) -> np.ndarray:
        """Clip LST values to physically realistic range"""
        n_below = (lst_map < self.config["temp_min"]).sum()
        n_above = (lst_map > self.config["temp_max"]).sum()
        
        if n_below + n_above > 0:
            logger.info(f"Clipping {n_below + n_above} pixels to "
                       f"[{self.config['temp_min']}, {self.config['temp_max']}]°C")
        
        return np.clip(lst_map, self.config["temp_min"], self.config["temp_max"])
    
    def process(self, lst_map: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply full post-processing pipeline"""
        logger.info("Post-processing LST predictions...")
        logger.info(f"  Input: mean={lst_map.mean():.2f}°C, std={lst_map.std():.2f}°C")
        
        processed = self.fill_nodata(lst_map, mask)
        # Bilateral filter deliberately skipped: it reduces spatial variance
        # which worsens slope/std_ratio at evaluation time.
        # Call apply_bilateral_filter() explicitly only for visualization output.
        processed = self.clip_values(processed)
        
        logger.info(f"  Output: mean={processed.mean():.2f}°C, std={processed.std():.2f}°C")
        logger.info("✓ Post-processing complete")
        
        return processed


def main():
    """Example usage of inference pipeline"""
    logger.info("="*70)
    logger.info("UHI INFERENCE PIPELINE - Expects NORMALIZED Data")
    logger.info("="*70)
    logger.info("Key features:")
    logger.info("  1. Expects NORMALIZED input (mean≈0, std≈1)")
    logger.info("  2. Same preprocessing as training")
    logger.info("  3. Automatic denormalization to Celsius")
    logger.info("  4. Input validation")
    logger.info("  5. MC Dropout uncertainty estimation")
    logger.info("="*70)
    
    # Define paths
    model_dir = MODEL_DIR
    cnn_model_path = model_dir / "final_cnn.pth"
    gbm_model_path = model_dir / "gbm_model.pkl"
    ensemble_config_path = model_dir / "ensemble_config.json"
    normalization_stats_path = PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
    
    # Check if all required files exist
    required_files = [
        cnn_model_path,
        gbm_model_path,
        ensemble_config_path,
        normalization_stats_path
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        logger.error("❌ Missing required files:")
        for f in missing_files:
            logger.error(f"  - {f}")
        logger.error("\nPlease run train_ensemble.py first!")
        return
    
    logger.info("✓ All required files found")
    
    # Initialize predictor
    predictor = EnsemblePredictor(
        cnn_model_path=cnn_model_path,
        gbm_model_path=gbm_model_path,
        ensemble_config_path=ensemble_config_path,
        normalization_stats_path=normalization_stats_path,
        device="cuda",
        mc_dropout_rate=0.05  # CHANGED: reduced from 0.1
    )
    
    logger.info("\n✓ Ensemble predictor initialized and ready for inference")
    logger.info("\nTo use this predictor:")
    logger.info("  # Load NORMALIZED test data")
    logger.info("  X_test = np.load('data/processed/cnn_dataset/test/X.npy')")
    logger.info("  ")
    logger.info("  # Make predictions (X_test should be normalized)")
    logger.info("  results = predictor.predict_ensemble(X_test, return_uncertainty=True)")
    logger.info("  ")
    logger.info("  # Results will be in Celsius (automatically denormalized)")
    logger.info("  ensemble_pred = results['ensemble']")


if __name__ == "__main__":
    main()