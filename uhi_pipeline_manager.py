"""
UHI Pipeline Manager — Fully Integrated
Wires together: inference → post-processing → UHI analysis → hotspot detection
               → validation → all diagnostic plots → visualizations → web map → report
"""
import sys
import json
import traceback
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import logging

import matplotlib
matplotlib.use("Agg")   # non-interactive / headless-safe
import matplotlib.pyplot as plt

from config import *
from uhi_inference import EnsemblePredictor, PostProcessor, DataNormalizer
from uhi_analysis import (
    UHIAnalyzer,
    HotspotDetector,
    ValidationAnalyzer,
    generate_report,
    run_all_diagnostics,
)
from uhi_visualization import UHIVisualizer, OutputGenerator

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_json_serializable(obj):
    """Recursively convert numpy scalars / arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def _set_deterministic_mode(seed: int = 42) -> None:
    """
    Force deterministic behaviour across numpy, Python random, and PyTorch.

    ROOT CAUSE of LST map changing on every run
    ────────────────────────────────────────────
    The map changes because several stochastic components reset differently:

      1. MC Dropout  — each forward pass samples a DIFFERENT dropout mask unless
                        you seed torch before inference. Fix: torch.manual_seed().
      2. cuDNN auto-tuner — when cudnn.benchmark=True (PyTorch default), CUDA
                        selects the 'fastest' convolution algorithm at runtime.
                        The fastest algorithm can change between runs / GPU states,
                        producing tiny numerical differences that compound across
                        layers. Fix: cudnn.deterministic=True, benchmark=False.
      3. numpy ops   — random subsampling in plots, KDE bandwidth estimation.
                        Fix: np.random.seed().
      4. GBM threads — LightGBM histogram binning is non-deterministic with
                        multiple threads. Fix: set num_threads=1 in GBM config
                        during TRAINING (not fixable purely at inference).
    """
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True   # ← KEY FIX
        torch.backends.cudnn.benchmark     = False  # ← disables auto-tuner
    logger.info(f"  ✓ Deterministic mode enabled (seed={seed})")
    logger.info("    numpy, torch, cuda all seeded; cudnn.benchmark=False")


def _section(title: str, width: int = 70) -> None:
    bar = "=" * width
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def _try(label: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), log errors but never crash the pipeline."""
    try:
        result = fn(*args, **kwargs)
        logger.info(f"✓ {label}")
        return result
    except Exception as exc:
        logger.error(f"✗ {label} — {exc}")
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the full UHI analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--model-dir",   default="models",
                   help="Directory containing trained models")
    p.add_argument("--output-dir",  default="outputs",
                   help="Root directory for all output products")
    p.add_argument("--test-data",   default="data/processed/cnn_dataset/test",
                   help="Directory containing NORMALIZED test data (X.npy [, y.npy])")

    # Inference
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--mc-samples",  type=int, default=50,
                   help="MC Dropout forward passes for uncertainty estimation")
    p.add_argument("--mc-dropout-rate", type=float, default=0.05)
    p.add_argument("--use-spatial-ensemble", action="store_true", default=True)
    p.add_argument("--use-tta",     action="store_true", default=False,
                   help="Use Test-Time Augmentation for final predictions")
    p.add_argument("--tta-augs",    type=int, default=8,
                   help="Number of TTA augmentations")
    p.add_argument("--device",      default="cuda",
                   help="Torch device ('cuda' or 'cpu')")
    p.add_argument("--seed",        type=int, default=42,
                   help="Global random seed for reproducible inference")

    # Analysis
    p.add_argument("--hotspot-radius",       type=float, default=500.0,
                   help="Search radius for Gi* hotspot detection (metres)")
    p.add_argument("--hotspot-confidence",   type=float, default=0.95,
                   choices=[0.90, 0.95, 0.99])
    p.add_argument("--urban-percentile",     type=float, default=75.0,
                   help="Percentile threshold for urban mask")
    p.add_argument("--rural-percentile",     type=float, default=25.0,
                   help="Percentile threshold for rural mask")
    p.add_argument("--max-patches-analysis", type=int, default=10,
                   help="Max number of patches post-processed for analysis")
    p.add_argument("--patch-grid-cols", type=int, default=None,
                   help="Number of patch columns in the spatial grid used during dataset "
                        "creation. When omitted the pipeline reads patch_grid_shape from "
                        "the dataset metadata; if absent it falls back to sqrt(N) layout. "
                        "Setting this explicitly is most reliable -- check the dataset "
                        "builder log for the grid dimensions.")

    # Diagnostics / output switches
    p.add_argument("--skip-diagnostics",  action="store_true",
                   help="Skip all diagnostic matplotlib figures")
    p.add_argument("--skip-geotiff",      action="store_true")
    p.add_argument("--skip-webmap",       action="store_true")
    p.add_argument("--skip-validation",   action="store_true")

    return p


# ══════════════════════════════════════════════════════════════════════════════
# Sub-pipeline stages
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Load data ────────────────────────────────────────────────────────

def stage_load_data(args) -> dict:
    _section("STAGE 1 — LOAD DATA")
    test_dir = Path(args.test_data)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_dir}")

    X_test = np.load(test_dir / "X.npy")
    logger.info(f"  X_test: {X_test.shape}  mean={X_test.mean():.4f}  std={X_test.std():.4f}")

    if not (-0.5 < X_test.mean() < 0.5 and 0.5 < X_test.std() < 1.5):
        logger.warning("  ⚠ X_test does not appear to be normalized (expected mean≈0, std≈1)")
    else:
        logger.info("  ✓ X_test is properly normalized")

    y_test = None
    y_path = test_dir / "y.npy"
    if y_path.exists() and not args.skip_validation:
        y_test = np.load(y_path)
        logger.info(f"  y_test: {y_test.shape}  mean={y_test.mean():.4f}  std={y_test.std():.4f}")

    return dict(X_test=X_test, y_test=y_test)


# ── Stage 2: Initialise predictor ────────────────────────────────────────────

def stage_init_predictor(args, norm_stats_path: Path) -> EnsemblePredictor:
    _section("STAGE 2 — INITIALISE PREDICTOR")
    model_dir = Path(args.model_dir)

    best_cnn  = model_dir / "checkpoints" / "best_r2.pth"
    final_cnn = model_dir / "final_cnn.pth"
    cnn_path  = best_cnn if best_cnn.exists() else final_cnn
    logger.info(f"  CNN weights : {cnn_path}")

    best_gbm  = model_dir / "best_gbm_model.pkl"
    gbm_path  = best_gbm if best_gbm.exists() else model_dir / "gbm_model.pkl"
    logger.info(f"  GBM model   : {gbm_path}")

    required = [cnn_path, gbm_path,
                model_dir / "ensemble_config.json",
                norm_stats_path]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model files:\n  " + "\n  ".join(missing)
            + "\n\nPlease run train_ensemble.py first!"
        )

    predictor = EnsemblePredictor(
        cnn_model_path=cnn_path,
        gbm_model_path=gbm_path,
        ensemble_config_path=model_dir / "ensemble_config.json",
        normalization_stats_path=norm_stats_path,
        device=args.device,
        mc_dropout_rate=args.mc_dropout_rate,
    )
    logger.info("  ✓ Predictor ready")
    return predictor


# ── Stage 3: Adaptive calibration ────────────────────────────────────────────

def stage_adaptive_calibration(predictor: EnsemblePredictor,
                                y_test: np.ndarray,
                                norm_stats_path: Path) -> None:
    """Adjust the static calibration slope/intercept for test-set LST shift."""
    if y_test is None:
        return

    _section("STAGE 3 — ADAPTIVE CALIBRATION")
    normalizer_tmp = DataNormalizer(norm_stats_path)
    y_celsius = normalizer_tmp.denormalize_predictions(
        y_test.squeeze().flatten(), "target")

    test_std  = float(y_celsius.std())
    test_mean = float(y_celsius.mean())
    train_std = float(predictor.normalizer.stats["target"]["std"])
    train_mean = float(predictor.normalizer.stats["target"]["mean"])

    adaptive_scale = test_std / train_std

    old_slope     = predictor.cal_slope
    old_intercept = predictor.cal_intercept

    predictor.cal_slope     = old_slope * adaptive_scale
    predictor.cal_intercept = (test_mean - train_mean) / train_std

    logger.info(f"  Train target  : mean={train_mean:.3f}°C  std={train_std:.3f}°C")
    logger.info(f"  Test target   : mean={test_mean:.3f}°C  std={test_std:.3f}°C")
    logger.info(f"  Adaptive scale: {adaptive_scale:.4f}")
    logger.info(f"  Old cal : slope={old_slope:.4f}  intercept={old_intercept:.4f}")
    logger.info(f"  New cal : slope={predictor.cal_slope:.4f}  "
                f"intercept={predictor.cal_intercept:.4f}")


# ── Stage 4: Inference ───────────────────────────────────────────────────────

def stage_inference(args, predictor: EnsemblePredictor,
                    X_test: np.ndarray, data_dir: Path) -> dict:
    _section("STAGE 4 — INFERENCE")

    # ── Propagate grid_cols to the inference plotter BEFORE running predictions
    # so that any diagnostic mosaic (03b) saved during predict_ensemble[_with_tta]
    # uses the correct spatial layout rather than the jumbling sqrt fallback.
    grid_cols = getattr(args, "patch_grid_cols", None)
    if grid_cols is None:
        _gs = _load_patch_grid_shape(Path(args.test_data))
        if _gs is not None:
            grid_cols = _gs[1]
    if grid_cols is not None:
        predictor.plotter.set_grid_cols(grid_cols)
        logger.info(f"  Plotter grid_cols set to {grid_cols}")
    else:
        logger.warning(
            "  ⚠ grid_cols unknown at inference time — diagnostic mosaic 03b will "            "use sqrt fallback.  Pass --patch-grid-cols=<N> to fix."
        )

    # ── Load per-patch spatial positions so the mosaic assembler places each
    # patch at its original grid cell (fixes jumbled output when QC filtering
    # has removed some patches from the full scan grid).
    _pos_path = Path(args.test_data) / "patch_positions.npy"
    if _pos_path.exists():
        try:
            _patch_positions = np.load(_pos_path)
            predictor.plotter.set_patch_positions(_patch_positions)
            logger.info(f"  Loaded patch_positions.npy ({len(_patch_positions)} entries)")
        except Exception as _pe:
            logger.warning(f"  ⚠ Could not load patch_positions.npy: {_pe}")
    else:
        logger.warning(
            "  ⚠ patch_positions.npy not found in test data dir — mosaic will use "
            "sequential placement (patches shifted if any were QC-filtered). "
            "Re-run preprocessing to regenerate the dataset with position metadata."
        )

    if args.use_tta:
        logger.info(f"  Mode: TTA  (n_augmentations={args.tta_augs})")
        results = predictor.predict_ensemble_with_tta(
            X_test,
            batch_size=args.batch_size,
            n_augmentations=args.tta_augs,
            return_uncertainty=True,
            use_spatial_ensemble=args.use_spatial_ensemble,
        )
    else:
        logger.info("  Mode: standard ensemble  (return_uncertainty=True)")
        results = predictor.predict_ensemble(
            X_test,
            batch_size=args.batch_size,
            return_uncertainty=True,
            use_spatial_ensemble=args.use_spatial_ensemble,
        )

    # Persist predictions
    for key in ("ensemble", "cnn", "gbm"):
        if key in results:
            np.save(data_dir / f"predictions_{key}.npy", results[key])

    unc_key = "combined_uncertainty" if "combined_uncertainty" in results else "uncertainty"
    if unc_key in results and results[unc_key] is not None:
        np.save(data_dir / "uncertainty.npy", results[unc_key])
        results["_unc_key"] = unc_key   # bubble up which key to use later

    ens = results["ensemble"]
    logger.info(f"  Ensemble shape : {ens.shape}")
    logger.info(f"  Ensemble range : [{ens.min():.2f}, {ens.max():.2f}] °C")
    logger.info(f"  Ensemble mean  : {ens.mean():.2f} ± {ens.std():.2f} °C")

    return results


# ── Stage 5: Post-processing ─────────────────────────────────────────────────

def _load_patch_grid_shape(test_data_dir: Path) -> tuple:
    """
    Try to read the patch grid shape (grid_rows, grid_cols) saved by the
    dataset builder.  Returns None if no metadata file is found.

    The dataset builder is expected to write one of:
      <test_data_dir>/patch_grid_shape.npy   — np.array([rows, cols])
      <test_data_dir>/dataset_metadata.json  — {"patch_grid_rows": R, "patch_grid_cols": C}
      <test_data_dir>/../patch_grid_shape.npy  (parent dir, i.e. cnn_dataset/)
      <test_data_dir>/../dataset_metadata.json
    """
    candidates = [
        test_data_dir / "patch_grid_shape.npy",
        test_data_dir / "dataset_metadata.json",
        test_data_dir.parent / "patch_grid_shape.npy",
        test_data_dir.parent / "dataset_metadata.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix == ".npy":
                shape = tuple(int(x) for x in np.load(path))
                logger.info(f"  Loaded patch grid shape {shape} from {path.name}")
                return shape   # (grid_rows, grid_cols)
            else:
                meta = json.loads(path.read_text())
                if "patch_grid_rows" in meta and "patch_grid_cols" in meta:
                    shape = (int(meta["patch_grid_rows"]), int(meta["patch_grid_cols"]))
                    logger.info(f"  Loaded patch grid shape {shape} from {path.name}")
                    return shape
        except Exception as exc:
            logger.warning(f"  Could not parse {path}: {exc}")
    return None


def _mosaic_patches(patches: np.ndarray,
                    grid_cols: int = None,
                    grid_rows: int = None,
                    patch_positions: np.ndarray = None) -> np.ndarray:
    """
    Assemble an (N, H, W) array of equal-sized patches into a single 2-D mosaic
    that faithfully reproduces the spatial layout of the study area.

    Args:
        patches:         np.ndarray of shape (N, H, W)
        grid_cols:       number of patch columns in the original spatial grid.
        grid_rows:       number of patch rows (alternative to grid_cols).
        patch_positions: (N, 2) int32 array of (grid_row, grid_col) per patch.
                         When supplied, patches are placed at their correct spatial
                         positions (NaN-gaps for QC-filtered patches) rather than
                         being packed sequentially — which shifts all downstream
                         patches when any patches were filtered out.

    Returns:
        mosaic: np.ndarray of shape (grid_rows*H, grid_cols*W), NaN-padded.
    """
    N, H, W = patches.shape

    if patch_positions is not None and np.any(patch_positions >= 0):
        valid_mask = (patch_positions[:, 0] >= 0) & (patch_positions[:, 1] >= 0)
        rows = int(patch_positions[valid_mask, 0].max()) + 1
        cols = int(patch_positions[valid_mask, 1].max()) + 1
        if grid_cols is not None:
            cols = max(cols, int(grid_cols))
        if grid_rows is not None:
            rows = max(rows, int(grid_rows))
        canvas = np.full((rows * H, cols * W), np.nan, dtype=patches.dtype)
        for idx in range(N):
            gr, gc = int(patch_positions[idx, 0]), int(patch_positions[idx, 1])
            if gr < 0 or gc < 0 or gr >= rows or gc >= cols:
                continue
            canvas[gr * H:(gr + 1) * H, gc * W:(gc + 1) * W] = patches[idx]
        return canvas

    if grid_cols is not None:
        cols = int(grid_cols)
        rows = int(np.ceil(N / cols))
    elif grid_rows is not None:
        rows = int(grid_rows)
        cols = int(np.ceil(N / rows))
    else:
        cols = int(np.ceil(np.sqrt(N)))
        rows = int(np.ceil(N / cols))

    # Pad so we have a full rows×cols grid
    padded = np.full((rows * cols, H, W), np.nan, dtype=patches.dtype)
    padded[:N] = patches

    # Reshape: (rows, cols, H, W) → interleave spatial axes → (rows*H, cols*W)
    mosaic = padded.reshape(rows, cols, H, W).transpose(0, 2, 1, 3).reshape(rows * H, cols * W)
    return mosaic


def stage_postprocess(args, results: dict, data_dir: Path) -> tuple:
    _section("STAGE 5 — POST-PROCESSING")
    post_processor = PostProcessor()

    # Process ALL available patches
    n_total    = len(results["ensemble"])
    n_analysis = min(n_total, args.max_patches_analysis)

    processed = []
    for i in range(n_total):
        processed.append(post_processor.process(results["ensemble"][i]))
    lst_maps_processed = np.array(processed)
    np.save(data_dir / "lst_processed.npy", lst_maps_processed)

    # ── Determine grid layout ──────────────────────────────────────────────────
    # Priority: CLI arg  >  saved metadata  >  sqrt fallback (may jumble patches!)
    grid_cols = getattr(args, "patch_grid_cols", None)

    if grid_cols is not None:
        grid_shape = None   # derive rows inside _mosaic_patches
        logger.info(f"  Grid layout  : {grid_cols} cols (from --patch-grid-cols)")
    else:
        grid_shape = _load_patch_grid_shape(Path(args.test_data))
        if grid_shape is not None:
            grid_cols = grid_shape[1]
            logger.info(f"  Grid layout  : {grid_shape[0]}r × {grid_shape[1]}c "                        f"(from dataset metadata)")
        else:
            # sqrt fallback — warn loudly
            grid_cols = int(np.ceil(np.sqrt(n_total)))
            logger.warning(
                f"  ⚠ No patch grid metadata found and --patch-grid-cols not set. "                f"Falling back to sqrt grid ({grid_cols} cols for {n_total} patches). "                f"This will JUMBLE the mosaic unless the study area is square in "                f"patch-count terms.  Re-run with --patch-grid-cols=<N> to fix."            )

    # ── Mosaic ALL patches into a single full-area map ─────────────────────────
    # Load per-patch spatial positions for position-aware mosaic assembly.
    _pos_path = Path(args.test_data) / "patch_positions.npy"
    _patch_positions = None
    if _pos_path.exists():
        try:
            _patch_positions = np.load(_pos_path)
            logger.info(f"  Loaded patch_positions.npy ({len(_patch_positions)} entries)")
        except Exception as _pe:
            logger.warning(f"  Could not load patch_positions.npy: {_pe}")

    lst_processed = _mosaic_patches(lst_maps_processed, grid_cols=grid_cols,
                                    patch_positions=_patch_positions)
    np.save(data_dir / "lst_processed_mosaic.npy", lst_processed)

    logger.info(f"  Processed    : {n_total} patches  (analysis cap: {n_analysis})")
    logger.info(f"  Mosaic shape : {lst_processed.shape}  "                f"(from {n_total} patches of "                f"{lst_maps_processed.shape[1]}×{lst_maps_processed.shape[2]})")
    logger.info(f"  Mosaic stats : mean={np.nanmean(lst_processed):.2f}°C  "                f"std={np.nanstd(lst_processed):.2f}°C  "                f"range=[{np.nanmin(lst_processed):.1f}, "                f"{np.nanmax(lst_processed):.1f}]°C")

    return lst_processed, lst_maps_processed


# ── Stage 6: UHI analysis ────────────────────────────────────────────────────

def stage_uhi_analysis(args, lst_processed: np.ndarray, data_dir: Path) -> dict:
    _section("STAGE 6 — UHI ANALYSIS")

    logger.info(f"  Input map shape : {lst_processed.shape}  "
                f"(full mosaicked area — {lst_processed.shape[0]}×{lst_processed.shape[1]} px)")

    urban_thr = float(np.nanpercentile(lst_processed, args.urban_percentile))
    rural_thr = float(np.nanpercentile(lst_processed, args.rural_percentile))
    urban_mask = lst_processed >= urban_thr
    rural_mask = lst_processed <= rural_thr

    logger.info(f"  Urban threshold  : {urban_thr:.2f}°C  "
                f"({urban_mask.mean()*100:.1f}% of pixels)")
    logger.info(f"  Rural threshold  : {rural_thr:.2f}°C  "
                f"({rural_mask.mean()*100:.1f}% of pixels)")

    analyzer = UHIAnalyzer(lst_processed)
    ref_temps  = analyzer.define_reference_areas(urban_mask, rural_mask)
    uhi_map    = analyzer.calculate_uhi_intensity()
    classified, categories = analyzer.classify_uhi_intensity()
    uhi_stats  = analyzer.calculate_statistics()

    np.save(data_dir / "uhi_intensity.npy",  uhi_map)
    np.save(data_dir / "uhi_classified.npy", classified)

    logger.info(f"  Mean UHI : {uhi_stats['mean_intensity']:.2f}°C")
    logger.info(f"  Max  UHI : {uhi_stats['max_intensity']:.2f}°C")
    logger.info(f"  Extent>2°C : {uhi_stats['spatial_extent_km2']:.2f} km²")

    return dict(
        analyzer=analyzer,
        uhi_map=uhi_map,
        classified=classified,
        categories=categories,
        uhi_stats=uhi_stats,
        urban_mask=urban_mask,
        rural_mask=rural_mask,
        ref_temps=ref_temps,
    )


# ── Stage 7: Hotspot detection ────────────────────────────────────────────────

def stage_hotspots(args, lst_processed: np.ndarray, data_dir: Path) -> dict:
    _section("STAGE 7 — HOTSPOT DETECTION")

    detector     = HotspotDetector(lst_processed, resolution=50.0)
    gi_star      = detector.calculate_gi_star(search_radius=args.hotspot_radius)
    hotspot_mask, hotspot_list = detector.identify_hotspots(args.hotspot_confidence)

    hotspots_df = pd.DataFrame()
    if hotspot_list:
        hotspots_df = detector.prioritize_hotspots(hotspot_list)
        hotspots_df.to_csv(data_dir / "hotspots.csv", index=False)
        logger.info(f"  Detected {len(hotspots_df)} hotspot regions")
    else:
        logger.warning("  ⚠ No hotspots detected — try lowering --hotspot-confidence")

    np.save(data_dir / "gi_star.npy",       gi_star)
    np.save(data_dir / "hotspot_mask.npy",  hotspot_mask)

    return dict(
        detector=detector,
        gi_star=gi_star,
        hotspot_mask=hotspot_mask,
        hotspots_df=hotspots_df,
    )


# ── Stage 8: Validation ───────────────────────────────────────────────────────

def stage_validation(args, predictor, results, y_test,
                     norm_stats_path, maps_dir, reports_dir) -> dict | None:
    if y_test is None or args.skip_validation:
        logger.info("  Skipping validation (no y_test or --skip-validation)")
        return None

    _section("STAGE 8 — VALIDATION")

    normalizer  = DataNormalizer(norm_stats_path)
    y_celsius   = normalizer.denormalize_predictions(y_test.squeeze(), "target")
    y_celsius   = y_celsius.reshape(y_test.shape)

    # If TTA wasn't used, run TTA now for final metric predictions
    if args.use_tta:
        logger.info("  Using TTA predictions already in results for metrics")
        pred_arr = results["ensemble"]
    else:
        logger.info("  Running TTA pass for validation metrics …")
        try:
            tta_res = predictor.predict_ensemble_with_tta(
                _load_X_test(args),          # reload (avoid mutation side-effects)
                batch_size=args.batch_size,
                n_augmentations=args.tta_augs,
                return_uncertainty=True,
                use_spatial_ensemble=args.use_spatial_ensemble,
            )
            pred_arr = tta_res["ensemble"]
        except Exception as exc:
            logger.warning(f"  TTA for validation failed ({exc}), using non-TTA predictions")
            pred_arr = results["ensemble"]

    pred_flat = pred_arr.flatten()
    gt_flat   = y_celsius.flatten()
    mask      = ~(np.isnan(pred_flat) | np.isnan(gt_flat))
    pred_flat, gt_flat = pred_flat[mask], gt_flat[mask]

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2   = r2_score(gt_flat, pred_flat)
    rmse = float(np.sqrt(mean_squared_error(gt_flat, pred_flat)))
    mae  = float(mean_absolute_error(gt_flat, pred_flat))
    mbe  = float(pred_flat.mean() - gt_flat.mean())

    slope, intercept, r_val, p_val, std_err = stats.linregress(gt_flat, pred_flat)
    residuals = pred_flat - gt_flat
    sw_stat, sw_p = stats.shapiro(
        residuals[np.random.choice(len(residuals), min(5000, len(residuals)), replace=False)])

    metrics = dict(
        r2=float(r2), rmse=rmse, mae=mae, mbe=mbe,
        slope=float(slope), intercept=float(intercept),
        correlation=float(r_val), p_value=float(p_val),
        std_err=float(std_err),
        residual_std=float(residuals.std()),
        residual_skew=float(stats.skew(residuals)),
        residual_kurt=float(stats.kurtosis(residuals)),
        shapiro_stat=float(sw_stat), shapiro_p=float(sw_p),
    )

    logger.info(f"  R²   : {r2:.4f}")
    logger.info(f"  RMSE : {rmse:.4f}°C")
    logger.info(f"  MAE  : {mae:.4f}°C")
    logger.info(f"  MBE  : {mbe:.4f}°C")
    logger.info(f"  Slope: {slope:.4f}  Intercept: {intercept:.4f}")
    logger.info(f"  Residuals {'NORMAL' if sw_p > 0.05 else 'NON-NORMAL'} "
                f"(Shapiro p={sw_p:.2e})")

    with open(reports_dir / "validation_metrics.json", "w") as f:
        json.dump(make_json_serializable(metrics), f, indent=2)

    # Basic validation plot (compact — full diagnostics come in stage 9)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    idx = np.random.choice(len(pred_flat), min(8000, len(pred_flat)), replace=False)
    axes[0].scatter(gt_flat[idx], pred_flat[idx], alpha=0.3, s=1, c="steelblue")
    mn, mx = gt_flat.min(), gt_flat.max()
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="1:1")
    axes[0].plot([mn, mx], [slope*mn+intercept, slope*mx+intercept],
                 "b-", lw=1.5, label=f"fit y={slope:.3f}x+{intercept:.3f}")
    axes[0].set(xlabel="Ground Truth (°C)", ylabel="Predicted (°C)",
                title=f"Predicted vs Observed  R²={r2:.4f}  RMSE={rmse:.3f}°C")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].scatter(gt_flat[idx], residuals[idx], alpha=0.3, s=1, c="steelblue")
    axes[1].axhline(0,   color="red",    ls="--", lw=1.5)
    axes[1].axhline(mbe, color="orange", ls="--", lw=1.5, label=f"MBE={mbe:.3f}°C")
    axes[1].set(xlabel="Ground Truth (°C)", ylabel="Residual (°C)",
                title=f"Residuals  MAE={mae:.3f}°C")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(maps_dir / "validation_basic.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Basic validation plot → {maps_dir / 'validation_basic.png'}")

    return dict(metrics=metrics, pred_flat=pred_flat, gt_flat=gt_flat)


def _load_X_test(args) -> np.ndarray:
    return np.load(Path(args.test_data) / "X.npy")


# ── Stage 9: Diagnostics ─────────────────────────────────────────────────────

def stage_diagnostics(args, lst_processed, uhi_ctx, hotspot_ctx,
                       val_ctx, results, maps_dir) -> None:
    if args.skip_diagnostics:
        logger.info("  --skip-diagnostics set — skipping all diagnostic plots")
        return

    _section("STAGE 9 — DIAGNOSTIC PLOTS (uhi_analysis)")

    diag_dir = maps_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # ── 9a. run_all_diagnostics (covers LST distribution, UHI intensity,
    #         classification, autocorrelation, Gi*, hotspot ranking, validation)
    _try(
        "run_all_diagnostics",
        run_all_diagnostics,
        lst_map=lst_processed,
        uhi_map=uhi_ctx["uhi_map"],
        gi_star=hotspot_ctx["gi_star"],
        hotspots_df=hotspot_ctx["hotspots_df"] if len(hotspot_ctx["hotspots_df"]) else None,
        predictions=val_ctx["pred_flat"] if val_ctx else None,
        ground_truth=val_ctx["gt_flat"]  if val_ctx else None,
        output_dir=diag_dir,
        urban_mask=uhi_ctx["urban_mask"],
        rural_mask=uhi_ctx["rural_mask"],
    )

    # ── 9b. Extended ValidationAnalyzer plot (6-panel) ───────────────────────
    if val_ctx:
        val_analyzer = ValidationAnalyzer(val_ctx["pred_flat"], val_ctx["gt_flat"])
        _try(
            "ValidationAnalyzer.plot_validation (extended 6-panel)",
            val_analyzer.plot_validation,
            diag_dir / "diag_validation_extended.png",
        )

    logger.info(f"  Diagnostic figures → {diag_dir}")


# ── Stage 10: Standard visualizations ────────────────────────────────────────

def stage_visualizations(args, lst_processed, uhi_ctx, hotspot_ctx,
                          results, val_ctx, maps_dir) -> None:
    _section("STAGE 10 — STANDARD VISUALIZATIONS (uhi_visualization)")

    # ── Resolve grid_cols with the same 3-priority logic as stage_postprocess ─
    # getattr(args, "patch_grid_cols") alone is not enough: it returns None when
    # the CLI flag is omitted, causing all _mosaic_patches calls here to fall
    # back to the sqrt heuristic and produce the horizontal-stripe artefact.
    _gc = getattr(args, "patch_grid_cols", None)
    if _gc is None:
        _gs = _load_patch_grid_shape(Path(args.test_data))
        if _gs is not None:
            _gc = _gs[1]
            logger.info(f"  Visualizations grid_cols: {_gc} (from dataset metadata)")
        else:
            n_patches = len(results.get("ensemble", []))
            _gc = int(np.ceil(np.sqrt(n_patches))) if n_patches > 0 else None
            logger.warning(
                f"  ⚠ No patch grid metadata found for visualizations. "                f"Using sqrt fallback (grid_cols={_gc}). "                f"Re-run preprocessing or pass --patch-grid-cols=<N> to fix."
            )
    # Store so inner lambdas / _try calls can reference it
    _resolved_grid_cols = _gc

    # ── Load per-patch positions for position-aware mosaic assembly ───────────
    _pos_path = Path(args.test_data) / "patch_positions.npy"
    _patch_positions = None
    if _pos_path.exists():
        try:
            _patch_positions = np.load(_pos_path)
        except Exception:
            pass

    vis = UHIVisualizer(figsize=(12, 10), dpi=300)

    # 10a. Core maps ──────────────────────────────────────────────────────────
    _try("LST map",
         vis.create_lst_map, lst_processed,
         maps_dir / "lst_map.png",
         title=f"Land Surface Temperature — {STUDY_AREA['name']}")

    _try("UHI intensity map",
         vis.create_uhi_intensity_map, uhi_ctx["uhi_map"],
         maps_dir / "uhi_intensity_map.png",
         title="Urban Heat Island Intensity")

    _try("Hotspot map",
         vis.create_hotspot_map,
         lst_processed, hotspot_ctx["hotspot_mask"], hotspot_ctx["gi_star"],
         maps_dir / "hotspot_map.png",
         title="UHI Hotspot Analysis — Gi*")

    # 10b. Uncertainty ────────────────────────────────────────────────────────
    unc_key = results.get("_unc_key", "uncertainty")
    uncertainty = results.get(unc_key)
    if uncertainty is not None:
        # Mosaic uncertainty patches to match the full lst_processed mosaic
        if uncertainty.ndim == 3:
            unc_mosaic = _mosaic_patches(uncertainty, grid_cols=_resolved_grid_cols,
                                         patch_positions=_patch_positions)
        else:
            unc_mosaic = uncertainty
        _try("Uncertainty map",
             vis.create_uncertainty_map, lst_processed, unc_mosaic,
             maps_dir / "uncertainty_map.png",
             title="Prediction Uncertainty (σ)")

        # Use ensemble mosaic for uncertainty analysis plot
        ens_mosaic = _mosaic_patches(results["ensemble"], grid_cols=_resolved_grid_cols,
                                     patch_positions=_patch_positions)
        _try("Uncertainty analysis plot",
             vis.create_uncertainty_analysis_plot,
             ens_mosaic, unc_mosaic,
             maps_dir / "uncertainty_analysis.png",
             ground_truth=None)  # ground truth is patch-level; skip spatial reshape

    # 10c. Statistics dashboard ───────────────────────────────────────────────
    _try("Statistics dashboard",
         vis.create_statistics_dashboard,
         uhi_ctx["uhi_stats"], uhi_ctx["categories"],
         hotspot_ctx["hotspots_df"],
         maps_dir / "statistics_dashboard.png")

    # 10d. Comprehensive dashboard ────────────────────────────────────────────
    _try("Comprehensive dashboard",
         vis.create_comprehensive_dashboard,
         lst_processed,
         uhi_ctx["uhi_map"],
         hotspot_ctx["gi_star"],
         hotspot_ctx["hotspots_df"],
         uhi_ctx["uhi_stats"],
         uhi_ctx["categories"],
         maps_dir / "comprehensive_dashboard.png",
         uncertainty=unc_mosaic if uncertainty is not None else None)

    # 10e. Urban vs rural comparison ──────────────────────────────────────────
    _try("Urban vs rural comparison",
         vis.create_urban_rural_comparison_plot,
         lst_processed,
         uhi_ctx["urban_mask"],
         uhi_ctx["rural_mask"],
         maps_dir / "urban_rural_comparison.png")

    # 10f. Model comparison (CNN / GBM / Ensemble) ────────────────────────────
    # Build per-model mosaics so the comparison shows the full area
    model_results = {}
    for k in ("cnn", "gbm", "ensemble"):
        if k in results:
            arr = results[k]
            if arr.ndim == 3:          # (N, H, W)
                model_results[k] = _mosaic_patches(arr, grid_cols=_resolved_grid_cols,
                                                   patch_positions=_patch_positions)
            elif arr.ndim == 1:        # GBM patch-average scalar per patch
                # Broadcast each scalar to a patch-sized tile, then mosaic
                H, W = results["ensemble"].shape[1], results["ensemble"].shape[2]
                tiles = np.stack([np.full((H, W), v) for v in arr])
                model_results[k] = _mosaic_patches(tiles, grid_cols=_resolved_grid_cols,
                                                   patch_positions=_patch_positions)
    if model_results:
        _try("Model comparison plot",
             vis.create_model_comparison_plot,
             model_results,
             None,          # ground truth cannot be reshaped to mosaic size
             maps_dir / "model_comparison.png")

    # 10g. Ensemble weights ───────────────────────────────────────────────────
    ens_cfg_path = Path(args.model_dir) / "ensemble_config.json"
    if ens_cfg_path.exists():
        with open(ens_cfg_path) as f:
            ens_cfg = json.load(f)
        final_weights = ens_cfg.get("weights")
        _try("Ensemble weights plot",
             vis.create_ensemble_weights_plot,
             weights_history=None,
             final_weights=final_weights,
             output_path=maps_dir / "ensemble_weights.png")

    # 10h. Error heatmap (needs ground truth) ─────────────────────────────────
    if val_ctx:
        _try("Spatial error heatmap",
             vis.create_error_heatmap,
             results["ensemble"],
             val_ctx["gt_flat"].reshape(results["ensemble"].shape)
             if val_ctx["gt_flat"].size == results["ensemble"].size
             else val_ctx["gt_flat"].reshape(-1, *results["ensemble"].shape[1:])[:results["ensemble"].shape[0]],
             maps_dir / "spatial_error_heatmap.png")

    # 10i. Feature importance (GBM) ───────────────────────────────────────────
    gbm_path = (Path(args.model_dir) / "best_gbm_model.pkl"
                if (Path(args.model_dir) / "best_gbm_model.pkl").exists()
                else Path(args.model_dir) / "gbm_model.pkl")
    if gbm_path.exists():
        try:
            import pickle
            with open(gbm_path, "rb") as f:
                gbm_model = pickle.load(f)
            feat_names   = gbm_model.feature_name()
            importances  = np.array(gbm_model.feature_importance(importance_type="gain"))
            _try("Feature importance plot",
                 vis.create_feature_importance_plot,
                 feat_names, importances,
                 maps_dir / "feature_importance.png")
        except Exception as exc:
            logger.warning(f"  Feature importance plot skipped: {exc}")

    logger.info(f"  Visualization figures → {maps_dir}")


# ── Stage 11: GeoTIFF export ──────────────────────────────────────────────────

def stage_geotiff(args, lst_processed, uhi_ctx, hotspot_ctx,
                  bounds, geotiff_dir) -> None:
    if args.skip_geotiff:
        return

    _section("STAGE 11 — GEOTIFF EXPORT")
    gen = OutputGenerator(geotiff_dir)

    _try("LST GeoTIFF",
         gen.export_geotiff, lst_processed.astype(np.float32),
         "lst_map.tif", bounds=bounds,
         metadata={"product": "LST", "units": "celsius"})

    _try("UHI GeoTIFF",
         gen.export_geotiff, uhi_ctx["uhi_map"].astype(np.float32),
         "uhi_intensity.tif", bounds=bounds,
         metadata={"product": "UHI_intensity", "units": "celsius_difference"})

    _try("Gi* GeoTIFF",
         gen.export_geotiff, hotspot_ctx["gi_star"].astype(np.float32),
         "gi_star.tif", bounds=bounds,
         metadata={"product": "Getis_Ord_Gi_star"})

    _try("Hotspot CSV",
         gen.export_csv, hotspot_ctx["hotspots_df"],
         "hotspots.csv")

    logger.info(f"  GeoTIFF exports → {geotiff_dir}")


# ── Stage 12: Web map ─────────────────────────────────────────────────────────

def stage_webmap(args, lst_processed, uhi_ctx, hotspot_ctx,
                 bounds, output_dir) -> None:
    if args.skip_webmap:
        return

    _section("STAGE 12 — WEB MAP")
    try:
        from uhi_map_overlay import RealMapOverlay
        overlay   = RealMapOverlay(bounds)
        webmap_dir = output_dir / "webmap"
        overlay.export_to_webmap(
            lst_processed, uhi_ctx["uhi_map"],
            hotspot_ctx["hotspots_df"], webmap_dir)
        logger.info(f"  ✓ Web map → {webmap_dir / 'index.html'}")
        logger.info("    Open the .html file in any browser to explore interactively")
    except ImportError:
        logger.warning("  uhi_map_overlay not available — skipping web map")
    except Exception as exc:
        logger.error(f"  Web map failed: {exc}")


# ── Stage 13: Report ─────────────────────────────────────────────────────────

def stage_report(uhi_ctx, hotspot_ctx, val_ctx, reports_dir) -> None:
    _section("STAGE 13 — REPORT")
    val_metrics = val_ctx["metrics"] if val_ctx else None
    _try("generate_report",
         generate_report,
         uhi_ctx["uhi_stats"],
         hotspot_ctx["hotspots_df"],
         val_metrics,
         reports_dir / "uhi_analysis_report.json")


# ══════════════════════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(args, uhi_ctx, hotspot_ctx, val_ctx,
                  lst_processed, output_dir) -> None:
    _section("PIPELINE COMPLETE")

    maps_dir     = output_dir / "maps"
    data_dir     = output_dir / "data"
    reports_dir  = output_dir / "reports"
    geotiff_dir  = output_dir / "geotiff"
    diag_dir     = maps_dir   / "diagnostics"

    logger.info("\n📁  Output directories:")
    logger.info(f"    📊 Maps          : {maps_dir}")
    logger.info(f"    🔬 Diagnostics   : {diag_dir}")
    logger.info(f"    💾 Data          : {data_dir}")
    logger.info(f"    📄 Reports       : {reports_dir}")
    if not args.skip_geotiff:
        logger.info(f"    🗺  GeoTIFF      : {geotiff_dir}")
    if not args.skip_webmap:
        logger.info(f"    🌐 Web Map       : {output_dir / 'webmap' / 'index.html'}")

    logger.info("\n📈  Results:")
    logger.info(f"    LST  : {np.nanmean(lst_processed):.2f} ± {np.nanstd(lst_processed):.2f} °C  "
                f"(mosaic {lst_processed.shape[0]}×{lst_processed.shape[1]} px)")
    logger.info(f"    UHI  : mean={uhi_ctx['uhi_stats']['mean_intensity']:.2f}°C  "
                f"max={uhi_ctx['uhi_stats']['max_intensity']:.2f}°C  "
                f"extent={uhi_ctx['uhi_stats']['spatial_extent_km2']:.2f} km²")
    logger.info(f"    Hotspots : {len(hotspot_ctx['hotspots_df'])}")

    if val_ctx:
        m = val_ctx["metrics"]
        logger.info(f"\n🎯  Validation:")
        logger.info(f"    R²   : {m['r2']:.4f}")
        logger.info(f"    RMSE : {m['rmse']:.4f} °C")
        logger.info(f"    MAE  : {m['mae']:.4f} °C")
        logger.info(f"    MBE  : {m['mbe']:.4f} °C")

    # Enumerate saved figures
    all_pngs = sorted(output_dir.rglob("*.png"))
    logger.info(f"\n🖼   Saved figures : {len(all_pngs)}")
    for p in all_pngs:
        logger.info(f"    {p.relative_to(output_dir)}")

    logger.info("")
    logger.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# main()
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    _section("UHI ANALYSIS PIPELINE — FULLY INTEGRATED")
    logger.info(f"  Study area   : {STUDY_AREA['name']}")
    logger.info(f"  Model dir    : {args.model_dir}")
    logger.info(f"  Output dir   : {args.output_dir}")
    logger.info(f"  Test data    : {args.test_data}")
    logger.info(f"  TTA          : {args.use_tta}  (n={args.tta_augs})")
    logger.info(f"  MC samples   : {args.mc_samples}")
    logger.info(f"  Hotspot r    : {args.hotspot_radius} m  "
                f"CI={args.hotspot_confidence}")

    # ── Determinism — must be called BEFORE any torch / numpy random ops ──────
    _section("DETERMINISM SETUP")
    _set_deterministic_mode(seed=args.seed)

    # ── Resolve shared paths ───────────────────────────────────────────────────
    output_dir       = Path(args.output_dir)
    norm_stats_path  = PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
    bounds = (
        STUDY_AREA["bounds"]["min_lon"],
        STUDY_AREA["bounds"]["min_lat"],
        STUDY_AREA["bounds"]["max_lon"],
        STUDY_AREA["bounds"]["max_lat"],
    )

    # ── Create output sub-directories ─────────────────────────────────────────
    maps_dir    = output_dir / "maps"
    data_dir    = output_dir / "data"
    reports_dir = output_dir / "reports"
    geotiff_dir = output_dir / "geotiff"
    for d in (maps_dir, data_dir, reports_dir, geotiff_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Execute pipeline stages
    # ══════════════════════════════════════════════════════════════════════════

    # 1. Load data
    try:
        data = stage_load_data(args)
    except Exception as exc:
        logger.error(f"Cannot continue without data: {exc}")
        return 1
    X_test, y_test = data["X_test"], data["y_test"]

    # 2. Init predictor
    try:
        predictor = stage_init_predictor(args, norm_stats_path)
    except Exception as exc:
        logger.error(f"Cannot continue without predictor: {exc}")
        return 1

    # 3. Adaptive calibration
    stage_adaptive_calibration(predictor, y_test, norm_stats_path)

    # 4. Inference
    try:
        results = stage_inference(args, predictor, X_test, data_dir)
    except Exception as exc:
        logger.error(f"Inference failed: {exc}"); traceback.print_exc(); return 1

    # 5. Post-processing
    try:
        lst_processed, lst_maps_processed = stage_postprocess(args, results, data_dir)
    except Exception as exc:
        logger.error(f"Post-processing failed: {exc}"); return 1

    # 6. UHI analysis
    try:
        uhi_ctx = stage_uhi_analysis(args, lst_processed, data_dir)
    except Exception as exc:
        logger.error(f"UHI analysis failed: {exc}"); return 1

    # 7. Hotspot detection
    try:
        hotspot_ctx = stage_hotspots(args, lst_processed, data_dir)
    except Exception as exc:
        logger.error(f"Hotspot detection failed: {exc}"); return 1

    # 8. Validation
    val_ctx = _try(
        "Validation stage",
        stage_validation,
        args, predictor, results, y_test,
        norm_stats_path, maps_dir, reports_dir,
    )

    # 9. Diagnostic plots (uhi_analysis)
    _try("Diagnostic plots",
         stage_diagnostics,
         args, lst_processed, uhi_ctx, hotspot_ctx, val_ctx, results, maps_dir)

    # 10. Standard visualizations (uhi_visualization)
    _try("Standard visualizations",
         stage_visualizations,
         args, lst_processed, uhi_ctx, hotspot_ctx, results, val_ctx, maps_dir)

    # 11. GeoTIFF
    _try("GeoTIFF export",
         stage_geotiff,
         args, lst_processed, uhi_ctx, hotspot_ctx, bounds, geotiff_dir)

    # 12. Web map
    _try("Web map",
         stage_webmap,
         args, lst_processed, uhi_ctx, hotspot_ctx, bounds, output_dir)

    # 13. Report
    _try("Report",
         stage_report,
         uhi_ctx, hotspot_ctx, val_ctx, reports_dir)

    # ── Final summary ──────────────────────────────────────────────────────────
    print_summary(args, uhi_ctx, hotspot_ctx, val_ctx, lst_processed, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())