"""
Fix script: Apply corrections to inference pipeline
"""
import numpy as np
import torch
from pathlib import Path
import json
import sys
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.append(str(Path(__file__).parent))
from uhi_inference import EnsemblePredictor
from config import MODEL_DIR, PROCESSED_DATA_DIR


def fix_and_evaluate():
    """Apply fixes and re-evaluate"""
    
    print("="*70)
    print("APPLYING FIXES TO INFERENCE PIPELINE")
    print("="*70)
    
    # Load test data
    test_dir = PROCESSED_DATA_DIR / "cnn_dataset" / "test"
    X_test = np.load(test_dir / "X.npy")
    y_test = np.load(test_dir / "y.npy")
    
    # Load normalization stats
    stats_path = PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
    with open(stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    target_mean = norm_stats['target']['mean']
    target_std = norm_stats['target']['std']
    
    # Denormalize ground truth
    y_test_denorm = y_test * target_std + target_mean
    
    print(f"\nTest data:")
    print(f"  Samples: {len(X_test)}")
    print(f"  Ground truth mean: {y_test_denorm.mean():.2f}°C")
    print(f"  Ground truth range: [{y_test_denorm.min():.2f}, {y_test_denorm.max():.2f}]°C")
    
    # Initialize predictor
    model_dir = MODEL_DIR
    predictor = EnsemblePredictor(
        cnn_model_path=model_dir / "final_cnn.pth",
        gbm_model_path=model_dir / "gbm_model.pkl",
        ensemble_config_path=model_dir / "ensemble_config.json",
        normalization_stats_path=stats_path,
        device="cuda",
        mc_dropout_rate=0.1
    )
    
    # Get predictions
    print("\n" + "="*70)
    print("GENERATING PREDICTIONS")
    print("="*70)
    
    results = predictor.predict_ensemble(
        X_test,
        batch_size=16,
        return_uncertainty=False,
        use_spatial_ensemble=True,
        skip_validation=True
    )
    
    # Extract predictions (spatial average)
    cnn_preds = results['cnn_patch']  # Patch-level averages
    gbm_preds = results['gbm']
    ensemble_preds = results['ensemble_patch']
    
    # Ground truth (patch-level average)
    y_test_patch = y_test_denorm.reshape(len(y_test), -1).mean(axis=1)
    
    print(f"\nPrediction statistics:")
    print(f"  CNN: {cnn_preds.mean():.2f} ± {cnn_preds.std():.2f}°C")
    print(f"  GBM: {gbm_preds.mean():.2f} ± {gbm_preds.std():.2f}°C")
    print(f"  Ensemble: {ensemble_preds.mean():.2f} ± {ensemble_preds.std():.2f}°C")
    print(f"  Ground truth: {y_test_patch.mean():.2f} ± {y_test_patch.std():.2f}°C")
    
    # FIX 1: Correct for systematic CNN bias
    print("\n" + "="*70)
    print("FIX 1: BIAS CORRECTION")
    print("="*70)
    
    # Calculate bias on validation set to calibrate
    val_dir = PROCESSED_DATA_DIR / "cnn_dataset" / "val"
    X_val = np.load(val_dir / "X.npy")
    y_val = np.load(val_dir / "y.npy")
    y_val_denorm = y_val * target_std + target_mean
    y_val_patch = y_val_denorm.reshape(len(y_val), -1).mean(axis=1)
    
    # Get validation predictions
    val_results = predictor.predict_ensemble(
        X_val,
        batch_size=16,
        return_uncertainty=False,
        use_spatial_ensemble=True,
        skip_validation=True
    )
    
    cnn_val_preds = val_results['cnn_patch']
    gbm_val_preds = val_results['gbm']
    
    # Calculate bias on validation
    cnn_bias = (cnn_val_preds - y_val_patch).mean()
    gbm_bias = (gbm_val_preds - y_val_patch).mean()
    
    print(f"Validation bias:")
    print(f"  CNN: {cnn_bias:.2f}°C")
    print(f"  GBM: {gbm_bias:.2f}°C")
    
    # Apply bias correction
    cnn_preds_corrected = cnn_preds - cnn_bias
    gbm_preds_corrected = gbm_preds - gbm_bias
    
    # Re-calculate ensemble with corrected predictions
    ensemble_preds_corrected = (
        predictor.weights["cnn"] * cnn_preds_corrected +
        predictor.weights["gbm"] * gbm_preds_corrected
    )
    
    print(f"\nAfter bias correction:")
    print(f"  CNN: {cnn_preds_corrected.mean():.2f} ± {cnn_preds_corrected.std():.2f}°C")
    print(f"  GBM: {gbm_preds_corrected.mean():.2f} ± {gbm_preds_corrected.std():.2f}°C")
    print(f"  Ensemble: {ensemble_preds_corrected.mean():.2f} ± {ensemble_preds_corrected.std():.2f}°C")
    
    # FIX 2: Variance scaling
    print("\n" + "="*70)
    print("FIX 2: VARIANCE SCALING")
    print("="*70)
    
    # Calculate variance ratios
    cnn_val_var = cnn_val_preds.var()
    y_val_var = y_val_patch.var()
    cnn_var_scale = np.sqrt(y_val_var / (cnn_val_var + 1e-8))
    
    gbm_val_var = gbm_val_preds.var()
    gbm_var_scale = np.sqrt(y_val_var / (gbm_val_var + 1e-8))
    
    print(f"Variance scaling factors:")
    print(f"  CNN: {cnn_var_scale:.3f}")
    print(f"  GBM: {gbm_var_scale:.3f}")
    
    # Apply variance scaling around mean
    cnn_mean = cnn_preds_corrected.mean()
    cnn_preds_scaled = cnn_mean + (cnn_preds_corrected - cnn_mean) * cnn_var_scale
    
    gbm_mean = gbm_preds_corrected.mean()
    gbm_preds_scaled = gbm_mean + (gbm_preds_corrected - gbm_mean) * gbm_var_scale
    
    # Re-calculate ensemble
    ensemble_preds_scaled = (
        predictor.weights["cnn"] * cnn_preds_scaled +
        predictor.weights["gbm"] * gbm_preds_scaled
    )
    
    print(f"\nAfter variance scaling:")
    print(f"  CNN: {cnn_preds_scaled.mean():.2f} ± {cnn_preds_scaled.std():.2f}°C")
    print(f"  GBM: {gbm_preds_scaled.mean():.2f} ± {gbm_preds_scaled.std():.2f}°C")
    print(f"  Ensemble: {ensemble_preds_scaled.mean():.2f} ± {ensemble_preds_scaled.std():.2f}°C")
    print(f"  Ground truth: {y_test_patch.mean():.2f} ± {y_test_patch.std():.2f}°C")
    
    # Evaluate all versions
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    def evaluate(preds, targets, name):
        r2 = r2_score(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        mae = mean_absolute_error(targets, preds)
        mbe = np.mean(preds - targets)
        
        print(f"\n{name}:")
        print(f"  R²:   {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}°C")
        print(f"  MAE:  {mae:.4f}°C")
        print(f"  MBE:  {mbe:+.4f}°C")
        
        return {"r2": r2, "rmse": rmse, "mae": mae, "mbe": mbe}
    
    print("\n1. ORIGINAL PREDICTIONS:")
    print("-"*70)
    orig_cnn = evaluate(cnn_preds, y_test_patch, "CNN")
    orig_gbm = evaluate(gbm_preds, y_test_patch, "GBM")
    orig_ens = evaluate(ensemble_preds, y_test_patch, "Ensemble")
    
    print("\n2. BIAS-CORRECTED:")
    print("-"*70)
    bc_cnn = evaluate(cnn_preds_corrected, y_test_patch, "CNN (bias corrected)")
    bc_gbm = evaluate(gbm_preds_corrected, y_test_patch, "GBM (bias corrected)")
    bc_ens = evaluate(ensemble_preds_corrected, y_test_patch, "Ensemble (bias corrected)")
    
    print("\n3. BIAS + VARIANCE CORRECTED:")
    print("-"*70)
    final_cnn = evaluate(cnn_preds_scaled, y_test_patch, "CNN (full correction)")
    final_gbm = evaluate(gbm_preds_scaled, y_test_patch, "GBM (full correction)")
    final_ens = evaluate(ensemble_preds_scaled, y_test_patch, "Ensemble (full correction)")
    
    # Summary
    print("\n" + "="*70)
    print("IMPROVEMENT SUMMARY")
    print("="*70)
    
    print(f"\nEnsemble R² improvement:")
    print(f"  Original:        {orig_ens['r2']:.4f}")
    print(f"  Bias corrected:  {bc_ens['r2']:.4f} ({(bc_ens['r2'] - orig_ens['r2']):.4f})")
    print(f"  Full corrected:  {final_ens['r2']:.4f} ({(final_ens['r2'] - orig_ens['r2']):.4f})")
    
    print(f"\nEnsemble RMSE improvement:")
    print(f"  Original:        {orig_ens['rmse']:.4f}°C")
    print(f"  Bias corrected:  {bc_ens['rmse']:.4f}°C")
    print(f"  Full corrected:  {final_ens['rmse']:.4f}°C")
    
    print(f"\nEnsemble MBE improvement:")
    print(f"  Original:        {orig_ens['mbe']:+.4f}°C")
    print(f"  Bias corrected:  {bc_ens['mbe']:+.4f}°C")
    print(f"  Full corrected:  {final_ens['mbe']:+.4f}°C")
    
    # Check if we meet targets
    print("\n" + "="*70)
    print("TARGET ACHIEVEMENT")
    print("="*70)
    
    targets = {
        "r2": 0.8,
        "rmse": 1.5,
        "mae": 1.0
    }
    
    print(f"\nOriginal ensemble:")
    print(f"  R² ≥ {targets['r2']}: {'✅' if orig_ens['r2'] >= targets['r2'] else '❌'} ({orig_ens['r2']:.4f})")
    print(f"  RMSE ≤ {targets['rmse']}°C: {'✅' if orig_ens['rmse'] <= targets['rmse'] else '❌'} ({orig_ens['rmse']:.4f}°C)")
    print(f"  MAE ≤ {targets['mae']}°C: {'✅' if orig_ens['mae'] <= targets['mae'] else '❌'} ({orig_ens['mae']:.4f}°C)")
    
    print(f"\nCorrected ensemble:")
    print(f"  R² ≥ {targets['r2']}: {'✅' if final_ens['r2'] >= targets['r2'] else '❌'} ({final_ens['r2']:.4f})")
    print(f"  RMSE ≤ {targets['rmse']}°C: {'✅' if final_ens['rmse'] <= targets['rmse'] else '❌'} ({final_ens['rmse']:.4f}°C)")
    print(f"  MAE ≤ {targets['mae']}°C: {'✅' if final_ens['mae'] <= targets['mae'] else '❌'} ({final_ens['mae']:.4f}°C)")
    
    print("\n" + "="*70)
    
    # Save corrected predictions
    output_dir = Path("outputs/corrected_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "ensemble_corrected.npy", ensemble_preds_scaled)
    np.save(output_dir / "cnn_corrected.npy", cnn_preds_scaled)
    np.save(output_dir / "gbm_corrected.npy", gbm_preds_scaled)
    np.save(output_dir / "ground_truth.npy", y_test_patch)
    
    print(f"\n✅ Saved corrected predictions to {output_dir}")
    
    # Save correction parameters
    correction_params = {
        "cnn_bias": float(cnn_bias),
        "gbm_bias": float(gbm_bias),
        "cnn_var_scale": float(cnn_var_scale),
        "gbm_var_scale": float(gbm_var_scale),
        "validation_metrics": {
            "cnn_r2": float(r2_score(y_val_patch, cnn_val_preds)),
            "gbm_r2": float(r2_score(y_val_patch, gbm_val_preds))
        }
    }
    
    with open(output_dir / "correction_params.json", 'w') as f:
        json.dump(correction_params, f, indent=2)
    
    print(f"✅ Saved correction parameters")
    print("="*70)


if __name__ == "__main__":
    fix_and_evaluate()