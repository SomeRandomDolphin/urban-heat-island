"""
Deep diagnosis: Check for preprocessing mismatches and model behavior
"""
import numpy as np
import torch
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent))
from uhi_inference import EnsemblePredictor
from config import MODEL_DIR, PROCESSED_DATA_DIR

def diagnose_inference_pipeline():
    """Diagnose the inference pipeline for issues"""
    
    print("="*70)
    print("DEEP INFERENCE DIAGNOSIS")
    print("="*70)
    
    # Load test data
    test_dir = PROCESSED_DATA_DIR / "cnn_dataset" / "test"
    X_test = np.load(test_dir / "X.npy")
    y_test = np.load(test_dir / "y.npy")
    
    print("\n1. TEST DATA INSPECTION")
    print("-"*70)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"X_test: mean={X_test.mean():.4f}, std={X_test.std():.4f}")
    print(f"y_test: mean={y_test.mean():.4f}, std={y_test.std():.4f}")
    
    # Load normalization stats
    stats_path = PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
    with open(stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    target_mean = norm_stats['target']['mean']
    target_std = norm_stats['target']['std']
    
    print(f"\nNormalization stats:")
    print(f"  Target mean: {target_mean:.2f}°C")
    print(f"  Target std: {target_std:.2f}°C")
    
    # Denormalize ground truth
    y_test_denorm = y_test * target_std + target_mean
    print(f"\nGround truth (denormalized):")
    print(f"  Mean: {y_test_denorm.mean():.2f}°C")
    print(f"  Std: {y_test_denorm.std():.2f}°C")
    print(f"  Range: [{y_test_denorm.min():.2f}, {y_test_denorm.max():.2f}]°C")
    
    # Initialize predictor
    print("\n2. INITIALIZING PREDICTOR")
    print("-"*70)
    
    model_dir = MODEL_DIR
    predictor = EnsemblePredictor(
        cnn_model_path=model_dir / "final_cnn.pth",
        gbm_model_path=model_dir / "gbm_model.pkl",
        ensemble_config_path=model_dir / "ensemble_config.json",
        normalization_stats_path=stats_path,
        device="cuda",
        mc_dropout_rate=0.1
    )
    
    # Test on a small batch
    print("\n3. TESTING PREDICTION PIPELINE")
    print("-"*70)
    
    # Take first 5 samples
    X_sample = X_test[:5]
    y_sample = y_test[:5]
    y_sample_denorm = y_test_denorm[:5]
    
    print(f"Sample batch: {X_sample.shape}")
    print(f"Ground truth mean: {y_sample_denorm.mean():.2f}°C")
    
    # Get predictions
    results = predictor.predict_ensemble(
        X_sample,
        batch_size=5,
        return_uncertainty=False,
        use_spatial_ensemble=True,
        skip_validation=True  # Skip validation for this test
    )
    
    # Analyze predictions
    print("\n4. PREDICTION ANALYSIS")
    print("-"*70)
    
    cnn_pred = results['cnn']
    gbm_pred = results['gbm']
    ensemble_pred = results['ensemble']
    
    print(f"\nCNN predictions:")
    print(f"  Shape: {cnn_pred.shape}")
    print(f"  Mean: {cnn_pred.mean():.2f}°C")
    print(f"  Std: {cnn_pred.std():.2f}°C")
    print(f"  Range: [{cnn_pred.min():.2f}, {cnn_pred.max():.2f}]°C")
    
    print(f"\nGBM predictions:")
    print(f"  Shape: {gbm_pred.shape}")
    print(f"  Mean: {gbm_pred.mean():.2f}°C")
    print(f"  Std: {gbm_pred.std():.2f}°C")
    print(f"  Range: [{gbm_pred.min():.2f}, {gbm_pred.max():.2f}]°C")
    
    print(f"\nEnsemble predictions:")
    print(f"  Mean: {ensemble_pred.mean():.2f}°C")
    print(f"  Std: {ensemble_pred.std():.2f}°C")
    print(f"  Range: [{ensemble_pred.min():.2f}, {ensemble_pred.max():.2f}]°C")
    
    print(f"\nGround truth:")
    print(f"  Mean: {y_sample_denorm.mean():.2f}°C")
    print(f"  Std: {y_sample_denorm.std():.2f}°C")
    
    # Check prediction variance
    print("\n5. VARIANCE ANALYSIS")
    print("-"*70)
    
    gt_variance = y_sample_denorm.var()
    cnn_variance = cnn_pred.var()
    gbm_variance = gbm_pred.var()
    ensemble_variance = ensemble_pred.var()
    
    print(f"Variance comparison:")
    print(f"  Ground truth: {gt_variance:.2f}")
    print(f"  CNN: {cnn_variance:.2f} (ratio: {cnn_variance/gt_variance:.2f})")
    print(f"  GBM: {gbm_variance:.2f} (ratio: {gbm_variance/gt_variance:.2f})")
    print(f"  Ensemble: {ensemble_variance:.2f} (ratio: {ensemble_variance/gt_variance:.2f})")
    
    if cnn_variance < gt_variance * 0.3:
        print("\n🔴 CRITICAL: CNN predictions have very low variance!")
        print("   Model is predicting almost constant values.")
        print("   This explains the slope of 0.14.")
    
    # Check per-sample predictions
    print("\n6. PER-SAMPLE ANALYSIS")
    print("-"*70)
    
    for i in range(min(5, len(X_sample))):
        gt_mean = y_sample_denorm[i].mean()
        cnn_mean = cnn_pred[i].mean()
        gbm_val = gbm_pred[i]
        ens_mean = ensemble_pred[i].mean()
        
        print(f"\nSample {i}:")
        print(f"  Ground truth: {gt_mean:.2f}°C")
        print(f"  CNN:          {cnn_mean:.2f}°C (error: {cnn_mean - gt_mean:+.2f}°C)")
        print(f"  GBM:          {gbm_val:.2f}°C (error: {gbm_val - gt_mean:+.2f}°C)")
        print(f"  Ensemble:     {ens_mean:.2f}°C (error: {ens_mean - gt_mean:+.2f}°C)")
    
    # Check if CNN is stuck
    print("\n7. CNN OUTPUT INSPECTION")
    print("-"*70)
    
    # Get raw CNN output (before denormalization)
    cnn_model = predictor.cnn_model
    cnn_model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_sample).permute(0, 3, 1, 2).to(predictor.device)
        raw_output = cnn_model(X_tensor)
        raw_output_np = raw_output.cpu().numpy()
    
    print(f"Raw CNN output (normalized space):")
    print(f"  Mean: {raw_output_np.mean():.4f}")
    print(f"  Std: {raw_output_np.std():.4f}")
    print(f"  Range: [{raw_output_np.min():.4f}, {raw_output_np.max():.4f}]")
    
    print(f"\nExpected normalized output:")
    print(f"  Mean: ~0.0")
    print(f"  Std: ~1.0")
    
    if abs(raw_output_np.mean()) > 0.5:
        print(f"\n🔴 CRITICAL: CNN output mean is {raw_output_np.mean():.4f}, not ~0!")
        print("   Model is not predicting in normalized space correctly.")
    
    if raw_output_np.std() < 0.3:
        print(f"\n🔴 CRITICAL: CNN output std is {raw_output_np.std():.4f}, very low!")
        print("   Model is collapsing to constant predictions.")
        print("\n   Possible causes:")
        print("   1. Model trained poorly (underfitting)")
        print("   2. Weights initialized incorrectly")
        print("   3. Early stopping triggered too early")
        print("   4. Learning rate too low")
    
    # Check model weights
    print("\n8. MODEL WEIGHT INSPECTION")
    print("-"*70)
    
    total_params = sum(p.numel() for p in cnn_model.parameters())
    zero_params = sum((p == 0).sum().item() for p in cnn_model.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Zero parameters: {zero_params:,} ({zero_params/total_params*100:.2f}%)")
    
    # Check weight statistics
    weight_stats = []
    for name, param in cnn_model.named_parameters():
        if 'weight' in name:
            weight_stats.append({
                'name': name,
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item()
            })
    
    print("\nFirst few layer weight statistics:")
    for stat in weight_stats[:3]:
        print(f"  {stat['name']}:")
        print(f"    mean={stat['mean']:.4f}, std={stat['std']:.4f}")
        print(f"    range=[{stat['min']:.4f}, {stat['max']:.4f}]")
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    issues = []
    
    if cnn_variance < gt_variance * 0.3:
        issues.append("CNN predictions have very low variance (collapsing)")
    
    if abs(raw_output_np.mean()) > 0.5:
        issues.append("CNN not outputting in normalized space")
    
    if raw_output_np.std() < 0.3:
        issues.append("CNN output variance too low (model stuck)")
    
    if issues:
        print("\n🔴 CRITICAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n💡 RECOMMENDED FIXES:")
        print("  1. Check training logs - did CNN actually learn?")
        print("     - Look for decreasing training loss")
        print("     - Check if early stopping triggered too early")
        
        print("\n  2. Try re-training with:")
        print("     - Higher learning rate (current: 0.001 → try 0.005)")
        print("     - More epochs (50 → 100)")
        print("     - Smaller model (fewer layers)")
        print("     - Disable early stopping initially")
        
        print("\n  3. Check if model is actually loaded:")
        print("     - Verify final_cnn.pth exists and is not corrupted")
        print("     - Try loading best_cnn.pth instead")
        
        print("\n  4. Test with validation data:")
        print("     - Run inference on validation set")
        print("     - Should match training validation R² (~0.65)")
    else:
        print("\n✅ No critical issues in inference pipeline")
        print("   Problem might be in training convergence")
    
    print("="*70)


if __name__ == "__main__":
    diagnose_inference_pipeline()