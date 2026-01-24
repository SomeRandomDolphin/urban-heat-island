"""
Diagnostic Tool: Find Training-Inference Mismatch Issues
Run this to identify what's causing the performance gap
"""
import torch
import numpy as np
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_training_inference_mismatch(
    train_data_path: Path,
    test_data_path: Path,
    model_dir: Path,
    norm_stats_path: Path
):
    """
    Comprehensive diagnostic to find training-inference issues
    """
    logger.info("="*80)
    logger.info("TRAINING-INFERENCE MISMATCH DIAGNOSTIC")
    logger.info("="*80)
    
    issues_found = []
    
    # ==================================================================
    # CHECK 1: Normalization Statistics
    # ==================================================================
    logger.info("\n[CHECK 1] Normalization Statistics")
    logger.info("-"*80)
    
    if not norm_stats_path.exists():
        issues_found.append("❌ CRITICAL: Normalization stats file missing!")
        logger.error(f"  File not found: {norm_stats_path}")
        logger.error("  This means inference doesn't normalize data correctly!")
    else:
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        
        logger.info(f"✅ Normalization stats found")
        
        # Check for required fields
        if 'target' not in norm_stats:
            issues_found.append("❌ Target normalization stats missing!")
        else:
            target_mean = norm_stats['target']['mean']
            target_std = norm_stats['target']['std']
            logger.info(f"  Target: mean={target_mean:.4f}, std={target_std:.4f}")
        
        if 'features' not in norm_stats:
            issues_found.append("⚠️ Feature normalization stats missing!")
            logger.warning("  Inference might not normalize features correctly")
        else:
            n_features = len(norm_stats['features'])
            logger.info(f"  Features: {n_features} channels found")
    
    # ==================================================================
    # CHECK 2: Data Distribution Comparison
    # ==================================================================
    logger.info("\n[CHECK 2] Data Distribution Comparison")
    logger.info("-"*80)
    
    try:
        # Load training data
        X_train = np.load(train_data_path / "X.npy")
        y_train = np.load(train_data_path / "y.npy")
        
        # Load test data
        X_test = np.load(test_data_path / "X.npy")
        y_test = np.load(test_data_path / "y.npy")
        
        logger.info("Training data:")
        logger.info(f"  X range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        logger.info(f"  X mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
        logger.info(f"  y range: [{y_train.min():.2f}, {y_train.max():.2f}]°C")
        logger.info(f"  y mean: {y_train.mean():.2f}°C, std: {y_train.std():.2f}°C")
        
        logger.info("\nTest data:")
        logger.info(f"  X range: [{X_test.min():.4f}, {X_test.max():.4f}]")
        logger.info(f"  X mean: {X_test.mean():.4f}, std: {X_test.std():.4f}")
        logger.info(f"  y range: [{y_test.min():.2f}, {y_test.max():.2f}]°C")
        logger.info(f"  y mean: {y_test.mean():.2f}°C, std: {y_test.std():.2f}°C")
        
        # Check if training data is normalized
        if abs(X_train.mean()) < 0.1 and abs(X_train.std() - 1.0) < 0.1:
            logger.info("\n✅ Training data appears normalized (mean≈0, std≈1)")
        else:
            issues_found.append("⚠️ Training data doesn't look normalized!")
            logger.warning("  Expected: mean≈0, std≈1")
            logger.warning(f"  Got: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
        
        # Check if test data is NOT normalized (should be raw)
        if abs(X_test.mean()) > 0.5 or X_test.std() > 2.0:
            logger.info("✅ Test data appears to be raw (not normalized)")
        else:
            issues_found.append("❌ Test data might be pre-normalized!")
            logger.error("  Inference should receive RAW data and normalize it")
        
        # Check for distribution shift
        train_y_mean = y_train.mean()
        test_y_mean = y_test.mean()
        
        if abs(train_y_mean - test_y_mean) > 5.0:
            issues_found.append("⚠️ Large distribution shift between train and test!")
            logger.warning(f"  Train mean: {train_y_mean:.2f}°C")
            logger.warning(f"  Test mean: {test_y_mean:.2f}°C")
            logger.warning(f"  Difference: {abs(train_y_mean - test_y_mean):.2f}°C")
        
    except Exception as e:
        issues_found.append(f"❌ Error loading data: {e}")
        logger.error(f"  {e}")
    
    # ==================================================================
    # CHECK 3: Model Configuration
    # ==================================================================
    logger.info("\n[CHECK 3] Model Configuration")
    logger.info("-"*80)
    
    ensemble_config_path = model_dir / "ensemble_config.json"
    if not ensemble_config_path.exists():
        issues_found.append("❌ Ensemble config missing!")
    else:
        with open(ensemble_config_path, 'r') as f:
            ensemble_config = json.load(f)
        
        weights = ensemble_config.get("weights", {})
        cnn_weight = weights.get("cnn", 0.5)
        gbm_weight = weights.get("gbm", 0.5)
        
        logger.info(f"Ensemble weights:")
        logger.info(f"  CNN: {cnn_weight:.4f}")
        logger.info(f"  GBM: {gbm_weight:.4f}")
        
        if cnn_weight == 0.0:
            issues_found.append("❌ CNN weight is 0! CNN model completely ignored!")
            logger.error("  This suggests CNN failed during training")
            logger.error("  Using only GBM predictions")
        elif cnn_weight < 0.3:
            issues_found.append("⚠️ CNN weight very low - CNN barely used")
            logger.warning(f"  CNN contributes only {cnn_weight*100:.1f}% to ensemble")
    
    # ==================================================================
    # CHECK 4: Training History
    # ==================================================================
    logger.info("\n[CHECK 4] Training History")
    logger.info("-"*80)
    
    history_path = model_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        if "ensemble_metrics" in history:
            metrics = history["ensemble_metrics"]
            logger.info(f"Final training metrics:")
            logger.info(f"  R²: {metrics.get('r2', 'N/A'):.4f}")
            logger.info(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}°C")
            logger.info(f"  MAE: {metrics.get('mae', 'N/A'):.4f}°C")
        
        # Check CNN learning
        if "cnn_metrics" in history and len(history["cnn_metrics"]) > 0:
            cnn_r2_progression = [m['r2'] for m in history["cnn_metrics"]]
            best_cnn_r2 = max(cnn_r2_progression) if cnn_r2_progression else 0
            final_cnn_r2 = cnn_r2_progression[-1] if cnn_r2_progression else 0
            
            logger.info(f"\nCNN performance:")
            logger.info(f"  Best R²: {best_cnn_r2:.4f}")
            logger.info(f"  Final R²: {final_cnn_r2:.4f}")
            
            if best_cnn_r2 < 0.5:
                issues_found.append("❌ CNN never learned well (R² < 0.5)")
                logger.error("  CNN training failed - investigate preprocessing")
    else:
        issues_found.append("⚠️ Training history not found")
    
    # ==================================================================
    # CHECK 5: Prediction Pipeline Test
    # ==================================================================
    logger.info("\n[CHECK 5] Prediction Pipeline Test")
    logger.info("-"*80)
    
    try:
        # Test normalization on sample
        if norm_stats_path.exists():
            X_sample = X_test[:1]  # Single sample
            
            logger.info("Testing normalization on sample:")
            logger.info(f"  Raw input range: [{X_sample.min():.4f}, {X_sample.max():.4f}]")
            
            # Simulate normalization
            with open(norm_stats_path, 'r') as f:
                stats = json.load(f)
            
            if 'features' in stats:
                X_norm = X_sample.copy()
                for ch in range(X_sample.shape[-1]):
                    ch_key = f'channel_{ch}'
                    if ch_key in stats['features']:
                        mean = stats['features'][ch_key]['mean']
                        std = stats['features'][ch_key]['std']
                        X_norm[:, :, :, ch] = (X_sample[:, :, :, ch] - mean) / std
                
                logger.info(f"  Normalized range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")
                
                if abs(X_norm.mean()) < 0.2 and 0.8 < X_norm.std() < 1.2:
                    logger.info("✅ Normalization working correctly")
                else:
                    issues_found.append("❌ Normalization produces unexpected values!")
                    logger.error(f"  Expected: mean≈0, std≈1")
                    logger.error(f"  Got: mean={X_norm.mean():.4f}, std={X_norm.std():.4f}")
    
    except Exception as e:
        logger.error(f"  Error testing normalization: {e}")
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*80)
    
    if not issues_found:
        logger.info("✅ NO MAJOR ISSUES FOUND")
        logger.info("\nIf you still see poor inference, check:")
        logger.info("  1. Model loading (correct checkpoint?)")
        logger.info("  2. Batch processing (memory issues?)")
        logger.info("  3. Device placement (CPU vs GPU differences?)")
    else:
        logger.error(f"\n❌ FOUND {len(issues_found)} ISSUE(S):")
        for i, issue in enumerate(issues_found, 1):
            logger.error(f"  {i}. {issue}")
        
        logger.info("\n🔧 RECOMMENDED FIXES:")
        
        if any("Normalization stats" in issue for issue in issues_found):
            logger.info("\n1. CREATE NORMALIZATION STATS during training:")
            logger.info("   - Save mean/std for each feature channel")
            logger.info("   - Save mean/std for target (LST)")
            logger.info("   - Store in normalization_stats.json")
        
        if any("CNN weight is 0" in issue for issue in issues_found):
            logger.info("\n2. FIX CNN TRAINING:")
            logger.info("   - Check learning rate (might be too low)")
            logger.info("   - Verify input data preprocessing")
            logger.info("   - Ensure targets are properly normalized")
            logger.info("   - Try simpler architecture first")
        
        if any("distribution shift" in issue for issue in issues_found):
            logger.info("\n3. ADDRESS DISTRIBUTION SHIFT:")
            logger.info("   - Check train/test split strategy")
            logger.info("   - Consider domain adaptation techniques")
            logger.info("   - Verify data collection consistency")
        
        if any("normalized" in issue.lower() for issue in issues_found):
            logger.info("\n4. FIX DATA PIPELINE:")
            logger.info("   - Training: Load normalized data")
            logger.info("   - Inference: Load RAW data, then normalize")
            logger.info("   - Always denormalize predictions before evaluation")
    
    logger.info("\n" + "="*80)
    
    return issues_found


if __name__ == "__main__":
    from config import PROCESSED_DATA_DIR, MODEL_DIR
    
    # Run diagnostic
    issues = diagnose_training_inference_mismatch(
        train_data_path=PROCESSED_DATA_DIR / "cnn_dataset" / "train",
        test_data_path=PROCESSED_DATA_DIR / "cnn_dataset" / "test",
        model_dir=MODEL_DIR,
        norm_stats_path=PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
    )
    
    print(f"\n\n{'='*80}")
    print(f"Total issues found: {len(issues)}")
    print(f"{'='*80}")