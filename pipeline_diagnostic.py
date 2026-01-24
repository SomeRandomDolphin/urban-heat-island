"""
Diagnostic and fix for UHI pipeline issues
"""
import sys
import numpy as np
import torch
from pathlib import Path
import logging
import matplotlib.pyplot as plt

from config import *

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def diagnose_predictions(predictions_dir: Path):
    """Diagnose prediction issues"""
    logger.info("="*60)
    logger.info("DIAGNOSTIC ANALYSIS")
    logger.info("="*60)
    
    # Load predictions
    ensemble_preds = np.load(predictions_dir / "predictions_ensemble.npy")
    cnn_preds = np.load(predictions_dir / "predictions_cnn.npy")
    gbm_preds = np.load(predictions_dir / "predictions_gbm.npy")
    
    logger.info("\n1. PREDICTION STATISTICS")
    logger.info("-"*60)
    
    def analyze_array(arr, name):
        logger.info(f"\n{name}:")
        logger.info(f"  Shape: {arr.shape}")
        logger.info(f"  Mean: {arr.mean():.4f}")
        logger.info(f"  Std: {arr.std():.6f}")
        logger.info(f"  Min: {arr.min():.4f}")
        logger.info(f"  Max: {arr.max():.4f}")
        logger.info(f"  Range: {arr.max() - arr.min():.6f}")
        logger.info(f"  Unique values: {len(np.unique(arr))}")
        
        if arr.std() < 0.01:
            logger.error(f"  ⚠️ CRITICAL: {name} has near-zero variance!")
            return False
        return True
    
    cnn_ok = analyze_array(cnn_preds, "CNN Predictions")
    gbm_ok = analyze_array(gbm_preds, "GBM Predictions")
    ens_ok = analyze_array(ensemble_preds, "Ensemble Predictions")
    
    # Check ground truth
    test_dir = PROCESSED_DATA_DIR / "cnn_dataset" / "test"
    if (test_dir / "y.npy").exists():
        y_test = np.load(test_dir / "y.npy")
        analyze_array(y_test, "Ground Truth")
    
    logger.info("\n2. PROBLEM IDENTIFICATION")
    logger.info("-"*60)
    
    issues = []
    
    if not cnn_ok:
        issues.append("CNN predictions have no variance - model collapsed")
    if not gbm_ok:
        issues.append("GBM predictions have no variance - feature issue")
    if not ens_ok:
        issues.append("Ensemble predictions have no variance")
    
    if len(issues) > 0:
        logger.error("\n⛔ CRITICAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            logger.error(f"  {i}. {issue}")
    else:
        logger.info("\n✓ Predictions have adequate variance")
    
    return {
        "cnn_ok": cnn_ok,
        "gbm_ok": gbm_ok,
        "ensemble_ok": ens_ok,
        "issues": issues
    }


def diagnose_model_outputs():
    """Diagnose model architecture issues"""
    logger.info("\n3. MODEL ARCHITECTURE CHECK")
    logger.info("-"*60)
    
    # Load test data
    test_dir = PROCESSED_DATA_DIR / "cnn_dataset" / "test"
    X_test = np.load(test_dir / "X.npy")
    y_test = np.load(test_dir / "y.npy") if (test_dir / "y.npy").exists() else None
    
    logger.info(f"Test data shape: {X_test.shape}")
    logger.info(f"Test data range: [{X_test.min():.4f}, {X_test.max():.4f}]")
    
    if y_test is not None:
        logger.info(f"Test labels shape: {y_test.shape}")
        logger.info(f"Test labels range: [{y_test.min():.4f}, {y_test.max():.4f}]")
        logger.info(f"Test labels mean: {y_test.mean():.4f} ± {y_test.std():.4f}")
    
    # Check if data is normalized
    if np.abs(X_test.mean()) < 0.1 and np.abs(X_test.std() - 1.0) < 0.2:
        logger.info("✓ Input data appears normalized")
    else:
        logger.warning(f"⚠️ Input data may not be normalized: mean={X_test.mean():.4f}, std={X_test.std():.4f}")
    
    # Load CNN model and check
    from models import UNet
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=CNN_CONFIG["input_channels"], out_channels=1)
    model.load_state_dict(torch.load(MODEL_DIR / "final_cnn.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # Test forward pass
    with torch.no_grad():
        sample = torch.FloatTensor(X_test[:1]).permute(0, 3, 1, 2).to(device)
        output = model(sample)
        
        logger.info(f"\nCNN forward pass test:")
        logger.info(f"  Input shape: {sample.shape}")
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        logger.info(f"  Output mean: {output.mean().item():.4f}")
        logger.info(f"  Output std: {output.std().item():.6f}")
        
        if output.std().item() < 0.01:
            logger.error("⚠️ CRITICAL: CNN output has no variance!")
            logger.error("   Possible causes:")
            logger.error("   1. Model weights not properly loaded")
            logger.error("   2. Model collapsed during training (dead ReLUs)")
            logger.error("   3. Output layer initialized incorrectly")


def check_denormalization():
    """Check if predictions need denormalization"""
    logger.info("\n4. NORMALIZATION CHECK")
    logger.info("-"*60)
    
    stats_path = PROCESSED_DATA_DIR / "normalization_stats.json"
    
    if stats_path.exists():
        import json
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        logger.info("Normalization statistics found:")
        if "target" in stats:
            logger.info(f"  Target mean: {stats['target'].get('mean', 'N/A')}")
            logger.info(f"  Target std: {stats['target'].get('std', 'N/A')}")
            logger.info(f"  Target min: {stats['target'].get('min', 'N/A')}")
            logger.info(f"  Target max: {stats['target'].get('max', 'N/A')}")
            
            return stats['target']
    else:
        logger.warning("⚠️ No normalization statistics found")
        logger.warning("   Predictions may need denormalization!")
    
    return None


def fix_predictions_with_denormalization(predictions_dir: Path, norm_stats: dict):
    """Denormalize predictions if needed"""
    logger.info("\n5. DENORMALIZATION FIX")
    logger.info("-"*60)
    
    # Load predictions
    ensemble_preds = np.load(predictions_dir / "predictions_ensemble.npy")
    
    logger.info(f"Original predictions - Mean: {ensemble_preds.mean():.4f}, Std: {ensemble_preds.std():.6f}")
    
    # Denormalize
    mean = norm_stats['mean']
    std = norm_stats['std']
    
    denormalized = ensemble_preds * std + mean
    
    logger.info(f"After denormalization - Mean: {denormalized.mean():.4f}, Std: {denormalized.std():.4f}")
    logger.info(f"Expected range: [{norm_stats['min']:.2f}, {norm_stats['max']:.2f}]°C")
    logger.info(f"Actual range: [{denormalized.min():.2f}, {denormalized.max():.2f}]°C")
    
    # Save denormalized predictions
    np.save(predictions_dir / "predictions_ensemble_denormalized.npy", denormalized)
    logger.info("✓ Saved denormalized predictions")
    
    return denormalized


def check_ensemble_weights():
    """Check if ensemble weights make sense"""
    logger.info("\n6. ENSEMBLE WEIGHTS CHECK")
    logger.info("-"*60)
    
    import json
    with open(MODEL_DIR / "ensemble_config.json", 'r') as f:
        config = json.load(f)
    
    weights = config.get("weights", {})
    logger.info(f"CNN weight: {weights.get('cnn', 'N/A')}")
    logger.info(f"GBM weight: {weights.get('gbm', 'N/A')}")
    
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        logger.warning(f"⚠️ Weights don't sum to 1.0: {total}")


def visualize_predictions(predictions_dir: Path, output_dir: Path):
    """Create diagnostic visualizations"""
    logger.info("\n7. CREATING DIAGNOSTIC PLOTS")
    logger.info("-"*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    ensemble_preds = np.load(predictions_dir / "predictions_ensemble.npy")
    cnn_preds = np.load(predictions_dir / "predictions_cnn.npy")
    gbm_preds = np.load(predictions_dir / "predictions_gbm.npy")
    
    # Load ground truth if available
    test_dir = PROCESSED_DATA_DIR / "cnn_dataset" / "test"
    y_test = None
    if (test_dir / "y.npy").exists():
        y_test = np.load(test_dir / "y.npy")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # CNN predictions
    ax = axes[0, 0]
    ax.hist(cnn_preds.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title(f'CNN Predictions\nMean: {cnn_preds.mean():.4f}, Std: {cnn_preds.std():.6f}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # GBM predictions
    ax = axes[0, 1]
    ax.hist(gbm_preds.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.set_title(f'GBM Predictions\nMean: {gbm_preds.mean():.4f}, Std: {gbm_preds.std():.6f}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Ensemble predictions
    ax = axes[1, 0]
    ax.hist(ensemble_preds.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    ax.set_title(f'Ensemble Predictions\nMean: {ensemble_preds.mean():.4f}, Std: {ensemble_preds.std():.6f}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Ground truth
    ax = axes[1, 1]
    if y_test is not None:
        ax.hist(y_test.flatten(), bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax.set_title(f'Ground Truth\nMean: {y_test.mean():.4f}, Std: {y_test.std():.4f}')
    else:
        ax.text(0.5, 0.5, 'No ground truth available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Ground Truth (N/A)')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_distributions.png", dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_dir / 'prediction_distributions.png'}")
    plt.close()
    
    # Plot spatial maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Take first patch
    cnn_map = cnn_preds[0] if len(cnn_preds.shape) == 3 else cnn_preds[0]
    ens_map = ensemble_preds[0] if len(ensemble_preds.shape) == 3 else ensemble_preds[0]
    
    # CNN
    im0 = axes[0].imshow(cnn_map, cmap='RdYlBu_r')
    axes[0].set_title(f'CNN Prediction\nRange: [{cnn_map.min():.4f}, {cnn_map.max():.4f}]')
    plt.colorbar(im0, ax=axes[0])
    
    # Ensemble
    im1 = axes[1].imshow(ens_map, cmap='RdYlBu_r')
    axes[1].set_title(f'Ensemble Prediction\nRange: [{ens_map.min():.4f}, {ens_map.max():.4f}]')
    plt.colorbar(im1, ax=axes[1])
    
    # Ground truth
    if y_test is not None:
        gt_map = y_test[0] if len(y_test.shape) == 4 else y_test[0]
        if len(gt_map.shape) > 2:
            gt_map = gt_map[:, :, 0]
        im2 = axes[2].imshow(gt_map, cmap='RdYlBu_r')
        axes[2].set_title(f'Ground Truth\nRange: [{gt_map.min():.4f}, {gt_map.max():.4f}]')
        plt.colorbar(im2, ax=axes[2])
    else:
        axes[2].text(0.5, 0.5, 'No ground truth', ha='center', va='center')
        axes[2].set_title('Ground Truth (N/A)')
    
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_predictions.png", dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_dir / 'spatial_predictions.png'}")
    plt.close()


def create_fixed_pipeline():
    """Create a fixed version of the pipeline with proper denormalization"""
    logger.info("\n8. RECOMMENDATIONS")
    logger.info("-"*60)
    
    logger.info("\n📋 RECOMMENDED FIXES:")
    logger.info("\n1. ADD DENORMALIZATION IN INFERENCE:")
    logger.info("   Modify uhi_inference.py predict_ensemble() method:")
    logger.info("""
   # After getting predictions
   if self.norm_stats and 'target' in self.norm_stats:
       target_stats = self.norm_stats['target']
       ensemble_preds = ensemble_preds * target_stats['std'] + target_stats['mean']
       cnn_preds = cnn_preds * target_stats['std'] + target_stats['mean']
    """)
    
    logger.info("\n2. CHECK MODEL TRAINING:")
    logger.info("   If predictions still have no variance after denormalization:")
    logger.info("   - Retrain CNN with lower dropout rates")
    logger.info("   - Increase learning rate")
    logger.info("   - Check if loss is decreasing")
    logger.info("   - Verify input data preprocessing")
    
    logger.info("\n3. VERIFY DATA PREPROCESSING:")
    logger.info("   Check prepare_cnn_dataset.py:")
    logger.info("   - Are targets properly saved?")
    logger.info("   - Is normalization applied correctly?")
    logger.info("   - Are normalization stats saved?")
    
    logger.info("\n4. ENSEMBLE WEIGHTS:")
    logger.info("   Consider adjusting weights if GBM performs better:")
    logger.info("   - Increase GBM weight if it has better variance")
    logger.info("   - Or use only GBM if CNN completely failed")


def main():
    """Run complete diagnostic"""
    logger.info("="*70)
    logger.info("UHI PIPELINE DIAGNOSTIC TOOL")
    logger.info("="*70)
    
    predictions_dir = OUTPUT_DIR / "data"
    diagnostic_dir = OUTPUT_DIR / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Diagnose predictions
    diagnosis = diagnose_predictions(predictions_dir)
    
    # Step 2: Check model
    diagnose_model_outputs()
    
    # Step 3: Check normalization
    norm_stats = check_denormalization()
    
    # Step 4: Try denormalization fix
    if norm_stats and not diagnosis["ensemble_ok"]:
        logger.info("\n" + "="*60)
        logger.info("ATTEMPTING FIX: DENORMALIZATION")
        logger.info("="*60)
        
        try:
            fixed_preds = fix_predictions_with_denormalization(predictions_dir, norm_stats)
            logger.info("\n✓ Denormalization applied successfully!")
            logger.info(f"  New mean: {fixed_preds.mean():.2f}°C")
            logger.info(f"  New std: {fixed_preds.std():.2f}°C")
            logger.info(f"  New range: [{fixed_preds.min():.2f}, {fixed_preds.max():.2f}]°C")
            
            # Re-run pipeline with fixed predictions
            logger.info("\n" + "="*60)
            logger.info("RECOMMENDATION: Re-run pipeline with:")
            logger.info("  python run_fixed_pipeline.py --use-denormalized")
            logger.info("="*60)
        except Exception as e:
            logger.error(f"Failed to apply denormalization: {e}")
    
    # Step 5: Check ensemble config
    check_ensemble_weights()
    
    # Step 6: Create visualizations
    visualize_predictions(predictions_dir, diagnostic_dir)
    
    # Step 7: Provide recommendations
    create_fixed_pipeline()
    
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("="*70)
    logger.info(f"\nDiagnostic outputs saved to: {diagnostic_dir}")
    logger.info(f"  - prediction_distributions.png")
    logger.info(f"  - spatial_predictions.png")


if __name__ == "__main__":
    main()