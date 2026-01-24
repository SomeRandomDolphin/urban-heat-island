"""
Comprehensive diagnostic script for UHI model debugging
Run this BEFORE training to identify data quality issues
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_normalization(data_dir: Path):
    """Check if data is properly normalized"""
    logger.info("\n" + "="*70)
    logger.info("NORMALIZATION DIAGNOSTIC")
    logger.info("="*70)
    
    # Load normalization stats
    stats_path = data_dir / "normalization_stats.json"
    if not stats_path.exists():
        logger.error("❌ NO NORMALIZATION STATS FOUND!")
        logger.error("   This is likely the root cause of poor performance.")
        logger.error("   Rerun preprocessing.py to create normalized data.")
        return False
    
    with open(stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    logger.info("Normalization stats loaded:")
    logger.info(f"  Target mean: {norm_stats['target']['mean']:.2f}°C")
    logger.info(f"  Target std:  {norm_stats['target']['std']:.2f}°C")
    
    # Load actual data
    for split in ['train', 'val', 'test']:
        X = np.load(data_dir / split / "X.npy")
        y = np.load(data_dir / split / "y.npy")
        
        logger.info(f"\n{split.upper()} split:")
        logger.info(f"  X - mean: {X.mean():.4f}, std: {X.std():.4f}")
        logger.info(f"  y - mean: {y.mean():.4f}, std: {y.std():.4f}")
        
        # Check normalization quality
        if split == 'train':
            if abs(X.mean()) > 0.2 or abs(y.mean()) > 0.2:
                logger.error(f"  ❌ {split} data NOT normalized (mean should be ≈0)")
                return False
            if not (0.8 < X.std() < 1.2) or not (0.8 < y.std() < 1.2):
                logger.error(f"  ❌ {split} data std out of range (should be ≈1)")
                return False
    
    logger.info("\n✅ Normalization looks good")
    return True

def diagnose_data_quality(data_dir: Path):
    """Check for data quality issues"""
    logger.info("\n" + "="*70)
    logger.info("DATA QUALITY DIAGNOSTIC")
    logger.info("="*70)
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        X = np.load(data_dir / split / "X.npy")
        y = np.load(data_dir / split / "y.npy")
        
        logger.info(f"\n{split.upper()} split ({len(X)} samples):")
        
        # Check for NaN/Inf
        if np.isnan(X).any() or np.isinf(X).any():
            nan_pct = np.isnan(X).sum() / X.size * 100
            issues.append(f"{split}: X contains {nan_pct:.2f}% NaN/Inf")
        
        if np.isnan(y).any() or np.isinf(y).any():
            nan_pct = np.isnan(y).sum() / y.size * 100
            issues.append(f"{split}: y contains {nan_pct:.2f}% NaN/Inf")
        
        # Check variance
        y_var = np.var(y)
        if y_var < 0.01:
            issues.append(f"{split}: y has very low variance ({y_var:.6f})")
            logger.warning(f"  ⚠️  Target variance too low: {y_var:.6f}")
        
        # Check for constant patches
        y_flat = y.reshape(len(y), -1)
        constant_patches = 0
        for i in range(len(y_flat)):
            if np.std(y_flat[i]) < 0.01:
                constant_patches += 1
        
        constant_ratio = constant_patches / len(y) * 100
        if constant_ratio > 10:
            issues.append(f"{split}: {constant_ratio:.1f}% patches are nearly constant")
            logger.warning(f"  ⚠️  {constant_ratio:.1f}% of patches have std < 0.01")
        
        logger.info(f"  Shape: X={X.shape}, y={y.shape}")
        logger.info(f"  Target variance: {y_var:.6f}")
        logger.info(f"  Constant patches: {constant_ratio:.1f}%")
    
    if issues:
        logger.error("\n❌ DATA QUALITY ISSUES:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("\n✅ Data quality looks good")
    return True

def diagnose_temperature_distribution(data_dir: Path):
    """Analyze temperature distribution"""
    logger.info("\n" + "="*70)
    logger.info("TEMPERATURE DISTRIBUTION DIAGNOSTIC")
    logger.info("="*70)
    
    # Load normalization stats
    stats_path = data_dir / "normalization_stats.json"
    with open(stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    target_mean = norm_stats['target']['mean']
    target_std = norm_stats['target']['std']
    
    for split in ['train', 'val']:
        y = np.load(data_dir / split / "y.npy")
        
        # Denormalize to Celsius
        y_celsius = y * target_std + target_mean
        
        logger.info(f"\n{split.upper()} temperatures (°C):")
        logger.info(f"  Range: [{y_celsius.min():.2f}, {y_celsius.max():.2f}]")
        logger.info(f"  Mean: {y_celsius.mean():.2f}")
        logger.info(f"  Std: {y_celsius.std():.2f}")
        logger.info(f"  Median: {np.median(y_celsius):.2f}")
        
        # Check for Jakarta realism
        if y_celsius.mean() < 25 or y_celsius.mean() > 40:
            logger.warning(f"  ⚠️  Mean temperature {y_celsius.mean():.2f}°C seems unrealistic for Jakarta")
        
        # Temperature percentiles
        p10, p25, p50, p75, p90 = np.percentile(y_celsius, [10, 25, 50, 75, 90])
        logger.info(f"  Percentiles: 10%={p10:.1f}, 25%={p25:.1f}, 50%={p50:.1f}, 75%={p75:.1f}, 90%={p90:.1f}")

def diagnose_model_architecture(input_channels: int):
    """Check if model architecture is appropriate"""
    logger.info("\n" + "="*70)
    logger.info("MODEL ARCHITECTURE DIAGNOSTIC")
    logger.info("="*70)
    
    # Calculate parameters
    filters = [64, 128, 256, 512, 1024]
    
    # Rough parameter count for UNet
    params = 0
    # Encoder
    params += (input_channels * filters[0] * 9 * 2)  # First conv block
    for i in range(len(filters) - 1):
        params += (filters[i] * filters[i+1] * 9 * 2)
    
    # Bottleneck
    params += (filters[-2] * filters[-1] * 9 * 2)
    
    # Decoder (roughly same as encoder)
    params *= 2
    
    logger.info(f"Estimated CNN parameters: ~{params:,}")
    logger.info(f"Input channels: {input_channels}")
    
    if params > 50_000_000:
        logger.warning("⚠️  Model might be too large (>50M params) - risk of overfitting")
    
    if input_channels < 5:
        logger.warning("⚠️  Very few input channels - model might lack information")

def recommend_improvements():
    """Provide improvement recommendations"""
    logger.info("\n" + "="*70)
    logger.info("IMPROVEMENT RECOMMENDATIONS")
    logger.info("="*70)
    
    recommendations = [
        {
            "issue": "Poor R² (0.64) and high RMSE (2.54°C)",
            "likely_causes": [
                "Data not properly normalized",
                "Insufficient training data",
                "Model too complex (overfitting) or too simple (underfitting)",
                "Loss function not appropriate"
            ],
            "fixes": [
                "1. VERIFY NORMALIZATION: Check that preprocessing saved normalized data",
                "2. INCREASE DATA: Extract more patches with lower stride (32 instead of 48)",
                "3. REDUCE COMPLEXITY: Try fewer filters [32, 64, 128, 256, 512]",
                "4. SIMPLIFY LOSS: Use pure MSE first, add complexity only if needed",
                "5. INCREASE LEARNING RATE: Try 0.001 instead of 0.0005"
            ]
        },
        {
            "issue": "Ensemble doesn't help much",
            "likely_causes": [
                "CNN predictions have wrong scale",
                "GBM and CNN learning different things",
                "Fusion weights not optimal"
            ],
            "fixes": [
                "1. DIAGNOSE SCALES: Print pred/target means & stds during training",
                "2. USE ADAPTIVE WEIGHTS: Let performance determine ensemble weights",
                "3. TRY GBM ONLY: If GBM R²>0.8, focus on improving it instead"
            ]
        },
        {
            "issue": "Training instability",
            "likely_causes": [
                "Gradient explosion/vanishing",
                "Learning rate too high or too low",
                "Batch size effects"
            ],
            "fixes": [
                "1. ADD GRADIENT MONITORING: Log gradient norms each epoch",
                "2. USE LEARNING RATE FINDER: Find optimal LR before training",
                "3. TRY DIFFERENT BATCH SIZES: 16, 32, 64",
                "4. USE GRADIENT CLIPPING: Already at 1.0, try 0.5"
            ]
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"\n{i}. {rec['issue']}")
        logger.info("   Likely causes:")
        for cause in rec['likely_causes']:
            logger.info(f"      - {cause}")
        logger.info("   Recommended fixes:")
        for fix in rec['fixes']:
            logger.info(f"      {fix}")

def main():
    """Run all diagnostics"""
    data_dir = Path("data/processed/cnn_dataset")
    
    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        logger.error("   Run preprocessing.py first!")
        return
    
    logger.info("="*70)
    logger.info("UHI MODEL DIAGNOSTIC TOOL")
    logger.info("="*70)
    
    # Run diagnostics
    norm_ok = diagnose_normalization(data_dir)
    quality_ok = diagnose_data_quality(data_dir)
    
    if norm_ok and quality_ok:
        diagnose_temperature_distribution(data_dir)
        
        # Check model config
        try:
            with open(data_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            input_channels = metadata.get('n_channels', 10)
            diagnose_model_architecture(input_channels)
        except:
            logger.warning("⚠️  Could not load metadata.json")
    
    # Always provide recommendations
    recommend_improvements()
    
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("="*70)
    
    if not norm_ok:
        logger.error("\n❌ CRITICAL: Data normalization issue detected!")
        logger.error("   ACTION REQUIRED: Rerun preprocessing.py with normalization enabled")
    elif not quality_ok:
        logger.warning("\n⚠️  Data quality issues detected")
        logger.warning("   ACTION SUGGESTED: Review preprocessing parameters")
    else:
        logger.info("\n✅ Data looks good - try recommended model improvements")

if __name__ == "__main__":
    main()