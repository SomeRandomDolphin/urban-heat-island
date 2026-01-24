"""
Diagnostic script to investigate train-test distribution mismatch
Run this to identify the root cause of poor inference performance
"""
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def diagnose_dataset(data_dir="data/processed/cnn_dataset"):
    """Diagnose potential issues in the dataset"""
    
    print("="*70)
    print("DATASET DIAGNOSIS")
    print("="*70)
    
    data_dir = Path(data_dir)
    
    # Load normalization stats
    stats_path = data_dir / "normalization_stats.json"
    with open(stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    print("\n1. NORMALIZATION STATISTICS")
    print("-"*70)
    target_mean = norm_stats['target']['mean']
    target_std = norm_stats['target']['std']
    print(f"Target LST used for normalization:")
    print(f"  Mean: {target_mean:.2f}°C")
    print(f"  Std:  {target_std:.2f}°C")
    
    # Load all splits
    splits = {}
    for split in ['train', 'val', 'test']:
        X = np.load(data_dir / split / "X.npy")
        y = np.load(data_dir / split / "y.npy")
        splits[split] = {'X': X, 'y': y}
        
        print(f"\n{split.upper()} split (NORMALIZED):")
        print(f"  X: mean={X.mean():.4f}, std={X.std():.4f}")
        print(f"  y: mean={y.mean():.4f}, std={y.std():.4f}")
        print(f"  y range: [{y.min():.4f}, {y.max():.4f}]")
        
        # Denormalize to see actual temperatures
        y_denorm = y * target_std + target_mean
        print(f"  y (denormalized): mean={y_denorm.mean():.2f}°C, std={y_denorm.std():.2f}°C")
        print(f"  y (denormalized) range: [{y_denorm.min():.2f}, {y_denorm.max():.2f}]°C")
    
    # Check for distribution shifts
    print("\n2. DISTRIBUTION SHIFT ANALYSIS")
    print("-"*70)
    
    y_train = splits['train']['y']
    y_val = splits['val']['y']
    y_test = splits['test']['y']
    
    # Denormalize
    y_train_denorm = y_train * target_std + target_mean
    y_val_denorm = y_val * target_std + target_mean
    y_test_denorm = y_test * target_std + target_mean
    
    print("Temperature distributions (°C):")
    print(f"  Train: {y_train_denorm.mean():.2f} ± {y_train_denorm.std():.2f}")
    print(f"  Val:   {y_val_denorm.mean():.2f} ± {y_val_denorm.std():.2f}")
    print(f"  Test:  {y_test_denorm.mean():.2f} ± {y_test_denorm.std():.2f}")
    
    # Statistical tests
    train_mean = y_train_denorm.mean()
    test_mean = y_test_denorm.mean()
    mean_diff = abs(test_mean - train_mean)
    
    if mean_diff > 2.0:
        print(f"\n⚠️  CRITICAL: Test mean differs by {mean_diff:.2f}°C from train!")
        print("   This explains the systematic bias in predictions.")
    
    train_std = y_train_denorm.std()
    test_std = y_test_denorm.std()
    std_ratio = test_std / train_std
    
    if std_ratio < 0.7 or std_ratio > 1.3:
        print(f"\n⚠️  WARNING: Test std/train std ratio = {std_ratio:.2f}")
        print("   Test data has different variance than training.")
    
    # Check temporal split
    print("\n3. TEMPORAL SPLIT ANALYSIS")
    print("-"*70)
    
    dates_train = np.load(data_dir / "train" / "dates.npy", allow_pickle=True)
    dates_val = np.load(data_dir / "val" / "dates.npy", allow_pickle=True)
    dates_test = np.load(data_dir / "test" / "dates.npy", allow_pickle=True)
    
    print(f"Train date range: {dates_train.min()} to {dates_train.max()}")
    print(f"Val date range:   {dates_val.min()} to {dates_val.max()}")
    print(f"Test date range:  {dates_test.min()} to {dates_test.max()}")
    
    # Check for seasonal effects
    train_months = [d.month for d in dates_train]
    test_months = [d.month for d in dates_test]
    
    print(f"\nTrain months: {set(train_months)}")
    print(f"Test months:  {set(test_months)}")
    
    # Check if test is from different season
    wet_season = [11, 12, 1, 2, 3]  # Nov-Mar for Jakarta
    train_wet = sum(1 for m in train_months if m in wet_season) / len(train_months)
    test_wet = sum(1 for m in test_months if m in wet_season) / len(test_months)
    
    print(f"\nWet season fraction:")
    print(f"  Train: {train_wet:.2%}")
    print(f"  Test:  {test_wet:.2%}")
    
    if abs(train_wet - test_wet) > 0.3:
        print("\n⚠️  CRITICAL: Test data from different season!")
        print("   This causes systematic temperature differences.")
    
    # Visualize distributions
    print("\n4. CREATING DIAGNOSTIC PLOTS")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Histograms
    axes[0, 0].hist(y_train_denorm.flatten(), bins=50, alpha=0.5, label='Train', density=True)
    axes[0, 0].hist(y_val_denorm.flatten(), bins=50, alpha=0.5, label='Val', density=True)
    axes[0, 0].hist(y_test_denorm.flatten(), bins=50, alpha=0.5, label='Test', density=True)
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Temperature Distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Box plots
    data_to_plot = [
        y_train_denorm.flatten(),
        y_val_denorm.flatten(),
        y_test_denorm.flatten()
    ]
    axes[0, 1].boxplot(data_to_plot, labels=['Train', 'Val', 'Test'])
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Temperature Range Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Temporal trend
    train_temps = [y_train_denorm.mean() for _ in dates_train[:100]]  # Sample
    test_temps = [y_test_denorm.mean() for _ in dates_test[:100]]
    
    axes[1, 0].plot(dates_train[:100], train_temps, 'o', alpha=0.5, label='Train')
    axes[1, 0].plot(dates_test[:100], test_temps, 'o', alpha=0.5, label='Test')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_title('Temporal Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Feature distributions (NDVI example)
    X_train = splits['train']['X']
    X_test = splits['test']['X']
    
    # Check NDVI channel (assuming it's channel 4)
    if X_train.shape[-1] > 4:
        axes[1, 1].hist(X_train[:, :, :, 4].flatten(), bins=50, alpha=0.5, 
                       label='Train', density=True)
        axes[1, 1].hist(X_test[:, :, :, 4].flatten(), bins=50, alpha=0.5, 
                       label='Test', density=True)
        axes[1, 1].set_xlabel('NDVI (normalized)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Feature Distribution (NDVI)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/distribution_diagnosis.png', dpi=150, bbox_inches='tight')
    print("✅ Saved diagnostic plots to outputs/distribution_diagnosis.png")
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    issues_found = []
    
    if mean_diff > 2.0:
        issues_found.append("CRITICAL: Large temperature difference between train and test")
    
    if abs(train_wet - test_wet) > 0.3:
        issues_found.append("CRITICAL: Test data from different season")
    
    if std_ratio < 0.7 or std_ratio > 1.3:
        issues_found.append("WARNING: Different variance in test data")
    
    if issues_found:
        print("\n🔴 ISSUES FOUND:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        
        print("\n💡 RECOMMENDATIONS:")
        print("  1. Use RANDOM split instead of TEMPORAL split")
        print("  2. Ensure test data includes all seasons")
        print("  3. Re-run preprocessing with stratified sampling")
        print("  4. Consider domain adaptation techniques")
        
        print("\n📝 To fix, modify preprocessing.py:")
        print("  Change split_method='temporal' to split_method='random'")
    else:
        print("\n✅ No major distribution issues detected")
        print("   The problem may be in model architecture or training")
    
    print("="*70)


if __name__ == "__main__":
    diagnose_dataset()