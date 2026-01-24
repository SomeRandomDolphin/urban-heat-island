"""
Comprehensive analysis of the entire UHI pipeline
Identifies ALL potential improvements beyond just learning rate
"""
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

def analyze_pipeline():
    """Comprehensive pipeline analysis"""
    
    print("="*70)
    print("COMPREHENSIVE PIPELINE ANALYSIS")
    print("="*70)
    
    data_dir = Path("data/processed/cnn_dataset")
    
    # ============================================================
    # 1. DATA ANALYSIS
    # ============================================================
    print("\n" + "="*70)
    print("1. DATA QUALITY ANALYSIS")
    print("="*70)
    
    # Load all splits
    splits = {}
    for split in ['train', 'val', 'test']:
        X = np.load(data_dir / split / "X.npy")
        y = np.load(data_dir / split / "y.npy")
        dates = np.load(data_dir / split / "dates.npy", allow_pickle=True)
        splits[split] = {'X': X, 'y': y, 'dates': dates}
    
    # Check sample sizes
    print("\n1.1 Dataset Size Analysis")
    print("-"*70)
    for split, data in splits.items():
        n_samples = len(data['X'])
        n_pixels = n_samples * 64 * 64
        print(f"{split.upper()}: {n_samples:,} samples ({n_pixels:,} pixels)")
    
    total_samples = sum(len(data['X']) for data in splits.values())
    train_ratio = len(splits['train']['X']) / total_samples
    
    if total_samples < 1000:
        print(f"\n⚠️  WARNING: Only {total_samples} total samples!")
        print("   Recommendation: Collect more data or reduce patch stride")
        print("   Current stride: 48 → Try stride: 32 (more overlap)")
    
    if train_ratio < 0.65:
        print(f"\n⚠️  WARNING: Training set is only {train_ratio:.1%} of data")
        print("   Recommendation: Use 70-80% for training")
    
    # Check class balance (temperature distribution)
    print("\n1.2 Temperature Distribution Analysis")
    print("-"*70)
    
    with open(data_dir / "normalization_stats.json", 'r') as f:
        norm_stats = json.load(f)
    
    target_mean = norm_stats['target']['mean']
    target_std = norm_stats['target']['std']
    
    for split, data in splits.items():
        y_denorm = data['y'] * target_std + target_mean
        
        # Calculate temperature bins
        bins = [20, 25, 30, 35, 40, 45, 50]
        hist, _ = np.histogram(y_denorm.flatten(), bins=bins)
        
        print(f"\n{split.upper()} temperature distribution:")
        for i in range(len(bins)-1):
            pct = hist[i] / hist.sum() * 100
            print(f"  {bins[i]}-{bins[i+1]}°C: {pct:5.1f}%")
        
        # Check for imbalance
        max_pct = hist.max() / hist.sum() * 100
        if max_pct > 60:
            print(f"  ⚠️  WARNING: Imbalanced ({max_pct:.0f}% in one bin)")
            print(f"     Recommendation: Use weighted loss or stratified sampling")
    
    # Check temporal coverage
    print("\n1.3 Temporal Coverage Analysis")
    print("-"*70)
    
    for split, data in splits.items():
        dates = data['dates']
        months = [d.month for d in dates]
        month_counts = Counter(months)
        
        print(f"\n{split.upper()} months:")
        for month in sorted(month_counts.keys()):
            count = month_counts[month]
            pct = count / len(dates) * 100
            print(f"  Month {month:2d}: {count:3d} samples ({pct:5.1f}%)")
        
        # Check for missing seasons
        unique_months = set(months)
        if len(unique_months) < 6:
            print(f"  ⚠️  WARNING: Only {len(unique_months)} months covered")
            print(f"     Recommendation: Collect data from more months")
    
    # ============================================================
    # 2. FEATURE ANALYSIS
    # ============================================================
    print("\n" + "="*70)
    print("2. FEATURE QUALITY ANALYSIS")
    print("="*70)
    
    X_train = splits['train']['X']
    
    # Load metadata for channel names
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    channel_order = metadata.get('channel_order', [f'ch_{i}' for i in range(X_train.shape[-1])])
    
    print("\n2.1 Feature Statistics")
    print("-"*70)
    
    for i, name in enumerate(channel_order):
        channel_data = X_train[:, :, :, i]
        
        mean = channel_data.mean()
        std = channel_data.std()
        min_val = channel_data.min()
        max_val = channel_data.max()
        
        # Check for dead features
        if std < 0.01:
            print(f"  ⚠️  {name}: DEAD FEATURE (std={std:.4f})")
        elif std < 0.1:
            print(f"  ⚠️  {name}: Low variance (std={std:.4f})")
        else:
            print(f"  ✅ {name}: mean={mean:+.3f}, std={std:.3f}, range=[{min_val:.2f}, {max_val:.2f}]")
    
    # Feature correlation analysis
    print("\n2.2 Feature Correlation Analysis")
    print("-"*70)
    
    # Sample 10,000 pixels for correlation
    n_samples = min(10000, X_train.shape[0] * X_train.shape[1] * X_train.shape[2])
    X_flat = X_train.reshape(-1, X_train.shape[-1])
    indices = np.random.choice(len(X_flat), n_samples, replace=False)
    X_sample = X_flat[indices]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X_sample.T)
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(channel_order)):
        for j in range(i+1, len(channel_order)):
            if abs(corr_matrix[i, j]) > 0.9:
                high_corr_pairs.append((channel_order[i], channel_order[j], corr_matrix[i, j]))
    
    if high_corr_pairs:
        print("\n  ⚠️  Highly correlated features (|r| > 0.9):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"     {feat1} <-> {feat2}: r={corr:.3f}")
        print("\n  Recommendation: Consider removing redundant features")
    else:
        print("  ✅ No highly correlated features detected")
    
    # ============================================================
    # 3. MODEL ARCHITECTURE ANALYSIS
    # ============================================================
    print("\n" + "="*70)
    print("3. MODEL ARCHITECTURE ANALYSIS")
    print("="*70)
    
    import torch
    from models import UNet
    
    n_channels = X_train.shape[-1]
    model = UNet(in_channels=n_channels, out_channels=1)
    
    # Count parameters by layer type
    total_params = 0
    conv_params = 0
    bn_params = 0
    
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        
        if 'conv' in name and 'weight' in name:
            conv_params += n
        elif 'bn' in name or 'norm' in name:
            bn_params += n
    
    print(f"\n3.1 Parameter Count")
    print("-"*70)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Conv parameters: {conv_params:,} ({conv_params/total_params*100:.1f}%)")
    print(f"  BN parameters: {bn_params:,} ({bn_params/total_params*100:.1f}%)")
    
    # Calculate parameters per sample
    params_per_sample = total_params / len(X_train)
    print(f"\n  Parameters per training sample: {params_per_sample:,.0f}")
    
    if params_per_sample > 100:
        print(f"  ⚠️  WARNING: Very high params/sample ratio!")
        print(f"     Model may be too complex for dataset size")
        print(f"     Recommendation: Reduce model capacity")
    
    # Check receptive field
    print(f"\n3.2 Receptive Field Analysis")
    print("-"*70)
    print(f"  Input size: 64×64")
    print(f"  U-Net with 4 levels → Receptive field ≈ 62×62")
    print(f"  Coverage: ~96% of input")
    
    if total_params > 20_000_000:
        print(f"\n  ⚠️  Model has {total_params/1e6:.1f}M parameters")
        print(f"     For 64×64 patches, this is overkill")
        print(f"     Recommendation: Use smaller model (see suggestions below)")
    
    # ============================================================
    # 4. TRAINING CONFIGURATION ANALYSIS
    # ============================================================
    print("\n" + "="*70)
    print("4. TRAINING CONFIGURATION ANALYSIS")
    print("="*70)
    
    # Check if training history exists
    history_path = Path("models/training_history.json")
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        print("\n4.1 Training Convergence Analysis")
        print("-"*70)
        
        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])
        
        if len(train_losses) > 0:
            print(f"  Epochs trained: {len(train_losses)}")
            print(f"  Initial train loss: {train_losses[0]:.4f}")
            print(f"  Final train loss: {train_losses[-1]:.4f}")
            
            # Check convergence
            if len(train_losses) >= 10:
                recent_losses = train_losses[-10:]
                loss_std = np.std(recent_losses)
                
                if loss_std < 0.01:
                    print(f"  ✅ Training converged (loss stable)")
                else:
                    print(f"  ⚠️  Training not converged (loss still changing)")
            
            # Check overfitting
            if len(val_losses) > 0:
                train_final = train_losses[-1]
                val_final = val_losses[-1]
                gap = val_final - train_final
                
                print(f"  Train-val gap: {gap:.4f}")
                
                if gap > 0.5:
                    print(f"  ⚠️  Significant overfitting detected")
                    print(f"     Recommendation: Increase regularization")
                elif gap < 0.1:
                    print(f"  ⚠️  Underfitting (val loss ≈ train loss)")
                    print(f"     Recommendation: Increase model capacity or train longer")
                else:
                    print(f"  ✅ Reasonable generalization gap")
        
        # Check CNN metrics
        cnn_metrics = history.get('cnn_metrics', [])
        if len(cnn_metrics) > 0:
            final_r2 = cnn_metrics[-1]['r2']
            final_rmse = cnn_metrics[-1]['rmse']
            
            print(f"\n4.2 Final CNN Performance")
            print("-"*70)
            print(f"  Validation R²: {final_r2:.4f}")
            print(f"  Validation RMSE: {final_rmse:.4f}°C")
            
            if final_r2 < 0.5:
                print(f"  ⚠️  Poor R² score - model not learning well")
            elif final_r2 < 0.7:
                print(f"  ⚠️  Mediocre R² score - room for improvement")
            else:
                print(f"  ✅ Good R² score")
    
    # ============================================================
    # 5. LOSS FUNCTION ANALYSIS
    # ============================================================
    print("\n" + "="*70)
    print("5. LOSS FUNCTION ANALYSIS")
    print("="*70)
    
    # Simulate loss calculation
    from models import LSTLoss
    
    # Load loss config
    try:
        from config import TRAINING_CONFIG
        loss_weights = TRAINING_CONFIG.get('loss_weights', {})
        
        print(f"\nCurrent loss weights:")
        print(f"  MSE (alpha): {loss_weights.get('mse', 1.0)}")
        print(f"  Spatial (beta): {loss_weights.get('spatial', 0.1)}")
        print(f"  Physical (gamma): {loss_weights.get('physical', 0.1)}")
        
        # Check if weights make sense
        alpha = loss_weights.get('mse', 1.0)
        beta = loss_weights.get('spatial', 0.1)
        gamma = loss_weights.get('physical', 0.1)
        
        if beta > alpha:
            print(f"\n  ⚠️  Spatial loss weight > MSE weight")
            print(f"     This may cause over-smoothing")
        
        if gamma > alpha:
            print(f"\n  ⚠️  Physical loss weight > MSE weight")
            print(f"     This may hurt accuracy for edge cases")
    except:
        pass
    
    # ============================================================
    # 6. RECOMMENDATIONS SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("6. COMPREHENSIVE RECOMMENDATIONS")
    print("="*70)
    
    recommendations = []
    
    # Data recommendations
    if total_samples < 1000:
        recommendations.append({
            "category": "DATA",
            "priority": "HIGH",
            "issue": "Small dataset",
            "solution": "Reduce patch stride from 48 to 32 (more overlap)"
        })
    
    # Model recommendations
    if params_per_sample > 100:
        recommendations.append({
            "category": "MODEL",
            "priority": "HIGH",
            "issue": "Model too complex for dataset size",
            "solution": "Use smaller architecture (fewer layers or channels)"
        })
    
    # Training recommendations
    if history_path.exists() and len(cnn_metrics) > 0 and cnn_metrics[-1]['r2'] < 0.6:
        recommendations.append({
            "category": "TRAINING",
            "priority": "HIGH",
            "issue": "Poor convergence (R² < 0.6)",
            "solution": "Increase learning rate, train longer, or simplify model"
        })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['category']}")
        print(f"   Issue: {rec['issue']}")
        print(f"   Solution: {rec['solution']}")
    
    print("\n" + "="*70)
    print("Analysis complete! See detailed recommendations above.")
    print("="*70)


if __name__ == "__main__":
    analyze_pipeline()