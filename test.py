import numpy as np
import json

# Load your predictions
pred = np.load("outputs/data/predictions_ensemble.npy")
gt = np.load("data/processed/cnn_dataset/test/y.npy")

print(f"Predictions: mean={pred.mean():.2f}, std={pred.std():.2f}")
print(f"Ground truth: mean={gt.mean():.2f}, std={gt.std():.2f}")

# Load normalization stats
with open("data/processed/normalization_stats.json") as f:
    stats = json.load(f)

mean = stats['target']['mean']
std_dev = stats['target']['std']

# Try denormalizing
fixed = pred * std_dev + mean
print(f"After denorm: mean={fixed.mean():.2f}, std={fixed.std():.2f}")

# Check MAE
mae_before = np.abs(pred - gt).mean()
mae_after = np.abs(fixed - gt).mean()

print(f"\nMAE before: {mae_before:.2f}")
print(f"MAE after: {mae_after:.2f}")