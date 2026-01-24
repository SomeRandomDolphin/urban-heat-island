import numpy as np

# Load predictions and ground truth
pred = np.load("outputs/data/predictions_ensemble.npy")
y_test = np.load("data/processed/cnn_dataset/test/y.npy")

print("Predictions (raw from model):")
print(f"  Mean: {pred.mean():.2f}, Std: {pred.std():.2f}")
print(f"  Range: [{pred.min():.2f}, {pred.max():.2f}]")

print("\nGround truth (normalized):")
print(f"  Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
print(f"  Range: [{y_test.min():.4f}, {y_test.max():.4f}]")

# Denormalize ground truth
from uhi_inference import DataNormalizer
from config import PROCESSED_DATA_DIR

normalizer = DataNormalizer(PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json")
y_test_celsius = normalizer.denormalize_predictions(y_test.squeeze(), prediction_type="target")

print("\nGround truth (Celsius):")
print(f"  Mean: {y_test_celsius.mean():.2f}, Std: {y_test_celsius.std():.2f}")
print(f"  Range: [{y_test_celsius.min():.2f}, {y_test_celsius.max():.2f}]")