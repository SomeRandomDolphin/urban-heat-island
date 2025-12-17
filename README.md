# Urban Heat Island Detection System

Deep learning-based system for mapping and analyzing urban heat islands using satellite imagery and ground sensor data.

## 📋 Overview

This project implements a comprehensive pipeline for:
- Satellite data acquisition and preprocessing (Landsat 8, Sentinel-2)
- Land Surface Temperature (LST) calculation
- Deep learning model training (U-Net + Gradient Boosting ensemble)
- High-resolution UHI mapping (30-100m resolution)
- Hotspot identification and analysis
- Cooling strategy simulation

**Target Area:** Jakarta Metropolitan Area, Indonesia

**Performance Targets:**
- R² ≥ 0.80
- RMSE ≤ 1.5°C
- MAE ≤ 1.0°C

## 🗂️ Project Structure

```
uhi-detection/
├── config.py                 # Configuration settings
├── preprocessing.py          # Data preprocessing pipeline
├── earth_engine_loader.py    # Satellite data loader (Google Earth Engine)
├── models.py                 # Deep learning models (U-Net, losses)
├── train.py                  # Training pipeline
├── utils.py                  # Utility functions
├── data/
│   ├── raw/                  # Raw satellite data
│   └── processed/            # Preprocessed training data
│       ├── train/
│       ├── val/
│       └── test/
├── models/                   # Saved model checkpoints
├── outputs/                  # Generated maps and reports
└── logs/                     # Training logs
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Google Earth Engine account

### Setup

```bash
# Clone repository
git clone <repository-url>
cd uhi-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Authenticate Google Earth Engine
earthengine authenticate
```

### Requirements

```txt
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Deep learning
torch>=1.10.0
torchvision>=0.11.0

# Geospatial
rasterio>=1.2.0
geopandas>=0.10.0
shapely>=1.8.0
pyproj>=3.2.0
earthengine-api>=0.1.300

# Machine learning
scikit-learn>=1.0.0
lightgbm>=3.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Utilities
tqdm>=4.62.0
```

## 📊 Data Acquisition

### 1. Download Satellite Data

```python
from earth_engine_loader import EarthEngineLoader

# Initialize loader
loader = EarthEngineLoader(satellite_type="landsat")

# Download monthly data
loader.download_monthly_data(year=2024, month=6, output_dir="data/raw/landsat")
```

### 2. Preprocess Data

```python
from preprocessing import DatasetCreator

# Create dataset
creator = DatasetCreator()

# Create grid
grid = creator.create_grid(
    bounds=STUDY_AREA["bounds"],
    resolution=50  # meters
)

# Extract patches and create training samples
# (See preprocessing.py for full pipeline)
```

## 🏋️ Training

### Quick Start

```bash
# Train model with default settings
python train.py
```

### Custom Training

```python
from train import Trainer
from models import UNet
import torch

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = UNet(in_channels=15, out_channels=1)

# Create trainer
trainer = Trainer(model, device)

# Train
history = trainer.train(train_loader, val_loader, save_dir="models/")
```

### Training Configuration

Edit `config.py` to modify:

```python
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "initial_lr": 1e-4,
    "patience": 10,
    # ... more options
}
```

## 📈 Evaluation

```python
from utils import calculate_metrics, plot_prediction_scatter

# Load test data
X_test, y_test = load_data("test")

# Make predictions
predictions = model.predict(X_test)

# Calculate metrics
metrics = calculate_metrics(y_test, predictions)
print(metrics)

# Visualize results
plot_prediction_scatter(y_test, predictions, save_path="outputs/scatter.png")
```

## 🗺️ Generate UHI Maps

```python
from utils import calculate_uhi_intensity, calculate_hotspots, save_geotiff

# Calculate UHI intensity
uhi_stats = calculate_uhi_intensity(lst, ndvi, building_density)

# Identify hotspots
hotspots = calculate_hotspots(lst, coords, distance_threshold=500)

# Save as GeoTIFF
save_geotiff(
    uhi_stats["uhi_map"],
    output_path="outputs/uhi_map.tif",
    bounds=STUDY_AREA["bounds"]
)
```

## 📊 Model Architecture

### U-Net CNN

```
Input: (128, 128, 15) channels
├── Encoder: 4 blocks [64, 128, 256, 512]
├── Bottleneck: 1024 filters
└── Decoder: 4 blocks with skip connections
Output: (128, 128, 1) - LST prediction

Total Parameters: ~31 million
```

### Multi-Task Loss

```
Total Loss = 0.7 × MSE + 0.2 × Spatial + 0.1 × Physical
```

## 🎯 Key Features

### Input Features (15 channels)

**Spectral Bands (4):**
- Red, NIR, SWIR1, SWIR2

**Spectral Indices (7):**
- NDVI, NDBI, MNDWI, BSI, UI, EBBI, Albedo

**Urban Morphology (3):**
- Building density, Road density, Impervious surface

**Meteorological (1):**
- Air temperature (interpolated)

### Outputs

1. **LST Map** - High-resolution temperature map
2. **UHI Intensity Map** - Temperature above rural reference
3. **Hotspot Map** - Statistical hot spots (Gi* statistic)
4. **Uncertainty Map** - Prediction confidence
5. **Feature Importance** - Key UHI drivers

## 🔧 Configuration

Key settings in `config.py`:

```python
# Study area
STUDY_AREA = {
    "bounds": {
        "min_lon": 106.6, "max_lon": 107.1,
        "min_lat": -6.4, "max_lat": -6.0
    },
    "epsg": 32748  # WGS84 / UTM Zone 48S
}

# Grid resolution
GRID_CONFIG = {
    "resolution": 50,  # meters
    "patch_size": 64,  # pixels
}

# Model targets
VALIDATION_CONFIG = {
    "targets": {
        "r2": 0.80,
        "rmse": 1.5,  # °C
        "mae": 1.0    # °C
    }
}
```

## 📖 Usage Examples

### Example 1: Complete Pipeline

```python
# 1. Download data
loader = EarthEngineLoader("landsat")
loader.download_monthly_data(2024, 6, "data/raw")

# 2. Preprocess
creator = DatasetCreator()
X, y = creator.create_training_samples(...)

# 3. Train
trainer = Trainer(model, device)
history = trainer.train(train_loader, val_loader, "models/")

# 4. Generate maps
predictions = model.predict(X_test)
uhi_stats = calculate_uhi_intensity(predictions, ndvi)

# 5. Save results
save_geotiff(uhi_stats["uhi_map"], "outputs/uhi.tif", bounds)
```

### Example 2: Batch Processing

```python
# Process multiple months
for year in [2023, 2024]:
    for month in range(1, 13):
        loader.download_monthly_data(year, month, "data/raw")
```

### Example 3: Custom Analysis

```python
# Calculate hotspots with custom threshold
hotspots = calculate_hotspots(
    lst,
    coords,
    distance_threshold=1000  # 1km
)

# Identify areas above threshold
critical_zones = hotspots > 2.58  # 99% confidence
```

## 🧪 Testing

```bash
# Test model architecture
python models.py

# Test preprocessing
python preprocessing.py

# Test utilities
python utils.py
```

## 📊 Expected Results

**Model Performance:**
- Training time: ~2-4 hours (on NVIDIA A100)
- Inference time: ~1 minute for full Jakarta area
- Memory: ~10GB GPU RAM

**Output Specifications:**
- Resolution: 50m × 50m
- Coverage: ~7,000 km² (Jakarta + buffer)
- Total grid cells: ~2.8 million
- File sizes: ~500MB (GeoTIFF)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 📚 References

1. Landsat 8 Collection 2 Level-2 Science Product Guide
2. Sentinel-2 User Handbook
3. U-Net: Convolutional Networks for Biomedical Image Segmentation
4. Urban Heat Island detection methodologies

## 🆘 Support

For issues and questions:
- Open a GitHub issue
- Check documentation in `docs/`
- Contact: [your-email]

## 🎯 Roadmap

- [ ] Add Sentinel-2 LST estimation
- [ ] Implement temporal analysis
- [ ] Add real-time prediction API
- [ ] Web dashboard integration
- [ ] Expand to other ASEAN cities

## ⚙️ Advanced Configuration

### GPU Memory Optimization

```python
# Reduce batch size
TRAINING_CONFIG["batch_size"] = 16

# Use gradient accumulation
accumulation_steps = 2

# Mixed precision training
use_amp = True
```

### Distributed Training

```python
# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{uhi_detection,
  title={Urban Heat Island Detection System},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

---

**Last Updated:** December 2025  
**Version:** 1.0.0