"""
UPDATED: UHI Pipeline Manager - Works with NORMALIZED test data
"""
import sys
import numpy as np
from pathlib import Path
import logging
import argparse

from config import *
from uhi_inference import EnsemblePredictor, PostProcessor, DataNormalizer
from uhi_analysis import UHIAnalyzer, HotspotDetector, ValidationAnalyzer, generate_report
from uhi_visualization import UHIVisualizer, OutputGenerator

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def make_json_serializable(obj):
    """Convert numpy types to JSON-serializable types"""
    import numpy as np

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="Run UHI analysis pipeline")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory containing trained models")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Directory for output products")
    parser.add_argument("--test-data", type=str, 
                       default="data/processed/cnn_dataset/test",
                       help="Directory containing NORMALIZED test data")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--use-spatial-ensemble", action="store_true", default=True,
                       help="Use spatial ensemble (preserves detail)")
    parser.add_argument("--hotspot-radius", type=float, default=500.0,
                       help="Search radius for hotspot detection (meters)")
    parser.add_argument("--create-webmap", action="store_true", default=True,
                       help="Create interactive web map overlay")
    parser.add_argument("--mc-samples", type=int, default=50,
                       help="Number of MC Dropout samples for uncertainty")
    
    args = parser.parse_args()
    
    # Extract bounds from config
    bounds = (
        STUDY_AREA["bounds"]["min_lon"],
        STUDY_AREA["bounds"]["min_lat"],
        STUDY_AREA["bounds"]["max_lon"],
        STUDY_AREA["bounds"]["max_lat"]
    )
    
    logger.info("="*70)
    logger.info("UHI ANALYSIS PIPELINE")
    logger.info("="*70)
    logger.info(f"Study Area: {STUDY_AREA['name']}")
    logger.info(f"Bounds: {bounds}")
    logger.info(f"Spatial ensemble: {args.use_spatial_ensemble}")
    logger.info(f"MC Dropout samples: {args.mc_samples}")
    
    # Setup paths
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    test_data_dir = Path(args.test_data)
    normalization_stats_path = PROCESSED_DATA_DIR / "cnn_dataset" / "normalization_stats.json"
    
    # Verify required files exist
    required_files = [
        model_dir / "final_cnn.pth",
        model_dir / "gbm_model.pkl",
        model_dir / "ensemble_config.json",
        normalization_stats_path
    ]
    
    missing_files = [f for f in required_files if not f.exists()]
    if missing_files:
        logger.error("❌ Missing required files:")
        for f in missing_files:
            logger.error(f"  - {f}")
        logger.error("\nPlease run train_ensemble.py first!")
        return 1
    
    # Create output directories
    maps_dir = output_dir / "maps"
    data_dir = output_dir / "data"
    reports_dir = output_dir / "reports"
    geotiff_dir = output_dir / "geotiff"
    
    for directory in [maps_dir, data_dir, reports_dir, geotiff_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Load test data (should be NORMALIZED)
    logger.info("\n" + "="*70)
    logger.info("LOADING TEST DATA")
    logger.info("="*70)
    
    if not test_data_dir.exists():
        logger.error(f"❌ Test data directory not found: {test_data_dir}")
        logger.error("Please run preprocessing.py first!")
        return 1
    
    X_test_path = test_data_dir / "X.npy"
    y_test_path = test_data_dir / "y.npy"
    
    if not X_test_path.exists():
        logger.error(f"❌ Test features not found: {X_test_path}")
        return 1
    
    X_test = np.load(X_test_path)
    logger.info(f"✓ Test data loaded: {X_test.shape}")
    logger.info(f"  Mean: {X_test.mean():.4f}, Std: {X_test.std():.4f}")
    logger.info(f"  Range: [{X_test.min():.4f}, {X_test.max():.4f}]")
    
    # Validate that data is normalized
    if not (-0.5 < X_test.mean() < 0.5 and 0.5 < X_test.std() < 1.5):
        logger.warning("⚠️ Test data does not appear to be normalized!")
        logger.warning("   Expected: mean≈0, std≈1")
        logger.warning(f"   Got: mean={X_test.mean():.4f}, std={X_test.std():.4f}")
        logger.warning("   Predictions may be incorrect!")
    else:
        logger.info("✓ Test data is properly normalized")
    
    # Load ground truth (also normalized)
    y_test = None
    if y_test_path.exists():
        y_test = np.load(y_test_path)
        logger.info(f"✓ Ground truth loaded: {y_test.shape}")
        logger.info(f"  Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
        
        if not (-0.5 < y_test.mean() < 0.5 and 0.5 < y_test.std() < 1.5):
            logger.warning("⚠️ Ground truth does not appear to be normalized!")
        else:
            logger.info("✓ Ground truth is properly normalized")
    
    # Initialize predictor
    logger.info("\n" + "="*70)
    logger.info("INITIALIZING PREDICTOR")
    logger.info("="*70)
    
    try:
        predictor = EnsemblePredictor(
            cnn_model_path=model_dir / "final_cnn.pth",
            gbm_model_path=model_dir / "gbm_model.pkl",
            ensemble_config_path=model_dir / "ensemble_config.json",
            normalization_stats_path=normalization_stats_path,
            device="cuda",
            mc_dropout_rate=0.05
        )
        logger.info("✓ Predictor initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Generate predictions
    logger.info("\n" + "="*70)
    logger.info("GENERATING PREDICTIONS")
    logger.info("="*70)
    logger.info(f"Processing {len(X_test)} test samples...")
    logger.info("Note: Input is NORMALIZED, outputs will be in CELSIUS")
    
    try:
        results = predictor.predict_ensemble(
            X_test,  # NORMALIZED data
            batch_size=args.batch_size,
            return_uncertainty=True,
            use_spatial_ensemble=args.use_spatial_ensemble
        )
        
        logger.info("✓ Predictions generated successfully")
        logger.info(f"  Shape: {results['ensemble'].shape}")
        logger.info(f"  Range: [{results['ensemble'].min():.2f}, "
                   f"{results['ensemble'].max():.2f}]°C")
        logger.info(f"  Mean: {results['ensemble'].mean():.2f}°C ± "
                   f"{results['ensemble'].std():.2f}°C")
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save predictions
    np.save(data_dir / "predictions_ensemble.npy", results["ensemble"])
    np.save(data_dir / "predictions_cnn.npy", results["cnn"])
    np.save(data_dir / "predictions_gbm.npy", results["gbm"])
    
    if "uncertainty" in results and results["uncertainty"] is not None:
        np.save(data_dir / "uncertainty.npy", results["uncertainty"])
        logger.info(f"  Uncertainty: {results['uncertainty'].mean():.3f}°C ± "
                   f"{results['uncertainty'].std():.3f}°C")
    
    logger.info(f"✓ Predictions saved to {data_dir}")
    
    # Post-processing
    logger.info("\n" + "="*70)
    logger.info("POST-PROCESSING")
    logger.info("="*70)
    
    post_processor = PostProcessor()
    
    # Process all patches
    lst_maps_processed = []
    for i in range(min(len(results["ensemble"]), 10)):  # Limit to first 10 patches
        lst_map = results["ensemble"][i]
        lst_processed = post_processor.process(lst_map)
        lst_maps_processed.append(lst_processed)
    
    lst_maps_processed = np.array(lst_maps_processed)
    np.save(data_dir / "lst_processed.npy", lst_maps_processed)
    
    # Use first patch for detailed analysis
    lst_processed = lst_maps_processed[0]
    
    logger.info(f"✓ Post-processed {len(lst_maps_processed)} patches")
    logger.info(f"  Mean: {lst_processed.mean():.2f}°C, Std: {lst_processed.std():.2f}°C")
    
    # UHI Analysis
    logger.info("\n" + "="*70)
    logger.info("UHI ANALYSIS")
    logger.info("="*70)
    
    analyzer = UHIAnalyzer(lst_processed)
    
    # Create urban/rural masks
    urban_threshold = np.percentile(lst_processed, 75)
    rural_threshold = np.percentile(lst_processed, 25)
    
    urban_mask = lst_processed >= urban_threshold
    rural_mask = lst_processed <= rural_threshold
    
    logger.info(f"  Urban threshold: {urban_threshold:.2f}°C")
    logger.info(f"  Rural threshold: {rural_threshold:.2f}°C")
    logger.info(f"  Urban area: {urban_mask.sum()/urban_mask.size*100:.1f}%")
    logger.info(f"  Rural area: {rural_mask.sum()/rural_mask.size*100:.1f}%")
    
    ref_temps = analyzer.define_reference_areas(urban_mask, rural_mask)
    uhi_map = analyzer.calculate_uhi_intensity()
    classified, categories = analyzer.classify_uhi_intensity()
    uhi_stats = analyzer.calculate_statistics()
    
    np.save(data_dir / "uhi_intensity.npy", uhi_map)
    np.save(data_dir / "uhi_classified.npy", classified)
    
    logger.info(f"✓ UHI analysis complete")
    logger.info(f"  Mean intensity: {uhi_stats['mean_intensity']:.2f}°C")
    logger.info(f"  Max intensity: {uhi_stats['max_intensity']:.2f}°C")
    
    # Hotspot Detection
    logger.info("\n" + "="*70)
    logger.info("HOTSPOT DETECTION")
    logger.info("="*70)
    
    detector = HotspotDetector(lst_processed, resolution=50.0)
    gi_star = detector.calculate_gi_star(search_radius=args.hotspot_radius)
    hotspot_mask, hotspot_list = detector.identify_hotspots(confidence_level=0.95)
    
    import pandas as pd
    hotspots_df = pd.DataFrame()
    if len(hotspot_list) > 0:
        hotspots_df = detector.prioritize_hotspots(hotspot_list)
        hotspots_df.to_csv(data_dir / "hotspots.csv", index=False)
        logger.info(f"✓ Detected {len(hotspots_df)} hotspots")
    else:
        logger.warning("⚠️ No hotspots detected")
    
    np.save(data_dir / "gi_star.npy", gi_star)
    np.save(data_dir / "hotspot_mask.npy", hotspot_mask)
    
    # Validation
    validation_metrics = None
    if y_test is not None:
        logger.info("\n" + "="*70)
        logger.info("VALIDATION")
        logger.info("="*70)
        
        try:
            # Denormalize ground truth to Celsius for comparison
            normalizer = DataNormalizer(normalization_stats_path)
            y_test_celsius = normalizer.denormalize_predictions(
                y_test.squeeze(), prediction_type="target"
            )
            # Keep original shape for proper comparison
            y_test_celsius = y_test_celsius.reshape(y_test.shape)
            
            logger.info(f"  Denormalized ground truth: "
                    f"mean={y_test_celsius.mean():.2f}°C")
            
            # CRITICAL FIX: Compare ALL samples, not just the first one!
            logger.info(f"\n  Comparing all {len(results['ensemble'])} test samples...")
            
            # Flatten all predictions and ground truth
            predictions_flat = results["ensemble"].flatten()
            ground_truth_flat = y_test_celsius.flatten()
            
            # Remove NaN values
            mask = ~(np.isnan(predictions_flat) | np.isnan(ground_truth_flat))
            predictions_flat = predictions_flat[mask]
            ground_truth_flat = ground_truth_flat[mask]
            
            logger.info(f"  Valid pixels: {len(predictions_flat):,}")
            logger.info(f"  Predictions: mean={predictions_flat.mean():.2f}°C, std={predictions_flat.std():.2f}°C")
            logger.info(f"  Ground truth: mean={ground_truth_flat.mean():.2f}°C, std={ground_truth_flat.std():.2f}°C")
            
            # Calculate metrics manually
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            r2 = r2_score(ground_truth_flat, predictions_flat)
            rmse = np.sqrt(mean_squared_error(ground_truth_flat, predictions_flat))
            mae = mean_absolute_error(ground_truth_flat, predictions_flat)
            mbe = predictions_flat.mean() - ground_truth_flat.mean()
            
            # Calculate additional metrics
            correlation = np.corrcoef(predictions_flat, ground_truth_flat)[0, 1]
            
            # Linear regression for slope/intercept
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(ground_truth_flat, predictions_flat)
            
            validation_metrics = {
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'mbe': float(mbe),
                'correlation': float(correlation),
                'slope': float(slope),
                'intercept': float(intercept),
                'p_value': float(p_value)
            }
            
            logger.info(f"\n✅ Validation complete (ALL SAMPLES)")
            logger.info(f"  R²: {validation_metrics['r2']:.4f}")
            logger.info(f"  RMSE: {validation_metrics['rmse']:.4f}°C")
            logger.info(f"  MAE: {validation_metrics['mae']:.4f}°C")
            logger.info(f"  Bias (MBE): {validation_metrics['mbe']:.4f}°C")
            logger.info(f"  Correlation: {validation_metrics['correlation']:.4f}")
            logger.info(f"  Slope: {validation_metrics['slope']:.4f}")
            logger.info(f"  Intercept: {validation_metrics['intercept']:.4f}°C")
            
            # Check if performance is acceptable
            if validation_metrics['r2'] < 0.3:
                logger.warning("⚠️ Low R² - model needs improvement")
            elif validation_metrics['r2'] < 0.6:
                logger.info("✓ Moderate R² - acceptable performance")
            else:
                logger.info("✅ Good R² score!")
            
            # Create validation plot
            logger.info(f"\n  Creating validation plot...")
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Scatter plot
            axes[0].scatter(ground_truth_flat, predictions_flat, alpha=0.3, s=1, c='blue')
            axes[0].plot([ground_truth_flat.min(), ground_truth_flat.max()], 
                        [ground_truth_flat.min(), ground_truth_flat.max()], 
                        'r--', linewidth=2, label='Perfect prediction')
            axes[0].plot([ground_truth_flat.min(), ground_truth_flat.max()],
                        [slope * ground_truth_flat.min() + intercept,
                        slope * ground_truth_flat.max() + intercept],
                        'g-', linewidth=2, label=f'Fit (slope={slope:.3f})')
            axes[0].set_xlabel('Ground Truth (°C)', fontsize=12)
            axes[0].set_ylabel('Predictions (°C)', fontsize=12)
            axes[0].set_title(f'Predictions vs Ground Truth\nR²={r2:.4f}, RMSE={rmse:.2f}°C', fontsize=14)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Residual plot
            residuals = predictions_flat - ground_truth_flat
            axes[1].scatter(ground_truth_flat, residuals, alpha=0.3, s=1, c='blue')
            axes[1].axhline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
            axes[1].axhline(mbe, color='g', linestyle='--', linewidth=2, label=f'Bias={mbe:.2f}°C')
            axes[1].set_xlabel('Ground Truth (°C)', fontsize=12)
            axes[1].set_ylabel('Residual (°C)', fontsize=12)
            axes[1].set_title(f'Residual Plot\nMAE={mae:.2f}°C', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(maps_dir / "validation_plot.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  ✅ Validation plot saved to {maps_dir / 'validation_plot.png'}")
            
            # Save metrics
            import json
            serializable_metrics = make_json_serializable(validation_metrics)
            with open(reports_dir / "validation_metrics.json", 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            
            logger.info(f"  ✅ Metrics saved to {reports_dir / 'validation_metrics.json'}")
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Visualizations
    logger.info("\n" + "="*70)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*70)
    
    try:
        visualizer = UHIVisualizer(figsize=(12, 10), dpi=300)
        
        visualizer.create_lst_map(
            lst_processed,
            maps_dir / "lst_map.png",
            title=f"Land Surface Temperature - {STUDY_AREA['name']}"
        )
        
        visualizer.create_uhi_intensity_map(
            uhi_map,
            maps_dir / "uhi_intensity_map.png",
            title="Urban Heat Island Intensity"
        )
        
        visualizer.create_hotspot_map(
            lst_processed,
            hotspot_mask,
            gi_star,
            maps_dir / "hotspot_map.png",
            title="UHI Hotspot Analysis"
        )
        
        if "uncertainty" in results and results["uncertainty"] is not None:
            visualizer.create_uncertainty_map(
                lst_processed,
                results["uncertainty"][0],
                maps_dir / "uncertainty_map.png",
                title="Prediction Uncertainty"
            )
        
        visualizer.create_statistics_dashboard(
            uhi_stats,
            categories,
            hotspots_df,
            maps_dir / "statistics_dashboard.png"
        )
        
        logger.info(f"✓ Visualizations saved to {maps_dir}")
        
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate Report
    logger.info("\n" + "="*70)
    logger.info("GENERATING REPORT")
    logger.info("="*70)
    
    try:
        generate_report(
            uhi_stats,
            hotspots_df,
            validation_metrics,
            reports_dir / "uhi_analysis_report.json"
        )
        logger.info(f"✓ Report saved to {reports_dir}")
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
    
    # Export GeoTIFF
    logger.info("\n" + "="*70)
    logger.info("EXPORTING GEOTIFF")
    logger.info("="*70)
    
    try:
        generator = OutputGenerator(geotiff_dir)
        
        generator.export_geotiff(
            lst_processed.astype(np.float32),
            "lst_map.tif",
            bounds=bounds,
            metadata={"date": "2024-12-22", "units": "celsius"}
        )
        
        generator.export_geotiff(
            uhi_map.astype(np.float32),
            "uhi_intensity.tif",
            bounds=bounds,
            metadata={"units": "celsius_difference"}
        )
        
        logger.info(f"✓ GeoTIFF files saved to {geotiff_dir}")
        
    except Exception as e:
        logger.error(f"❌ GeoTIFF export failed: {e}")
    
    # Create Web Map
    if args.create_webmap:
        logger.info("\n" + "="*70)
        logger.info("CREATING WEB MAP")
        logger.info("="*70)
        
        try:
            from uhi_map_overlay import RealMapOverlay
            
            overlay = RealMapOverlay(bounds)
            webmap_dir = output_dir / "webmap"
            overlay.export_to_webmap(lst_processed, uhi_map, hotspots_df, webmap_dir)
            
            logger.info(f"✓ Web map created: {webmap_dir / 'index.html'}")
            logger.info("🌐 Open the HTML file in your browser to view!")
            
        except Exception as e:
            logger.error(f"❌ Web map creation failed: {e}")
    
    # Final Summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*70)
    
    logger.info(f"\n📁 Outputs:")
    logger.info(f"  📊 Maps: {maps_dir}")
    logger.info(f"  💾 Data: {data_dir}")
    logger.info(f"  📄 Reports: {reports_dir}")
    logger.info(f"  🗺️  GeoTIFF: {geotiff_dir}")
    if args.create_webmap:
        logger.info(f"  🌐 Web Map: {output_dir / 'webmap' / 'index.html'}")
    
    logger.info(f"\n📈 Results Summary:")
    logger.info(f"  LST: {lst_processed.mean():.2f}°C ± {lst_processed.std():.2f}°C")
    logger.info(f"  UHI: {uhi_stats['mean_intensity']:.2f}°C (max: {uhi_stats['max_intensity']:.2f}°C)")
    logger.info(f"  Hotspots: {len(hotspots_df)}")
    
    if validation_metrics:
        logger.info(f"\n🎯 Validation:")
        logger.info(f"  R²: {validation_metrics['r2']:.4f}")
        logger.info(f"  RMSE: {validation_metrics['rmse']:.4f}°C")
        logger.info(f"  MAE: {validation_metrics['mae']:.4f}°C")
    
    logger.info("\n" + "="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())