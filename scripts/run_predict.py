#!/usr/bin/env python3
"""
Main execution script for Shark Foraging Hotspot Prediction
NASA Space Apps 2025 - Team Solo Scientist

This script orchestrates the complete workflow for computing shark foraging hotspots
using NASA satellite data and generating interactive visualizations.
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add the foi_backend to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'foi_backend'))

from foi_backend.shark_hotspots.predictor import SharkHotspotPredictor
from foi_backend.shark_hotspots.visualize_hotspots import HotspotVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/shark_hotspots.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Shark Foraging Hotspot Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python run_predict.py

  # Run with custom date range
  python run_predict.py --start-date 2025-03-01 --end-date 2025-03-14

  # Run with custom region
  python run_predict.py --bbox 94.0 -11.0 142.0 6.0

  # Run with custom config
  python run_predict.py --config custom_config.yaml

  # Generate visualizations only
  python run_predict.py --visualize-only --date 20250301
        """
    )
    
    # Date parameters
    parser.add_argument('--start-date', type=str, default='2025-03-01',
                       help='Start date in YYYY-MM-DD format (default: 2025-03-01)')
    parser.add_argument('--end-date', type=str, default='2025-03-14',
                       help='End date in YYYY-MM-DD format (default: 2025-03-14)')
    
    # Region parameters
    parser.add_argument('--bbox', nargs=4, type=float, 
                       default=[94.0, -11.0, 142.0, 6.0],
                       metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'),
                       help='Bounding box: lon_min lat_min lon_max lat_max (default: 94.0 -11.0 142.0 6.0)')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='data/outputs',
                       help='Output directory (default: data/outputs)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output files')
    
    # Visualization options
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only generate visualizations (requires existing FOI data)')
    parser.add_argument('--date', type=str,
                       help='Date for visualization-only mode (YYYYMMDD format)')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Skip interactive dashboard generation')
    parser.add_argument('--no-static', action='store_true',
                       help='Skip static snapshot generation')
    
    # Processing options
    parser.add_argument('--compute-cps', action='store_true',
                       help='Also compute Conservation Priority Surface')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments"""
    # Validate dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        if (end_date - start_date).days > 30:
            logger.warning("Date range is longer than 30 days. This may take a while.")
            
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)
    
    # Validate bounding box
    lon_min, lat_min, lon_max, lat_max = args.bbox
    
    if lon_min >= lon_max:
        raise ValueError("lon_min must be less than lon_max")
    
    if lat_min >= lat_max:
        raise ValueError("lat_min must be less than lat_max")
    
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
        raise ValueError("Longitude values must be between -180 and 180")
    
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        raise ValueError("Latitude values must be between -90 and 90")
    
    # Validate visualization-only mode
    if args.visualize_only:
        if not args.date:
            logger.error("--date is required for --visualize-only mode")
            sys.exit(1)
        
        try:
            datetime.strptime(args.date, '%Y%m%d')
        except ValueError:
            logger.error("Invalid date format for --date. Use YYYYMMDD format.")
            sys.exit(1)


def setup_directories(output_dir):
    """Setup required directories"""
    directories = [
        Path(output_dir),
        Path(output_dir) / 'foi',
        Path(output_dir) / 'cps',
        Path(output_dir) / 'visualizations',
        Path('logs'),
        Path('data/inputs')
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Setup directories in {output_dir}")


def run_foi_computation(args):
    """Run FOI computation"""
    logger.info("Starting FOI computation...")
    
    try:
        # Initialize predictor
        predictor = SharkHotspotPredictor(args.config)
        
        # Compute FOI map
        foi, summary = predictor.compute_foi_map(
            args.start_date,
            args.end_date,
            args.bbox,
            save_output=not args.no_save
        )
        
        logger.info("FOI computation completed successfully")
        
        # Print summary
        print("\n" + "="*60)
        print("FOI COMPUTATION RESULTS")
        print("="*60)
        print(f"Date range: {args.start_date} to {args.end_date}")
        print(f"Region: {args.bbox}")
        print(f"FOI range: [{foi.min().values:.4f}, {foi.max().values:.4f}]")
        print(f"FOI mean: {foi.mean().values:.4f}")
        print(f"Spatial coverage: {summary['spatial_coverage']['coverage_percentage']:.2f}%")
        
        print("\nHotspot Statistics:")
        for threshold in ['low', 'medium', 'high']:
            stats = summary[f'{threshold}_hotspots']
            print(f"  {threshold.capitalize()} hotspots: {stats['count']} pixels ({stats['percentage']:.2f}%)")
        
        # Compute CPS if requested
        if args.compute_cps:
            logger.info("Computing Conservation Priority Surface...")
            cps, cps_summary = predictor.compute_cps_map(
                foi, 
                start_date=args.start_date, 
                end_date=args.end_date,
                save_output=not args.no_save
            )
            
            print(f"\nCPS range: [{cps.min().values:.4f}, {cps.max().values:.4f}]")
            print(f"CPS mean: {cps.mean().values:.4f}")
        
        return foi, summary
        
    except Exception as e:
        logger.error(f"Error in FOI computation: {e}")
        raise


def run_visualization(args, foi=None, summary=None):
    """Run visualization generation"""
    logger.info("Starting visualization generation...")
    
    try:
        visualizer = HotspotVisualizer()
        
        # Load FOI data if not provided
        if foi is None:
            if not args.date:
                args.date = args.start_date.replace('-', '')
            
            predictor = SharkHotspotPredictor(args.config)
            foi, summary = predictor.load_foi_map(args.date)
        
        # Generate static snapshot
        if not args.no_static:
            snapshot_path = visualizer.create_static_snapshot(foi, args.date)
            print(f"Static snapshot: {snapshot_path}")
        
        # Generate interactive dashboard
        if not args.no_interactive:
            dashboard_path = visualizer.generate_dashboard(foi, args.date)
            if dashboard_path:
                print(f"Interactive dashboard: {dashboard_path}")
            else:
                print("Interactive dashboard not generated (Folium not available)")
        
        # Generate statistics plot
        stats_path = visualizer.create_statistics_plot(foi, args.date)
        print(f"Statistics plot: {stats_path}")
        
        logger.info("Visualization generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in visualization generation: {e}")
        raise


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    validate_arguments(args)
    
    # Setup directories
    setup_directories(args.output_dir)
    
    logger.info("Starting Shark Foraging Hotspot Prediction System")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        if args.visualize_only:
            # Visualization-only mode
            logger.info("Running in visualization-only mode")
            run_visualization(args)
        else:
            # Full computation mode
            foi, summary = run_foi_computation(args)
            
            # Generate visualizations
            if not (args.no_static and args.no_interactive):
                run_visualization(args, foi, summary)
        
        logger.info("Shark Foraging Hotspot Prediction completed successfully!")
        
        # Print final status
        print("\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("="*60)
        
        predictor = SharkHotspotPredictor(args.config)
        status = predictor.get_processing_status()
        
        print(f"Total FOI maps: {status['total_foi_maps']}")
        print(f"Total CPS maps: {status['total_cps_maps']}")
        
        if status['foi_maps']:
            print("\nAvailable FOI maps:")
            for foi_info in status['foi_maps']:
                print(f"  {foi_info['date']}: {foi_info['size_mb']:.2f} MB")
        
        if status['cps_maps']:
            print("\nAvailable CPS maps:")
            for cps_info in status['cps_maps']:
                print(f"  {cps_info['date']}: {cps_info['size_mb']:.2f} MB")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
