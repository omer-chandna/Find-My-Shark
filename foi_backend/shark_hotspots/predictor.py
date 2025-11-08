"""
Predictor Module
Main orchestration module for FOI computation and output generation
"""

import logging
import os
import yaml
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings

# Import local modules
from .data_loader import SatelliteDataLoader
from .harmonize import DataHarmonizer
from .derived_fields import DerivedFieldsComputer
from .normalization import DataNormalizer
from .model_core import FOIModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SharkHotspotPredictor:
    """
    Main predictor class for shark foraging hotspot analysis
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize predictor with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_directories()
        
        # Initialize components
        self.data_loader = SatelliteDataLoader(config_path)
        self.harmonizer = DataHarmonizer(
            target_resolution=self.config.get('grid', {}).get('resolution', 0.1)
        )
        self.derived_computer = DerivedFieldsComputer(self.config)
        self.normalizer = DataNormalizer()
        self.foi_model = FOIModel(self.config)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'data': {
                'start_date': '2025-03-01',
                'end_date': '2025-03-14',
                'composite_days': 7
            },
            'region': {
                'name': 'coral_triangle',
                'bbox': [94.0, -11.0, 142.0, 6.0]
            },
            'grid': {
                'resolution': 0.1
            },
            'output': {
                'format': 'netcdf',
                'output_dir': 'data/outputs'
            }
        }
    
    def setup_directories(self):
        """Setup output directories"""
        output_dir = Path(self.config.get('output', {}).get('output_dir', 'data/outputs'))
        
        # Create directory structure
        self.output_dir = output_dir
        self.foi_dir = output_dir / 'foi'
        self.cps_dir = output_dir / 'cps'
        self.logs_dir = Path('logs')
        
        # Create directories
        for directory in [self.output_dir, self.foi_dir, self.cps_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def compute_foi_map(self, start_date: str, end_date: str, 
                        bbox: Optional[List[float]] = None,
                        save_output: bool = True) -> Tuple[xr.DataArray, Dict]:
        """
        Compute Foraging Opportunity Index map
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
            save_output: Whether to save output files
            
        Returns:
            Tuple of (FOI DataArray, model_summary)
        """
        logger.info(f"Computing FOI map from {start_date} to {end_date}")
        
        # Use config bbox if not provided
        if bbox is None:
            bbox = self.config['region']['bbox']
        
        try:
            # Step 1: Load satellite data
            logger.info("Step 1: Loading satellite data...")
            datasets = self.data_loader.load_satellite_data(start_date, end_date, bbox)
            
            # Step 2: Harmonize datasets
            logger.info("Step 2: Harmonizing datasets...")
            composite_days = self.config.get('data', {}).get('composite_days', 7)
            composite_mode = self.config.get('data', {}).get('composite_mode', 'mean')
            harmonized = self.harmonizer.harmonize_datasets(
                datasets, bbox, start_date, end_date, composite_days, composite_mode
            )
            
            # Validate harmonization
            if not self.harmonizer.validate_harmonization(harmonized):
                raise ValueError("Dataset harmonization validation failed")
            
            # Step 3: Compute derived fields
            logger.info("Step 3: Computing derived fields...")
            derived_fields = self.derived_computer.compute_derived_fields(harmonized)
            
            # Step 4: Normalize fields
            logger.info("Step 4: Normalizing fields...")
            normalized_fields = self.normalizer.normalize_datasets(derived_fields)
            
            # Step 5: Compute FOI
            logger.info("Step 5: Computing FOI...")
            foi = self.foi_model.compute_foi(
                harmonized, derived_fields, normalized_fields
            )
            
            # Get model summary
            summary = self.foi_model.get_model_summary(foi)
            
            # Validate FOI
            if not self.foi_model.validate_model(foi):
                logger.warning("FOI model validation failed, but continuing...")
            
            # Step 6: Save output
            if save_output:
                logger.info("Step 6: Saving output...")
                self._save_foi_output(foi, summary, start_date, end_date)
            
            logger.info("FOI computation completed successfully")
            return foi, summary
            
        except Exception as e:
            logger.error(f"Error in FOI computation: {e}")
            raise
    
    def compute_cps_map(self, foi: xr.DataArray, 
                       fishing_effort: Optional[xr.DataArray] = None,
                       start_date: str = None, end_date: str = None,
                       save_output: bool = True) -> Tuple[xr.DataArray, Dict]:
        """
        Compute Conservation Priority Surface map
        
        Args:
            foi: Foraging Opportunity Index
            fishing_effort: Fishing effort data (optional)
            start_date: Start date for output naming
            end_date: End date for output naming
            save_output: Whether to save output files
            
        Returns:
            Tuple of (CPS DataArray, summary)
        """
        logger.info("Computing Conservation Priority Surface (CPS)...")
        
        try:
            # Compute CPS
            cps = self.foi_model.compute_cps(foi, fishing_effort)
            
            # Create summary
            summary = {
                'cps_statistics': {
                    'min': float(cps.min().values),
                    'max': float(cps.max().values),
                    'mean': float(cps.mean().values),
                    'std': float(cps.std().values)
                },
                'spatial_coverage': {
                    'total_pixels': int(cps.size),
                    'valid_pixels': int(np.sum(~np.isnan(cps.values))),
                    'coverage_percentage': float((np.sum(~np.isnan(cps.values)) / cps.size) * 100)
                },
                'fishing_effort_included': fishing_effort is not None
            }
            
            # Save output
            if save_output and start_date and end_date:
                self._save_cps_output(cps, summary, start_date, end_date)
            
            logger.info("CPS computation completed successfully")
            return cps, summary
            
        except Exception as e:
            logger.error(f"Error in CPS computation: {e}")
            raise
    
    def _save_foi_output(self, foi: xr.DataArray, summary: Dict, 
                        start_date: str, end_date: str):
        """Save FOI output files"""
        # Create date-based subdirectory
        date_str = start_date.replace('-', '')
        foi_date_dir = self.foi_dir / date_str
        foi_date_dir.mkdir(exist_ok=True)
        
        # Add temporal metadata to FOI
        foi_with_metadata = foi.copy()
        foi_with_metadata.attrs.update({
            'time_window': f"{start_date}→{end_date}",
            'composite_days': self.config.get('data', {}).get('composite_days', 7),
            'composite_mode': self.config.get('data', {}).get('composite_mode', 'mean'),
            'temporal_synchronization': 'enforced',
            'synchronization_method': 'xr.align_join_inner'
        })
        
        # Save FOI as NetCDF
        foi_filename = foi_date_dir / 'foi_map.nc'
        foi_with_metadata.to_netcdf(foi_filename)
        logger.info(f"Saved FOI map to {foi_filename}")
        
        # Save summary as JSON
        import json
        summary_filename = foi_date_dir / 'foi_summary.json'
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved FOI summary to {summary_filename}")
        
        # Save metadata
        metadata = {
            'computation_date': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'time_window': f"{start_date}→{end_date}",
            'composite_days': self.config.get('data', {}).get('composite_days', 7),
            'composite_mode': self.config.get('data', {}).get('composite_mode', 'mean'),
            'temporal_synchronization': 'enforced',
            'synchronization_method': 'xr.align_join_inner',
            'region': self.config['region'],
            'model_version': '1.0',
            'data_sources': ['PACE', 'MODIS-Aqua', 'SWOT'],
            'resolution': self.config.get('grid', {}).get('resolution', 0.1)
        }
        
        metadata_filename = foi_date_dir / 'metadata.json'
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_filename}")
    
    def _save_cps_output(self, cps: xr.DataArray, summary: Dict, 
                        start_date: str, end_date: str):
        """Save CPS output files"""
        # Create date-based subdirectory
        date_str = start_date.replace('-', '')
        cps_date_dir = self.cps_dir / date_str
        cps_date_dir.mkdir(exist_ok=True)
        
        # Save CPS as NetCDF
        cps_filename = cps_date_dir / 'cps_map.nc'
        cps.to_netcdf(cps_filename)
        logger.info(f"Saved CPS map to {cps_filename}")
        
        # Save summary as JSON
        import json
        summary_filename = cps_date_dir / 'cps_summary.json'
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved CPS summary to {summary_filename}")
    
    def load_foi_map(self, date_str: str) -> Tuple[xr.DataArray, Dict]:
        """
        Load previously computed FOI map
        
        Args:
            date_str: Date string in YYYYMMDD format
            
        Returns:
            Tuple of (FOI DataArray, summary)
        """
        foi_date_dir = self.foi_dir / date_str
        
        if not foi_date_dir.exists():
            raise FileNotFoundError(f"FOI data not found for date {date_str}")
        
        # Load FOI data
        foi_filename = foi_date_dir / 'foi_map.nc'
        foi = xr.open_dataarray(foi_filename)
        
        # Load summary
        summary_filename = foi_date_dir / 'foi_summary.json'
        import json
        with open(summary_filename, 'r') as f:
            summary = json.load(f)
        
        logger.info(f"Loaded FOI map for {date_str}")
        return foi, summary
    
    def get_processing_status(self) -> Dict:
        """
        Get processing status and available outputs
        
        Returns:
            Dictionary of processing status
        """
        status = {
            'foi_maps': [],
            'cps_maps': [],
            'total_foi_maps': 0,
            'total_cps_maps': 0
        }
        
        # Check FOI maps
        if self.foi_dir.exists():
            for foi_date_dir in self.foi_dir.iterdir():
                if foi_date_dir.is_dir():
                    foi_info = {
                        'date': foi_date_dir.name,
                        'files': list(foi_date_dir.glob('*')),
                        'size_mb': sum(f.stat().st_size for f in foi_date_dir.glob('*')) / (1024*1024)
                    }
                    status['foi_maps'].append(foi_info)
                    status['total_foi_maps'] += 1
        
        # Check CPS maps
        if self.cps_dir.exists():
            for cps_date_dir in self.cps_dir.iterdir():
                if cps_date_dir.is_dir():
                    cps_info = {
                        'date': cps_date_dir.name,
                        'files': list(cps_date_dir.glob('*')),
                        'size_mb': sum(f.stat().st_size for f in cps_date_dir.glob('*')) / (1024*1024)
                    }
                    status['cps_maps'].append(cps_info)
                    status['total_cps_maps'] += 1
        
        return status


def compute_foi_map(start_date: str, end_date: str, 
                   bbox: Optional[List[float]] = None,
                   config_path: str = "config.yaml",
                   save_output: bool = True) -> Tuple[xr.DataArray, Dict]:
    """
    Convenience function to compute FOI map
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
        config_path: Path to configuration file
        
    Returns:
        Tuple of (FOI DataArray, model_summary)
    """
    predictor = SharkHotspotPredictor(config_path)
    return predictor.compute_foi_map(start_date, end_date, bbox=bbox, save_output=save_output)


def compute_cps_map(foi: xr.DataArray, 
                   fishing_effort: Optional[xr.DataArray] = None,
                   start_date: str = None, end_date: str = None,
                   config_path: str = "config.yaml") -> Tuple[xr.DataArray, Dict]:
    """
    Convenience function to compute CPS map
    
    Args:
        foi: Foraging Opportunity Index
        fishing_effort: Fishing effort data (optional)
        start_date: Start date for output naming
        end_date: End date for output naming
        config_path: Path to configuration file
        
    Returns:
        Tuple of (CPS DataArray, summary)
    """
    predictor = SharkHotspotPredictor(config_path)
    return predictor.compute_cps_map(foi, fishing_effort, start_date, end_date)


if __name__ == "__main__":
    # Test the predictor
    bbox = [94.0, -11.0, 142.0, 6.0]
    start_date = "2025-03-01"
    end_date = "2025-03-14"
    
    # Compute FOI map
    foi, summary = compute_foi_map(start_date, end_date, bbox)
    
    print("\nFOI Computation Results:")
    print(f"FOI range: [{foi.min().values:.4f}, {foi.max().values:.4f}]")
    print(f"FOI mean: {foi.mean().values:.4f}")
    print(f"Spatial coverage: {summary['spatial_coverage']['coverage_percentage']:.2f}%")
    
    # Compute CPS map
    cps, cps_summary = compute_cps_map(foi, start_date=start_date, end_date=end_date)
    
    print("\nCPS Computation Results:")
    print(f"CPS range: [{cps.min().values:.4f}, {cps.max().values:.4f}]")
    print(f"CPS mean: {cps.mean().values:.4f}")
    
    # Check processing status
    predictor = SharkHotspotPredictor()
    status = predictor.get_processing_status()
    print(f"\nProcessing Status:")
    print(f"FOI maps: {status['total_foi_maps']}")
    print(f"CPS maps: {status['total_cps_maps']}")
