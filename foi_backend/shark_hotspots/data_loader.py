"""
Data Loading Module for NASA Satellite Datasets
Handles ingestion of PACE, MODIS-Aqua, and SWOT data with temporal alignment
"""

import os
import logging
import requests
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteDataLoader:
    """
    Loads and manages NASA satellite datasets for shark foraging analysis
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data loader with configuration"""
        self.config = self._load_config(config_path)
        self.data_dir = Path("data/inputs")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
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
            'data_sources': {
                'pace': {'base_url': 'https://oceandata.sci.gsfc.nasa.gov/api/file_search/'},
                'modis_aqua': {'base_url': 'https://oceandata.sci.gsfc.nasa.gov/ob/getfile/'},
                'swot': {'base_url': 'https://podaac.jpl.nasa.gov/dataset/SWOT_L2_LR_SSH_1.0'}
            }
        }
    
    def load_satellite_data(self, start_date: str, end_date: str, 
                          bbox: List[float]) -> Dict[str, xr.Dataset]:
        """
        Load all required satellite datasets for the specified time window and region
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
            
        Returns:
            Dictionary containing loaded datasets
        """
        logger.info(f"Loading satellite data from {start_date} to {end_date}")
        logger.info(f"Region: {bbox}")
        
        datasets = {}
        
        try:
            # Load PACE Ocean Color data (chlorophyll-a)
            datasets['chl'] = self._load_pace_data(start_date, end_date, bbox)
            
            # Load MODIS-Aqua SST data
            datasets['sst'] = self._load_modis_sst(start_date, end_date, bbox)
            
            # Load SWOT SSH data
            datasets['ssh'] = self._load_swot_ssh(start_date, end_date, bbox)
            
            # Load MODIS Kd490 data
            datasets['kd490'] = self._load_modis_kd490(start_date, end_date, bbox)
            
            # Validate temporal overlap
            self._validate_temporal_overlap(datasets, start_date, end_date)
            
            logger.info("Successfully loaded all satellite datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"Error loading satellite data: {e}")
            raise
    
    def _load_pace_data(self, start_date: str, end_date: str, 
                       bbox: List[float]) -> xr.Dataset:
        """Load PACE Ocean Color data"""
        logger.info("Loading PACE Ocean Color data...")
        
        # For demo purposes, create synthetic PACE data
        # In production, this would query NASA OB.DAAC API
        return self._create_synthetic_dataset(
            'chl', start_date, end_date, bbox, 
            var_name='chlor_a', units='mg/m^3'
        )
    
    def _load_modis_sst(self, start_date: str, end_date: str, 
                       bbox: List[float]) -> xr.Dataset:
        """Load MODIS-Aqua SST data"""
        logger.info("Loading MODIS-Aqua SST data...")
        
        # For demo purposes, create synthetic MODIS SST data
        return self._create_synthetic_dataset(
            'sst', start_date, end_date, bbox,
            var_name='sst', units='degrees_C'
        )
    
    def _load_swot_ssh(self, start_date: str, end_date: str, 
                      bbox: List[float]) -> xr.Dataset:
        """Load SWOT SSH data"""
        logger.info("Loading SWOT SSH data...")
        
        # For demo purposes, create synthetic SWOT SSH data
        return self._create_synthetic_dataset(
            'ssh', start_date, end_date, bbox,
            var_name='ssh', units='m'
        )
    
    def _load_modis_kd490(self, start_date: str, end_date: str, 
                         bbox: List[float]) -> xr.Dataset:
        """Load MODIS Kd490 data"""
        logger.info("Loading MODIS Kd490 data...")
        
        # For demo purposes, create synthetic Kd490 data
        return self._create_synthetic_dataset(
            'kd490', start_date, end_date, bbox,
            var_name='kd490', units='1/m'
        )
    
    def _create_synthetic_dataset(self, data_type: str, start_date: str, 
                                end_date: str, bbox: List[float],
                                var_name: str, units: str) -> xr.Dataset:
        """
        Create synthetic satellite data for demonstration purposes
        In production, this would be replaced with actual data loading
        """
        lon_min, lat_min, lon_max, lat_max = bbox
        
        # Create coordinate arrays
        lon = np.arange(lon_min, lon_max + 0.1, 0.1)
        lat = np.arange(lat_min, lat_max + 0.1, 0.1)
        
        # Create time array
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        time = pd.date_range(start_dt, end_dt, freq='D')
        
        # Create synthetic data based on data type
        if data_type == 'chl':
            # Chlorophyll-a: higher near coasts, lower in open ocean
            data = self._generate_chl_pattern(lon, lat, time)
        elif data_type == 'sst':
            # SST: warmer in tropics, cooler in higher latitudes
            data = self._generate_sst_pattern(lon, lat, time)
        elif data_type == 'ssh':
            # SSH: dynamic oceanographic features
            data = self._generate_ssh_pattern(lon, lat, time)
        elif data_type == 'kd490':
            # Kd490: higher in turbid waters
            data = self._generate_kd490_pattern(lon, lat, time)
        else:
            data = np.random.rand(len(time), len(lat), len(lon))
        
        # Create xarray Dataset
        ds = xr.Dataset(
            {var_name: (['time', 'lat', 'lon'], data)},
            coords={
                'time': time,
                'lat': lat,
                'lon': lon
            }
        )
        
        # Add attributes
        ds[var_name].attrs = {
            'long_name': f'{data_type.upper()} data',
            'units': units,
            'source': 'Synthetic data for demonstration'
        }
        
        ds.attrs = {
            'title': f'Synthetic {data_type.upper()} data',
            'description': f'Generated synthetic data for {data_type}',
            'creation_date': datetime.now().isoformat()
        }
        
        return ds
    
    def _generate_chl_pattern(self, lon: np.ndarray, lat: np.ndarray, 
                            time: pd.DatetimeIndex) -> np.ndarray:
        """Generate realistic chlorophyll-a pattern"""
        data = np.zeros((len(time), len(lat), len(lon)))
        
        for t in range(len(time)):
            for i, la in enumerate(lat):
                for j, lo in enumerate(lon):
                    # Higher chlorophyll near coasts and in upwelling regions
                    coast_dist = min(abs(lo - 94), abs(lo - 142), 
                                   abs(la - (-11)), abs(la - 6))
                    coast_effect = np.exp(-coast_dist / 5.0)
                    
                    # Seasonal variation
                    seasonal = 0.5 + 0.3 * np.sin(2 * np.pi * t / 365)
                    
                    # Random noise
                    noise = np.random.normal(0, 0.1)
                    
                    data[t, i, j] = 0.1 + coast_effect * seasonal + noise
                    data[t, i, j] = max(0.01, data[t, i, j])  # Ensure positive
        
        return data
    
    def _generate_sst_pattern(self, lon: np.ndarray, lat: np.ndarray, 
                            time: pd.DatetimeIndex) -> np.ndarray:
        """Generate realistic SST pattern"""
        data = np.zeros((len(time), len(lat), len(lon)))
        
        for t in range(len(time)):
            for i, la in enumerate(lat):
                for j, lo in enumerate(lon):
                    # Base temperature gradient
                    base_temp = 30 - 0.5 * abs(la)
                    
                    # Seasonal variation
                    seasonal = 2 * np.sin(2 * np.pi * t / 365)
                    
                    # Random noise
                    noise = np.random.normal(0, 0.5)
                    
                    data[t, i, j] = base_temp + seasonal + noise
        
        return data
    
    def _generate_ssh_pattern(self, lon: np.ndarray, lat: np.ndarray, 
                            time: pd.DatetimeIndex) -> np.ndarray:
        """Generate realistic SSH pattern"""
        data = np.zeros((len(time), len(lat), len(lon)))
        
        for t in range(len(time)):
            for i, la in enumerate(lat):
                for j, lo in enumerate(lon):
                    # Oceanographic features (eddies, currents)
                    eddy1 = 0.1 * np.sin(2 * np.pi * (lo - 100) / 20) * \
                           np.cos(2 * np.pi * (la - (-5)) / 10)
                    eddy2 = 0.05 * np.sin(2 * np.pi * (lo - 120) / 15) * \
                           np.cos(2 * np.pi * (la - 2) / 8)
                    
                    # Random noise
                    noise = np.random.normal(0, 0.02)
                    
                    data[t, i, j] = eddy1 + eddy2 + noise
        
        return data
    
    def _generate_kd490_pattern(self, lon: np.ndarray, lat: np.ndarray, 
                              time: pd.DatetimeIndex) -> np.ndarray:
        """Generate realistic Kd490 pattern"""
        data = np.zeros((len(time), len(lat), len(lon)))
        
        for t in range(len(time)):
            for i, la in enumerate(lat):
                for j, lo in enumerate(lon):
                    # Higher attenuation near coasts
                    coast_dist = min(abs(lo - 94), abs(lo - 142), 
                                   abs(la - (-11)), abs(la - 6))
                    coast_effect = np.exp(-coast_dist / 3.0)
                    
                    # Random noise
                    noise = np.random.normal(0, 0.05)
                    
                    data[t, i, j] = 0.05 + coast_effect * 0.1 + noise
                    data[t, i, j] = max(0.01, data[t, i, j])  # Ensure positive
        
        return data
    
    def _validate_temporal_overlap(self, datasets: Dict[str, xr.Dataset], 
                                 start_date: str, end_date: str):
        """
        Validate that all datasets cover the target time window
        
        Args:
            datasets: Dictionary of loaded datasets
            start_date: Target start date
            end_date: Target end date
        """
        logger.info("Validating temporal overlap across datasets...")
        
        start_dt = np.datetime64(start_date)
        end_dt = np.datetime64(end_date)
        
        for name, ds in datasets.items():
            if 'time' not in ds.dims:
                logger.warning(f"{name} dataset has no time dimension")
                continue
            
            ds_start = ds.time.min().values
            ds_end = ds.time.max().values
            
            # Check if dataset fully covers the target window
            if not ((ds_start <= start_dt) and (ds_end >= end_dt)):
                error_msg = (f"{name} dataset does not fully cover the target window "
                           f"{start_date}→{end_date}. Dataset covers {ds_start}→{ds_end}")
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"{name}: {ds_start} → {ds_end} ✓")
        
        logger.info("Temporal overlap validation passed")
    
    def save_dataset(self, dataset: xr.Dataset, filename: str) -> str:
        """Save dataset to NetCDF file"""
        filepath = self.data_dir / filename
        dataset.to_netcdf(filepath)
        logger.info(f"Saved dataset to {filepath}")
        return str(filepath)
    
    def load_dataset(self, filename: str) -> xr.Dataset:
        """Load dataset from NetCDF file"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        dataset = xr.open_dataset(filepath)
        logger.info(f"Loaded dataset from {filepath}")
        return dataset


def load_satellite_data(bbox: List[float], start_date: str, end_date: str) -> Dict[str, xr.Dataset]:
    """
    Convenience function to load satellite data
    
    Args:
        bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Dictionary containing loaded datasets
    """
    loader = SatelliteDataLoader()
    return loader.load_satellite_data(start_date, end_date, bbox)


if __name__ == "__main__":
    # Test the data loader
    bbox = [94.0, -11.0, 142.0, 6.0]
    start_date = "2025-03-01"
    end_date = "2025-03-14"
    
    datasets = load_satellite_data(bbox, start_date, end_date)
    
    for name, ds in datasets.items():
        print(f"\n{name.upper()} Dataset:")
        print(ds)
        print(f"Shape: {ds[list(ds.data_vars)[0]].shape}")
