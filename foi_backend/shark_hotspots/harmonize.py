"""
Data Harmonization Module
Handles regridding and temporal alignment of multi-satellite datasets
"""

import logging
import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import griddata
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataHarmonizer:
    """
    Harmonizes multi-satellite datasets for temporal and spatial alignment
    """
    
    def __init__(self, target_resolution: float = 0.1):
        """
        Initialize harmonizer with target grid resolution
        
        Args:
            target_resolution: Target grid resolution in degrees
        """
        self.target_resolution = target_resolution
        self.target_grid = None
        
    def harmonize_datasets(self, datasets: Dict[str, xr.Dataset], 
                          bbox: List[float], 
                          start_date: str, 
                          end_date: str,
                          composite_days: int = 7,
                          composite_mode: str = "mean") -> Dict[str, xr.Dataset]:
        """
        Harmonize multiple satellite datasets to common grid and time
        
        Args:
            datasets: Dictionary of datasets to harmonize
            bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
            start_date: Start date for temporal alignment
            end_date: End date for temporal alignment
            composite_days: Number of days for temporal compositing
            composite_mode: Temporal aggregation mode (mean, median, rolling)
            
        Returns:
            Dictionary of harmonized datasets
        """
        logger.info("Starting dataset harmonization...")
        
        # Synchronize time axes first
        datasets = self.synchronize_time_axes(datasets)
        
        # Create target grid
        self._create_target_grid(bbox)
        
        # Harmonize each dataset
        harmonized = {}
        
        for name, dataset in datasets.items():
            logger.info(f"Harmonizing {name} dataset...")
            
            # Temporal alignment
            temp_aligned = self._temporal_alignment(dataset, start_date, end_date, composite_days, composite_mode)
            
            # Spatial regridding
            spatial_aligned = self._spatial_regridding(temp_aligned, name)
            
            harmonized[name] = spatial_aligned
            
        logger.info("Dataset harmonization completed")
        return harmonized
    
    def _create_target_grid(self, bbox: List[float]):
        """Create target grid for harmonization"""
        lon_min, lat_min, lon_max, lat_max = bbox
        
        # Create regular grid
        lon = np.arange(lon_min, lon_max + self.target_resolution, self.target_resolution)
        lat = np.arange(lat_min, lat_max + self.target_resolution, self.target_resolution)
        
        self.target_grid = {
            'lon': lon,
            'lat': lat,
            'bbox': bbox
        }
        
        logger.info(f"Created target grid: {len(lon)} x {len(lat)} points")
        logger.info(f"Resolution: {self.target_resolution}°")
    
    def synchronize_time_axes(self, datasets: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        """
        Ensure all input datasets share identical time indices using intersection alignment
        
        Args:
            datasets: Dictionary of datasets to synchronize
            
        Returns:
            Dictionary of synchronized datasets
        """
        logger.info("Synchronizing time axes across datasets...")
        
        # Extract datasets that have time dimensions
        datasets_with_time = {}
        datasets_without_time = {}
        
        for name, ds in datasets.items():
            if 'time' in ds.dims:
                datasets_with_time[name] = ds
            else:
                datasets_without_time[name] = ds
                logger.info(f"{name} has no time dimension, skipping synchronization")
        
        if not datasets_with_time:
            logger.warning("No datasets with time dimensions found")
            return datasets
        
        # Align datasets using intersection (inner join)
        aligned_datasets = xr.align(*datasets_with_time.values(), join="inner")
        
        # Create synchronized dataset dictionary
        synchronized = {}
        dataset_names = list(datasets_with_time.keys())
        
        for i, aligned_ds in enumerate(aligned_datasets):
            name = dataset_names[i]
            synchronized[name] = aligned_ds
            logger.info(f"{name}: synchronized to {len(aligned_ds.time)} time points")
        
        # Add datasets without time dimensions back
        synchronized.update(datasets_without_time)
        
        logger.info("Time axis synchronization completed")
        return synchronized
    
    def _temporal_alignment(self, dataset: xr.Dataset, 
                          start_date: str, end_date: str, 
                          composite_days: int, composite_mode: str = "mean") -> xr.Dataset:
        """
        Perform temporal alignment and compositing
        
        Args:
            dataset: Input dataset
            start_date: Start date
            end_date: End date
            composite_days: Days for compositing
            composite_mode: Temporal aggregation mode (mean, median, rolling)
            
        Returns:
            Temporally aligned dataset
        """
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Select time slice
        dataset_slice = dataset.sel(time=slice(start_dt, end_dt))
        
        # Perform temporal compositing based on mode
        if composite_mode == "rolling":
            # Continuous rolling window
            dataset_composite = self._rolling_composite(dataset_slice, composite_days)
        elif composite_mode == "median":
            # Median compositing for outlier robustness
            dataset_composite = self._median_composite(dataset_slice, composite_days)
        else:  # default: "mean"
            # Mean compositing
            dataset_composite = self._mean_composite(dataset_slice, composite_days)
        
        # Add temporal metadata
        dataset_composite.attrs.update({
            'temporal_composite_days': composite_days,
            'temporal_composite_mode': composite_mode,
            'start_date': start_date,
            'end_date': end_date,
            'harmonization_date': pd.Timestamp.now().isoformat()
        })
        
        return dataset_composite
    
    def _rolling_composite(self, dataset: xr.Dataset, window_days: int) -> xr.Dataset:
        """
        Create rolling composite of dataset
        
        Args:
            dataset: Input dataset
            window_days: Rolling window size in days
            
        Returns:
            Composited dataset
        """
        # Calculate number of windows
        time_length = len(dataset.time)
        n_windows = max(1, time_length // window_days)
        
        # Create composite by averaging over windows
        composite_data = {}
        
        for var_name in dataset.data_vars:
            var_data = dataset[var_name]
            
            # Reshape for windowed averaging
            if len(var_data.dims) == 3:  # time, lat, lon
                reshaped = var_data.values.reshape(n_windows, window_days, 
                                                 var_data.shape[1], var_data.shape[2])
                composite_data[var_name] = np.nanmean(reshaped, axis=1)
            elif len(var_data.dims) == 2:  # lat, lon
                composite_data[var_name] = var_data.values
            else:
                composite_data[var_name] = var_data.values
        
        # Create new dataset with composite data
        composite_ds = xr.Dataset()
        
        for var_name, data in composite_data.items():
            if len(dataset[var_name].dims) == 3:
                composite_ds[var_name] = (['time', 'lat', 'lon'], data)
            elif len(dataset[var_name].dims) == 2:
                composite_ds[var_name] = (['lat', 'lon'], data)
            else:
                composite_ds[var_name] = dataset[var_name]
        
        # Set coordinates
        if 'time' in dataset.dims:
            # Create new time coordinate for composite
            time_values = dataset.time.values[::window_days][:n_windows]
            composite_ds = composite_ds.assign_coords(time=time_values)
        
        composite_ds = composite_ds.assign_coords({
            'lat': dataset.lat,
            'lon': dataset.lon
        })
        
        # Copy attributes
        composite_ds.attrs = dataset.attrs.copy()
        
        return composite_ds
    
    def _mean_composite(self, dataset: xr.Dataset, composite_days: int) -> xr.Dataset:
        """
        Create mean composite of dataset
        
        Args:
            dataset: Input dataset
            composite_days: Number of days for compositing
            
        Returns:
            Mean composited dataset
        """
        if composite_days > 1:
            # Calculate number of windows
            time_length = len(dataset.time)
            n_windows = max(1, time_length // composite_days)
            
            # Create composite by averaging over windows
            composite_data = {}
            
            for var_name in dataset.data_vars:
                var_data = dataset[var_name]
                
                # Reshape for windowed averaging
                if len(var_data.dims) == 3:  # time, lat, lon
                    reshaped = var_data.values.reshape(n_windows, composite_days, 
                                                     var_data.shape[1], var_data.shape[2])
                    composite_data[var_name] = np.nanmean(reshaped, axis=1)
                elif len(var_data.dims) == 2:  # lat, lon
                    composite_data[var_name] = var_data.values
                else:
                    composite_data[var_name] = var_data.values
            
            # Create new dataset with composite data
            composite_ds = xr.Dataset()
            
            for var_name, data in composite_data.items():
                if len(dataset[var_name].dims) == 3:
                    composite_ds[var_name] = (['time', 'lat', 'lon'], data)
                elif len(dataset[var_name].dims) == 2:
                    composite_ds[var_name] = (['lat', 'lon'], data)
                else:
                    composite_ds[var_name] = dataset[var_name]
            
            # Set coordinates
            if 'time' in dataset.dims:
                # Create new time coordinate for composite
                time_values = dataset.time.values[::composite_days][:n_windows]
                composite_ds = composite_ds.assign_coords(time=time_values)
            
            composite_ds = composite_ds.assign_coords({
                'lat': dataset.lat,
                'lon': dataset.lon
            })
            
            # Copy attributes
            composite_ds.attrs = dataset.attrs.copy()
            
            return composite_ds
        else:
            # Simple time mean
            return dataset.mean('time', keep_attrs=True)
    
    def _median_composite(self, dataset: xr.Dataset, composite_days: int) -> xr.Dataset:
        """
        Create median composite of dataset (outlier-robust)
        
        Args:
            dataset: Input dataset
            composite_days: Number of days for compositing
            
        Returns:
            Median composited dataset
        """
        if composite_days > 1:
            # Calculate number of windows
            time_length = len(dataset.time)
            n_windows = max(1, time_length // composite_days)
            
            # Create composite by taking median over windows
            composite_data = {}
            
            for var_name in dataset.data_vars:
                var_data = dataset[var_name]
                
                # Reshape for windowed median
                if len(var_data.dims) == 3:  # time, lat, lon
                    reshaped = var_data.values.reshape(n_windows, composite_days, 
                                                     var_data.shape[1], var_data.shape[2])
                    composite_data[var_name] = np.nanmedian(reshaped, axis=1)
                elif len(var_data.dims) == 2:  # lat, lon
                    composite_data[var_name] = var_data.values
                else:
                    composite_data[var_name] = var_data.values
            
            # Create new dataset with composite data
            composite_ds = xr.Dataset()
            
            for var_name, data in composite_data.items():
                if len(dataset[var_name].dims) == 3:
                    composite_ds[var_name] = (['time', 'lat', 'lon'], data)
                elif len(dataset[var_name].dims) == 2:
                    composite_ds[var_name] = (['lat', 'lon'], data)
                else:
                    composite_ds[var_name] = dataset[var_name]
            
            # Set coordinates
            if 'time' in dataset.dims:
                # Create new time coordinate for composite
                time_values = dataset.time.values[::composite_days][:n_windows]
                composite_ds = composite_ds.assign_coords(time=time_values)
            
            composite_ds = composite_ds.assign_coords({
                'lat': dataset.lat,
                'lon': dataset.lon
            })
            
            # Copy attributes
            composite_ds.attrs = dataset.attrs.copy()
            
            return composite_ds
        else:
            # Simple time median
            return dataset.median('time', keep_attrs=True)
    
    def _spatial_regridding(self, dataset: xr.Dataset, dataset_name: str) -> xr.Dataset:
        """
        Regrid dataset to target grid
        
        Args:
            dataset: Input dataset
            dataset_name: Name of dataset for logging
            
        Returns:
            Regridded dataset
        """
        logger.info(f"Regridding {dataset_name} to {self.target_resolution}° resolution")
        
        # Create target coordinates
        target_lon = self.target_grid['lon']
        target_lat = self.target_grid['lat']
        
        # Create meshgrid for interpolation
        target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat)
        
        regridded_data = {}
        
        for var_name in dataset.data_vars:
            var_data = dataset[var_name]
            
            if len(var_data.dims) == 3:  # time, lat, lon
                # Handle 3D data (time, lat, lon)
                regridded_var = np.zeros((var_data.shape[0], len(target_lat), len(target_lon)))
                
                for t in range(var_data.shape[0]):
                    regridded_var[t] = self._interpolate_2d(
                        var_data.isel(time=t).values,
                        dataset.lat.values,
                        dataset.lon.values,
                        target_lat_grid,
                        target_lon_grid
                    )
                
                regridded_data[var_name] = (['time', 'lat', 'lon'], regridded_var)
                
            elif len(var_data.dims) == 2:  # lat, lon
                # Handle 2D data (lat, lon)
                regridded_var = self._interpolate_2d(
                    var_data.values,
                    dataset.lat.values,
                    dataset.lon.values,
                    target_lat_grid,
                    target_lon_grid
                )
                
                regridded_data[var_name] = (['lat', 'lon'], regridded_var)
            
            else:
                # Handle 1D or scalar data
                regridded_data[var_name] = var_data
        
        # Create regridded dataset
        regridded_ds = xr.Dataset(regridded_data)
        
        # Set coordinates
        if 'time' in dataset.dims:
            regridded_ds = regridded_ds.assign_coords(time=dataset.time)
        
        regridded_ds = regridded_ds.assign_coords({
            'lat': target_lat,
            'lon': target_lon
        })
        
        # Copy attributes
        regridded_ds.attrs = dataset.attrs.copy()
        regridded_ds.attrs.update({
            'regridded_resolution': self.target_resolution,
            'regridding_method': 'linear_interpolation'
        })
        
        return regridded_ds
    
    def _interpolate_2d(self, data: np.ndarray, source_lat: np.ndarray, 
                       source_lon: np.ndarray, target_lat: np.ndarray, 
                       target_lon: np.ndarray) -> np.ndarray:
        """
        Interpolate 2D data to target grid
        
        Args:
            data: Source data array
            source_lat: Source latitude coordinates
            source_lon: Source longitude coordinates
            target_lat: Target latitude grid
            target_lon: Target longitude grid
            
        Returns:
            Interpolated data on target grid
        """
        # Create source coordinate arrays
        source_lat_grid, source_lon_grid = np.meshgrid(source_lat, source_lon, indexing='ij')
        
        # Flatten for interpolation
        source_points = np.column_stack((source_lat_grid.ravel(), source_lon_grid.ravel()))
        source_values = data.ravel()
        
        # Remove NaN values
        valid_mask = ~np.isnan(source_values)
        source_points = source_points[valid_mask]
        source_values = source_values[valid_mask]
        
        if len(source_values) == 0:
            logger.warning("No valid data points for interpolation")
            return np.full_like(target_lat, np.nan)
        
        # Flatten target coordinates
        target_points = np.column_stack((target_lat.ravel(), target_lon.ravel()))
        
        # Perform interpolation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interpolated = griddata(
                source_points, source_values, target_points,
                method='linear', fill_value=np.nan
            )
        
        # Reshape to target grid
        return interpolated.reshape(target_lat.shape)
    
    def validate_harmonization(self, harmonized_datasets: Dict[str, xr.Dataset]) -> bool:
        """
        Validate that all datasets have consistent dimensions
        
        Args:
            harmonized_datasets: Dictionary of harmonized datasets
            
        Returns:
            True if validation passes
        """
        logger.info("Validating harmonized datasets...")
        
        # Check that all datasets have same spatial dimensions
        spatial_dims = None
        for name, dataset in harmonized_datasets.items():
            if 'lat' in dataset.dims and 'lon' in dataset.dims:
                current_dims = (len(dataset.lat), len(dataset.lon))
                if spatial_dims is None:
                    spatial_dims = current_dims
                elif spatial_dims != current_dims:
                    logger.error(f"Spatial dimensions mismatch in {name}: {current_dims} vs {spatial_dims}")
                    return False
        
        # Check coordinate consistency
        reference_lat = None
        reference_lon = None
        
        for name, dataset in harmonized_datasets.items():
            if 'lat' in dataset.coords and 'lon' in dataset.coords:
                if reference_lat is None:
                    reference_lat = dataset.lat.values
                    reference_lon = dataset.lon.values
                else:
                    if not np.allclose(dataset.lat.values, reference_lat):
                        logger.error(f"Latitude coordinates mismatch in {name}")
                        return False
                    if not np.allclose(dataset.lon.values, reference_lon):
                        logger.error(f"Longitude coordinates mismatch in {name}")
                        return False
        
        logger.info("Harmonization validation passed")
        return True


def harmonize_datasets(datasets: Dict[str, xr.Dataset], 
                      bbox: List[float], 
                      start_date: str, 
                      end_date: str,
                      composite_days: int = 7,
                      composite_mode: str = "mean",
                      resolution: float = 0.1) -> Dict[str, xr.Dataset]:
    """
    Convenience function to harmonize datasets
    
    Args:
        datasets: Dictionary of datasets to harmonize
        bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
        start_date: Start date for temporal alignment
        end_date: End date for temporal alignment
        composite_days: Number of days for temporal compositing
        composite_mode: Temporal aggregation mode (mean, median, rolling)
        resolution: Target grid resolution in degrees
        
    Returns:
        Dictionary of harmonized datasets
    """
    harmonizer = DataHarmonizer(target_resolution=resolution)
    return harmonizer.harmonize_datasets(datasets, bbox, start_date, end_date, composite_days, composite_mode)


if __name__ == "__main__":
    # Test the harmonizer
    from foi_backend.shark_hotspots.data_loader import load_satellite_data
    
    # Load test data
    bbox = [94.0, -11.0, 142.0, 6.0]
    start_date = "2025-03-01"
    end_date = "2025-03-14"
    
    datasets = load_satellite_data(bbox, start_date, end_date)
    
    # Harmonize datasets
    harmonized = harmonize_datasets(datasets, bbox, start_date, end_date)
    
    # Print results
    for name, ds in harmonized.items():
        print(f"\n{name.upper()} Harmonized Dataset:")
        print(f"Shape: {ds[list(ds.data_vars)[0]].shape}")
        print(f"Coordinates: {list(ds.coords.keys())}")
        print(f"Data variables: {list(ds.data_vars.keys())}")
