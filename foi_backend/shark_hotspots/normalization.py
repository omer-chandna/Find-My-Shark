"""
Normalization Module
Implements robust 5-95 percentile scaling for satellite data normalization
"""

import logging
import xarray as xr
import numpy as np
from typing import Dict, Union, Optional, Tuple
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Normalizes satellite data using robust percentile-based scaling
    """
    
    def __init__(self, percentile_range: Tuple[float, float] = (5.0, 95.0)):
        """
        Initialize normalizer with percentile range
        
        Args:
            percentile_range: Tuple of (lower, upper) percentiles for scaling
        """
        self.percentile_range = percentile_range
        self.normalization_stats = {}
        
    def normalize_datasets(self, datasets: Dict[str, Union[xr.Dataset, xr.DataArray]]) -> Dict[str, Union[xr.Dataset, xr.DataArray]]:
        """
        Normalize multiple datasets using robust percentile scaling
        
        Args:
            datasets: Dictionary of datasets/arrays to normalize
            
        Returns:
            Dictionary of normalized datasets/arrays
        """
        logger.info("Starting data normalization...")
        
        normalized = {}
        
        for name, data in datasets.items():
            logger.info(f"Normalizing {name}...")
            
            if isinstance(data, xr.Dataset):
                normalized[name] = self._normalize_dataset(data, name)
            elif isinstance(data, xr.DataArray):
                normalized[name] = self._normalize_dataarray(data, name)
            else:
                logger.warning(f"Unsupported data type for {name}: {type(data)}")
                normalized[name] = data
        
        logger.info("Data normalization completed")
        return normalized
    
    def _normalize_dataset(self, dataset: xr.Dataset, name: str) -> xr.Dataset:
        """
        Normalize all variables in a dataset
        
        Args:
            dataset: Input dataset
            name: Dataset name for logging
            
        Returns:
            Normalized dataset
        """
        normalized_vars = {}
        
        for var_name in dataset.data_vars:
            var_data = dataset[var_name]
            normalized_var = self._normalize_dataarray(var_data, f"{name}_{var_name}")
            normalized_vars[var_name] = normalized_var
        
        # Create normalized dataset
        normalized_ds = xr.Dataset(normalized_vars)
        
        # Copy coordinates
        normalized_ds = normalized_ds.assign_coords(dataset.coords)
        
        # Copy attributes
        normalized_ds.attrs = dataset.attrs.copy()
        normalized_ds.attrs.update({
            'normalization_method': 'robust_percentile_scaling',
            'percentile_range': self.percentile_range,
            'normalization_date': np.datetime64('now').astype(str)
        })
        
        return normalized_ds
    
    def _normalize_dataarray(self, dataarray: xr.DataArray, name: str) -> xr.DataArray:
        """
        Normalize a single DataArray using robust percentile scaling
        
        Args:
            dataarray: Input DataArray
            name: DataArray name for logging
            
        Returns:
            Normalized DataArray
        """
        # Compute percentiles
        p_lower, p_upper = self._compute_percentiles(dataarray)
        
        # Store normalization statistics
        self.normalization_stats[name] = {
            'p_lower': p_lower,
            'p_upper': p_upper,
            'original_min': float(dataarray.min().values),
            'original_max': float(dataarray.max().values),
            'original_mean': float(dataarray.mean().values),
            'original_std': float(dataarray.std().values)
        }
        
        # Apply robust normalization
        normalized_data = self._apply_robust_normalization(dataarray, p_lower, p_upper)
        
        # Create normalized DataArray
        normalized_da = xr.DataArray(
            normalized_data,
            coords=dataarray.coords,
            dims=dataarray.dims,
            name=dataarray.name,
            attrs=dataarray.attrs.copy()
        )
        
        # Update attributes
        normalized_da.attrs.update({
            'normalization_method': 'robust_percentile_scaling',
            'percentile_range': self.percentile_range,
            'p_lower': float(p_lower),
            'p_upper': float(p_upper),
            'normalized_min': float(normalized_data.min()),
            'normalized_max': float(normalized_data.max())
        })
        
        logger.info(f"Normalized {name}: [{p_lower:.4f}, {p_upper:.4f}] -> [0, 1]")
        
        return normalized_da
    
    def _compute_percentiles(self, dataarray: xr.DataArray) -> Tuple[float, float]:
        """
        Compute robust percentiles for normalization
        
        Args:
            dataarray: Input DataArray
            
        Returns:
            Tuple of (lower_percentile, upper_percentile)
        """
        # Flatten data and remove NaN values
        flat_data = dataarray.values.flatten()
        valid_data = flat_data[~np.isnan(flat_data)]
        
        if len(valid_data) == 0:
            logger.warning("No valid data found for percentile computation")
            return 0.0, 1.0
        
        # Compute percentiles
        p_lower = np.percentile(valid_data, self.percentile_range[0])
        p_upper = np.percentile(valid_data, self.percentile_range[1])
        
        # Ensure p_upper > p_lower
        if p_upper <= p_lower:
            logger.warning(f"Upper percentile ({p_upper}) <= lower percentile ({p_lower}), adjusting...")
            p_upper = p_lower + 1e-6
        
        return float(p_lower), float(p_upper)
    
    def _apply_robust_normalization(self, dataarray: xr.DataArray, 
                                  p_lower: float, p_upper: float) -> np.ndarray:
        """
        Apply robust normalization: X~ = clip((X - P5) / (P95 - P5), 0, 1)
        
        Args:
            dataarray: Input DataArray
            p_lower: Lower percentile value
            p_upper: Upper percentile value
            
        Returns:
            Normalized data array
        """
        # Compute normalized values
        normalized = (dataarray - p_lower) / (p_upper - p_lower)
        
        # Clip to [0, 1] range
        normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    def denormalize_dataarray(self, normalized_dataarray: xr.DataArray, 
                            original_name: str) -> xr.DataArray:
        """
        Denormalize a normalized DataArray back to original scale
        
        Args:
            normalized_dataarray: Normalized DataArray
            original_name: Original name used for normalization
            
        Returns:
            Denormalized DataArray
        """
        if original_name not in self.normalization_stats:
            logger.error(f"No normalization statistics found for {original_name}")
            return normalized_dataarray
        
        stats = self.normalization_stats[original_name]
        p_lower = stats['p_lower']
        p_upper = stats['p_upper']
        
        # Reverse normalization: X = X~ * (P95 - P5) + P5
        denormalized_data = normalized_dataarray * (p_upper - p_lower) + p_lower
        
        # Create denormalized DataArray
        denormalized_da = xr.DataArray(
            denormalized_data,
            coords=normalized_dataarray.coords,
            dims=normalized_dataarray.dims,
            name=normalized_dataarray.name,
            attrs=normalized_dataarray.attrs.copy()
        )
        
        # Update attributes
        denormalized_da.attrs.update({
            'denormalized': True,
            'original_p_lower': p_lower,
            'original_p_upper': p_upper
        })
        
        return denormalized_da
    
    def get_normalization_stats(self) -> Dict:
        """
        Get normalization statistics for all processed datasets
        
        Returns:
            Dictionary of normalization statistics
        """
        return self.normalization_stats.copy()
    
    def save_normalization_stats(self, filepath: str):
        """
        Save normalization statistics to file
        
        Args:
            filepath: Path to save statistics
        """
        import json
        
        # Convert numpy types to Python types for JSON serialization
        stats_json = {}
        for name, stats in self.normalization_stats.items():
            stats_json[name] = {k: float(v) for k, v in stats.items()}
        
        with open(filepath, 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        logger.info(f"Normalization statistics saved to {filepath}")
    
    def load_normalization_stats(self, filepath: str):
        """
        Load normalization statistics from file
        
        Args:
            filepath: Path to load statistics from
        """
        import json
        
        with open(filepath, 'r') as f:
            stats_json = json.load(f)
        
        self.normalization_stats = stats_json
        logger.info(f"Normalization statistics loaded from {filepath}")


class AdaptiveNormalizer(DataNormalizer):
    """
    Adaptive normalizer that adjusts percentile ranges based on data characteristics
    """
    
    def __init__(self, percentile_range: Tuple[float, float] = (5.0, 95.0),
                 adaptive_threshold: float = 0.1):
        """
        Initialize adaptive normalizer
        
        Args:
            percentile_range: Default percentile range
            adaptive_threshold: Threshold for adaptive adjustment
        """
        super().__init__(percentile_range)
        self.adaptive_threshold = adaptive_threshold
    
    def _compute_percentiles(self, dataarray: xr.DataArray) -> Tuple[float, float]:
        """
        Compute adaptive percentiles based on data distribution
        
        Args:
            dataarray: Input DataArray
            
        Returns:
            Tuple of (lower_percentile, upper_percentile)
        """
        # Flatten data and remove NaN values
        flat_data = dataarray.values.flatten()
        valid_data = flat_data[~np.isnan(flat_data)]
        
        if len(valid_data) == 0:
            logger.warning("No valid data found for percentile computation")
            return 0.0, 1.0
        
        # Compute data statistics
        data_mean = np.mean(valid_data)
        data_std = np.std(valid_data)
        data_skew = self._compute_skewness(valid_data)
        
        # Adjust percentiles based on data characteristics
        if abs(data_skew) > self.adaptive_threshold:
            # Skewed data - use more extreme percentiles
            if data_skew > 0:  # Right-skewed
                p_lower = np.percentile(valid_data, 2.0)
                p_upper = np.percentile(valid_data, 98.0)
            else:  # Left-skewed
                p_lower = np.percentile(valid_data, 2.0)
                p_upper = np.percentile(valid_data, 98.0)
        else:
            # Normal distribution - use default percentiles
            p_lower = np.percentile(valid_data, self.percentile_range[0])
            p_upper = np.percentile(valid_data, self.percentile_range[1])
        
        # Ensure p_upper > p_lower
        if p_upper <= p_lower:
            p_upper = p_lower + 1e-6
        
        logger.info(f"Adaptive percentiles for skewed data (skew={data_skew:.3f}): [{p_lower:.4f}, {p_upper:.4f}]")
        
        return float(p_lower), float(p_upper)
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """
        Compute skewness of data distribution
        
        Args:
            data: Input data array
            
        Returns:
            Skewness value
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness


def normalize_datasets(datasets: Dict[str, Union[xr.Dataset, xr.DataArray]], 
                       percentile_range: Tuple[float, float] = (5.0, 95.0),
                       adaptive: bool = False) -> Tuple[Dict[str, Union[xr.Dataset, xr.DataArray]], DataNormalizer]:
    """
    Convenience function to normalize datasets
    
    Args:
        datasets: Dictionary of datasets/arrays to normalize
        percentile_range: Percentile range for normalization
        adaptive: Whether to use adaptive normalization
        
    Returns:
        Tuple of (normalized_datasets, normalizer_instance)
    """
    if adaptive:
        normalizer = AdaptiveNormalizer(percentile_range)
    else:
        normalizer = DataNormalizer(percentile_range)
    
    normalized = normalizer.normalize_datasets(datasets)
    
    return normalized, normalizer


if __name__ == "__main__":
    # Test the normalizer
    from foi_backend.shark_hotspots.data_loader import load_satellite_data
    from foi_backend.shark_hotspots.harmonize import harmonize_datasets
    from foi_backend.shark_hotspots.derived_fields import compute_derived_fields
    
    # Load and process test data
    bbox = [94.0, -11.0, 142.0, 6.0]
    start_date = "2025-03-01"
    end_date = "2025-03-14"
    
    datasets = load_satellite_data(bbox, start_date, end_date)
    harmonized = harmonize_datasets(datasets, bbox, start_date, end_date)
    derived = compute_derived_fields(harmonized)
    
    # Test normalization
    normalized, normalizer = normalize_datasets(derived)
    
    # Print results
    print("\nNormalization Results:")
    for name, data in normalized.items():
        if isinstance(data, xr.DataArray):
            print(f"{name}: [{data.min().values:.4f}, {data.max().values:.4f}]")
        else:
            for var_name in data.data_vars:
                var_data = data[var_name]
                print(f"{name}_{var_name}: [{var_data.min().values:.4f}, {var_data.max().values:.4f}]")
    
    # Print normalization statistics
    print("\nNormalization Statistics:")
    stats = normalizer.get_normalization_stats()
    for name, stat in stats.items():
        print(f"{name}: P5={stat['p_lower']:.4f}, P95={stat['p_upper']:.4f}")
