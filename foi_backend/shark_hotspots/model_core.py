"""
Model Core Module
Implements the Foraging Opportunity Index (FOI) mathematical model
"""

import logging
import xarray as xr
import numpy as np
from typing import Dict, Optional, Tuple
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FOIModel:
    """
    Foraging Opportunity Index (FOI) model for shark foraging hotspots
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FOI model with configuration
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or self._get_default_config()
        
        # Extract model parameters
        self.coefficients = self.config.get('coefficients', {})
        self.derived_params = self.config.get('derived_fields', {})
        self.species_params = self.config.get('species', {})
        
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'coefficients': {
                'b0': 0.0,   # intercept
                'b1': 0.3,   # thermal suitability
                'b2': 0.25,  # productivity proxy
                'b3': 0.25,  # twilight access
                'b4': 0.2    # front strength
            },
            'derived_fields': {
                'alpha': 0.5,      # eddy relief coefficient
                'beta1': 0.5,      # twilight access - euphotic depth weight
                'beta2': 0.5       # twilight access - EKE weight
            },
            'species': {
                'preferred_temp': 26.0,  # °C
                'temp_tolerance': 2.5    # °C
            }
        }
    
    def compute_foi(self, datasets: Dict[str, xr.Dataset], 
                   derived_fields: Dict[str, xr.DataArray],
                   normalized_fields: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Compute Foraging Opportunity Index (FOI)
        
        Args:
            datasets: Original harmonized datasets
            derived_fields: Computed derived fields
            normalized_fields: Normalized fields
            
        Returns:
            FOI DataArray
        """
        logger.info("Computing Foraging Opportunity Index (FOI)...")
        
        # Compute individual components
        thermal_suitability = self._compute_thermal_suitability(datasets['sst'])
        eddy_relief = self._compute_eddy_relief(thermal_suitability, normalized_fields['eke'])
        productivity_proxy = self._compute_productivity_proxy(datasets['chl'])
        twilight_access = self._compute_twilight_access(
            normalized_fields['euphotic_depth'], 
            normalized_fields['eke']
        )
        front_strength = self._compute_front_strength(normalized_fields['sst_gradient'])
        
        # Compute FOI using logistic regression
        foi = self._compute_logistic_foi(
            eddy_relief, productivity_proxy, twilight_access, front_strength
        )
        
        logger.info("FOI computation completed")
        return foi
    
    def _compute_thermal_suitability(self, sst_ds: xr.Dataset) -> xr.DataArray:
        """
        Compute thermal suitability for sharks
        
        Args:
            sst_ds: SST dataset
            
        Returns:
            Thermal suitability index
        """
        sst_var = list(sst_ds.data_vars)[0]
        sst = sst_ds[sst_var]
        
        preferred_temp = self.species_params.get('preferred_temp', 26.0)
        temp_tolerance = self.species_params.get('temp_tolerance', 2.5)
        
        # Thermal suitability: ST = exp(-(T - Tpref)² / (2σT²))
        thermal_suitability = np.exp(-((sst - preferred_temp)**2) / (2 * temp_tolerance**2))
        
        thermal_suitability.attrs = {
            'long_name': 'Thermal suitability',
            'units': 'dimensionless',
            'description': f'Thermal suitability for sharks (preferred: {preferred_temp}°C, tolerance: {temp_tolerance}°C)'
        }
        
        return thermal_suitability
    
    def _compute_eddy_relief(self, thermal_suitability: xr.DataArray, 
                           eke_normalized: xr.DataArray) -> xr.DataArray:
        """
        Compute eddy relief effect on thermal suitability
        
        Args:
            thermal_suitability: Thermal suitability index
            eke_normalized: Normalized Eddy Kinetic Energy
            
        Returns:
            Eddy-relief adjusted thermal suitability
        """
        alpha = self.derived_params.get('alpha', 0.5)
        
        # Eddy relief: ST,eff = 1 - (1 - ST)(1 - α·EKE~)
        eddy_relief = 1 - (1 - thermal_suitability) * (1 - alpha * eke_normalized)
        
        eddy_relief.attrs = {
            'long_name': 'Eddy-relief thermal suitability',
            'units': 'dimensionless',
            'description': 'Thermal suitability adjusted for eddy relief effects'
        }
        
        return eddy_relief
    
    def _compute_productivity_proxy(self, chl_ds: xr.Dataset) -> xr.DataArray:
        """
        Compute productivity proxy from chlorophyll-a
        
        Args:
            chl_ds: Chlorophyll-a dataset
            
        Returns:
            Productivity proxy
        """
        chl_var = list(chl_ds.data_vars)[0]
        chl = chl_ds[chl_var]
        
        # Productivity proxy: P = log(1 + Chl)
        productivity = np.log(1 + chl)
        
        productivity.attrs = {
            'long_name': 'Productivity proxy',
            'units': 'log(mg/m³)',
            'description': 'Log-transformed chlorophyll-a concentration as productivity proxy'
        }
        
        return productivity
    
    def _compute_twilight_access(self, euphotic_depth_normalized: xr.DataArray, 
                               eke_normalized: xr.DataArray) -> xr.DataArray:
        """
        Compute twilight access index
        
        Args:
            euphotic_depth_normalized: Normalized euphotic depth
            eke_normalized: Normalized Eddy Kinetic Energy
            
        Returns:
            Twilight access index
        """
        beta1 = self.derived_params.get('beta1', 0.5)
        beta2 = self.derived_params.get('beta2', 0.5)
        
        # Twilight access: Atw = β₁·Z~eu + β₂·EKE~
        twilight_access = beta1 * euphotic_depth_normalized + beta2 * eke_normalized
        
        twilight_access.attrs = {
            'long_name': 'Twilight access',
            'units': 'dimensionless',
            'description': 'Combined euphotic depth and EKE for twilight access'
        }
        
        return twilight_access
    
    def _compute_front_strength(self, sst_gradient_normalized: xr.DataArray) -> xr.DataArray:
        """
        Compute front strength from SST gradient
        
        Args:
            sst_gradient_normalized: Normalized SST gradient
            
        Returns:
            Front strength index
        """
        # Front strength: Ffront = |∇SST|~
        front_strength = sst_gradient_normalized
        
        front_strength.attrs = {
            'long_name': 'Front strength',
            'units': 'dimensionless',
            'description': 'Normalized SST gradient magnitude as front strength'
        }
        
        return front_strength
    
    def _compute_logistic_foi(self, eddy_relief: xr.DataArray, 
                            productivity_proxy: xr.DataArray,
                            twilight_access: xr.DataArray,
                            front_strength: xr.DataArray) -> xr.DataArray:
        """
        Compute FOI using logistic regression
        
        Args:
            eddy_relief: Eddy-relief adjusted thermal suitability
            productivity_proxy: Productivity proxy
            twilight_access: Twilight access index
            front_strength: Front strength index
            
        Returns:
            FOI DataArray
        """
        # Extract coefficients
        b0 = self.coefficients.get('b0', 0.0)
        b1 = self.coefficients.get('b1', 0.3)
        b2 = self.coefficients.get('b2', 0.25)
        b3 = self.coefficients.get('b3', 0.25)
        b4 = self.coefficients.get('b4', 0.2)
        
        # Normalize productivity proxy (log-transformed data needs normalization)
        productivity_normalized = self._normalize_productivity(productivity_proxy)
        
        # Compute linear combination: η = b₀ + b₁·ST,eff + b₂·P~ + b₃·Atw + b₄·Ffront
        eta = (b0 + 
               b1 * eddy_relief + 
               b2 * productivity_normalized + 
               b3 * twilight_access + 
               b4 * front_strength)
        
        # Apply logistic function: FOI = 1 / (1 + e^(-η))
        foi = 1 / (1 + np.exp(-eta))
        
        # Create FOI DataArray
        foi_da = xr.DataArray(
            foi,
            coords=eddy_relief.coords,
            dims=eddy_relief.dims,
            name='foi',
            attrs={
                'long_name': 'Foraging Opportunity Index',
                'units': 'dimensionless',
                'description': 'Shark foraging opportunity index (0-1 scale)',
                'model_coefficients': {
                    'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4
                },
                'model_version': '1.0',
                'creation_date': np.datetime64('now').astype(str)
            }
        )
        
        return foi_da
    
    def _normalize_productivity(self, productivity: xr.DataArray) -> xr.DataArray:
        """
        Normalize productivity proxy using robust scaling
        
        Args:
            productivity: Productivity proxy
            
        Returns:
            Normalized productivity proxy
        """
        # Compute robust percentiles
        flat_data = productivity.values.flatten()
        valid_data = flat_data[~np.isnan(flat_data)]
        
        if len(valid_data) == 0:
            return productivity
        
        p5 = np.percentile(valid_data, 5)
        p95 = np.percentile(valid_data, 95)
        
        # Apply robust normalization
        if p95 > p5:
            normalized = (productivity - p5) / (p95 - p5)
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = productivity * 0  # All zeros if no variation
        
        return normalized
    
    def compute_cps(self, foi: xr.DataArray, fishing_effort: Optional[xr.DataArray] = None) -> xr.DataArray:
        """
        Compute Conservation Priority Surface (CPS)
        
        Args:
            foi: Foraging Opportunity Index
            fishing_effort: Fishing effort data (optional)
            
        Returns:
            Conservation Priority Surface
        """
        logger.info("Computing Conservation Priority Surface (CPS)...")
        
        if fishing_effort is None:
            logger.warning("No fishing effort data provided, CPS = FOI")
            cps = foi.copy()
            cps.name = 'cps'
            cps.attrs.update({
                'long_name': 'Conservation Priority Surface',
                'description': 'Conservation Priority Surface (FOI only, no fishing effort data)'
            })
        else:
            # Normalize fishing effort
            fishing_normalized = self._normalize_productivity(fishing_effort)
            
            # Compute CPS: CPS = FOI × E~fish
            cps = foi * fishing_normalized
            
            cps.attrs = {
                'long_name': 'Conservation Priority Surface',
                'units': 'dimensionless',
                'description': 'Conservation Priority Surface combining FOI and fishing effort risk'
            }
        
        logger.info("CPS computation completed")
        return cps
    
    def validate_model(self, foi: xr.DataArray) -> bool:
        """
        Validate FOI model output
        
        Args:
            foi: FOI DataArray
            
        Returns:
            True if validation passes
        """
        logger.info("Validating FOI model output...")
        
        # Check value range
        foi_min = float(foi.min().values)
        foi_max = float(foi.max().values)
        
        if foi_min < 0 or foi_max > 1:
            logger.error(f"FOI values outside valid range [0,1]: [{foi_min:.4f}, {foi_max:.4f}]")
            return False
        
        # Check for NaN values
        nan_count = np.isnan(foi.values).sum()
        total_count = foi.size
        
        if nan_count > 0:
            nan_percentage = (nan_count / total_count) * 100
            logger.warning(f"FOI contains {nan_count} NaN values ({nan_percentage:.2f}%)")
            
            if nan_percentage > 50:
                logger.error(f"Too many NaN values in FOI: {nan_percentage:.2f}%")
                return False
        
        # Check spatial coverage
        valid_pixels = np.sum(~np.isnan(foi.values))
        coverage_percentage = (valid_pixels / total_count) * 100
        
        if coverage_percentage < 10:
            logger.error(f"Insufficient spatial coverage: {coverage_percentage:.2f}%")
            return False
        
        logger.info(f"FOI validation passed: range=[{foi_min:.4f}, {foi_max:.4f}], coverage={coverage_percentage:.2f}%")
        return True
    
    def get_model_summary(self, foi: xr.DataArray) -> Dict:
        """
        Get model summary statistics
        
        Args:
            foi: FOI DataArray
            
        Returns:
            Dictionary of summary statistics
        """
        valid_data = foi.values[~np.isnan(foi.values)]
        
        summary = {
            'foi_statistics': {
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'mean': float(np.mean(valid_data)),
                'median': float(np.median(valid_data)),
                'std': float(np.std(valid_data)),
                'q25': float(np.percentile(valid_data, 25)),
                'q75': float(np.percentile(valid_data, 75))
            },
            'spatial_coverage': {
                'total_pixels': int(foi.size),
                'valid_pixels': int(len(valid_data)),
                'coverage_percentage': float((len(valid_data) / foi.size) * 100)
            },
            'model_parameters': self.coefficients.copy(),
            'hotspot_thresholds': {
                'low': 0.3,
                'medium': 0.5,
                'high': 0.7
            }
        }
        
        # Count hotspots by threshold
        for threshold_name, threshold_value in summary['hotspot_thresholds'].items():
            count = np.sum(valid_data >= threshold_value)
            percentage = (count / len(valid_data)) * 100
            summary[f'{threshold_name}_hotspots'] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        return summary


def compute_foi_model(datasets: Dict[str, xr.Dataset], 
                     derived_fields: Dict[str, xr.DataArray],
                     normalized_fields: Dict[str, xr.DataArray],
                     config: Optional[Dict] = None) -> Tuple[xr.DataArray, Dict]:
    """
    Convenience function to compute FOI model
    
    Args:
        datasets: Original harmonized datasets
        derived_fields: Computed derived fields
        normalized_fields: Normalized fields
        config: Model configuration
        
    Returns:
        Tuple of (FOI DataArray, model_summary)
    """
    model = FOIModel(config)
    foi = model.compute_foi(datasets, derived_fields, normalized_fields)
    
    # Validate model
    if not model.validate_model(foi):
        logger.warning("FOI model validation failed")
    
    # Get summary
    summary = model.get_model_summary(foi)
    
    return foi, summary


if __name__ == "__main__":
    # Test the FOI model
    from foi_backend.shark_hotspots.data_loader import load_satellite_data
    from foi_backend.shark_hotspots.harmonize import harmonize_datasets
    from foi_backend.shark_hotspots.derived_fields import compute_derived_fields
    from foi_backend.shark_hotspots.normalization import normalize_datasets
    
    # Load and process test data
    bbox = [94.0, -11.0, 142.0, 6.0]
    start_date = "2025-03-01"
    end_date = "2025-03-14"
    
    datasets = load_satellite_data(bbox, start_date, end_date)
    harmonized = harmonize_datasets(datasets, bbox, start_date, end_date)
    derived = compute_derived_fields(harmonized)
    normalized, _ = normalize_datasets(derived)
    
    # Compute FOI
    foi, summary = compute_foi_model(harmonized, derived, normalized)
    
    print("\nFOI Model Results:")
    print(f"FOI range: [{foi.min().values:.4f}, {foi.max().values:.4f}]")
    print(f"FOI mean: {foi.mean().values:.4f}")
    print(f"Spatial coverage: {summary['spatial_coverage']['coverage_percentage']:.2f}%")
    
    print("\nHotspot Statistics:")
    for threshold in ['low', 'medium', 'high']:
        stats = summary[f'{threshold}_hotspots']
        print(f"{threshold.capitalize()} hotspots: {stats['count']} pixels ({stats['percentage']:.2f}%)")
