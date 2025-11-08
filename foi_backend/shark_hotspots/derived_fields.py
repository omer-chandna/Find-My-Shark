"""
Derived Fields Module
Computes oceanographic derived fields from satellite data for shark foraging analysis
"""

import logging
import xarray as xr
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DerivedFieldsComputer:
    """
    Computes derived oceanographic fields from satellite data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize derived fields computer
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or self._get_default_config()
        
        # Physical constants
        self.g = 9.81  # gravitational acceleration (m/s²)
        self.omega = 7.2921e-5  # Earth's angular velocity (rad/s)
        
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'derived_fields': {
                'alpha': 0.5,  # eddy relief coefficient
                'beta1': 0.5,  # twilight access - euphotic depth weight
                'beta2': 0.5   # twilight access - EKE weight
            }
        }
    
    def compute_derived_fields(self, datasets: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        """
        Compute all derived fields from input datasets
        
        Args:
            datasets: Dictionary containing harmonized satellite datasets
            
        Returns:
            Dictionary containing derived fields
        """
        logger.info("Computing derived oceanographic fields...")
        
        derived_fields = {}
        
        # Extract required datasets
        ssh_ds = datasets.get('ssh')
        sst_ds = datasets.get('sst')
        chl_ds = datasets.get('chl')
        kd490_ds = datasets.get('kd490')
        
        if ssh_ds is None:
            raise ValueError("SSH dataset required for derived field computation")
        
        # Compute geostrophic velocities
        logger.info("Computing geostrophic velocities...")
        ug, vg = self._compute_geostrophic_velocities(ssh_ds)
        
        # Compute Eddy Kinetic Energy (EKE)
        logger.info("Computing Eddy Kinetic Energy...")
        eke = self._compute_eke(ug, vg)
        
        # Compute vorticity
        logger.info("Computing relative vorticity...")
        vorticity = self._compute_vorticity(ug, vg)
        
        # Store derived fields
        derived_fields['ug'] = ug
        derived_fields['vg'] = vg
        derived_fields['eke'] = eke
        derived_fields['vorticity'] = vorticity
        
        # Compute SST gradient if SST data available
        if sst_ds is not None:
            logger.info("Computing SST gradient...")
            sst_gradient = self._compute_sst_gradient(sst_ds)
            derived_fields['sst_gradient'] = sst_gradient
        
        # Compute euphotic depth if Kd490 data available
        if kd490_ds is not None:
            logger.info("Computing euphotic depth...")
            euphotic_depth = self._compute_euphotic_depth(kd490_ds)
            derived_fields['euphotic_depth'] = euphotic_depth
        
        logger.info("Derived field computation completed")
        return derived_fields
    
    def _compute_geostrophic_velocities(self, ssh_ds: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute geostrophic velocities from SSH
        
        Args:
            ssh_ds: SSH dataset
            
        Returns:
            Tuple of (ug, vg) geostrophic velocity components
        """
        ssh_var = list(ssh_ds.data_vars)[0]  # Get first data variable
        ssh = ssh_ds[ssh_var]
        
        # Get coordinates
        lat = ssh.lat.values
        lon = ssh.lon.values
        
        # Convert to radians
        lat_rad = np.deg2rad(lat)
        
        # Compute Coriolis parameter
        f = 2 * self.omega * np.sin(lat_rad)
        
        # Create coordinate grids
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        f_grid = 2 * self.omega * np.sin(np.deg2rad(lat_grid))
        
        # Compute gradients
        # Convert longitude to meters (approximate)
        dlon_m = float(np.cos(np.deg2rad(lat_grid.mean())) * 111320 * np.deg2rad(lon[1] - lon[0]))
        dlat_m = float(111320 * np.deg2rad(lat[1] - lat[0]))
        
        # Compute gradients using numpy.gradient
        ssh_grad_lon, ssh_grad_lat = np.gradient(ssh.values, dlon_m, dlat_m, axis=(2, 1))
        
        # Geostrophic velocities
        ug = -(self.g / f_grid) * ssh_grad_lat
        vg = (self.g / f_grid) * ssh_grad_lon
        
        # Create DataArrays with proper dimensions
        if 'time' in ssh.dims:
            coords = {'time': ssh.time, 'lat': lat, 'lon': lon}
            dims = ['time', 'lat', 'lon']
        else:
            coords = {'lat': lat, 'lon': lon}
            dims = ['lat', 'lon']
        
        ug_da = xr.DataArray(
            ug,
            coords=coords,
            dims=dims,
            name='ug',
            attrs={
                'long_name': 'Geostrophic velocity (eastward)',
                'units': 'm/s',
                'description': 'Eastward geostrophic velocity derived from SSH'
            }
        )
        
        vg_da = xr.DataArray(
            vg,
            coords=coords,
            dims=dims,
            name='vg',
            attrs={
                'long_name': 'Geostrophic velocity (northward)',
                'units': 'm/s',
                'description': 'Northward geostrophic velocity derived from SSH'
            }
        )
        
        return ug_da, vg_da
    
    def _compute_eke(self, ug: xr.DataArray, vg: xr.DataArray) -> xr.DataArray:
        """
        Compute Eddy Kinetic Energy from geostrophic velocities
        
        Args:
            ug: Eastward geostrophic velocity
            vg: Northward geostrophic velocity
            
        Returns:
            Eddy Kinetic Energy
        """
        # Compute velocity anomalies (remove mean)
        ug_mean = ug.mean()
        vg_mean = vg.mean()
        
        ug_prime = ug - ug_mean
        vg_prime = vg - vg_mean
        
        # Compute EKE
        eke = 0.5 * (ug_prime**2 + vg_prime**2)
        
        eke.attrs = {
            'long_name': 'Eddy Kinetic Energy',
            'units': 'm²/s²',
            'description': 'Eddy Kinetic Energy computed from geostrophic velocity anomalies'
        }
        
        return eke
    
    def _compute_vorticity(self, ug: xr.DataArray, vg: xr.DataArray) -> xr.DataArray:
        """
        Compute relative vorticity from geostrophic velocities
        
        Args:
            ug: Eastward geostrophic velocity
            vg: Northward geostrophic velocity
            
        Returns:
            Relative vorticity
        """
        # Get coordinates
        lat = ug.lat.values
        lon = ug.lon.values
        
        # Convert to meters
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        dlon_m = float(np.cos(np.deg2rad(lat_grid.mean())) * 111320 * np.deg2rad(lon[1] - lon[0]))
        dlat_m = float(111320 * np.deg2rad(lat[1] - lat[0]))
        
        # Compute velocity gradients
        dug_dx, dug_dy = np.gradient(ug.values, dlon_m, dlat_m, axis=(2, 1))
        dvg_dx, dvg_dy = np.gradient(vg.values, dlon_m, dlat_m, axis=(2, 1))
        
        # Compute relative vorticity: ζ = ∂v/∂x - ∂u/∂y
        vorticity = dvg_dx - dug_dy
        
        # Create DataArray with proper dimensions
        if 'time' in ug.dims:
            coords = {'time': ug.time, 'lat': lat, 'lon': lon}
            dims = ['time', 'lat', 'lon']
        else:
            coords = {'lat': lat, 'lon': lon}
            dims = ['lat', 'lon']
        
        vorticity_da = xr.DataArray(
            vorticity,
            coords=coords,
            dims=dims,
            name='vorticity',
            attrs={
                'long_name': 'Relative vorticity',
                'units': 's⁻¹',
                'description': 'Relative vorticity computed from geostrophic velocity gradients'
            }
        )
        
        return vorticity_da
    
    def _compute_sst_gradient(self, sst_ds: xr.Dataset) -> xr.DataArray:
        """
        Compute SST gradient magnitude
        
        Args:
            sst_ds: SST dataset
            
        Returns:
            SST gradient magnitude
        """
        sst_var = list(sst_ds.data_vars)[0]  # Get first data variable
        sst = sst_ds[sst_var]
        
        # Get coordinates
        lat = sst.lat.values
        lon = sst.lon.values
        
        # Convert to meters
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
        dlon_m = float(np.cos(np.deg2rad(lat_grid.mean())) * 111320 * np.deg2rad(lon[1] - lon[0]))
        dlat_m = float(111320 * np.deg2rad(lat[1] - lat[0]))
        
        # Compute SST gradients
        dsst_dx, dsst_dy = np.gradient(sst.values, dlon_m, dlat_m, axis=(2, 1))
        
        # Compute gradient magnitude
        sst_gradient = np.sqrt(dsst_dx**2 + dsst_dy**2)
        
        # Create DataArray with proper dimensions
        if 'time' in sst.dims:
            coords = {'time': sst.time, 'lat': lat, 'lon': lon}
            dims = ['time', 'lat', 'lon']
        else:
            coords = {'lat': lat, 'lon': lon}
            dims = ['lat', 'lon']
        
        sst_gradient_da = xr.DataArray(
            sst_gradient,
            coords=coords,
            dims=dims,
            name='sst_gradient',
            attrs={
                'long_name': 'SST gradient magnitude',
                'units': '°C/m',
                'description': 'Magnitude of sea surface temperature gradient'
            }
        )
        
        return sst_gradient_da
    
    def _compute_euphotic_depth(self, kd490_ds: xr.Dataset) -> xr.DataArray:
        """
        Compute euphotic depth from Kd490
        
        Args:
            kd490_ds: Kd490 dataset
            
        Returns:
            Euphotic depth
        """
        kd490_var = list(kd490_ds.data_vars)[0]  # Get first data variable
        kd490 = kd490_ds[kd490_var]
        
        # Euphotic depth: Zeu = 4.6 / Kd490
        euphotic_depth = 4.6 / kd490
        
        euphotic_depth.attrs = {
            'long_name': 'Euphotic depth',
            'units': 'm',
            'description': 'Depth of 1% light penetration (euphotic zone)'
        }
        
        return euphotic_depth
    
    def compute_thermal_suitability(self, sst_ds: xr.Dataset, 
                                  preferred_temp: float = 26.0,
                                  temp_tolerance: float = 2.5) -> xr.DataArray:
        """
        Compute thermal suitability for sharks
        
        Args:
            sst_ds: SST dataset
            preferred_temp: Preferred temperature (°C)
            temp_tolerance: Temperature tolerance (°C)
            
        Returns:
            Thermal suitability index
        """
        sst_var = list(sst_ds.data_vars)[0]  # Get first data variable
        sst = sst_ds[sst_var]
        
        # Thermal suitability: ST = exp(-(T - Tpref)² / (2σT²))
        thermal_suitability = np.exp(-((sst - preferred_temp)**2) / (2 * temp_tolerance**2))
        
        thermal_suitability.attrs = {
            'long_name': 'Thermal suitability',
            'units': 'dimensionless',
            'description': f'Thermal suitability for sharks (preferred: {preferred_temp}°C, tolerance: {temp_tolerance}°C)'
        }
        
        return thermal_suitability
    
    def compute_eddy_relief(self, thermal_suitability: xr.DataArray, 
                           eke: xr.DataArray, alpha: float = 0.5) -> xr.DataArray:
        """
        Compute eddy relief effect on thermal suitability
        
        Args:
            thermal_suitability: Thermal suitability index
            eke: Eddy Kinetic Energy
            alpha: Eddy relief coefficient
            
        Returns:
            Eddy-relief adjusted thermal suitability
        """
        # Eddy relief: ST,eff = 1 - (1 - ST)(1 - α·EKE~)
        # For now, use EKE directly (normalization will be applied later)
        eddy_relief = 1 - (1 - thermal_suitability) * (1 - alpha * eke)
        
        eddy_relief.attrs = {
            'long_name': 'Eddy-relief thermal suitability',
            'units': 'dimensionless',
            'description': 'Thermal suitability adjusted for eddy relief effects'
        }
        
        return eddy_relief
    
    def compute_productivity_proxy(self, chl_ds: xr.Dataset) -> xr.DataArray:
        """
        Compute productivity proxy from chlorophyll-a
        
        Args:
            chl_ds: Chlorophyll-a dataset
            
        Returns:
            Productivity proxy
        """
        chl_var = list(chl_ds.data_vars)[0]  # Get first data variable
        chl = chl_ds[chl_var]
        
        # Productivity proxy: P = log(1 + Chl)
        productivity = np.log(1 + chl)
        
        productivity.attrs = {
            'long_name': 'Productivity proxy',
            'units': 'log(mg/m³)',
            'description': 'Log-transformed chlorophyll-a concentration as productivity proxy'
        }
        
        return productivity
    
    def compute_twilight_access(self, euphotic_depth: xr.DataArray, 
                              eke: xr.DataArray,
                              beta1: float = 0.5, beta2: float = 0.5) -> xr.DataArray:
        """
        Compute twilight access index
        
        Args:
            euphotic_depth: Euphotic depth
            eke: Eddy Kinetic Energy
            beta1: Euphotic depth weight
            beta2: EKE weight
            
        Returns:
            Twilight access index
        """
        # Twilight access: Atw = β₁·Z~eu + β₂·EKE~
        # For now, use raw values (normalization will be applied later)
        twilight_access = beta1 * euphotic_depth + beta2 * eke
        
        twilight_access.attrs = {
            'long_name': 'Twilight access',
            'units': 'mixed',
            'description': 'Combined euphotic depth and EKE for twilight access'
        }
        
        return twilight_access


def compute_derived_fields(datasets: Dict[str, xr.Dataset], 
                         config: Optional[Dict] = None) -> Dict[str, xr.Dataset]:
    """
    Convenience function to compute derived fields
    
    Args:
        datasets: Dictionary containing harmonized satellite datasets
        config: Configuration dictionary (optional)
        
    Returns:
        Dictionary containing derived fields
    """
    computer = DerivedFieldsComputer(config)
    return computer.compute_derived_fields(datasets)


if __name__ == "__main__":
    # Test the derived fields computer
    from foi_backend.shark_hotspots.data_loader import load_satellite_data
    from foi_backend.shark_hotspots.harmonize import harmonize_datasets
    
    # Load and harmonize test data
    bbox = [94.0, -11.0, 142.0, 6.0]
    start_date = "2025-03-01"
    end_date = "2025-03-14"
    
    datasets = load_satellite_data(bbox, start_date, end_date)
    harmonized = harmonize_datasets(datasets, bbox, start_date, end_date)
    
    # Compute derived fields
    derived = compute_derived_fields(harmonized)
    
    # Print results
    for name, field in derived.items():
        print(f"\n{name.upper()} Field:")
        print(f"Shape: {field.shape}")
        print(f"Min: {field.min().values:.4f}")
        print(f"Max: {field.max().values:.4f}")
        print(f"Mean: {field.mean().values:.4f}")
