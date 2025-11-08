"""
Visualization Module
Creates interactive visualizations with NASA GIBS MODIS True Color overlays
"""

import logging
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Try to import folium, but don't fail if not available
try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logging.warning("Folium not available. Interactive maps will not be generated.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HotspotVisualizer:
    """
    Creates visualizations for shark foraging hotspots
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize visualizer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.output_dir = Path(self.config.get('output_dir', 'data/outputs'))
        
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'visualization': {
                'opacity': 0.55,
                'colormap': 'viridis',
                'overlay_bounds': [[-11, 94], [6, 142]]
            },
            'gibs': {
                'base_url': 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/',
                'layer': 'MODIS_Terra_CorrectedReflectance_TrueColor'
            }
        }
    
    def compute_global_foi(self, start_date: str, end_date: str) -> xr.DataArray:
        """
        Compute FOI for the entire globe
        
        Args:
            start_date: Start date for computation
            end_date: End date for computation
            
        Returns:
            Global FOI DataArray
        """
        from foi_backend.shark_hotspots.predictor import compute_foi_map
        
        logger.info("Computing global FOI...")
        
        # Global bounding box
        global_bbox = [-180.0, -90.0, 180.0, 90.0]
        
        # Compute global FOI
        global_foi, _ = compute_foi_map(start_date, end_date, global_bbox, save_output=False)
        
        logger.info(f"Global FOI computed: {global_foi.shape}")
        return global_foi
    
    def get_top_foraging_areas(self, foi: xr.DataArray, top_n: int = 10) -> list:
        """
        Get top foraging areas from FOI data
        
        Args:
            foi: FOI DataArray
            top_n: Number of top areas to return
            
        Returns:
            List of top foraging areas with coordinates and values
        """
        # Handle time dimension if present
        if 'time' in foi.dims:
            foi_2d = foi.isel(time=0)  # Take first time slice
        else:
            foi_2d = foi
        
        # Convert to numpy array for easier manipulation
        foi_values = foi_2d.values
        lat_coords = foi_2d.lat.values
        lon_coords = foi_2d.lon.values
        
        # Flatten and get indices of top values
        foi_flat = foi_values.flatten()
        top_indices_flat = np.argsort(foi_flat)[-top_n:]
        
        # Convert flat indices back to 2D indices
        lat_indices, lon_indices = np.unravel_index(top_indices_flat, foi_values.shape)
        
        # Get top areas
        top_areas = []
        for i, (lat_idx, lon_idx) in enumerate(zip(lat_indices, lon_indices)):
            lat = lat_coords[lat_idx]
            lon = lon_coords[lon_idx]
            foi_value = foi_values[lat_idx, lon_idx]
            
            top_areas.append({
                'rank': i + 1,
                'lat': float(lat),
                'lon': float(lon),
                'foi_value': float(foi_value),
                'region': self._get_region_name(lat, lon)
            })
        
        # Sort by FOI value (highest first)
        top_areas.sort(key=lambda x: x['foi_value'], reverse=True)
        
        # Update ranks
        for i, area in enumerate(top_areas):
            area['rank'] = i + 1
        
        return top_areas
    
    def _get_region_name(self, lat: float, lon: float) -> str:
        """Get region name from coordinates"""
        if -11 <= lat <= 6 and 94 <= lon <= 142:
            return "Indonesia-Malaysia"
        elif 20 <= lat <= 50 and 120 <= lon <= 180:
            return "North Pacific"
        elif -60 <= lat <= -20 and 0 <= lon <= 60:
            return "South Atlantic"
        elif 30 <= lat <= 60 and -80 <= lon <= -40:
            return "North Atlantic"
        elif -40 <= lat <= -10 and 140 <= lon <= 180:
            return "South Pacific"
        else:
            return "Global Ocean"

    def generate_dashboard(self, foi: xr.DataArray, date: str = None,
                          output_dir: Optional[Path] = None) -> str:
        """
        Generate interactive dashboard with NASA GIBS basemap
        
        Args:
            foi: FOI DataArray
            date: Date for the visualization
            output_dir: Output directory (optional)
            
        Returns:
            Path to generated HTML file
        """
        if not FOLIUM_AVAILABLE:
            logger.error("Folium not available. Cannot generate interactive dashboard.")
            return None
        
        logger.info("Generating interactive dashboard...")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir / 'visualizations'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        html_filename = output_dir / f'foi_dashboard_{date}.html'
        
        try:
            # Create a clean copy of FOI without problematic attributes
            foi_clean = foi.copy()
            foi_clean.attrs = {k: v for k, v in foi.attrs.items() 
                             if isinstance(v, (str, int, float, bool))}
            
            # Create map
            map_obj = self._create_folium_map(foi_clean, date)
            
            # Save map
            map_obj.save(str(html_filename))
            
            logger.info(f"Interactive dashboard saved to {html_filename}")
            return str(html_filename)
            
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
            raise
    
    def _create_folium_map(self, foi: xr.DataArray, date: str) -> 'folium.Map':
        """
        Create Folium map with NASA GIBS basemap and FOI overlay
        
        Args:
            foi: FOI DataArray
            date: Date for the visualization
            
        Returns:
            Folium map object
        """
        # Get map center and bounds
        center_lat = float((foi.lat.min().values + foi.lat.max().values) / 2)
        center_lon = float((foi.lon.min().values + foi.lon.max().values) / 2)
        
        # Create base map with standard tiles first
        map_obj = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'  # Use standard tiles as fallback
        )
        
        # Try to add NASA GIBS MODIS True Color basemap
        try:
            self._add_gibs_basemap(map_obj, date)
        except Exception as e:
            logger.warning(f"NASA GIBS basemap failed, using standard tiles: {e}")
        
        # Add alternative basemaps for cloud-free viewing
        self._add_alternative_basemaps(map_obj)
        
        # Add FOI overlay
        self._add_foi_overlay(map_obj, foi)
        
        # Add Apple Find My style controls
        self._add_apple_controls(map_obj)
        
        # Add tabs for different views
        self._add_dashboard_tabs(map_obj, foi, date)
        
        # Add layer control
        folium.LayerControl().add_to(map_obj)
        
        # Add fullscreen plugin
        plugins.Fullscreen().add_to(map_obj)
        
        # Add scale bar
        plugins.MeasureControl().add_to(map_obj)
        
        return map_obj
    
    def _add_gibs_basemap(self, map_obj: 'folium.Map', date: str):
        """
        Add NASA GIBS basemap with cloud-free options
        
        Args:
            map_obj: Folium map object
            date: Date for the basemap
        """
        gibs_config = self.config.get('gibs', {})
        base_url = gibs_config.get('base_url', 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/')
        
        # Try multiple cloud-free layers
        cloud_free_layers = [
            'MODIS_Terra_CorrectedReflectance_Bands721',  # False color (cloud-free)
            'MODIS_Terra_CorrectedReflectance_Bands143',  # Another false color option
            'MODIS_Terra_CorrectedReflectance_TrueColor', # True color (fallback)
        ]
        
        # Format date for GIBS URL
        formatted_date = datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
        
        # Add multiple basemap options
        for i, layer in enumerate(cloud_free_layers):
            gibs_url = f"{base_url}{layer}/default/{formatted_date}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
            
            layer_name = f"NASA GIBS - {layer.split('_')[-1]}"
            if i == 0:
                layer_name += " (Cloud-Free)"
            
            folium.TileLayer(
                tiles=gibs_url,
                attr='NASA GIBS',
                name=layer_name,
                overlay=False,
                control=True,
                show=(i == 0)  # Show first layer by default
            ).add_to(map_obj)
    
    def _add_alternative_basemaps(self, map_obj: 'folium.Map'):
        """
        Add alternative basemaps for cloud-free viewing
        """
        # Add satellite imagery without clouds
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite (Cloud-Free Composite)',
            overlay=False,
            control=True
        ).add_to(map_obj)
        
        # Add bathymetry/topography for ocean context
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Ocean Basemap',
            overlay=False,
            control=True
        ).add_to(map_obj)
        
        # Add terrain for land context
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Terrain',
            overlay=False,
            control=True
        ).add_to(map_obj)
    
    def _add_foi_overlay(self, map_obj: 'folium.Map', foi: xr.DataArray):
        """
        Add FOI overlay to map
        
        Args:
            map_obj: Folium map object
            foi: FOI DataArray
        """
        # Convert FOI to image
        foi_image = self._foi_to_image(foi)
        
        # Get bounds
        bounds = [
            [float(foi.lat.min().values), float(foi.lon.min().values)],
            [float(foi.lat.max().values), float(foi.lon.max().values)]
        ]
        
        # Add image overlay with better transparency for cloud penetration
        folium.raster_layers.ImageOverlay(
            image=foi_image,
            bounds=bounds,
            opacity=self.config.get('visualization', {}).get('opacity', 0.75),  # Increased opacity
            name='Shark Foraging Hotspots (FOI)',
            overlay=True,
            control=True,
            show=True,
            interactive=True
        ).add_to(map_obj)
        
        # Add legend
        self._add_foi_legend(map_obj)
    
    def _foi_to_image(self, foi: xr.DataArray) -> np.ndarray:
        """
        Convert FOI DataArray to RGBA image array for map overlay
        
        Args:
            foi: FOI DataArray
            
        Returns:
            Image array (RGBA) with values in 0-255 range
        """
        # Select first time slice if 3D
        foi_to_use = foi.isel(time=0) if 'time' in foi.dims else foi
        
        # Get FOI values
        foi_values = foi_to_use.values
        
        # Handle NaN values
        foi_values = np.nan_to_num(foi_values, nan=0.0)
        
        # Normalize to 0-1 range (all areas)
        foi_min, foi_max = foi_values.min(), foi_values.max()
        if foi_max > foi_min:
            foi_normalized = (foi_values - foi_min) / (foi_max - foi_min)
        else:
            foi_normalized = foi_values * 0
        
        # Apply color smoothing to reduce noise in large regions
        foi_smoothed = self._smooth_foi_colors(foi_normalized)
        
        # Apply colormap - use custom FOI colormap
        colormap = self._create_foi_colormap()
        
        # Convert to RGBA
        rgba_image = colormap(foi_smoothed)
        
        # Make the overlay more opaque to improve color distinction
        # Higher opacity makes colors more distinct against the blue background
        rgba_image[:, :, 3] = 0.9  # Set alpha to 90% opacity (10% transparency)
        
        # Convert to uint8
        rgba_image = (rgba_image * 255).astype(np.uint8)
        
        return rgba_image
    
    def _smooth_foi_colors(self, foi_normalized: np.ndarray) -> np.ndarray:
        """
        Smooth FOI colors to fill in isolated low-value pixels within high-value regions
        
        Uses morphological operations and Gaussian filtering to reduce noise and fill gaps
        in high-activity zones, improving visual clarity of foraging hotspots.
        
        Args:
            foi_normalized: Normalized FOI values (0-1 range)
            
        Returns:
            Smoothed FOI values with reduced noise
        """
        from scipy import ndimage
        
        # Create a binary mask for high-value regions (red areas)
        # Use 0.5 as threshold - values above this are considered "high"
        high_value_mask = foi_normalized > 0.5
        
        # Apply more aggressive morphological operations to fill holes in high-value regions
        # Use larger kernel for better hole filling
        kernel_size = 5  # Increased from 3 to 5
        closed_mask = ndimage.binary_closing(high_value_mask, structure=np.ones((kernel_size, kernel_size)))
        
        # Apply opening to remove small isolated high-value pixels
        opened_mask = ndimage.binary_opening(closed_mask, structure=np.ones((3, 3)))  # Increased from 2x2
        
        # Create smoothed values by taking the maximum of original and a smoothed version
        # This preserves high values while filling in small gaps
        # Use more aggressive smoothing to fill in medium activity dots
        smoothed = ndimage.gaussian_filter(foi_normalized, sigma=1.2)  # Increased from 0.8 to 1.2
        
        # For pixels in the smoothed high-value regions, use the smoothed values
        # For other pixels, use original values
        result = np.where(opened_mask, 
                         np.maximum(foi_normalized, smoothed), 
                         foi_normalized)
        
        # Apply additional aggressive smoothing specifically for high-value regions
        high_value_smoothed = ndimage.gaussian_filter(result, sigma=0.8)  # Increased from 0.5 to 0.8
        final_result = np.where(opened_mask, high_value_smoothed, result)
        
        # Additional step: fill in any remaining medium activity dots within high zones
        # Create a more aggressive high-value mask
        very_high_mask = final_result > 0.6  # Higher threshold
        if very_high_mask.any():
            # Dilate the very high mask to expand high activity zones
            dilated_high = ndimage.binary_dilation(very_high_mask, structure=np.ones((4, 4)))
            # For pixels near high activity, boost their values
            near_high = ndimage.binary_dilation(dilated_high, structure=np.ones((2, 2)))
            final_result = np.where(near_high, 
                                  np.maximum(final_result, 0.7),  # Boost to high activity
                                  final_result)
        
        return final_result
    
    
    def _create_foi_colormap(self):
        """Create custom FOI colormap matching the Apple Find My legend"""
        from matplotlib.colors import LinearSegmentedColormap
        
        # Define colors with maximum distinction between yellow and red
        colors = ['#059669', '#FFD700', '#FF0000']  # Low (darker green), Medium (golden yellow), High (bright red)
        n_bins = 256
        
        # Create colormap
        cmap = LinearSegmentedColormap.from_list('foi', colors, N=n_bins)
        return cmap
    
    def _add_apple_controls(self, map_obj: 'folium.Map'):
        """Add Apple Find My style header and controls"""
        header_html = """
        <div style="position: fixed; 
                    top: 0; left: 0; right: 0;
                    background: rgba(255, 255, 255, 0.95); 
                    backdrop-filter: blur(20px);
                    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
                    padding: 16px 20px;
                    z-index: 10000;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center;">
                    <div style="width: 12px; height: 12px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); border-radius: 50%; margin-right: 12px;"></div>
                    <div>
                        <div style="font-size: 18px; font-weight: 600; color: #1d1d1f; margin: 0;">Find My Shark</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <!-- Clean header without clutter -->
                </div>
            </div>
        </div>
        """
        map_obj.get_root().html.add_child(folium.Element(header_html))
        
        # Add CSS for proper spacing
        css_html = """
        <style>
            body { 
                margin: 0; 
                padding-top: 80px; 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .folium-map { 
                margin-top: 0 !important; 
            }
            .leaflet-control-container { 
                margin-top: 20px; 
            }
        </style>
        """
        map_obj.get_root().html.add_child(folium.Element(css_html))
    
    
    def _add_dashboard_tabs(self, map_obj: 'folium.Map', foi: xr.DataArray, date: str):
        """Add collapsible foraging areas tab under zoom controls"""
        # Get top foraging areas
        top_areas = self.get_top_foraging_areas(foi, top_n=5)
        
        # Create areas HTML
        areas_html = ""
        for area in top_areas:
            foi_percent = area['foi_value'] * 100
            areas_html += f"""
            <div style="display: flex; align-items: center; background: rgba(239, 68, 68, 0.05); border-radius: 8px; padding: 8px; margin-bottom: 8px;">
                <div style="width: 20px; height: 20px; background: linear-gradient(45deg, #EF4444, #F59E0B); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px;">
                    <span style="color: white; font-size: 10px; font-weight: 600;">{area['rank']}</span>
                </div>
                <div style="flex: 1;">
                    <div style="font-weight: 500; color: #1d1d1f; font-size: 12px;">{area['region']}</div>
                    <div style="color: #86868b; font-size: 10px;">{area['lat']:.2f}°, {area['lon']:.2f}°</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 600; color: #EF4444; font-size: 12px;">{foi_percent:.1f}%</div>
                </div>
            </div>
            """
        
        tabs_html = f"""
        <!-- Foraging Areas Tab -->
        <div style="position: fixed; 
                    top: 80px; left: 60px; 
                    background: rgba(255, 255, 255, 0.95); 
                    backdrop-filter: blur(20px);
                    border-radius: 16px; 
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    z-index: 9999; 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    font-size: 13px;
                    max-width: 300px;">
            
            <!-- Tab Header -->
            <div id="foraging-tab-header" style="display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; cursor: pointer; border-radius: 16px 16px 0 0;" onclick="toggleForagingTab()">
                <div style="display: flex; align-items: center;">
                    <div style="width: 8px; height: 8px; background: linear-gradient(45deg, #EF4444, #F59E0B); border-radius: 50%; margin-right: 8px;"></div>
                    <span style="font-weight: 600; color: #1d1d1f;">Top Foraging Areas</span>
                </div>
                <div id="foraging-tab-arrow" style="color: #86868b; font-size: 14px; transition: transform 0.3s ease;">▼</div>
            </div>
            
            <!-- Tab Content -->
            <div id="foraging-tab-content" style="display: none; padding: 0 16px 16px 16px; max-height: 50vh; overflow-y: auto;">
                <div style="font-size: 11px; color: #86868b; margin-bottom: 12px;">Top 5 Regional Hotspots</div>
                {areas_html}
                
                <div style="background: rgba(59, 130, 246, 0.05); border-radius: 8px; padding: 12px; margin-top: 12px;">
                    <div style="font-weight: 500; color: #1d1d1f; margin-bottom: 8px;">Indonesia-Malaysia Region</div>
                    <div style="color: #86868b; font-size: 11px; line-height: 1.4;">
                        Current view shows shark foraging hotspots in the Indonesian-Malaysian archipelago. 
                        This region is known for high marine biodiversity and shark activity.
                    </div>
                </div>
            </div>
        </div>
        
        <script>
        function toggleForagingTab() {{
            const content = document.getElementById('foraging-tab-content');
            const arrow = document.getElementById('foraging-tab-arrow');
            
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                arrow.style.transform = 'rotate(180deg)';
            }} else {{
                content.style.display = 'none';
                arrow.style.transform = 'rotate(0deg)';
            }}
        }}
        </script>
        """
        
        map_obj.get_root().html.add_child(folium.Element(tabs_html))
    
    def _add_foi_legend(self, map_obj: 'folium.Map'):
        """
        Add Apple Find My style FOI legend to map
        
        Args:
            map_obj: Folium map object
        """
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 20px; left: 20px; 
                    background: rgba(255, 255, 255, 0.95); 
                    backdrop-filter: blur(20px);
                    border-radius: 16px; 
                    padding: 16px 20px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    z-index: 9999; 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    font-size: 13px;
                    min-width: 240px;">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <div style="width: 8px; height: 8px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); border-radius: 50%; margin-right: 8px;"></div>
                <span style="font-weight: 600; color: #1d1d1f;">Shark Foraging Index</span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                  <div style="display: flex; align-items: center;">
                      <div style="width: 12px; height: 12px; background: #FF0000; border-radius: 3px; margin-right: 10px;"></div>
                      <span style="color: #1d1d1f; font-weight: 500;">High Activity</span>
                      <span style="color: #86868b; margin-left: auto; font-size: 11px;">0.7-1.0</span>
                  </div>
                  <div style="display: flex; align-items: center;">
                      <div style="width: 12px; height: 12px; background: #FFD700; border-radius: 3px; margin-right: 10px;"></div>
                      <span style="color: #1d1d1f; font-weight: 500;">Medium Activity</span>
                      <span style="color: #86868b; margin-left: auto; font-size: 11px;">0.3-0.7</span>
                  </div>
                  <div style="display: flex; align-items: center;">
                      <div style="width: 12px; height: 12px; background: #059669; border-radius: 3px; margin-right: 10px;"></div>
                      <span style="color: #1d1d1f; font-weight: 500;">Low Activity</span>
                      <span style="color: #86868b; margin-left: auto; font-size: 11px;">0.0-0.3</span>
                  </div>
            </div>
        </div>
        '''
        
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def create_static_snapshot(self, foi: xr.DataArray, date: str = None,
                             output_dir: Optional[Path] = None) -> str:
        """
        Create static PNG snapshot of FOI map
        
        Args:
            foi: FOI DataArray
            date: Date for the visualization
            output_dir: Output directory (optional)
            
        Returns:
            Path to generated PNG file
        """
        logger.info("Creating static snapshot...")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir / 'visualizations'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        png_filename = output_dir / f'foi_snapshot_{date}.png'
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot FOI (select first time slice if 3D)
            foi_to_plot = foi.isel(time=0) if 'time' in foi.dims else foi
            foi_plot = foi_to_plot.plot(
                ax=ax,
                cmap='viridis',
                vmin=0,
                vmax=1,
                add_colorbar=True,
                cbar_kwargs={'label': 'Foraging Opportunity Index'}
            )
            
            # Customize plot
            ax.set_title(f'Shark Foraging Hotspots - {date}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            
            # Add coastlines (if available)
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            except (ImportError, AttributeError):
                logger.warning("Cartopy not available or incompatible. Coastlines not added.")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Static snapshot saved to {png_filename}")
            return str(png_filename)
            
        except Exception as e:
            logger.error(f"Error creating static snapshot: {e}")
            raise
    
    def create_comparison_plot(self, foi_maps: Dict[str, xr.DataArray],
                             output_dir: Optional[Path] = None) -> str:
        """
        Create comparison plot of multiple FOI maps
        
        Args:
            foi_maps: Dictionary of {date: foi_dataarray}
            output_dir: Output directory (optional)
            
        Returns:
            Path to generated PNG file
        """
        logger.info("Creating comparison plot...")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir / 'visualizations'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        png_filename = output_dir / 'foi_comparison.png'
        
        try:
            n_maps = len(foi_maps)
            fig, axes = plt.subplots(1, n_maps, figsize=(5*n_maps, 5))
            
            if n_maps == 1:
                axes = [axes]
            
            for i, (date, foi) in enumerate(foi_maps.items()):
                foi_to_plot = foi.isel(time=0) if 'time' in foi.dims else foi
                foi_to_plot.plot(
                    ax=axes[i],
                    cmap='viridis',
                    vmin=0,
                    vmax=1,
                    add_colorbar=True,
                    cbar_kwargs={'label': 'FOI'}
                )
                axes[i].set_title(f'FOI - {date}')
                axes[i].set_xlabel('Longitude')
                axes[i].set_ylabel('Latitude')
            
            plt.tight_layout()
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comparison plot saved to {png_filename}")
            return str(png_filename)
            
        except Exception as e:
            logger.error(f"Error creating comparison plot: {e}")
            raise
    
    def create_statistics_plot(self, foi: xr.DataArray, date: str = None,
                             output_dir: Optional[Path] = None) -> str:
        """
        Create statistics plot for FOI data
        
        Args:
            foi: FOI DataArray
            date: Date for the visualization
            output_dir: Output directory (optional)
            
        Returns:
            Path to generated PNG file
        """
        logger.info("Creating statistics plot...")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir / 'visualizations'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        png_filename = output_dir / f'foi_statistics_{date}.png'
        
        try:
            # Get valid data
            valid_data = foi.values[~np.isnan(foi.values)]
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Histogram
            ax1.hist(valid_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_xlabel('FOI Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('FOI Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(valid_data, vert=True)
            ax2.set_ylabel('FOI Value')
            ax2.set_title('FOI Box Plot')
            ax2.grid(True, alpha=0.3)
            
            # Spatial statistics
            foi_mean = foi.mean(dim=['lat', 'lon'])
            foi_std = foi.std(dim=['lat', 'lon'])
            
            if 'time' in foi.dims:
                ax3.plot(foi.time.values, foi_mean.values, 'b-', linewidth=2)
                ax3.fill_between(foi.time.values, 
                               foi_mean.values - foi_std.values,
                               foi_mean.values + foi_std.values,
                               alpha=0.3)
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Mean FOI')
                ax3.set_title('Temporal FOI Variation')
                ax3.grid(True, alpha=0.3)
            else:
                # Latitudinal profile
                lat_profile = foi.mean(dim='lon')
                ax3.plot(lat_profile.lat.values, lat_profile.values, 'b-', linewidth=2)
                ax3.set_xlabel('Latitude')
                ax3.set_ylabel('Mean FOI')
                ax3.set_title('Latitudinal FOI Profile')
                ax3.grid(True, alpha=0.3)
            
            # Hotspot statistics
            thresholds = [0.3, 0.5, 0.7]
            hotspot_counts = []
            hotspot_percentages = []
            
            for threshold in thresholds:
                count = np.sum(valid_data >= threshold)
                percentage = (count / len(valid_data)) * 100
                hotspot_counts.append(count)
                hotspot_percentages.append(percentage)
            
            bars = ax4.bar([f'>{t}' for t in thresholds], hotspot_percentages, 
                          color=['yellow', 'orange', 'red'], alpha=0.7)
            ax4.set_ylabel('Percentage of Area')
            ax4.set_title('Hotspot Coverage')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars, hotspot_counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}\npixels', ha='center', va='bottom')
            
            plt.suptitle(f'FOI Statistics - {date}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Statistics plot saved to {png_filename}")
            return str(png_filename)
            
        except Exception as e:
            logger.error(f"Error creating statistics plot: {e}")
            raise

    def generate_enhanced_dashboard(self, start_date: str, end_date: str, 
                                   regional_bbox: list = None,
                                   output_dir: Optional[Path] = None) -> str:
        """
        Generate enhanced dashboard with global FOI computation and regional focus
        
        Args:
            start_date: Start date for computation
            end_date: End date for computation
            regional_bbox: Regional bounding box (default: Indonesia-Malaysia)
            output_dir: Output directory (optional)
            
        Returns:
            Path to generated dashboard HTML file
        """
        if regional_bbox is None:
            regional_bbox = [94.0, -11.0, 142.0, 6.0]  # Indonesia-Malaysia
        
        logger.info("Generating enhanced dashboard with global FOI...")
        
        # Compute global FOI
        global_foi = self.compute_global_foi(start_date, end_date)
        
        # Compute regional FOI for detailed view
        from foi_backend.shark_hotspots.predictor import compute_foi_map
        regional_foi, _ = compute_foi_map(start_date, end_date, regional_bbox, save_output=False)
        
        # Generate dashboard with regional FOI but global context
        dashboard_path = self.generate_dashboard(regional_foi, start_date, output_dir)
        
        logger.info(f"Enhanced dashboard saved to {dashboard_path}")
        return dashboard_path


def generate_dashboard(foi: xr.DataArray, date: str = None,
                      output_dir: Optional[Path] = None,
                      config: Optional[Dict] = None) -> str:
    """
    Convenience function to generate interactive dashboard
    
    Args:
        foi: FOI DataArray
        date: Date for the visualization
        output_dir: Output directory (optional)
        config: Configuration dictionary
        
    Returns:
        Path to generated HTML file
    """
    visualizer = HotspotVisualizer(config)
    return visualizer.generate_dashboard(foi, date, output_dir)


def create_static_snapshot(foi: xr.DataArray, date: str = None,
                          output_dir: Optional[Path] = None,
                          config: Optional[Dict] = None) -> str:
    """
    Convenience function to create static snapshot
    
    Args:
        foi: FOI DataArray
        date: Date for the visualization
        output_dir: Output directory (optional)
        config: Configuration dictionary
        
    Returns:
        Path to generated PNG file
    """
    visualizer = HotspotVisualizer(config)
    return visualizer.create_static_snapshot(foi, date, output_dir)


if __name__ == "__main__":
    """Test the visualizer with sample data"""
    from foi_backend.shark_hotspots.predictor import compute_foi_map
    
    # Test configuration
    bbox = [94.0, -11.0, 142.0, 6.0]  # Indonesia-Malaysia region
    start_date = "2025-03-01"
    end_date = "2025-03-14"
    
    logger.info("Computing test FOI map...")
    foi, summary = compute_foi_map(start_date, end_date, bbox)
    
    # Test visualization
    visualizer = HotspotVisualizer()
    
    # Create static snapshot
    snapshot_path = visualizer.create_static_snapshot(foi, "20250301")
    logger.info(f"Static snapshot created: {snapshot_path}")
    
    # Create statistics plot
    stats_path = visualizer.create_statistics_plot(foi, "20250301")
    logger.info(f"Statistics plot created: {stats_path}")
    
    # Create interactive dashboard (if folium available)
    if FOLIUM_AVAILABLE:
        dashboard_path = visualizer.generate_dashboard(foi, "20250301")
        logger.info(f"Interactive dashboard created: {dashboard_path}")
    else:
        logger.warning("Folium not available. Interactive dashboard not created.")
