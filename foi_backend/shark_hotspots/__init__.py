"""
Shark Hotspots Module - Core functionality for FOI computation
"""

from .predictor import compute_foi_map, compute_cps_map
from .visualize_hotspots import generate_dashboard, create_static_snapshot

__all__ = ['compute_foi_map', 'compute_cps_map', 'generate_dashboard', 'create_static_snapshot']
