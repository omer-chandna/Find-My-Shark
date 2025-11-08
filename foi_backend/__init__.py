"""
Sharks from Space - Foraging Opportunity Index Backend
NASA Space Apps 2025 - Indonesia-Malaysia Team

A modular Python backend for ingesting NASA satellite data and generating
shark foraging hotspot predictions using multi-satellite temporal alignment.
"""

__version__ = "1.0.0"
__author__ = "NASA Space Apps 2025 - Indonesia-Malaysia Team"

from .shark_hotspots import predictor, visualize_hotspots

__all__ = ['predictor', 'visualize_hotspots']
