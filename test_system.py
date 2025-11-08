#!/usr/bin/env python3
"""
Test script for Sharks from Space system
Verifies that all modules work correctly with synthetic data
"""

import sys
import os
from pathlib import Path

# Add the foi_backend to Python path
sys.path.insert(0, str(Path(__file__).parent / 'foi_backend'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from foi_backend.shark_hotspots.data_loader import SatelliteDataLoader
        from foi_backend.shark_hotspots.harmonize import DataHarmonizer
        from foi_backend.shark_hotspots.derived_fields import DerivedFieldsComputer
        from foi_backend.shark_hotspots.normalization import DataNormalizer
        from foi_backend.shark_hotspots.model_core import FOIModel
        from foi_backend.shark_hotspots.predictor import SharkHotspotPredictor
        from foi_backend.shark_hotspots.visualize_hotspots import HotspotVisualizer
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    
    try:
        from foi_backend.shark_hotspots.data_loader import load_satellite_data
        
        bbox = [94.0, -11.0, 142.0, 6.0]
        start_date = "2025-03-01"
        end_date = "2025-03-14"
        
        datasets = load_satellite_data(bbox, start_date, end_date)
        
        print(f"✅ Loaded {len(datasets)} datasets:")
        for name, ds in datasets.items():
            print(f"   - {name}: {ds[list(ds.data_vars)[0]].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_harmonization():
    """Test data harmonization"""
    print("\nTesting harmonization...")
    
    try:
        from foi_backend.shark_hotspots.data_loader import load_satellite_data
        from foi_backend.shark_hotspots.harmonize import harmonize_datasets
        
        bbox = [94.0, -11.0, 142.0, 6.0]
        start_date = "2025-03-01"
        end_date = "2025-03-14"
        
        datasets = load_satellite_data(bbox, start_date, end_date)
        harmonized = harmonize_datasets(datasets, bbox, start_date, end_date)
        
        print(f"✅ Harmonized {len(harmonized)} datasets")
        for name, ds in harmonized.items():
            print(f"   - {name}: {ds[list(ds.data_vars)[0]].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Harmonization error: {e}")
        return False

def test_derived_fields():
    """Test derived field computation"""
    print("\nTesting derived fields...")
    
    try:
        from foi_backend.shark_hotspots.data_loader import load_satellite_data
        from foi_backend.shark_hotspots.harmonize import harmonize_datasets
        from foi_backend.shark_hotspots.derived_fields import compute_derived_fields
        
        bbox = [94.0, -11.0, 142.0, 6.0]
        start_date = "2025-03-01"
        end_date = "2025-03-14"
        
        datasets = load_satellite_data(bbox, start_date, end_date)
        harmonized = harmonize_datasets(datasets, bbox, start_date, end_date)
        derived = compute_derived_fields(harmonized)
        
        print(f"✅ Computed {len(derived)} derived fields:")
        for name, field in derived.items():
            print(f"   - {name}: {field.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Derived fields error: {e}")
        return False

def test_foi_computation():
    """Test FOI computation"""
    print("\nTesting FOI computation...")
    
    try:
        from foi_backend.shark_hotspots.predictor import compute_foi_map
        
        bbox = [94.0, -11.0, 142.0, 6.0]
        start_date = "2025-03-01"
        end_date = "2025-03-14"
        
        foi, summary = compute_foi_map(start_date, end_date, bbox, save_output=False)
        
        print(f"✅ FOI computation successful:")
        print(f"   - FOI range: [{foi.min().values:.4f}, {foi.max().values:.4f}]")
        print(f"   - FOI mean: {foi.mean().values:.4f}")
        print(f"   - Spatial coverage: {summary['spatial_coverage']['coverage_percentage']:.2f}%")
        
        return True
    except Exception as e:
        print(f"❌ FOI computation error: {e}")
        return False

def test_visualization():
    """Test visualization generation"""
    print("\nTesting visualization...")
    
    try:
        from foi_backend.shark_hotspots.predictor import compute_foi_map
        from foi_backend.shark_hotspots.visualize_hotspots import HotspotVisualizer
        
        bbox = [94.0, -11.0, 142.0, 6.0]
        start_date = "2025-03-01"
        end_date = "2025-03-14"
        
        foi, summary = compute_foi_map(start_date, end_date, bbox, save_output=False)
        
        visualizer = HotspotVisualizer()
        
        # Test static snapshot
        snapshot_path = visualizer.create_static_snapshot(foi, "20250301")
        print(f"✅ Static snapshot: {snapshot_path}")
        
        # Test statistics plot
        stats_path = visualizer.create_statistics_plot(foi, "20250301")
        print(f"✅ Statistics plot: {stats_path}")
        
        return True
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def main():
    """Run all tests"""
    print("🦈 Sharks from Space - System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_loading,
        test_harmonization,
        test_derived_fields,
        test_foi_computation,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
