#!/usr/bin/env python3
"""
Test script for Temporal Synchronization Enforcement
Verifies that the temporal synchronization amendment works correctly
"""

import sys
import os
from pathlib import Path

# Add the foi_backend to Python path
sys.path.insert(0, str(Path(__file__).parent / 'foi_backend'))
sys.path.insert(0, str(Path(__file__).parent))

def test_temporal_validation():
    """Test temporal overlap validation"""
    print("Testing temporal overlap validation...")
    
    try:
        from foi_backend.shark_hotspots.data_loader import SatelliteDataLoader
        
        loader = SatelliteDataLoader()
        
        # Test with valid time window
        bbox = [94.0, -11.0, 142.0, 6.0]
        start_date = "2025-03-01"
        end_date = "2025-03-14"
        
        datasets = loader.load_satellite_data(start_date, end_date, bbox)
        
        print(f"✅ Temporal validation passed for {len(datasets)} datasets")
        for name, ds in datasets.items():
            if 'time' in ds.dims:
                print(f"   - {name}: {ds.time.min().values} → {ds.time.max().values}")
        
        return True
    except Exception as e:
        print(f"❌ Temporal validation error: {e}")
        return False

def test_time_synchronization():
    """Test time axis synchronization"""
    print("\nTesting time axis synchronization...")
    
    try:
        from foi_backend.shark_hotspots.data_loader import load_satellite_data
        from foi_backend.shark_hotspots.harmonize import DataHarmonizer
        
        # Load test data
        bbox = [94.0, -11.0, 142.0, 6.0]
        start_date = "2025-03-01"
        end_date = "2025-03-14"
        
        datasets = load_satellite_data(bbox, start_date, end_date)
        
        # Test synchronization
        harmonizer = DataHarmonizer()
        synchronized = harmonizer.synchronize_time_axes(datasets)
        
        print(f"✅ Time synchronization completed for {len(synchronized)} datasets")
        
        # Check that all datasets have same time points
        time_lengths = []
        for name, ds in synchronized.items():
            if 'time' in ds.dims:
                time_length = len(ds.time)
                time_lengths.append(time_length)
                print(f"   - {name}: {time_length} time points")
        
        # Verify all datasets have same number of time points
        if len(set(time_lengths)) == 1:
            print("✅ All datasets synchronized to same time axis")
            return True
        else:
            print(f"❌ Time axis mismatch: {time_lengths}")
            return False
            
    except Exception as e:
        print(f"❌ Time synchronization error: {e}")
        return False

def test_composite_modes():
    """Test different composite modes"""
    print("\nTesting composite modes...")
    
    try:
        from foi_backend.shark_hotspots.data_loader import load_satellite_data
        from foi_backend.shark_hotspots.harmonize import harmonize_datasets
        
        # Load test data
        bbox = [94.0, -11.0, 142.0, 6.0]
        start_date = "2025-03-01"
        end_date = "2025-03-14"
        
        datasets = load_satellite_data(bbox, start_date, end_date)
        
        # Test different composite modes
        modes = ['mean', 'median', 'rolling']
        
        for mode in modes:
            print(f"   Testing {mode} composite mode...")
            harmonized = harmonize_datasets(
                datasets, bbox, start_date, end_date, 
                composite_days=7, composite_mode=mode
            )
            
            # Check that metadata includes composite mode
            for name, ds in harmonized.items():
                if 'temporal_composite_mode' in ds.attrs:
                    if ds.attrs['temporal_composite_mode'] == mode:
                        print(f"   ✅ {mode} mode verified for {name}")
                    else:
                        print(f"   ❌ {mode} mode mismatch for {name}")
                        return False
        
        print("✅ All composite modes working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Composite modes error: {e}")
        return False

def test_foi_with_temporal_sync():
    """Test FOI computation with temporal synchronization"""
    print("\nTesting FOI computation with temporal synchronization...")
    
    try:
        from foi_backend.shark_hotspots.predictor import compute_foi_map
        
        bbox = [94.0, -11.0, 142.0, 6.0]
        start_date = "2025-03-01"
        end_date = "2025-03-14"
        
        foi, summary = compute_foi_map(start_date, end_date, bbox, save_output=False)
        
        # Check temporal metadata
        if 'time_window' in foi.attrs:
            print(f"✅ FOI includes time window: {foi.attrs['time_window']}")
        else:
            print("❌ FOI missing time window metadata")
            return False
        
        if 'temporal_synchronization' in foi.attrs:
            print(f"✅ FOI includes synchronization info: {foi.attrs['temporal_synchronization']}")
        else:
            print("❌ FOI missing synchronization metadata")
            return False
        
        if 'composite_mode' in foi.attrs:
            print(f"✅ FOI includes composite mode: {foi.attrs['composite_mode']}")
        else:
            print("❌ FOI missing composite mode metadata")
            return False
        
        print(f"✅ FOI computation successful with temporal sync")
        print(f"   FOI range: [{foi.min().values:.4f}, {foi.max().values:.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ FOI computation error: {e}")
        return False

def test_config_composite_mode():
    """Test configuration file composite mode"""
    print("\nTesting configuration composite mode...")
    
    try:
        import yaml
        
        config_path = "foi_backend/shark_hotspots/config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'composite_mode' in config.get('data', {}):
            mode = config['data']['composite_mode']
            print(f"✅ Config includes composite_mode: {mode}")
            
            if mode in ['mean', 'median', 'rolling']:
                print(f"✅ Valid composite mode: {mode}")
                return True
            else:
                print(f"❌ Invalid composite mode: {mode}")
                return False
        else:
            print("❌ Config missing composite_mode")
            return False
            
    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False

def main():
    """Run all temporal synchronization tests"""
    print("🕒 Temporal Synchronization Enforcement - Test Suite")
    print("=" * 60)
    
    tests = [
        test_config_composite_mode,
        test_temporal_validation,
        test_time_synchronization,
        test_composite_modes,
        test_foi_with_temporal_sync
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Temporal Sync Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All temporal synchronization tests passed!")
        print("\n✅ Amendment Implementation Complete:")
        print("   - Temporal intersection alignment enforced")
        print("   - Composite mode configuration added")
        print("   - Validation checks implemented")
        print("   - Metadata provenance included")
        print("   - Synchronization step integrated")
        return 0
    else:
        print("❌ Some temporal synchronization tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
