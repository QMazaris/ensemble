#!/usr/bin/env python3
"""
Comprehensive pipeline flow debugging script for the new simplified architecture.
This script tests the data flow where pipeline results are stored directly in API memory.
"""

import sys
import os
import time
import requests
import traceback
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

def test_api_connection():
    """Test if API is running and accessible."""
    print("🔗 Testing API connection...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and accessible")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def test_api_memory_storage():
    """Test API memory storage system."""
    print("\n🧪 Testing API memory storage...")
    
    try:
        # Test sample data storage
        response = requests.post("http://localhost:8000/test/store-sample-data", timeout=10)
        if response.status_code == 200:
            print("✅ Sample data storage successful")
            
            # Test data retrieval
            response = requests.get("http://localhost:8000/results/metrics", timeout=5)
            if response.status_code == 200:
                print("✅ Data retrieval successful")
                return True
            else:
                print(f"❌ Data retrieval failed: {response.status_code}")
                return False
        else:
            print(f"❌ Sample data storage failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API memory storage test failed: {e}")
        return False

def test_pipeline_execution():
    """Test pipeline execution through API."""
    print("\n🚀 Testing pipeline execution...")
    
    try:
        # Clear any existing data first
        print("🗑️ Clearing existing data...")
        requests.delete("http://localhost:8000/data/clear")
        
        # Run pipeline
        print("⚙️ Running pipeline...")
        response = requests.post("http://localhost:8000/pipeline/run", timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Pipeline executed successfully")
            print(f"   Status: {result.get('status', {}).get('status', 'unknown')}")
            print(f"   Message: {result.get('status', {}).get('message', 'No message')}")
            
            # Check data summary from response
            data_summary = result.get('data_summary', {})
            print(f"   Data Summary:")
            print(f"     Metrics stored: {data_summary.get('metrics_stored', False)}")
            print(f"     Predictions stored: {data_summary.get('predictions_stored', False)}")
            print(f"     Sweep stored: {data_summary.get('sweep_stored', False)}")
            
            return data_summary.get('metrics_stored', False)
        else:
            print(f"❌ Pipeline execution failed: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Pipeline execution test failed: {e}")
        return False

def test_data_endpoints():
    """Test all data endpoint accessibility."""
    print("\n📊 Testing data endpoint accessibility...")
    
    endpoints = [
        ("/results/metrics", "Metrics"),
        ("/results/predictions", "Predictions"),
        ("/results/threshold-sweep?model_name=kfold_avg_XGBoost&data_type=costs", "Threshold Sweep"),
        ("/results/model-comparison", "Model Comparison")
    ]
    
    results = {}
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
            if response.status_code == 200:
                print(f"✅ {name} endpoint accessible")
                results[name] = True
            else:
                print(f"❌ {name} endpoint failed: {response.status_code}")
                results[name] = False
        except Exception as e:
            print(f"❌ {name} endpoint error: {e}")
            results[name] = False
    
    return all(results.values())

def test_debug_endpoints():
    """Test debug endpoints."""
    print("\n🔍 Testing debug endpoints...")
    
    try:
        response = requests.get("http://localhost:8000/debug/pipeline-data", timeout=5)
        if response.status_code == 200:
            debug_data = response.json()
            print("✅ Debug endpoint accessible")
            print(f"   Storage type: {debug_data.get('storage_type', 'unknown')}")
            print(f"   Data summary: {debug_data.get('data_summary', {})}")
            return True
        else:
            print(f"❌ Debug endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Debug endpoint error: {e}")
        return False

def main():
    """Run comprehensive tests."""
    print("=" * 60)
    print("🧪 COMPREHENSIVE PIPELINE FLOW TEST")
    print("   New Simplified Architecture: Data stored directly in API memory")
    print("=" * 60)
    
    tests = [
        ("api_connection", test_api_connection),
        ("api_memory_storage", test_api_memory_storage),
        ("pipeline_execution", test_pipeline_execution),
        ("data_endpoints", test_data_endpoints),
        ("debug_endpoints", test_debug_endpoints)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
    
    print("\n🔍 ANALYSIS:")
    all_passed = all(results.values())
    if all_passed:
        print("🎉 ALL TESTS PASSED! The simplified pipeline system is working correctly.")
        print("   ✓ Data is stored directly in API memory")
        print("   ✓ No intermediate files or singleton patterns")
        print("   ✓ Clean, simple data flow from pipeline → API memory → frontend")
    else:
        failed_tests = [name for name, result in results.items() if not result]
        print(f"❌ Some tests failed: {', '.join(failed_tests)}")
        
        if not results.get('api_connection'):
            print("   💡 Make sure the API server is running: uvicorn backend.api.main:app --reload")
        elif not results.get('pipeline_execution'):
            print("   💡 Check backend/run.py and ensure it returns results properly")
        elif not results.get('data_endpoints'):
            print("   💡 Check pipeline data storage in backend/api/main.py")

if __name__ == "__main__":
    main() 