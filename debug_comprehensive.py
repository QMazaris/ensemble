#!/usr/bin/env python3

"""Comprehensive debug script to find the root cause of data export issues."""

import sys
import os
import requests
import time
import traceback
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_service_singleton():
    """Test if data service singleton works correctly."""
    print("🧪 Testing Data Service Singleton...")
    
    try:
        # Import from different contexts
        from shared import data_service as ds1
        from backend.run import data_service as ds2
        from backend.api.main import data_service as ds3
        
        print(f"   Shared data_service ID: {id(ds1)}")
        print(f"   Backend data_service ID: {id(ds2)}")
        print(f"   API data_service ID: {id(ds3)}")
        
        # Check if they're the same instance
        same_instance = (id(ds1) == id(ds2) == id(ds3))
        print(f"   Same singleton instance: {same_instance}")
        
        if not same_instance:
            print("   ❌ PROBLEM: Different data service instances!")
            return False
        
        # Test basic functionality
        print("   Testing basic functionality...")
        ds1.set_metrics_data({"test": "data"})
        
        # Check if other instances see the data
        data_from_ds2 = ds2.get_metrics_data()
        data_from_ds3 = ds3.get_metrics_data()
        
        print(f"   DS1 set data, DS2 can read: {data_from_ds2 is not None}")
        print(f"   DS1 set data, DS3 can read: {data_from_ds3 is not None}")
        
        ds1.clear_all_data()
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False

def test_api_server():
    """Test if API server is running and accessible."""
    print("\n🧪 Testing API Server...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"   Health check status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ API server is running")
            return True
        else:
            print(f"   ❌ API server returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Cannot connect to API server: {e}")
        return False

def test_pipeline_execution():
    """Test running the pipeline directly."""
    print("\n🧪 Testing Direct Pipeline Execution...")
    
    try:
        # Import and run pipeline
        from backend.run import main, data_service
        
        print("   Checking data service before pipeline...")
        print(f"   Data service has data: {data_service.has_data()}")
        print(f"   Data keys: {list(data_service._data.keys())}")
        
        # Mock a simple config for testing
        test_config = {
            "data": {"path": "data/training_data.csv", "target_column": "GT_Label"},
            "models": {"enabled": ["XGBoost"], "base_model_columns": []},
            "training": {"n_splits": 2},  # Smaller for faster testing
            "costs": {"false_positive": 1.0, "false_negative": 10.0},
            "output": {"base_dir": "output", "subdirs": {"models": "models", "plots": "plots"}},
            "export": {"save_predictions": True, "save_models": False},
            "logging": {"summary": True},
            "model_params": {"XGBoost": {"n_estimators": 10, "max_depth": 3}}  # Small for speed
        }
        
        print("   Running pipeline with test config...")
        main(test_config)
        
        print("   Checking data service after pipeline...")
        print(f"   Data service has data: {data_service.has_data()}")
        print(f"   Data keys: {list(data_service._data.keys())}")
        
        # Check specific data
        metrics = data_service.get_metrics_data()
        predictions = data_service.get_predictions_data()
        sweep = data_service.get_sweep_data()
        
        print(f"   Metrics available: {metrics is not None}")
        print(f"   Predictions available: {predictions is not None}")
        print(f"   Sweep data available: {sweep is not None}")
        
        if metrics:
            print(f"   Metrics count: {len(metrics.get('model_metrics', []))}")
        
        return data_service.has_data()
        
    except Exception as e:
        print(f"   ❌ Pipeline execution failed: {e}")
        traceback.print_exc()
        return False

def test_api_pipeline_execution():
    """Test running pipeline via API."""
    print("\n🧪 Testing Pipeline Execution via API...")
    
    try:
        # Start pipeline via API
        response = requests.post("http://localhost:8000/pipeline/run", timeout=10)
        print(f"   Pipeline start response: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ❌ Failed to start pipeline: {response.text}")
            return False
        
        # Wait for pipeline to complete
        print("   Waiting for pipeline to complete...")
        max_wait = 300  # 5 minutes max
        wait_time = 0
        
        while wait_time < max_wait:
            try:
                status_response = requests.get("http://localhost:8000/pipeline/status")
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"   Pipeline status: {status.get('status')} - {status.get('message')}")
                    
                    if status.get('status') == 'completed':
                        print("   ✅ Pipeline completed successfully")
                        break
                    elif status.get('status') == 'failed':
                        print(f"   ❌ Pipeline failed: {status.get('message')}")
                        return False
                
                time.sleep(5)
                wait_time += 5
                
            except Exception as e:
                print(f"   Error checking status: {e}")
                time.sleep(5)
                wait_time += 5
        
        if wait_time >= max_wait:
            print("   ❌ Pipeline timed out")
            return False
        
        # Check if data is available via API
        print("   Checking data availability via API...")
        
        try:
            metrics_response = requests.get("http://localhost:8000/results/metrics")
            print(f"   Metrics endpoint status: {metrics_response.status_code}")
            
            predictions_response = requests.get("http://localhost:8000/results/predictions")
            print(f"   Predictions endpoint status: {predictions_response.status_code}")
            
            debug_response = requests.get("http://localhost:8000/debug/data-service")
            print(f"   Debug endpoint status: {debug_response.status_code}")
            
            if debug_response.status_code == 200:
                debug_data = debug_response.json()
                print(f"   Debug info: {debug_data}")
            
            return metrics_response.status_code == 200 and predictions_response.status_code == 200
            
        except Exception as e:
            print(f"   Error checking API endpoints: {e}")
            return False
        
    except Exception as e:
        print(f"   ❌ API pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_import_issues():
    """Test for potential import issues."""
    print("\n🧪 Testing Import Issues...")
    
    try:
        # Test importing data service in different ways
        print("   Testing different import patterns...")
        
        # Import 1: Direct import
        from shared.data_service import DataService, data_service
        ds1 = data_service
        print(f"   Direct import ID: {id(ds1)}")
        
        # Import 2: Module import
        import shared.data_service as ds_module
        ds2 = ds_module.data_service
        print(f"   Module import ID: {id(ds2)}")
        
        # Import 3: From backend context
        import backend.run
        # The backend.run module imports data_service, let's see what it got
        ds3 = backend.run.data_service
        print(f"   Backend context ID: {id(ds3)}")
        
        # Check if all are the same
        same_instances = (id(ds1) == id(ds2) == id(ds3))
        print(f"   All imports same instance: {same_instances}")
        
        if not same_instances:
            print("   ❌ PROBLEM: Different instances from different imports!")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_export_function_directly():
    """Test the export function directly with mock data."""
    print("\n🧪 Testing Export Function Directly...")
    
    try:
        from backend.run import export_data_to_api_memory, data_service
        from backend.helpers import ModelEvaluationResult, ModelEvaluationRun
        import pandas as pd
        import numpy as np
        
        # Create mock data
        print("   Creating mock data...")
        
        # Mock ModelEvaluationResult
        mock_result = ModelEvaluationResult(
            model_name="MockModel",
            split="Full",
            threshold_type="cost",
            threshold=0.5,
            precision=0.8,
            recall=0.7,
            f1_score=0.75,
            accuracy=0.85,
            cost=100.0,
            tp=10,
            fp=5,
            tn=15,
            fn=3,
            is_base_model=False
        )
        
        # Mock ModelEvaluationRun
        mock_run = ModelEvaluationRun(
            model_name="MockModel",
            results=[mock_result],
            probabilities={'Full': np.array([0.1, 0.3, 0.7, 0.9, 0.2])},
            sweep_data={'Full': {
                0.3: {'cost': 120.0, 'accuracy': 0.8, 'f1_score': 0.7},
                0.5: {'cost': 100.0, 'accuracy': 0.85, 'f1_score': 0.75},
                0.7: {'cost': 110.0, 'accuracy': 0.82, 'f1_score': 0.72}
            }}
        )
        
        # Mock dataframe and target
        mock_df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        mock_df.index = [0, 1, 2, 3, 4]
        mock_y = pd.Series([0, 1, 1, 0, 1], index=[0, 1, 2, 3, 4])
        
        print("   Calling export function...")
        export_data_to_api_memory([mock_run], mock_df, mock_y, meta_model_names={"MockModel"})
        
        # Check if data was stored
        print("   Checking stored data...")
        stored_metrics = data_service.get_metrics_data()
        stored_predictions = data_service.get_predictions_data()
        stored_sweep = data_service.get_sweep_data()
        
        print(f"   Metrics stored: {stored_metrics is not None}")
        print(f"   Predictions stored: {stored_predictions is not None}")
        print(f"   Sweep data stored: {stored_sweep is not None}")
        
        if stored_metrics:
            print(f"   Metrics details: {len(stored_metrics.get('model_metrics', []))} metrics")
        
        success = all([stored_metrics, stored_predictions, stored_sweep])
        
        if success:
            print("   ✅ Export function works correctly")
        else:
            print("   ❌ Export function failed to store data")
        
        # Clear test data
        data_service.clear_all_data()
        
        return success
        
    except Exception as e:
        print(f"   ❌ Export function test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive debugging tests."""
    print("🔧 Comprehensive System Debug")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Data Service Singleton", test_data_service_singleton),
        ("API Server", test_api_server),
        ("Import Issues", test_import_issues),
        ("Export Function", test_export_function_directly),
        ("Direct Pipeline", test_pipeline_execution),
        ("API Pipeline", test_api_pipeline_execution),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("🔍 DEBUG SUMMARY")
    print("="*50)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    # Analysis
    print("\n🧠 ANALYSIS:")
    
    all_passed = all(results.values())
    if all_passed:
        print("   All tests passed - the issue may be with process isolation or timing")
    else:
        failed_tests = [name for name, success in results.items() if not success]
        print(f"   Failed tests: {', '.join(failed_tests)}")
        
        if not results.get("Data Service Singleton", True):
            print("   🚨 CRITICAL: Data service singleton is broken!")
        if not results.get("API Server", True):
            print("   🚨 CRITICAL: API server is not running!")
        if not results.get("Export Function", True):
            print("   🚨 CRITICAL: Export function is not working!")

if __name__ == "__main__":
    main()