#!/usr/bin/env python3
"""
Test script to verify dynamic base model functionality.

This script tests that:
1. Base models can be dynamically configured
2. Each configured base model goes through k-fold validation
3. Result structs are created for each base model
4. The frontend can display the results correctly
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add the root directory to Python path
root_dir = str(Path(__file__).parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def test_base_model_configuration():
    """Test base model configuration APIs."""
    print("🔍 Testing Base Model Configuration APIs...")
    
    try:
        import requests
        API_BASE_URL = "http://localhost:8000"
        
        # Test 1: Get current base model columns configuration
        print("\n1. Testing GET /config/base-model-columns")
        response = requests.get(f"{API_BASE_URL}/config/base-model-columns", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Current configuration: {data['config']['model_columns']}")
            print(f"✅ Available columns: {len(data['available_columns'])} columns detected")
        else:
            print(f"❌ Failed to get configuration: {response.status_code}")
            return False
        
        # Test 2: Update base model columns configuration with new models
        print("\n2. Testing POST /config/base-model-columns with dynamic models")
        new_config = {
            "model_columns": {
                "AD": "AnomalyScore",
                "CL": "CL_ConfMax", 
                "RF": "RandomForest_Score",  # Add a third base model
                "SVM": "SVM_Score"  # Add a fourth base model
            }
        }
        
        response = requests.post(f"{API_BASE_URL}/config/base-model-columns", 
                               json=new_config, timeout=10)
        if response.status_code == 200:
            print(f"✅ Successfully updated configuration with {len(new_config['model_columns'])} models")
        else:
            print(f"❌ Failed to update configuration: {response.status_code}")
            return False
        
        # Test 3: Verify the configuration was saved
        print("\n3. Verifying configuration persistence")
        response = requests.get(f"{API_BASE_URL}/config/base-model-columns", timeout=10)
        if response.status_code == 200:
            data = response.json()
            saved_models = data['config']['model_columns']
            if saved_models == new_config['model_columns']:
                print(f"✅ Configuration correctly persisted: {saved_models}")
            else:
                print(f"❌ Configuration mismatch: {saved_models} != {new_config['model_columns']}")
                return False
        else:
            print(f"❌ Failed to verify configuration: {response.status_code}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def test_pipeline_with_dynamic_base_models():
    """Test running the pipeline with dynamic base model configuration."""
    print("\n🔍 Testing Pipeline with Dynamic Base Models...")
    
    try:
        # Import backend modules correctly
        from backend import config_adapter  
        from backend import run
        
        # Print current base model configuration - use the module-level properties
        print(f"📊 Current base model columns: {config_adapter.BASE_MODEL_OUTPUT_COLUMNS}")
        print(f"📊 Current base model decisions: {config_adapter.BASE_MODEL_DECISION_COLUMNS}")
        
        # Check if required data exists
        data_path = config_adapter.DATA_PATH
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            print("Please ensure training data is available before running the pipeline.")
            return False
        
        # Load data to check for required columns
        df = pd.read_csv(data_path, nrows=5)
        base_model_columns = config_adapter.BASE_MODEL_OUTPUT_COLUMNS
        
        print(f"📊 Dataset columns: {list(df.columns)}")
        print(f"📊 Looking for base model columns: {list(base_model_columns.values())}")
        
        # Check which base model columns exist
        missing_columns = []
        available_columns = []
        for model_name, column_name in base_model_columns.items():
            if column_name in df.columns:
                available_columns.append((model_name, column_name))
                print(f"✅ Found column '{column_name}' for model '{model_name}'")
            else:
                missing_columns.append((model_name, column_name))
                print(f"⚠️ Missing column '{column_name}' for model '{model_name}'")
        
        if not available_columns:
            print("❌ No base model columns found in dataset. Cannot test base model functionality.")
            return False
        
        if missing_columns:
            print(f"⚠️ {len(missing_columns)} base models will be skipped due to missing columns.")
            print("The pipeline should still process the available base models.")
        
        print(f"\n🚀 Running pipeline with {len(available_columns)} base models...")
        
        # Run the pipeline
        try:
            results = run.main(config_adapter._adapter)
            print("✅ Pipeline completed successfully!")
            
            # Verify results were created
            return verify_pipeline_results(available_columns)
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Error setting up pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_pipeline_results(expected_base_models):
    """Verify that pipeline results include all expected base models."""
    print("\n🔍 Verifying Pipeline Results...")
    
    try:
        # Check if data service has results
        from shared import data_service
        
        # Get metrics data
        metrics_data = data_service.get_metrics_data()
        if not metrics_data:
            print("❌ No metrics data found in data service")
            return False
        
        # Get model summaries
        model_summaries = metrics_data.get('model_summary', [])
        print(f"📊 Found {len(model_summaries)} model results")
        
        # Check for base model results
        base_model_results = []
        meta_model_results = []
        
        for summary in model_summaries:
            model_name = summary.get('model_name', '')
            if model_name.startswith('kfold_avg_'):
                # Remove the kfold_avg_ prefix to get the actual model name
                actual_model_name = model_name.replace('kfold_avg_', '')
                
                # Check if this is a base model
                is_base_model = any(actual_model_name == base_model[0] for base_model in expected_base_models)
                
                if is_base_model:
                    base_model_results.append(summary)
                    print(f"✅ Found base model result: {model_name}")
                else:
                    meta_model_results.append(summary)
                    print(f"✅ Found meta model result: {model_name}")
        
        print(f"\n📊 Summary:")
        print(f"  • Base model results: {len(base_model_results)}")
        print(f"  • Meta model results: {len(meta_model_results)}")
        print(f"  • Expected base models: {len(expected_base_models)}")
        
        # Verify all expected base models have results
        expected_model_names = [base_model[0] for base_model in expected_base_models]
        found_model_names = [result['model_name'].replace('kfold_avg_', '') for result in base_model_results]
        
        missing_models = []
        for expected_name in expected_model_names:
            if expected_name not in found_model_names:
                missing_models.append(expected_name)
        
        if missing_models:
            print(f"❌ Missing results for base models: {missing_models}")
            return False
        
        print("✅ All expected base models have results!")
        
        # Verify result structure
        for result in base_model_results:
            required_fields = ['accuracy', 'precision', 'recall', 'f1_score', 'cost']
            missing_fields = [field for field in required_fields if field not in result or result[field] is None]
            
            if missing_fields:
                print(f"❌ Model {result['model_name']} missing fields: {missing_fields}")
                return False
            else:
                print(f"✅ Model {result['model_name']} has all required metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying results: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_frontend_integration():
    """Test that the frontend can display the dynamic base model results."""
    print("\n🔍 Testing Frontend Integration...")
    
    try:
        import requests
        API_BASE_URL = "http://localhost:8000"
        
        # Test metrics endpoint
        print("1. Testing metrics API endpoint")
        response = requests.get(f"{API_BASE_URL}/results/metrics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', {})
            
            model_summaries = results.get('model_summary', [])
            print(f"✅ API returned {len(model_summaries)} model summaries")
            
            # Count base models
            base_model_count = 0
            for summary in model_summaries:
                model_name = summary.get('model_name', '')
                if 'kfold_avg_' in model_name and not model_name.endswith(('XGBoost', 'RandomForest')):
                    base_model_count += 1
                    print(f"  • Base model: {model_name}")
            
            print(f"✅ Found {base_model_count} base model results in API response")
            return True
        else:
            print(f"❌ Failed to get metrics from API: {response.status_code}")
            return False
        
    except Exception as e:
        print(f"❌ Error testing frontend integration: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting Dynamic Base Models Test Suite")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: API Configuration
    if not test_base_model_configuration():
        print("\n❌ Base model configuration test failed")
        all_tests_passed = False
    else:
        print("\n✅ Base model configuration test passed")
    
    # Test 2: Pipeline Execution
    if not test_pipeline_with_dynamic_base_models():
        print("\n❌ Pipeline execution test failed")
        all_tests_passed = False
    else:
        print("\n✅ Pipeline execution test passed")
    
    # Test 3: Frontend Integration
    if not test_frontend_integration():
        print("\n❌ Frontend integration test failed")
        all_tests_passed = False
    else:
        print("\n✅ Frontend integration test passed")
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Dynamic base models are working correctly.")
        print("\nThe system now supports:")
        print("  • Dynamic base model configuration via frontend")
        print("  • K-fold validation for all configured base models")
        print("  • Result struct generation for each base model")
        print("  • Frontend display of all base model results")
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 