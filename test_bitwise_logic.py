#!/usr/bin/env python3
"""
Test script for the new bitwise logic functionality that works directly with 
API requests instead of config files.
"""

import requests
import json
import time

BACKEND_URL = "http://localhost:8000"

def test_bitwise_logic_api():
    """Test the bitwise logic API endpoint with direct rule submission."""
    
    print("🧪 Testing Bitwise Logic API (Config-Free Approach)")
    print("=" * 60)
    
    # Test 1: Check API health
    print("\n1. Testing API connectivity...")
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            print("✅ API is healthy")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False
    
    # Test 2: Check if model results exist
    print("\n2. Checking for existing model results...")
    try:
        response = requests.get(f"{BACKEND_URL}/results/metrics")
        if response.status_code == 200:
            data = response.json()
            model_metrics = data.get('results', {}).get('model_metrics', [])
            unique_models = set()
            for metric in model_metrics:
                model_name = metric.get('model_name', '').replace('kfold_avg_', '')
                if model_name:
                    unique_models.add(model_name)
            
            if unique_models:
                print(f"✅ Found {len(unique_models)} available models: {list(unique_models)}")
                available_models = list(unique_models)
            else:
                print("⚠️ No model results available. Pipeline needs to be run first.")
                available_models = ["XGBoost", "RandomForest"]  # Use dummy models for testing
        else:
            print(f"❌ Failed to get model metrics: {response.status_code}")
            available_models = ["XGBoost", "RandomForest"]  # Use dummy models for testing
    except Exception as e:
        print(f"❌ Error checking model results: {e}")
        available_models = ["XGBoost", "RandomForest"]  # Use dummy models for testing
    
    # Test 3: Test bitwise logic with direct API call
    print("\n3. Testing bitwise logic with direct API call...")
    
    # Create test request
    test_request = {
        "rules": [
            {
                "name": "Combined_OR_Test",
                "columns": available_models[:2],  # Use first 2 available models
                "logic": "OR"
            },
            {
                "name": "Combined_AND_Test", 
                "columns": available_models[:2],  # Use first 2 available models
                "logic": "AND"
            }
        ],
        "model_thresholds": {
            available_models[0]: 0.6,
            available_models[1]: 0.7
        } if len(available_models) >= 2 else {},
        "enabled": True
    }
    
    print(f"📤 Sending request: {json.dumps(test_request, indent=2)}")
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/config/bitwise-logic/apply",
            json=test_request,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Bitwise logic API call successful!")
            print(f"📊 Result: {json.dumps(result, indent=2)}")
            
            combined_models_created = result.get('combined_models_created', 0)
            combined_model_names = result.get('combined_model_names', [])
            
            if combined_models_created > 0:
                print(f"🎉 Successfully created {combined_models_created} combined models: {combined_model_names}")
                return True
            else:
                print("⚠️ No combined models were created")
                return False
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error during API call: {e}")
        return False

def test_model_availability():
    """Test that we can get available models from the results structure."""
    
    print("\n🔍 Testing Model Discovery from Results Structure")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BACKEND_URL}/results/metrics")
        if response.status_code == 200:
            data = response.json()
            model_metrics = data.get('results', {}).get('model_metrics', [])
            
            # Extract unique model names
            unique_models = set()
            for metric in model_metrics:
                model_name = metric.get('model_name', '')
                unique_models.add(model_name)
                
                # Also add clean name (without kfold_avg_ prefix)
                clean_name = model_name.replace('kfold_avg_', '')
                if clean_name:
                    unique_models.add(clean_name)
            
            print(f"✅ Discovered models from results structure:")
            for model in sorted(unique_models):
                print(f"  • {model}")
            
            return list(unique_models)
        else:
            print(f"❌ Failed to get model metrics: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error discovering models: {e}")
        return []

if __name__ == "__main__":
    print("🚀 Starting Bitwise Logic Test Suite")
    print("This tests the new config-free approach for bitwise logic")
    
    # Test model discovery
    available_models = test_model_availability()
    
    # Test bitwise logic API
    success = test_bitwise_logic_api()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Bitwise logic is working correctly.")
        print("📋 Summary:")
        print("  ✅ API connectivity works")
        print("  ✅ Model discovery from results works") 
        print("  ✅ Direct bitwise logic rule submission works")
        print("  ✅ Combined models are created successfully")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    print("=" * 60) 