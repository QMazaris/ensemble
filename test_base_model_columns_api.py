#!/usr/bin/env python3
"""Test script for base model columns API endpoints."""

import requests
import json

def test_base_model_columns_api():
    """Test the base model columns API endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing Base Model Columns API...")
    print("=" * 50)
    
    # Test GET endpoint
    print("\n1. Testing GET /config/base-model-columns")
    try:
        response = requests.get(f"{base_url}/config/base-model-columns")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test POST endpoint
    print("\n2. Testing POST /config/base-model-columns")
    test_config = {
        "model_columns": {
            "AD": "AnomalyScore",
            "CL": "CL_ConfMax", 
            "RF": "RandomForestScore"
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/config/base-model-columns",
            json=test_config,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test GET again to verify the update
    print("\n3. Testing GET after update")
    try:
        response = requests.get(f"{base_url}/config/base-model-columns")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_base_model_columns_api() 