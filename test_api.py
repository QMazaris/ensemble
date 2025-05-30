#!/usr/bin/env python3
"""
Test API endpoints for bitwise logic functionality
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_api_connection():
    """Test basic API connection."""
    print("🔍 Testing API connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API connection failed: {e}")
        return False

def test_bitwise_logic_config_get():
    """Test getting bitwise logic configuration."""
    print("\n🔍 Testing GET /config/bitwise-logic...")
    try:
        response = requests.get(f"{API_BASE_URL}/config/bitwise-logic", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_bitwise_logic_config_post():
    """Test posting bitwise logic configuration."""
    print("\n🔍 Testing POST /config/bitwise-logic...")
    
    test_config = {
        "rules": [
            {
                "name": "Test_Combined",
                "columns": ["AD_Decision", "CL_Decision"],
                "logic": "OR"
            }
        ],
        "enabled": True
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/config/bitwise-logic",
            json=test_config,
            timeout=10
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_pipeline_run():
    """Test running the pipeline."""
    print("\n🔍 Testing POST /pipeline/run...")
    try:
        response = requests.post(f"{API_BASE_URL}/pipeline/run", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Pipeline started: {data}")
            
            # Wait for pipeline to complete
            print("⏳ Waiting for pipeline to complete...")
            for i in range(30):  # Wait up to 5 minutes
                time.sleep(10)
                status_response = requests.get(f"{API_BASE_URL}/pipeline/status")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"📊 Status: {status_data['status']} - {status_data['message']}")
                    
                    if status_data['status'] == 'completed':
                        print("✅ Pipeline completed successfully!")
                        return True
                    elif status_data['status'] == 'failed':
                        print(f"❌ Pipeline failed: {status_data['message']}")
                        return False
                else:
                    print(f"❌ Failed to get status: {status_response.text}")
            
            print("⏰ Pipeline timeout - check manually")
            return False
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_bitwise_logic_apply():
    """Test applying bitwise logic rules."""
    print("\n🔍 Testing POST /config/bitwise-logic/apply...")
    try:
        response = requests.post(f"{API_BASE_URL}/config/bitwise-logic/apply", timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Run all API tests."""
    print("🚀 Testing Bitwise Logic API Endpoints")
    print("="*50)
    
    tests = [
        test_api_connection,
        test_bitwise_logic_config_get,
        test_bitwise_logic_config_post,
    ]
    
    # Optional tests (require data)
    pipeline_tests = [
        test_pipeline_run,
        test_bitwise_logic_apply,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test {test.__name__} failed")
    
    print(f"\n📊 Core Tests: {passed}/{len(tests)} tests passed")
    
    # Ask if we should run pipeline tests
    print("\n🤔 Run pipeline tests? (requires data and takes time) [y/N]:", end=" ")
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes']:
            pipeline_passed = 0
            for test in pipeline_tests:
                if test():
                    pipeline_passed += 1
                else:
                    print(f"❌ Test {test.__name__} failed")
            
            print(f"📊 Pipeline Tests: {pipeline_passed}/{len(pipeline_tests)} tests passed")
            passed += pipeline_passed
            tests.extend(pipeline_tests)
    except:
        print("Skipping pipeline tests")
    
    if passed == len(tests):
        print("🎉 All tests passed! Bitwise logic API is working.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 