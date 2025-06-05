#!/usr/bin/env python3

"""Test script to debug data service issues."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_data_service():
    print("Testing data service import...")
    
    try:
        from shared import data_service
        print("✅ Data service imported successfully")
        print(f"📍 Data service instance ID: {id(data_service)}")
        print(f"📊 Has data: {data_service.has_data()}")
        print(f"🔑 Data keys: {list(data_service._data.keys())}")
        
        # Test basic functionality
        print("\nTesting basic functionality...")
        data_service.set_metrics_data({"test": "data"})
        print(f"📊 After setting test data - Has data: {data_service.has_data()}")
        print(f"🔑 Data keys: {list(data_service._data.keys())}")
        
        # Clear data
        data_service.clear_all_data()
        print(f"🗑️ After clearing - Has data: {data_service.has_data()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing data service: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_from_backend():
    print("\nTesting import from backend context...")
    
    try:
        # Simulate backend import
        from backend.run import data_service as backend_data_service
        print("✅ Backend data service imported successfully")
        print(f"📍 Backend data service instance ID: {id(backend_data_service)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing from backend: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_from_api():
    print("\nTesting import from API context...")
    
    try:
        # Simulate API import  
        from backend.api.main import data_service as api_data_service
        print("✅ API data service imported successfully")
        print(f"📍 API data service instance ID: {id(api_data_service)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing from API: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Data Service Debug Test")
    print("=" * 50)
    
    success1 = test_data_service()
    success2 = test_import_from_backend()
    success3 = test_import_from_api()
    
    print("\n" + "=" * 50)
    if all([success1, success2, success3]):
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")