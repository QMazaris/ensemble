#!/usr/bin/env python3
"""
Test script to verify Streamlit frontend functionality.
"""

import sys
from pathlib import Path
import requests
import time

# Add the root directory to Python path
root_dir = str(Path(__file__).parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def test_streamlit_app():
    """Test if Streamlit app is accessible and working."""
    print("🔍 Testing Streamlit Frontend...")
    
    # Try to access the Streamlit app
    try:
        # Default Streamlit URL
        url = "http://localhost:8501"
        
        print(f"Attempting to connect to {url}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✅ Streamlit app is accessible")
            print(f"Response status: {response.status_code}")
            
            # Check if the response contains expected content
            content = response.text.lower()
            if "ai pipeline dashboard" in content or "streamlit" in content:
                print("✅ App appears to be loading correctly")
                return True
            else:
                print("⚠️ App is accessible but content may not be loading properly")
                return False
        else:
            print(f"❌ Streamlit app returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Streamlit app. Is it running on localhost:8501?")
        return False
    except requests.exceptions.Timeout:
        print("❌ Timeout connecting to Streamlit app")
        return False
    except Exception as e:
        print(f"❌ Error testing Streamlit app: {e}")
        return False

def test_frontend_components():
    """Test if frontend components can be imported and work."""
    print("\n🔍 Testing Frontend Components...")
    
    try:
        # Test importing main components
        from frontend.streamlit.app import main
        from frontend.streamlit.tabs import render_overview_tab
        from frontend.streamlit.utils import load_metrics_data
        from frontend.streamlit.sidebar import render_sidebar
        
        print("✅ All frontend components imported successfully")
        
        # Test data loading
        metrics_df, summary_df, cm_df, sweep_data = load_metrics_data()
        
        if summary_df is not None and not summary_df.empty:
            print("✅ Frontend can load data successfully")
            print(f"  - Data shape: {summary_df.shape}")
            print(f"  - Models available: {len(summary_df['model_name'].unique())}")
            return True
        else:
            print("❌ Frontend cannot load data")
            return False
            
    except Exception as e:
        print(f"❌ Error testing frontend components: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 Testing Streamlit Frontend Functionality\n")
    
    # Test frontend components
    components_ok = test_frontend_components()
    
    # Test Streamlit app accessibility
    app_ok = test_streamlit_app()
    
    print(f"\n📊 Test Results:")
    print(f"  Frontend Components: {'✅ PASS' if components_ok else '❌ FAIL'}")
    print(f"  Streamlit App Access: {'✅ PASS' if app_ok else '❌ FAIL'}")
    
    if components_ok and app_ok:
        print("\n🎉 SUCCESS: Streamlit frontend is working correctly!")
        print("You can access the dashboard at: http://localhost:8501")
    elif components_ok:
        print("\n⚠️ PARTIAL SUCCESS: Frontend components work but app may not be accessible")
        print("Try accessing: http://localhost:8501")
    else:
        print("\n❌ ISSUES DETECTED: Frontend has problems")

if __name__ == "__main__":
    main() 