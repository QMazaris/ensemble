import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
import time
import json
from datetime import datetime
import requests

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import from individual tab files
from tabs.overview import render_overview_tab
from tabs.model_analysis import render_model_analysis_tab
from tabs.model_zoo import render_model_zoo_tab
from tabs.data_management import render_data_management_tab
from tabs.preprocessing import render_preprocessing_tab
from tabs.downloads import render_downloads_tab

# Import from the same directory
from frontend.streamlit.utils import ensure_directories, clear_cache
from frontend.streamlit.sidebar import render_sidebar
from shared.config_manager import get_config

BACKEND_API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="AI Pipeline Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure directories exist
ensure_directories()

def load_config_from_api():
    """
    Load config from the FastAPI backend /config/load endpoint.
    This is the proper way to get the current config state.
    """
    try:
        print("🔄 [TERMINAL] Loading config from API...")
        response = requests.get(f"{BACKEND_API_URL}/config/load", timeout=10)
        if response.status_code == 200:
            # Backend returns config directly, not nested under "config" key
            config = response.json()
            print(f"✅ [TERMINAL] Loaded config from API: {len(config)} top-level keys")
            print(f"📊 [TERMINAL] Config keys: {list(config.keys())}")
            return config
        else:
            print(f"❌ [TERMINAL] Failed to load config: HTTP {response.status_code}")
            st.error(f"❌ Failed to load config: HTTP {response.status_code}")
            return {}
    except requests.exceptions.ConnectionError:
        print("❌ [TERMINAL] Cannot connect to backend API on http://localhost:8000")
        st.error("❌ Cannot connect to backend API. Please ensure the backend server is running on http://localhost:8000")
        return {}
    except Exception as e:
        print(f"❌ [TERMINAL] Error loading config from API: {e}")
        st.error(f"❌ Error loading config from API: {e}")
        return {}

def run_pipeline():
    """Run the main pipeline using the virtual environment's Python."""
    with st.spinner("Running pipeline..."):
        try:
            # Get the path to the virtual environment's Python
            venv_python = os.path.join("venv", "Scripts", "python.exe")
            if not os.path.exists(venv_python):
                # Fallback to system Python if venv not found
                venv_python = "python"
                
            # Run the backend pipeline
            subprocess.run([venv_python, "-m", "backend.run"], check=True, cwd=root_dir)
            
            # Clear all cached data to force fresh data loading
            clear_cache()
            
            # Use a single timestamp-based refresh mechanism instead of multiple flags
            # This prevents race conditions between tabs
            import time
            st.session_state.pipeline_completed_at = time.time()
            
            # Show success message
            st.success("✅ Pipeline completed successfully!")
            st.info("🔄 Data will refresh automatically. You may need to interact with the page to see updates.")
            
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Pipeline failed with error: {str(e)}")
        except FileNotFoundError:
            st.error("❌ Error: backend/run.py not found. Please ensure backend/run.py exists.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

def main():
    """Main application entry point."""
    print("\n" + "="*60)
    print("🚀 [TERMINAL] Starting Streamlit App")
    print("="*60)

    # Load config from API on startup
    if "config_settings" not in st.session_state:
        print("🔄 [TERMINAL] First load - getting config from API")
        st.session_state.config_settings = load_config_from_api()
        
    # Refresh config from API if it's empty (connection was down)
    if not st.session_state.config_settings:
        print("⚠️ [TERMINAL] Config is empty, retrying API call")
        st.session_state.config_settings = load_config_from_api()

    # Get config reference
    cfg = st.session_state.config_settings

    # Debug info - terminal and UI
    print(f"🔧 [TERMINAL] Config loaded: {len(cfg)} keys")
    if cfg:
        print(f"📊 [TERMINAL] Available sections: {list(cfg.keys())}")
    else:
        print("❌ [TERMINAL] Config is empty!")

    st.sidebar.write(f"🔧 Config loaded: {len(cfg)} keys")
    if cfg:
        st.sidebar.write(f"📊 Available sections: {list(cfg.keys())}")

    # Header
    st.title("🚀 AI Pipeline Dashboard")
    st.markdown("---")
    
    # Sidebar with automatic config saving
    config_updates = render_sidebar()
    
    # Sidebar controls
    st.sidebar.markdown("---")
    
    # Only keep the Run Pipeline button - config saves automatically
    if st.sidebar.button("▶️ Run Pipeline", use_container_width=True, type="primary"):
        run_pipeline()

    # Tabs - Restored with enhanced overview
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📁 Data Management", "⚙️ Preprocessing Config", 
        "🎯 Model Zoo", "📈 Model Analysis", "📥 Downloads"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_data_management_tab(cfg)
        
    with tab3:
        print("\n🔄 [TERMINAL] Rendering preprocessing tab...")
        render_preprocessing_tab()
        print("✅ [TERMINAL] Preprocessing tab rendered")
        
    with tab4:
        render_model_zoo_tab()
        
    with tab5:
        render_model_analysis_tab()
    
    with tab6:
        render_downloads_tab(cfg)

if __name__ == "__main__":
    main() 