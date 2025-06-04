import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
import time
import json
from datetime import datetime
import requests
import copy
from utils import calculate_config_diff

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import from individual tab files
from tabs.overview import render_overview_tab
# from tabs.model_analysis import render_model_analysis_tab
from tabs.model_zoo import render_model_zoo_tab
from tabs.data_management import render_data_management_tab
from tabs.preprocessing import render_preprocessing_tab
from tabs.bitwise_logic import render_bitwise_logic_tab
from tabs.model_analysis import render_model_analysis_tab
from tabs.downloads import render_downloads_tab

# Import from the same directory
from frontend.streamlit.utils import ensure_directories, clear_cache, sync_frontend_to_backend
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
        response = requests.get(f"{BACKEND_API_URL}/config/load", timeout=10)
        if response.status_code == 200:
            # Backend returns config directly, not nested under "config" key
            config = response.json()
            return config
        else:
            st.error(f"❌ Failed to load config: HTTP {response.status_code}")
            return {}
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Please ensure the backend server is running on http://localhost:8000")
        return {}
    except Exception as e:
        st.error(f"❌ Error loading config from API: {e}")
        return {}

def run_pipeline():
    """Run the main pipeline using the virtual environment's Python."""
    with st.spinner("Syncing configuration and running pipeline..."):
        try:
            # First, always sync config to backend and wait for completion
            st.info("🔄 Syncing configuration to backend...")
            sync_notification = st.empty()
            
            # Sync config and wait for completion
            sync_success = sync_frontend_to_backend(sync_notification)
            if not sync_success:
                st.error("❌ Failed to sync configuration to backend. Pipeline aborted.")
                return
            
            st.success("✅ Configuration synced successfully!")
            
            # Small delay to ensure sync is fully complete
            time.sleep(0.5)
            
            # Now run the pipeline
            st.info("🚀 Starting pipeline execution...")
            
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
            st.session_state.pipeline_completed_at = time.time()
            
            # Update sync tracking since we just synced
            st.session_state.last_synced_config = copy.deepcopy(st.session_state.config_settings)
            
            # Show success message
            st.success("✅ Pipeline completed successfully!")
            st.info("🔄 Data will refresh automatically. You may need to interact with the page to see updates.")
            st.balloons()
            
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Pipeline failed with error: {str(e)}")
        except FileNotFoundError:
            st.error("❌ Error: backend/run.py not found. Please ensure backend/run.py exists.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

def main():
    """Main application entry point."""
    
    # Load config from API on startup - only once per session
    if "config_settings" not in st.session_state:
        st.session_state.config_settings = load_config_from_api()
        
        # Initialize sync tracking - consider the initial load as "synced"
        st.session_state.last_synced_config = copy.deepcopy(st.session_state.config_settings)


    # Tabs - Restored with enhanced overview
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", "📁 Data Management", "⚙️ Preprocessing Config", 
        "🔗 Bitwise Logic", "🎯 Model Zoo", "📈 Model Analysis", "📥 Downloads"
    ])

    render_sidebar()

    # Sidebar controls
    st.sidebar.markdown("---")
    
    # Add a clear cache button
    if st.sidebar.button("🗑️ Clear All Cache", use_container_width=True, help="Clear all cached data to force fresh results"):
        try:
            # Clear frontend cache first
            clear_cache()
            
            # Force clear backend cache via API (more thorough than regular clear)
            response = requests.delete(f"{BACKEND_API_URL}/data/force-clear", timeout=10)
            if response.status_code == 200:
                st.sidebar.success("✅ All cache forcefully cleared!")
            else:
                st.sidebar.warning(f"⚠️ Backend cache clear returned status {response.status_code}")
                
            # Reset all session state cache-related items
            cache_keys_to_clear = [
                'pipeline_completed_at', 'api_cache', 'api_cache_timestamps',
                'last_synced_config'
            ]
            for key in cache_keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error clearing cache: {str(e)}")
    
    # Only keep the Run Pipeline button - config saves automatically
    if st.sidebar.button("▶️ Run Pipeline", use_container_width=True, type="primary"):
        run_pipeline()
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_data_management_tab()
        
    with tab3:
        render_preprocessing_tab()
        
    with tab4:
        render_bitwise_logic_tab()
        
    with tab5:
        render_model_zoo_tab()
        
    with tab6:
        render_model_analysis_tab()
    
    with tab7:
        render_downloads_tab()

    

if __name__ == "__main__":
    main() 