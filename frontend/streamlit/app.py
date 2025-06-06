import streamlit as st
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
from frontend.streamlit.utils import ensure_directories, sync_frontend_to_backend
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
    """Run the main pipeline via the API endpoint."""
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
            
            # Now run the pipeline via API
            st.info("🚀 Starting pipeline execution via API...")
            
            # Call the API pipeline endpoint
            response = requests.post(f"{BACKEND_API_URL}/pipeline/run", timeout=300)  # 5 minute timeout
            
            if response.status_code == 200:
                result = response.json()
                
                # Record completion time and clear cached data so fresh results are loaded
                st.session_state.pipeline_completed_at = time.time()
                st.cache_data.clear()
                
                # Update sync tracking since we just synced
                st.session_state.last_synced_config = copy.deepcopy(st.session_state.config_settings)
                
                # Show success message with data summary
                st.success("✅ Pipeline completed successfully!")
                
                # Show data summary if available
                if 'data_summary' in result:
                    summary = result['data_summary']
                    st.info(f"📊 Data stored: Metrics: {summary.get('metrics_stored', False)}, "
                           f"Predictions: {summary.get('predictions_stored', False)}, "
                           f"Sweep: {summary.get('sweep_stored', False)}")
                
                st.info("🔄 Data will refresh automatically. You may need to interact with the page to see updates.")
                st.balloons()
                
            else:
                error_detail = response.json().get('detail', 'Unknown error') if response.headers.get('content-type') == 'application/json' else response.text
                st.error(f"❌ Pipeline failed: {error_detail}")
                
        except requests.exceptions.Timeout:
            st.error("❌ Pipeline execution timed out. The pipeline may still be running in the background.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API. Please ensure the backend server is running on http://localhost:8000")
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
    
    # Only keep the Run Pipeline button - config saves automatically
    if st.sidebar.button("▶️ Run Pipeline", use_container_width=True, type="primary"):
        run_pipeline()
    
    # Add a refresh data button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True, help="Force refresh all data from backend"):
        try:
            # Force clear backend cache via API
            response = requests.delete(f"{BACKEND_API_URL}/data/force-clear", timeout=10)
            if response.status_code == 200:
                st.sidebar.success("✅ Backend data refreshed!")
            else:
                st.sidebar.warning(f"⚠️ Backend refresh returned status {response.status_code}")

            # Reset pipeline completion timestamp to force fresh data
            if 'pipeline_completed_at' in st.session_state:
                del st.session_state['pipeline_completed_at']
            st.cache_data.clear()

            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error refreshing data: {str(e)}")

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