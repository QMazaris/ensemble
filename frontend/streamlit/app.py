import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
import time

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

# Page config
st.set_page_config(
    page_title="AI Pipeline Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure directories exist
ensure_directories()

def load_config_settings():
    """Loads config settings using the new configuration manager."""
    try:
        config = get_config()
        return config.to_dict()
    except Exception as e:
        st.error(f"Error loading config settings: {e}")
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
            
            # Import utils for cache management
            from frontend.streamlit.utils import clear_cache
            
            # Clear all cached data to force fresh data loading
            clear_cache()
            
            # Use a single timestamp-based refresh mechanism instead of multiple flags
            # This prevents race conditions between tabs
            import time
            st.session_state.pipeline_completed_at = time.time()
            
            # Show success message
            st.success("✅ Pipeline completed successfully!")
            st.info("🔄 Data will refresh automatically. You may need to interact with the page to see updates.")
            
            # Don't use st.rerun() immediately as it can cause issues
            # Let the natural Streamlit refresh cycle handle the update
            
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Pipeline failed with error: {str(e)}")
        except FileNotFoundError:
            st.error("❌ Error: backend/run.py not found. Please ensure backend/run.py exists.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

def main():
    """Main application entry point."""
    # Header
    st.title("🚀 AI Pipeline Dashboard")
    st.markdown("---")
    
    # Sidebar with automatic config saving
    config_updates = render_sidebar() # Config is now automatically saved
    
    # Sidebar controls
    st.sidebar.markdown("---")
    
    # Only keep the Run Pipeline button - config saves automatically
    if st.sidebar.button("▶️ Run Pipeline", use_container_width=True, type="primary"):
        run_pipeline()

    # Load config settings for the app
    current_config_settings = load_config_settings()

    # Tabs - Restored with enhanced overview
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "📁 Data Management", "⚙️ Preprocessing Config", 
        "🎯 Model Zoo", "📈 Model Analysis", "📥 Downloads"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_data_management_tab()
        
    with tab3:
        render_preprocessing_tab(current_config_settings)
        
    with tab4:
        render_model_zoo_tab()
        
    with tab5:
        render_model_analysis_tab()
    
    with tab6:
        render_downloads_tab()

if __name__ == "__main__":
    main() 