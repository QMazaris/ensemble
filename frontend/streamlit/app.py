import streamlit as st
import subprocess
import os
import sys
from pathlib import Path

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import from the same directory
from frontend.streamlit.utils import ensure_directories
from frontend.streamlit.tabs import (
    render_overview_tab, render_model_analysis_tab,
    render_plots_gallery_tab, render_downloads_tab,
    render_data_management_tab,
    render_preprocessing_tab,
    render_model_zoo_tab
)
from frontend.streamlit.sidebar import render_sidebar, save_config
from shared.config_manager import get_config

# Page config
st.set_page_config(
    page_title="🚀 AI Pipeline Dashboard",
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
            st.success("✅ Pipeline completed successfully!")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Pipeline failed with error: {str(e)}")
        except FileNotFoundError:
            st.error("❌ Error: backend/run.py not found. Please ensure backend/run.py exists.")

def main():
    """Main application entry point."""
    # Header
    st.title("🚀 AI Pipeline Dashboard")
    st.markdown("---")
    
    # Sidebar
    config_updates = render_sidebar() # Get latest config settings
    
    # Sidebar buttons
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("💾 Save Config", use_container_width=True):
            save_config(config_updates)
    
    with col2:
        if st.button("▶️ Run Pipeline", use_container_width=True, type="primary"):
            run_pipeline()

    # Load config settings for the app
    current_config_settings = load_config_settings()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", "📁 Data Management", "⚙️ Preprocessing Config", 
        "🎯 Model Zoo", "📈 Model Analysis", "📊 Plots Gallery", "📥 Downloads"
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
        render_plots_gallery_tab()
        
    with tab7:
        render_downloads_tab()

if __name__ == "__main__":
    main() 