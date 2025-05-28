import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
# import config # Import config to pass its settings - REMOVED

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import from the same directory
from utils import ensure_directories
from tabs import (
    render_overview_tab, render_model_analysis_tab,
    render_plots_gallery_tab, render_downloads_tab,
    render_data_management_tab,
    render_preprocessing_tab,
    render_model_zoo_tab,
    render_model_metrics_cheat_tab
)
from sidebar import render_sidebar, save_config

# Page config
st.set_page_config(
    page_title="AI Pipeline Dash",
    page_icon="🚀",
    layout="wide"
)

# Ensure directories exist
ensure_directories()

def load_config_settings():
    """Loads config settings by reading the config.py file."""
    config_settings = {}
    try:
        config_path = Path(root_dir) / "config.py"
        if config_path.exists():
            # Safely execute the config file to load settings into a dictionary
            # This is a simple approach; for complex configs, a dedicated parser is better
            exec(config_path.read_text(), {}, config_settings)
    except Exception as e:
        st.error(f"Error loading config settings: {e}")
    return config_settings

def run_pipeline():
    """Run the main pipeline using the virtual environment's Python."""
    with st.spinner("Running pipeline..."):
        try:
            # Get the path to the virtual environment's Python
            venv_python = os.path.join("venv", "Scripts", "python.exe")
            if not os.path.exists(venv_python):
                st.error("Virtual environment Python not found. Please ensure venv is set up correctly.")
                return
                
            subprocess.run([venv_python, "run.py"], check=True)
            st.success("Pipeline complete and metrics exported!")
        except subprocess.CalledProcessError as e:
            st.error(f"Pipeline failed with error: {str(e)}")
        except FileNotFoundError:
            st.error("Error: run.py not found. Please ensure run.py exists in the root directory.")

def main():
    """Main application entry point."""
    # Sidebar
    config_updates = render_sidebar() # Get latest config settings
    
    if st.sidebar.button("💾 Save & Reload Config"):
        save_config(config_updates)
    
    if st.sidebar.button("▶️ Run Pipeline"):
        run_pipeline()

    # Load config settings for the app
    current_config_settings = load_config_settings()

    # Tabs
    # Passing current_config_settings dictionary to access all config settings
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Overview", "Data Management", "Preprocessing Config", "Model Zoo", "Model Analysis", "Cheat Metrics", "Plots Gallery", "Downloads"
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
        render_model_metrics_cheat_tab()
    
    with tab7:
        render_plots_gallery_tab()
        
    with tab8:
        render_downloads_tab()

if __name__ == "__main__":
    main() 