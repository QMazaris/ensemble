import streamlit as st
import subprocess
import os
from utils import ensure_directories
from tabs import (
    render_overview_tab, render_model_analysis_tab,
    render_plots_gallery_tab, render_downloads_tab
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
    config_updates = render_sidebar()
    
    if st.sidebar.button("💾 Save & Reload Config"):
        save_config(config_updates)
    
    if st.sidebar.button("▶️ Run Pipeline"):
        run_pipeline()

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Analysis", "Plots Gallery", "Downloads"])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_model_analysis_tab()
    
    with tab3:
        render_plots_gallery_tab()
    
    with tab4:
        render_downloads_tab()

if __name__ == "__main__":
    main() 