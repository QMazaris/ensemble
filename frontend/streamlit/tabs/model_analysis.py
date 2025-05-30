import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path to access utils
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add root directory to path to access shared modules
root_dir = str(Path(__file__).parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import utility functions
from utils import (
    get_cached_data, clear_cache, plot_confusion_matrix, 
    render_model_curves
)

def render_model_analysis_tab():
    """Render the model analysis tab content."""
    st.write("### Model Analysis")
    
    # Add cache refresh button
    if st.button("🔄 Refresh Data", key="model_analysis_refresh", help="Clear cache and reload model data"):
        st.session_state.force_refresh_metrics = True
        clear_cache()
        st.rerun()
    
    # Use cached data to avoid redundant API calls
    metrics_data = get_cached_data(
        cache_key="metrics_data",
        api_endpoint="/results/metrics",
        default_value={"results": {"model_metrics": [], "model_summary": [], "confusion_matrices": []}},
        force_refresh=st.session_state.get('force_refresh_metrics', False)
    )
    
    # Clear the force refresh flag
    if st.session_state.get('force_refresh_metrics', False):
        st.session_state.force_refresh_metrics = False
    
    if metrics_data and metrics_data.get('results'):
        results = metrics_data['results']
        
        # Convert to DataFrames for compatibility
        metrics_df = pd.DataFrame(results.get('model_metrics', []))
        summary_df = pd.DataFrame(results.get('model_summary', []))
        cm_df = pd.DataFrame(results.get('confusion_matrices', []))
        
        # Get sweep data from cache
        try:
            from shared import data_service
            sweep_data = data_service.get_sweep_data()
        except Exception as e:
            sweep_data = None
    
    if summary_df is not None and not summary_df.empty:
        # Model Selection
        model_options = summary_df['model_name'].unique()
        model = st.selectbox("Select Model", model_options)
        
        # Filter data for selected model
        model_data = summary_df[summary_df['model_name'] == model]
        model_cm = cm_df[cm_df['model_name'] == model]

        # Split Selection
        split_options = model_data['split'].unique()
        selected_split = st.selectbox("Select Split", split_options)

        # Filter data for selected split
        model_data_split = model_data[model_data['split'] == selected_split]
        model_cm_split = model_cm[model_cm['split'] == selected_split]
        
        if not model_data_split.empty:
            # Create two columns for metrics and confusion matrix
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"#### Performance Metrics ({selected_split} Split)")
                metrics_display = model_data_split[['threshold_type', 'accuracy', 'precision', 'recall', 'f1_score', 'cost', 'threshold']]
                st.dataframe(metrics_display.style.format({
                    'accuracy': '{:.1f}%',
                    'precision': '{:.1f}%',
                    'recall': '{:.1f}%',
                    'f1_score': '{:.1f}%',
                    'cost': '{:.1f}',
                    'threshold': '{:.3f}'
                }))
            
            with col2:
                st.write(f"#### Confusion Matrix ({selected_split} Split)")
                cm_threshold_options = model_cm_split['threshold_type'].unique()
                selected_cm_threshold = (st.selectbox("Select Threshold Type for CM", cm_threshold_options) 
                                       if len(cm_threshold_options) > 1 else cm_threshold_options[0])

                cm_data = model_cm_split[model_cm_split['threshold_type'] == selected_cm_threshold].iloc[0]
                st.plotly_chart(plot_confusion_matrix(cm_data), use_container_width=True, key=f"confusion_matrix_{model}_{selected_split}_{selected_cm_threshold}")

            # Model Curves and Threshold Analysis
            if sweep_data and model in sweep_data:
                render_model_curves(model, sweep_data, model_data_split)
            elif model_data_split['threshold_type'].isin(['base']).any():
                st.info("Threshold sweep data is not available for decision-based models.")
            else:
                st.info("Threshold sweep data is not available. This may be because the pipeline hasn't been run yet or the model doesn't have sweep data.")
        else:
            st.info("No model data available. Run the pipeline to see analysis here.")
    else:
        st.info("Run the pipeline to see analysis here.")

        