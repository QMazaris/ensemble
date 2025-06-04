import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import time

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
from config_util import on_config_change

def render_model_analysis_tab():
    """Render the model analysis tab content."""
    st.write("### Model Analysis")
    
    # Add cache refresh button
    if st.button("🔄 Refresh Data", key="model_analysis_refresh", help="Clear cache and reload model data"):
        clear_cache()
        st.session_state.pipeline_completed_at = time.time()
        st.rerun()
    
    # Use cached data to avoid redundant API calls
    # Check if pipeline completed recently (within last 30 seconds) to force refresh
    pipeline_completed_at = st.session_state.get('pipeline_completed_at', 0)
    force_refresh = (time.time() - pipeline_completed_at) < 30
    
    metrics_data = get_cached_data(
        cache_key="metrics_data",
        api_endpoint="/results/metrics",
        default_value={"results": {"model_metrics": [], "model_summary": [], "confusion_matrices": []}},
        force_refresh=force_refresh
    )
    
    # Clear the pipeline completion timestamp after first use to prevent constant refreshing
    if force_refresh and 'pipeline_completed_at' in st.session_state:
        st.session_state.pipeline_completed_at = 0
    
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
        try:
            # Optimization Type Selection - new dropdown for cost vs accuracy
            optimization_options = ['Cost Optimized', 'Accuracy Optimized']
            selected_optimization = st.selectbox(
                "Select Optimization", 
                optimization_options,
                index=0,  # Default to cost optimized
                key="model_analysis_optimization_select"
            )
            
            # Convert selection to threshold_type
            threshold_type = 'cost' if selected_optimization == 'Cost Optimized' else 'accuracy'
            
            # Filter data by threshold type
            summary_df_filtered = summary_df[summary_df['threshold_type'] == threshold_type]
            cm_df_filtered = cm_df[cm_df['threshold_type'] == threshold_type]
            
            # Model Selection - frontend only, resets to default
            model_options = summary_df_filtered['model_name'].unique()
            
            if len(model_options) == 0:
                st.warning(f"No data available for {selected_optimization} optimization.")
                return
            
            model = st.selectbox(
                "Select Model", 
                model_options,
                index=0,  # Always default to first model
                key="model_analysis_model_select"
            )
            
            # Filter data for selected model
            model_data = summary_df_filtered[summary_df_filtered['model_name'] == model]
            model_cm = cm_df_filtered[cm_df_filtered['model_name'] == model]

            # Split Selection - frontend only, resets to default
            split_options = model_data['split'].unique()
            
            selected_split = st.selectbox(
                "Select Split", 
                split_options,
                index=0,  # Always default to first split
                key="model_analysis_split_select"
            )

            # Filter data for selected split
            model_data_split = model_data[model_data['split'] == selected_split]
            model_cm_split = model_cm[model_cm['split'] == selected_split]
            
            if not model_data_split.empty:
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    accuracy = model_data_split['accuracy'].iloc[0]
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                    
                with col2:
                    precision = model_data_split['precision'].iloc[0]
                    st.metric("Precision", f"{precision:.1f}%")
                    
                with col3:
                    recall = model_data_split['recall'].iloc[0]
                    st.metric("Recall", f"{recall:.1f}%")
                    
                with col4:
                    f1 = model_data_split['f1_score'].iloc[0]
                    st.metric("F1 Score", f"{f1:.1f}%")

                # Confusion Matrix
                if not model_cm_split.empty:
                    st.write("#### Confusion Matrix")
                    st.plotly_chart(
                        plot_confusion_matrix(model_cm_split.iloc[0]),
                        use_container_width=True,
                        key=f"confusion_matrix_{model}_{selected_split}_{threshold_type}"
                    )

                # Model Curves
                if sweep_data and model in sweep_data:
                    render_model_curves(model, sweep_data, model_data_split)
                else:
                    st.info("Threshold sweep data not available for detailed analysis.")
            else:
                st.warning(f"No data available for model {model} on split {selected_split} with {selected_optimization} optimization")
        except Exception as e:
            st.error(f"❌ Error displaying model analysis: {str(e)}")
            st.write("**Error Details:** Please try refreshing the data or clearing the cache.")
            if st.button("🔄 Refresh Data", key="error_refresh_analysis"):
                clear_cache()
                st.session_state.pipeline_completed_at = time.time()
                st.rerun()
    else:
        st.info("No model data available. Run the pipeline to see analysis here.")

        