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
    get_cached_data, clear_cache, create_radar_chart,
    render_threshold_comparison_plots, BACKEND_API_URL
)
from config_util import on_config_change

def render_overview_tab():
    """Render the simplified overview tab with basic model analysis."""
    st.write("### 📊 Model Performance Dashboard")
    
    # Initialize variables
    summary_df = None
    metrics_df = None
    cm_df = None
    sweep_data = None
    
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
        sweep_data = get_cached_data(
            cache_key="sweep_data",
            api_endpoint="/debug/data-service",
            default_value=None
        )
        
        # Try to get actual sweep data from data service
        try:
            from shared import data_service
            sweep_data = data_service.get_sweep_data()
        except Exception as e:
            sweep_data = None
    
    # Check if we have data to display
    if summary_df is not None and not summary_df.empty:
        try:
            # ========= UPDATED THRESHOLD SELECTOR WITH "ALL" OPTION =========
            all_threshold_types = list(summary_df['threshold_type'].unique())
            available_threshold_types = ["All"] + all_threshold_types

            default_index = available_threshold_types.index('cost') if 'cost' in all_threshold_types else 0

            selected_threshold_type = st.selectbox(
                "Select Threshold Type for Analysis",
                options=available_threshold_types,
                index=default_index,
                help="Choose which threshold optimization method to display",
                key="overview_threshold_selector"
            )

            # Conditional filtering logic
            if selected_threshold_type == "All":
                overview_data_combined = summary_df[summary_df['split'] == 'Full'].copy()
            else:
                overview_data_combined = summary_df[
                    (summary_df['split'] == 'Full') &
                    (summary_df['threshold_type'] == selected_threshold_type)
                ].copy()

            
            if overview_data_combined.empty:
                st.warning(f"No model data found for threshold type: {selected_threshold_type}")
                return
            
            # Enhanced Performance Overview with F1 Score and Thresholds
            overview_data_combined['model_display'] = overview_data_combined.apply(
                lambda row: f"{row['model_name']} (τ={row['threshold']:.2f})", axis=1
            )

            # Reshape so each metric becomes its own bar - including F1 score
            metric_cols = ["accuracy", "precision", "recall", "f1_score"]
            plot_df = (
                overview_data_combined
                .melt(
                    id_vars=["model_name","model_display", "threshold"],
                    value_vars=metric_cols,
                    var_name="metric",
                    value_name="score"
                )
            )

            # Create the main performance chart
            fig = px.bar(
                plot_df,
                x="model_display",
                y="score",
                color="metric",
                barmode="group",
                labels={
                    "score": "Score (%)",
                    "model_display": "Model (with Threshold)",
                    "metric": "Metric"
                },
                title=f"Model Performance with {selected_threshold_type.title()} Threshold",
                hover_data={"threshold": True},
                color_discrete_map={
                    'accuracy': '#1f77b4',
                    'precision': '#ff7f0e', 
                    'recall': '#2ca02c',
                    'f1_score': '#d62728'
                }
            )

            fig.update_layout(
                yaxis_range=[0, 100],
                height=700,
                legend_title_text="Metric"
            )

            st.plotly_chart(fig, use_container_width=True, key=f"main_performance_bar_chart_{selected_threshold_type}")

            # Add threshold information table
            st.write("##### Threshold Values Used")
            threshold_info = overview_data_combined[['model_name', 'threshold', 'threshold_type']].copy()
            threshold_info['threshold'] = threshold_info['threshold'].round(4)
            
            st.dataframe(
                threshold_info.style.format({'threshold': '{:.4f}'}),
                use_container_width=True,
                hide_index=True
            )

            # Cost Analysis - uses same filtering as main chart
            st.write("#### 💰 Cost Analysis")
            
            # Create separate cost charts for accuracy and cost optimized models
            if selected_threshold_type == "All":
                # Show separate charts for each threshold type
                for threshold_type in summary_df['threshold_type'].unique():
                    threshold_data = summary_df[
                        (summary_df['split'] == 'Full') &
                        (summary_df['threshold_type'] == threshold_type)
                    ].copy()
                    
                    if not threshold_data.empty:
                        fig_cost_bar = px.bar(threshold_data,
                                        x='model_name',
                                        y='cost',
                                        title=f'Model Costs - {threshold_type.title()} Optimized (Lower is Better)',
                                        labels={'cost': 'Cost', 'model_name': 'Model'})
                        fig_cost_bar.update_layout(height=400)
                        st.plotly_chart(fig_cost_bar, use_container_width=True, key=f"cost_comparison_{threshold_type}")
            else:
                # Show single chart for selected threshold type
                fig_cost_bar = px.bar(overview_data_combined,
                                x='model_name',
                                y='cost',
                                title=f'Model Costs - {selected_threshold_type.title()} Optimized (Lower is Better)',
                                labels={'cost': 'Cost', 'model_name': 'Model'})
                fig_cost_bar.update_layout(height=500)
                st.plotly_chart(fig_cost_bar, use_container_width=True, key="cost_comparison_bar_chart")
            
            # Radar chart for model comparison (all models in filtered data)
            if not overview_data_combined.empty and len(overview_data_combined) > 1:
                st.write("#### 📈 Model Comparison Radar Chart")
                fig_radar = create_radar_chart(overview_data_combined)
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True, key="model_comparison_radar_chart")
            
            # Detailed tables section
            st.write("#### 🔍 Detailed Metrics Table")
            
            # Add filtering options - frontend only, reset to defaults on reload
            col1, col2, col3 = st.columns(3)
            
            with col1:
                split_options = ['All'] + list(summary_df['split'].unique())
                
                split_filter = st.selectbox(
                    "Filter by Split", 
                    options=split_options,
                    index=0,  # Always default to 'All'
                    key="overview_split_filter"
                )
                
            with col2:
                threshold_options = ['All'] + list(summary_df['threshold_type'].unique())
                    
                threshold_filter = st.selectbox(
                    "Filter by Threshold Type",
                    options=threshold_options,
                    index=0,  # Always default to 'All'
                    key="overview_threshold_filter"
                )
                
            with col3:
                # Keep model filter with config persistence since it wasn't mentioned to change
                config = st.session_state.get('config_settings', {})
                current_model_filter = config.get('overview', {}).get('model_filter', list(summary_df['model_name'].unique()))
                
                # Use a safe key for model filter
                model_filter_key = "overview_model_filter"
                
                # Initialize state only once
                if model_filter_key not in st.session_state:
                    st.session_state[model_filter_key] = current_model_filter
                
                model_filter = st.multiselect(
                    "Filter by Models",
                    options=summary_df['model_name'].unique(),
                    key=model_filter_key
                )
                
                # Button-based save for multiselect
                if st.button("Save Model Filter", key="save_model_filter"):
                    if 'overview' not in config:
                        config['overview'] = {}
                    config['overview']['model_filter'] = model_filter
                    st.session_state['config_settings'] = config
                    st.toast("Model filter saved!", icon="✅")

            
            # Apply filters directly to API data
            filtered_df = summary_df.copy()
            if split_filter != 'All':
                filtered_df = filtered_df[filtered_df['split'] == split_filter]
            if threshold_filter != 'All':
                filtered_df = filtered_df[filtered_df['threshold_type'] == threshold_filter]
            if model_filter:
                filtered_df = filtered_df[filtered_df['model_name'].isin(model_filter)]
            
            st.dataframe(filtered_df.style.format({
                'accuracy': '{:.1f}%',
                'precision': '{:.1f}%',
                'recall': '{:.1f}%',
                'f1_score': '{:.1f}%',
                'cost': '{:.1f}',
                'threshold': '{:.3f}'
            }), use_container_width=True)
            
            # Additional visualizations if sweep data is available
            if sweep_data:
                st.write("### 🔄 Threshold Analysis Dashboard")
                render_threshold_comparison_plots(sweep_data, summary_df)
                
        except Exception as e:
            st.error(f"❌ Error displaying overview data: {str(e)}")
            st.write("**Error Details:** Please try refreshing the page or clearing the cache.")
            if st.button("🔄 Refresh Page", key="error_refresh_overview"):
                clear_cache()
                st.session_state.pipeline_completed_at = time.time()
                st.rerun()
    else:
        # Enhanced error message with debugging information
        st.error("❌ **No model data found!**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if metrics_data:
                if 'results' in metrics_data:
                    results = metrics_data['results']
                    model_summary_count = len(results.get('model_summary', []))
                    if model_summary_count == 0:
                        st.warning("🔍 **Issue**: Backend API responded but returned no model summary data. This usually means:")
                        st.write("- Pipeline hasn't been run yet")
                        st.write("- Models failed to train properly") 
                        st.write("- There's an issue with the backend data processing")
                    else:
                        st.info(f"Found {model_summary_count} model summary entries, but they may be filtered out.")
                else:
                    st.warning("🔍 **Issue**: Backend API response doesn't contain 'results' field")
            else:
                st.warning("🔍 **Issue**: No response from backend API or cached data is empty")
                
        with col2:
            if st.button("🔄 Try Again", key="retry_overview_data"):
                clear_cache()
                st.session_state.pipeline_completed_at = time.time()
                st.rerun()
                