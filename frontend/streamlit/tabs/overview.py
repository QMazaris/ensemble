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
    get_cached_data, clear_cache, create_radar_chart,
    render_threshold_comparison_plots, BACKEND_API_URL
)

def render_overview_tab():
    """Render the enhanced overview tab with comprehensive model analysis."""
    st.write("### 📊 Model Performance Dashboard")
    
    # Initialize variables
    summary_df = None
    metrics_df = None
    cm_df = None
    sweep_data = None
    
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
        sweep_data = get_cached_data(
            cache_key="sweep_data",
            api_endpoint="/debug/data-service",  # This endpoint provides info about what data is available
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
        # Enhanced comprehensive model comparison chart
        
        # Threshold type selector
        available_threshold_types = summary_df['threshold_type'].unique()
        selected_threshold_type = st.selectbox(
            "Select Threshold Type for Analysis",
            options=available_threshold_types,
            index=0 if 'cost' not in available_threshold_types else list(available_threshold_types).index('cost'),
            help="Choose which threshold optimization method to display",
            key="overview_threshold_selector"
        )
        
        # Get ALL models for the selected threshold type, with fallbacks for base models
        overview_data = []
        
        # First, get all ML models with the selected threshold type  
        ml_models = summary_df[
            (summary_df['split'] == 'Full') &
            (summary_df['threshold_type'] == selected_threshold_type) &
            (~summary_df['model_name'].isin(['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail'])) &
            (~summary_df['model_name'].str.contains('_Combined|Combined_', na=False))  # Also include manually created combined models
        ].copy()
        
        if not ml_models.empty:
            overview_data.append(ml_models)
        
        # Then, get base models - try selected threshold type first, then fallback
        base_model_names = ['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail']
        
        for model_name in base_model_names:
            model_data = summary_df[
                (summary_df['split'] == 'Full') &
                (summary_df['model_name'] == model_name)
            ].copy()
            
            if not model_data.empty:
                # Try to get data with selected threshold type first
                preferred_data = model_data[model_data['threshold_type'] == selected_threshold_type]
                if not preferred_data.empty:
                    overview_data.append(preferred_data)
                else:
                    # Fallback to any available threshold type for this model
                    fallback_data = model_data.iloc[[0]]  # Take the first available
                    overview_data.append(fallback_data)
        
        # Get ALL combined models (created via bitwise logic)
        combined_models = summary_df[
            (summary_df['split'] == 'Full') &
            (summary_df['model_name'].str.contains('Combined|_Combined', na=False))
        ].copy()
        
        if not combined_models.empty:
            # For combined models, try selected threshold type first, then fallback
            preferred_combined = combined_models[combined_models['threshold_type'] == selected_threshold_type]
            if not preferred_combined.empty:
                overview_data.append(preferred_combined)
            else:
                # Group by model name and take first available threshold type for each
                for model_name in combined_models['model_name'].unique():
                    model_combined = combined_models[combined_models['model_name'] == model_name].iloc[[0]]
                    overview_data.append(model_combined)
        
        # Combine all data
        if overview_data:
            overview_data_combined = pd.concat(overview_data, ignore_index=True)
            
            # Add model type for better visualization
            overview_data_combined['model_type'] = overview_data_combined['model_name'].apply(
                lambda x: 'Base Model' if x in ['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail'] 
                         else 'Combined Model' if 'Combined' in x or '_Combined' in x
                         else 'ML Model'
            )
        else:
            st.warning(f"No model data found for threshold type: {selected_threshold_type}")
            overview_data_combined = pd.DataFrame()
        
        # Separate ML models and base models for visualization
        ml_overview_data = overview_data_combined[
            ~overview_data_combined['model_name'].isin(['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail'])
        ]
        
        base_models_data = overview_data_combined[
            overview_data_combined['model_name'].isin(['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail'])
        ]
        
        if not overview_data_combined.empty:
            # ============================================
            # 🔥 Enhanced Performance Overview with F1 Score and Thresholds
            # ============================================

            overview_data_combined['model_display'] = overview_data_combined.apply(
                lambda row: f"{row['model_name']} (τ={row['threshold']:.2f})", axis=1
            )

            # Reshape so each metric becomes its own bar - now including F1 score
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
            threshold_info = overview_data_combined[['model_name', 'threshold', 'threshold_type', 'model_type']].copy()
            threshold_info['threshold'] = threshold_info['threshold'].round(4)
            
            st.dataframe(
                threshold_info.style.format({'threshold': '{:.4f}'}),
                use_container_width=True,
                hide_index=True
            )

            # Cost Analysis
            st.write("#### 💰 Cost Analysis")
            
            # Cost comparison - make sure it includes all model types including combined models
            fig_cost_bar = px.bar(overview_data_combined,
                            x='model_name',
                            y='cost',
                            title='Model Costs (Lower is Better)',
                            labels={'cost': 'Cost', 'model_name': 'Model'},
                            color='model_type',
                            color_discrete_map={
                                'ML Model': '#1f77b4', 
                                'Base Model': '#ff7f0e',
                                'Combined Model': '#2ca02c'  # Green for combined models
                            })
            fig_cost_bar.update_layout(height=500)
            st.plotly_chart(fig_cost_bar, use_container_width=True, key="cost_comparison_bar_chart")
            
            # Radar chart for model comparison (ML models and combined models)
            radar_data = overview_data_combined[
                overview_data_combined['model_type'].isin(['ML Model', 'Combined Model'])
            ]
            
            if not radar_data.empty and len(radar_data) > 1:
                st.write("#### 📈 ML & Combined Models Radar Chart")
                fig_radar = create_radar_chart(radar_data)
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True, key="ml_combined_models_radar_chart")
            
            # Detailed tables section
            st.write("#### 🔍 Detailed Metrics Table")
            
            # Add filtering options
            col1, col2, col3 = st.columns(3)
            with col1:
                split_filter = st.selectbox("Filter by Split", 
                                          options=['All'] + list(summary_df['split'].unique()),
                                          index=0)
            with col2:
                threshold_filter = st.selectbox("Filter by Threshold Type",
                                              options=['All'] + list(summary_df['threshold_type'].unique()),
                                              index=0)
            with col3:
                model_filter = st.multiselect("Filter by Models",
                                            options=summary_df['model_name'].unique(),
                                            default=summary_df['model_name'].unique())
            
            # Apply filters
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
            
            # Summary statistics
            if not filtered_df.empty:
                st.write("##### Summary Statistics")
                summary_stats = filtered_df.groupby('model_name')[['accuracy', 'precision', 'recall', 'f1_score', 'cost']].agg(['mean', 'std']).round(2)
                st.dataframe(summary_stats)
        
        # Additional visualizations if sweep data is available
        if sweep_data:
            st.write("### 🔄 Threshold Analysis Dashboard")
            render_threshold_comparison_plots(sweep_data, summary_df)
            
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
                        st.write("- The pipeline hasn't been run yet")
                        st.write("- The pipeline failed during execution")
                        st.write("- The data wasn't saved properly")
                    else:
                        st.warning(f"🔍 **Issue**: Backend returned {model_summary_count} model summaries, but DataFrame creation failed")
                else:
                    st.warning("🔍 **Issue**: Backend API responded but didn't include 'results' key")
            else:
                st.warning(f"🔍 **Issue**: Cannot connect to backend API at {BACKEND_API_URL}")
                st.write("- Make sure the backend server is running")
                st.write("- Check if the API endpoint is correct")
        
        with col2:
            if st.button("🔄 Refresh Data", type="primary"):
                st.session_state.force_refresh_metrics = True
                st.rerun()
            
            if st.button("🗑️ Clear Cache"):
                clear_cache()
                st.session_state.force_refresh_metrics = True
                st.success("Cache cleared!")
                st.rerun()
        
        st.info("🔄 Run the pipeline to see comprehensive metrics and visualizations here.")
        
        # Show example of what will be displayed
        st.write("#### Preview: What you'll see after running the pipeline")
        st.write("- **All Models Performance Comparison**: Bar charts showing accuracy, precision, and recall for both ML models and base models")
        st.write("- **Enhanced Scatter Plots**: Precision vs Recall and Cost vs Accuracy for all models with model type distinction")
        st.write("- **Cost Analysis**: Detailed cost breakdowns and comparisons")
        st.write("- **Interactive Filtering**: Filter by split, threshold type, and specific models")
        st.write("- **Summary Statistics**: Statistical summaries of model performance")
        st.write("- **Threshold Analysis**: Advanced threshold sweep visualizations")