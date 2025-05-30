import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from .utils import (
    load_metrics_data, load_predictions_data, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_threshold_sweep, get_plot_groups,
    MODEL_DIR, PLOT_DIR, PREDICTIONS_DIR
)
from pathlib import Path
import inspect
import re
import ast
import importlib.util
import requests
import time
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8000"

# ========== CENTRALIZED DATA LOADING WITH CACHING ==========

def get_cached_data(cache_key, api_endpoint, default_value=None, force_refresh=False):
    """
    Centralized caching system for API calls to improve efficiency.
    
    Args:
        cache_key: Key to store data in session state
        api_endpoint: API endpoint to call
        default_value: Default value if API call fails
        force_refresh: Force refresh the cache
    """
    # Initialize cache if not exists
    if 'api_cache' not in st.session_state:
        st.session_state.api_cache = {}
    
    # Initialize cache timestamps
    if 'api_cache_timestamps' not in st.session_state:
        st.session_state.api_cache_timestamps = {}
    
    # Check if cache is expired (5 minutes cache time)
    cache_timeout = 300  # 5 minutes
    current_time = time.time()
    is_expired = (
        cache_key in st.session_state.api_cache_timestamps and
        current_time - st.session_state.api_cache_timestamps[cache_key] > cache_timeout
    )
    
    # Check if we need to refresh cache
    if force_refresh or cache_key not in st.session_state.api_cache or is_expired:
        try:
            response = requests.get(f"{API_BASE_URL}{api_endpoint}", timeout=10)
            if response.status_code == 200:
                st.session_state.api_cache[cache_key] = response.json()
                st.session_state.api_cache_timestamps[cache_key] = current_time
            else:
                st.session_state.api_cache[cache_key] = default_value
                st.session_state.api_cache_timestamps[cache_key] = current_time
        except Exception as e:
            st.session_state.api_cache[cache_key] = default_value
            st.session_state.api_cache_timestamps[cache_key] = current_time
    
    return st.session_state.api_cache[cache_key]

def debounced_auto_save(save_function, config_data, notification_container, debounce_key, delay=2.0):
    """
    Debounced auto-save function to prevent excessive API calls.
    
    Args:
        save_function: Function to call for saving
        config_data: Data to save
        notification_container: Streamlit container for notifications
        debounce_key: Unique key for this auto-save operation
        delay: Delay in seconds before saving
    """
    # Initialize debounce storage
    if 'debounce_timers' not in st.session_state:
        st.session_state.debounce_timers = {}
    
    if 'debounce_data' not in st.session_state:
        st.session_state.debounce_data = {}
    
    # Store the current time and data
    current_time = time.time()
    st.session_state.debounce_timers[debounce_key] = current_time
    st.session_state.debounce_data[debounce_key] = {
        'save_function': save_function,
        'config_data': config_data,
        'notification_container': notification_container
    }
    
    # Check if enough time has passed for any pending saves
    keys_to_process = []
    for key, timestamp in st.session_state.debounce_timers.items():
        if current_time - timestamp >= delay:
            keys_to_process.append(key)
    
    # Process debounced saves
    for key in keys_to_process:
        if key in st.session_state.debounce_data:
            data = st.session_state.debounce_data[key]
            try:
                result = data['save_function'](data['config_data'], data['notification_container'])
                if result:
                    # Only show success message for actual saves
                    pass  # The save function itself handles notifications
            except Exception as e:
                data['notification_container'].error(f"❌ Auto-save failed: {str(e)}")
            
            # Clean up
            del st.session_state.debounce_timers[key]
            del st.session_state.debounce_data[key]

def clear_cache():
    """Clear all cached data."""
    if 'api_cache' in st.session_state:
        st.session_state.api_cache = {}
    if 'api_cache_timestamps' in st.session_state:
        st.session_state.api_cache_timestamps = {}

def update_cache(cache_key, data):
    """Update specific cache entry."""
    if 'api_cache' not in st.session_state:
        st.session_state.api_cache = {}
    if 'api_cache_timestamps' not in st.session_state:
        st.session_state.api_cache_timestamps = {}
    
    st.session_state.api_cache[cache_key] = data
    st.session_state.api_cache_timestamps[cache_key] = time.time()

def render_overview_tab():
    """Render the enhanced overview tab with comprehensive model analysis."""
    st.write("### 📊 Model Performance Dashboard")
    
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
            import sys
            from pathlib import Path
            root_dir = str(Path(__file__).parent.parent.parent)
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
            from shared import data_service
            sweep_data = data_service.get_sweep_data()
        except Exception as e:
            sweep_data = None
    
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
        st.info("🔄 Run the pipeline to see comprehensive metrics and visualizations here.")
        
        # Show example of what will be displayed
        st.write("#### Preview: What you'll see after running the pipeline")
        st.write("- **All Models Performance Comparison**: Bar charts showing accuracy, precision, and recall for both ML models and base models")
        st.write("- **Enhanced Scatter Plots**: Precision vs Recall and Cost vs Accuracy for all models with model type distinction")
        st.write("- **Cost Analysis**: Detailed cost breakdowns and comparisons")
        st.write("- **Interactive Filtering**: Filter by split, threshold type, and specific models")
        st.write("- **Summary Statistics**: Statistical summaries of model performance")
        st.write("- **Threshold Analysis**: Advanced threshold sweep visualizations")

def create_radar_chart(data):
    """Create a radar chart for model comparison."""
    try:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig = go.Figure()
        
        for _, row in data.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[metric] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=row['model_name']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating radar chart: {e}")
        return None

def render_threshold_comparison_plots(sweep_data, summary_df):
    """Render threshold sweep comparison plots."""
    if not sweep_data:
        return
        
    # Model selection for threshold analysis
    available_models = list(sweep_data.keys())
    if not available_models:
        return
        
    selected_models = st.multiselect(
        "Select models for threshold comparison",
        options=available_models,
        default=available_models[:min(3, len(available_models))]  # Default to first 3 models
    )
    
    if not selected_models:
        return
        
    # Create threshold sweep comparison plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Cost vs Threshold")
        fig_cost = go.Figure()
        
        for model in selected_models:
            model_data = sweep_data[model]
            fig_cost.add_trace(go.Scatter(
                x=model_data['thresholds'],
                y=model_data['costs'],
                mode='lines+markers',
                name=model
            ))
        
        fig_cost.update_layout(
            title="Cost vs Threshold Comparison",
            xaxis_title="Threshold",
            yaxis_title="Cost"
        )
        st.plotly_chart(fig_cost, use_container_width=True, key="threshold_cost_comparison")
    
    with col2:
        st.write("#### Accuracy vs Threshold")
        fig_acc = go.Figure()
        
        for model in selected_models:
            model_data = sweep_data[model]
            fig_acc.add_trace(go.Scatter(
                x=model_data['thresholds'],
                y=model_data['accuracies'],
                mode='lines+markers',
                name=model
            ))
        
        fig_acc.update_layout(
            title="Accuracy vs Threshold Comparison",
            xaxis_title="Threshold",
            yaxis_title="Accuracy (%)"
        )
        st.plotly_chart(fig_acc, use_container_width=True, key="threshold_accuracy_comparison")

    # Add F1 Score vs Threshold plot
    st.write("#### F1 Score vs Threshold")
    fig_f1 = go.Figure()
    
    for model in selected_models:
        model_data = sweep_data[model]
        if 'f1_scores' in model_data:  # Check if F1 scores are available
            fig_f1.add_trace(go.Scatter(
                x=model_data['thresholds'],
                y=model_data['f1_scores'],
                mode='lines+markers',
                name=model
            ))
    
    fig_f1.update_layout(
        title="F1 Score vs Threshold Comparison",
        xaxis_title="Threshold",
        yaxis_title="F1 Score (%)"
    )
    st.plotly_chart(fig_f1, use_container_width=True, key="threshold_f1_comparison")

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
            import sys
            from pathlib import Path
            root_dir = str(Path(__file__).parent.parent.parent)
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
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

def render_model_curves(model, sweep_data, model_data_split):
    """Render model curves and threshold analysis."""
    st.write("#### Model Curves and Threshold Analysis")
    
    # Load predictions and ensure we have the data
    try:
        # Use cached predictions data instead of loading fresh each time
        predictions_data = get_cached_data(
            cache_key="predictions_data",
            api_endpoint="/results/predictions",
            default_value={"predictions": []},
            force_refresh=False  # Don't force refresh here
        )
        
        if predictions_data and predictions_data.get('predictions'):
            pred_df = pd.DataFrame(predictions_data['predictions'])
        else:
            st.warning(f"No predictions data available.")
            return
            
        if model not in pred_df.columns:
            st.warning(f"Model {model} not found in predictions data.")
            return
            
        # Get the probabilities and true labels for this model
        probs = np.array(sweep_data[model]['probabilities'])
        y_true = pred_df['GT'].values
        
        # Ensure lengths match
        if len(probs) != len(y_true):
            st.warning(f"Length mismatch between probabilities ({len(probs)}) and true labels ({len(y_true)}). Using the shorter length.")
            min_len = min(len(probs), len(y_true))
            probs = probs[:min_len]
            y_true = y_true[:min_len]
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("##### ROC Curve (Full Data)")
            st.plotly_chart(plot_roc_curve(y_true, probs), use_container_width=True, key=f"roc_curve_{model}")
        
        with col4:
            st.write("##### Precision-Recall Curve (Full Data)")
            st.plotly_chart(plot_precision_recall_curve(y_true, probs), use_container_width=True, key=f"pr_curve_{model}")

        # Threshold Analysis
        st.write("##### Probability Distribution and Threshold Sweep")
        
        fig_hist = px.histogram(
            x=probs,
            title=f'Probability Distribution - {model}',
            nbins=50,
            labels={'x': 'Probability', 'y': 'Count'}
        )
        st.plotly_chart(fig_hist, use_container_width=True, key=f"prob_distribution_{model}")

        # Threshold sweep plot
        sweep_fig = plot_threshold_sweep(sweep_data, model)
        if sweep_fig:
            st.plotly_chart(sweep_fig, use_container_width=True, key=f"threshold_sweep_{model}")

        # Get optimal thresholds
        cost_optimal_thr = (model_data_split[model_data_split['threshold_type'] == 'cost']['threshold'].iloc[0] 
                           if not model_data_split[model_data_split['threshold_type'] == 'cost'].empty else None)
        
    except Exception as e:
        st.error(f"Error plotting curves: {str(e)}")

def render_downloads_tab():
    """Render the downloads tab content."""
    st.write("### Download Files")
    
    # Download predictions - now using cached data instead of fresh API call
    try:
        # Use cached predictions data
        predictions_data = get_cached_data(
            cache_key="predictions_data",
            api_endpoint="/results/predictions",
            default_value={"predictions": []},
            force_refresh=False
        )
        
        if predictions_data and predictions_data.get('predictions'):
            predictions_df = pd.DataFrame(predictions_data['predictions'])
            
            # Convert DataFrame to CSV for download
            csv_data = predictions_df.to_csv(index=False)
            st.download_button(
                "Download Predictions CSV",
                csv_data.encode('utf-8'),
                file_name="all_model_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("No predictions data available. Please run the pipeline first.")
    except Exception as e:
        st.error(f"Error loading predictions data: {str(e)}")
    
    # Download models
    st.write("#### Download Models")
    
    # Get all model files and sort them
    model_files = sorted(MODEL_DIR.glob("*.*"))
    
    # Group files by model name (without extension)
    model_groups = {}
    for file in model_files:
        if file.suffix in ['.pkl', '.onnx']:
            base_name = file.stem
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append(file)
    
    # Display models in sorted order
    for model_name in sorted(model_groups.keys()):
        st.write(f"##### {model_name}")
        for model_file in sorted(model_groups[model_name]):
            with open(model_file, 'rb') as f:
                file_type = "ONNX" if model_file.suffix == '.onnx' else "Pickle"
                st.download_button(
                    f"Download {file_type} Model",
                    f.read(),
                    file_name=model_file.name,
                    mime="application/octet-stream"
                )

def render_data_management_tab():
    """Render the data management tab content."""
    st.write("### Data Management")
    
    # File Upload Section
    st.write("#### Upload New Training Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the data with explicit type checking
            df = pd.read_csv(uploaded_file, low_memory=False)
            
            # Data validation section
            st.write("#### Data Validation")
            
            problematic_cols = []
            categorical_cols = {}
            for col in df.columns:
                # Check for object dtype columns
                if df[col].dtype == 'object':
                    # Try to convert to numeric if possible
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        df[col] = pd.to_numeric(df[col])
                    except:
                        # If conversion fails, check if it's a date
                        try:
                            # Use a more robust date parsing approach if needed
                            df[col] = pd.to_datetime(df[col], errors='raise')
                        except:
                            # If still object, check if it's likely categorical
                            unique_vals = df[col].dropna().unique()
                            if len(unique_vals) > 0 and len(unique_vals) <= 50: # Heuristic for categorical
                                categorical_cols[col] = unique_vals.tolist()
                            else:
                                problematic_cols.append(col)
            
            if problematic_cols or categorical_cols:
                st.warning("⚠️ Potential data issues detected. Please review the details below.")
                st.info("Please review the 'Data Preview' and 'Column Information' sections below carefully to ensure structure is acceptable before saving.")
                
                if problematic_cols:
                    st.write("**Columns with potentially unhandled data types:**")
                    st.write("These columns could contain mixed types or formats that couldn't be automatically identified as numeric, datetime, or simple categorical. They might require cleaning.")
                    for col in problematic_cols:
                        st.write(f"- **{col}**: `{df[col].dtype}`")
                        # Optionally display sample values for problematic columns as well
                        sample_vals = df[col].dropna().unique()
                        if len(sample_vals) <= 10: # Show a few samples
                             st.write(f"  Sample values: {', '.join(str(x) for x in sample_vals[:10])}")
                        elif len(sample_vals) > 10:
                             st.write(f"  Contains {len(sample_vals)} unique values (showing first 10 sample values: {', '.join(str(x) for x in sample_vals[:10])})")
                    
                if categorical_cols:
                    st.write("**Columns identified as potentially categorical:**")
                    st.write("These columns have a limited number of unique string values and have been identified as potentially categorical.")
                    for col, unique_vals in categorical_cols.items():
                         st.write(f"- **{col}**: `{df[col].dtype}`")
                         st.write(f"  Unique values ({len(unique_vals)}): {', '.join(str(x) for x in unique_vals)}")
            
            # Display data preview with type information
            st.write("#### Data Preview")
            
            st.dataframe(df.head())
            
            st.write("##### Column Information")
            for col in df.columns:
                st.write(f"**{col}**")
                st.write(f"- Type: {df[col].dtype}")
                st.write(f"- Non-null values: {df[col].count()}")
                if df[col].dtype in ['int64', 'float64']:
                    st.write(f"- Range: {df[col].min():.2f} to {df[col].max():.2f}")
            
            # Basic data validation
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("##### Data Shape")
                st.write(f"Rows: {df.shape[0]}")
                st.write(f"Columns: {df.shape[1]}")
                
                # Check for missing values
                missing_values = df.isnull().sum()
                if missing_values.any():
                    st.warning("Missing values detected:")
                    st.write(missing_values[missing_values > 0])
                else:
                    st.success("No missing values detected")
            
            with col2:
                st.write("##### Data Quality Checks")
                # Check for duplicate rows
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    st.warning(f"Found {duplicates} duplicate rows")
                else:
                    st.success("No duplicate rows found")
                
                # Check for constant columns
                constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
                if constant_cols:
                    st.warning(f"Found {len(constant_cols)} constant columns: {', '.join(constant_cols)}")
            
            # Save options
            st.write("#### Save Options")
            if problematic_cols:
                st.warning("⚠️ Some columns have non-standard data types. Consider cleaning the data before saving.")
            
            save_name = st.text_input("Save as (without .csv extension)", "training_data")
            if st.button("Save Dataset"):
                if not save_name:
                    st.error("Please provide a name for the dataset")
                    return
                    
                # Create data directory if it doesn't exist
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                
                # Save the file
                file_path = data_dir / f"{save_name}.csv"
                try:
                    df.to_csv(file_path, index=False)
                    st.success(f"Dataset saved successfully to {file_path}")
                    
                    # Update config to use new dataset
                    st.info("Please update the config file to use the new dataset path if needed")
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please check that your CSV file is properly formatted and try again.")
    
    # Current Dataset Information
    st.write("#### Current Dataset Information")
    data_dir = Path("data")
    if data_dir.exists():
        training_files = list(data_dir.glob("*.csv"))
        if training_files:
            st.write("Available datasets:")
            for file in training_files:
                with st.expander(f"📊 {file.name}"):
                    try:
                        df = pd.read_csv(file, low_memory=False)
                        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                        st.write(f"Last modified: {file.stat().st_mtime}")
                        
                        # Show column types
                        st.write("Column types:")
                        for col, dtype in df.dtypes.items():
                            st.write(f"- {col}: {dtype}")
                            
                        # Quick preview
                        if st.button(f"Preview {file.name}", key=f"preview_{file.name}"):
                            st.dataframe(df.head())
                            
                        # Delete option
                        if st.button(f"Delete {file.name}", key=f"delete_{file.name}"):
                            try:
                                file.unlink()
                                st.success(f"Deleted {file.name}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting file: {str(e)}")
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {str(e)}")
        else:
            st.info("No datasets found in the data directory")
    else:
        st.info("Data directory not found")

def update_config_file(updates):
    """Update specific key-value pairs in the config.py file."""
    try:
        config_path = Path("config.py")
        if not config_path.exists():
            st.error("config.py not found.")
            return
            
        lines = config_path.read_text().splitlines()
        new_lines = []
        updated_keys = set()

        for line in lines:
            added = False
            for key, value in updates.items():
                # Check if the line defines this config key
                if line.strip().startswith(f"{key} ="):
                    if isinstance(value, str):
                        new_lines.append(f"{key} = '{value}'")
                    elif isinstance(value, list):
                         # Format list nicely, handle potential strings inside list
                        formatted_list_items = [f"'{item}'" if isinstance(item, str) else str(item) for item in value]
                        new_lines.append(f"{key} = [{', '.join(formatted_list_items)}]")
                    else:
                        new_lines.append(f"{key} = {value}")
                    updated_keys.add(key)
                    added = True
                    break # Move to the next line in original file
            if not added:
                new_lines.append(line)
                
        config_path.write_text("\n".join(new_lines))
        
        # Store success message in session state
        st.session_state.config_update_success = True
        st.rerun()
        
    except Exception as e:
        st.error(f"Error updating config file: {e}")

def auto_save_data_config(config_updates, notification_container):
    """Automatically save data configuration changes."""
    # Initialize session state for previous data config if not exists
    if 'previous_data_config' not in st.session_state:
        st.session_state.previous_data_config = {}
    
    # Check if configuration has changed
    config_changed = False
    for key, value in config_updates.items():
        if key not in st.session_state.previous_data_config or st.session_state.previous_data_config[key] != value:
            config_changed = True
            break
    
    # Only save if configuration actually changed
    if not config_changed:
        return False
    
    # Save if configuration changed
    if config_changed:
        try:
            update_config_file(config_updates)
            # Update previous config in session state
            st.session_state.previous_data_config = config_updates.copy()
            # Show auto-save notification with timestamp
            current_time = datetime.now().strftime("%H:%M:%S")
            notification_container.success(f"✅ Data config auto-saved at {current_time}!", icon="💾")
            return True
        except Exception as e:
            notification_container.error(f"❌ Auto-save failed: {str(e)}")
    return False

def _save_data_config_helper(config_data, notification_container):
    """Helper function for debounced data config saving."""
    return auto_save_data_config(config_data, notification_container)

def auto_save_base_model_config(config_data, notification_container):
    """Automatically save base model configuration changes."""
    # Initialize session state for previous base model config if not exists
    if 'previous_base_model_config' not in st.session_state:
        st.session_state.previous_base_model_config = {}
    
    # Check if configuration has changed
    config_changed = False
    for key, value in config_data.items():
        if key not in st.session_state.previous_base_model_config or st.session_state.previous_base_model_config[key] != value:
            config_changed = True
            break
    
    # Only save if configuration actually changed
    if not config_changed:
        return False
    
    # Save if configuration changed
    if config_changed:
        try:
            response = requests.post(f"{API_BASE_URL}/config/base-models", json=config_data, timeout=10)
            if response.status_code == 200:
                # Update previous config in session state
                st.session_state.previous_base_model_config = config_data.copy()
                # Show auto-save notification with timestamp
                current_time = datetime.now().strftime("%H:%M:%S")
                notification_container.success(f"✅ Base model config auto-saved at {current_time}!", icon="💾")
                return True
            else:
                notification_container.error(f"❌ Auto-save failed: {response.text}")
                return False
        except Exception as e:
            notification_container.error(f"❌ Auto-save failed: {str(e)}")
    return False

def _save_base_model_config_helper(config_data, notification_container):
    """Helper function for debounced base model config saving."""
    return auto_save_base_model_config(config_data, notification_container)

def auto_save_model_config(selected_model, edited_params, config_content, config_path, available_models, notification_container):
    """Automatically save model configuration changes."""
    # Initialize session state for previous model config if not exists
    if 'previous_model_config' not in st.session_state:
        st.session_state.previous_model_config = {}
    
    # Create a unique key for this model's configuration
    model_config_key = f"{selected_model}_config"
    
    # Check if configuration has changed
    config_changed = False
    if model_config_key not in st.session_state.previous_model_config or st.session_state.previous_model_config[model_config_key] != edited_params:
        config_changed = True
    
    # Save if configuration changed
    if config_changed:
        try:
            # Construct new model definition
            model_class = available_models[selected_model]['class']
            param_str = ', '.join(f"{k}={repr(v)}" for k, v in edited_params.items())
            new_model_def = f"'{selected_model}': {model_class.__name__}({param_str})"

            # Find and replace the model definition
            model_pattern = rf"'{selected_model}':\s*{model_class.__name__}\(.*?\)"
            new_config_content = re.sub(model_pattern, new_model_def, config_content, flags=re.DOTALL)
            
            # Write updated config
            config_path.write_text(new_config_content)
            
            # Update previous config in session state
            st.session_state.previous_model_config[model_config_key] = edited_params.copy()
            
            # Show auto-save notification
            notification_container.success(f"✅ {selected_model} config auto-saved!", icon="💾")
            return True
        except Exception as e:
            notification_container.error(f"❌ Auto-save failed: {str(e)}")
            return False
    return False

def render_model_zoo_tab():
    """Render the model zoo tab content with automatic saving."""
    st.write("### Model Zoo")

    # Import required model classes
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    config_path = Path("config.py")
    if not config_path.exists():
        st.error("config.py not found.")
        return

    # Read and parse the config file
    try:
        config_content = config_path.read_text()
        # Load the actual config module to get current values
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        current_models = config_module.MODELS

        # --- Use AST to extract source code for MODELS and HYPERPARAM_SPACE ---
        tree = ast.parse(config_content)
        models_code = "# MODELS section not found"
        hyperparams_code = "# HYPERPARAM_SPACE section not found"

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == 'MODELS':
                            # Extract the source code for the value node
                            # Use ast.get_source_segment which handles indentation within the segment
                            models_code = ast.get_source_segment(config_content, node) or "# MODELS section not found"
                        elif target.id == 'HYPERPARAM_SPACE':
                            # Extract the source code for the value node
                            hyperparams_code = ast.get_source_segment(config_content, node) or "# HYPERPARAM_SPACE section not found"

        # The AST approach is more reliable, removing the regex fallback.

    except Exception as e:
        st.error(f"Error loading or parsing config.py: {e}")
        return

    # Define available models and their parameters
    available_models = {
        'XGBoost': {
            'class': xgb.XGBClassifier,
            'params': {
                'objective': {'type': 'select', 'options': ['binary:logistic'], 'default': 'binary:logistic'},
                'max_depth': {'type': 'number', 'min': 1.0, 'max': 20.0, 'step': 1.0, 'default': 4.0},
                'learning_rate': {'type': 'number', 'min': 0.001, 'max': 0.5, 'step': 0.001, 'default': 0.1},
                'n_estimators': {'type': 'number', 'min': 50.0, 'max': 1000.0, 'step': 50.0, 'default': 400.0},
                'subsample': {'type': 'number', 'min': 0.1, 'max': 1.0, 'step': 0.05, 'default': 0.8},
                'colsample_bytree': {'type': 'number', 'min': 0.1, 'max': 1.0, 'step': 0.05, 'default': 0.8},
                'min_child_weight': {'type': 'number', 'min': 1.0, 'max': 20.0, 'step': 1.0, 'default': 5.0},
                'gamma': {'type': 'number', 'min': 0.0, 'max': 1.0, 'step': 0.1, 'default': 0.1},
                'scale_pos_weight': {'type': 'number', 'min': 1.0, 'max': 100.0, 'step': 1.0, 'default': 62.5},
                'random_state': {'type': 'number', 'min': 0.0, 'max': 1000.0, 'step': 1.0, 'default': 42.0},
                'n_jobs': {'type': 'number', 'min': -1.0, 'max': 16.0, 'step': 1.0, 'default': -1.0},
                'eval_metric': {'type': 'select', 'options': ['logloss', 'auc', 'error'], 'default': 'logloss'}
            }
        },
        'RandomForest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': {'type': 'number', 'min': 50.0, 'max': 1000.0, 'step': 50.0, 'default': 300.0},
                'max_depth': {'type': 'number', 'min': 1.0, 'max': 20.0, 'step': 1.0, 'default': 6.0},
                'min_samples_split': {'type': 'number', 'min': 2.0, 'max': 50.0, 'step': 1.0, 'default': 10.0},
                'min_samples_leaf': {'type': 'number', 'min': 1.0, 'max': 50.0, 'step': 1.0, 'default': 5.0},
                'class_weight': {'type': 'select', 'options': ['balanced', 'balanced_subsample', None], 'default': 'balanced_subsample'},
                'max_features': {'type': 'select', 'options': ['sqrt', 'log2', None], 'default': 'sqrt'},
                'bootstrap': {'type': 'select', 'options': [True, False], 'default': True},
                'oob_score': {'type': 'select', 'options': [True, False], 'default': True},
                'n_jobs': {'type': 'number', 'min': -1.0, 'max': 16.0, 'step': 1.0, 'default': -1.0},
                'random_state': {'type': 'number', 'min': 0.0, 'max': 1000.0, 'step': 1.0, 'default': 42.0}
            }
        }
    }

    # Model Selection
    st.write("#### Select and Configure Model")
    selected_model = st.selectbox("Choose a model to configure", list(available_models.keys()), key="model_zoo_selection")

    if selected_model:
        st.write(f"##### Configure {selected_model} Parameters")
        
        # Get current model configuration from loaded config module
        current_config = {}
        if selected_model in current_models:
            model_instance = current_models[selected_model]
            # Get parameters from the model instance
            current_config = model_instance.get_params()

        # Create parameter input widgets with unique keys
        edited_params = {}
        for param_name, param_info in available_models[selected_model]['params'].items():
            current_value = current_config.get(param_name, param_info['default'])
            
            if param_info['type'] == 'select':
                edited_params[param_name] = st.selectbox(
                    param_name,
                    options=param_info['options'],
                    index=param_info['options'].index(current_value) if current_value in param_info['options'] else 0,
                    key=f"model_zoo_{selected_model}_{param_name}_select"
                )
            elif param_info['type'] == 'number':
                # Convert all numeric values to float for consistency
                current_value = float(current_value)
                edited_params[param_name] = st.number_input(
                    param_name,
                    min_value=float(param_info['min']),
                    max_value=float(param_info['max']),
                    step=float(param_info.get('step', 1.0)),
                    value=current_value,
                    key=f"model_zoo_{selected_model}_{param_name}_number"
                )

        # Display current configuration
        st.write("##### Current Configuration")
        st.json(edited_params)

        # Auto-save model configuration
        model_config_notification = st.empty()
        auto_save_model_config(selected_model, edited_params, config_content, config_path, available_models, model_config_notification)

    # Advanced Configuration Editor
    st.write("#### Advanced Configuration Editor")
    st.info("""
    Edit the model configurations directly below. You can also modify these settings by editing `config.py` directly.
    The MODELS section defines the model instances and their parameters, while HYPERPARAM_SPACE defines the parameter ranges for optimization.
    """)
    
    # Display the cleaned-up sections
    st.write("##### Model Definitions")
    # Now using AST extracted code, no need for cleaning here
    st.code(models_code, language='python')
    edited_models_code = st.text_area("Edit MODELS", models_code, height=300, key="advanced_models_editor")

    st.write("##### Hyperparameter Search Space")
    # Now using AST extracted code, no need for cleaning here
    st.code(hyperparams_code, language='python')
    edited_hyperparams_code = st.text_area("Edit HYPERPARAM_SPACE", hyperparams_code, height=200, key="advanced_hyperparams_editor")

    # Auto-save advanced configuration
    if 'previous_advanced_config' not in st.session_state:
        st.session_state.previous_advanced_config = {'models': '', 'hyperparams': ''}
    
    advanced_config_changed = (
        st.session_state.previous_advanced_config['models'] != edited_models_code or
        st.session_state.previous_advanced_config['hyperparams'] != edited_hyperparams_code
    )
    
    if advanced_config_changed and edited_models_code and edited_hyperparams_code:
        try:
            # Replace the sections in the config file while preserving the rest
            new_config_content = config_content
            if models_code != "# MODELS section not found":
                new_models_section = f"MODELS = {{\n{edited_models_code.split('=', 1)[1].strip()}\n}}"
                new_config_content = re.sub(r"MODELS\s*=\s*\{[^}]*\}", new_models_section, new_config_content, flags=re.DOTALL)
            if hyperparams_code != "# HYPERPARAM_SPACE section not found":
                new_hyperparams_section = f"HYPERPARAM_SPACE = {{\n{edited_hyperparams_code.split('=', 1)[1].strip()}\n}}"
                new_config_content = re.sub(r"HYPERPARAM_SPACE\s*=\s*\{[^}]*\}", new_hyperparams_section, new_config_content, flags=re.DOTALL)
            
            config_path.write_text(new_config_content)
            
            # Update previous config in session state
            st.session_state.previous_advanced_config = {
                'models': edited_models_code,
                'hyperparams': edited_hyperparams_code
            }
            
            st.success("✅ Advanced configuration auto-saved! You may need to restart the app to see the changes take effect.", icon="💾")
        except Exception as e:
            st.error(f"❌ Auto-save failed: {str(e)}")

def render_model_metrics_cheat_tab():
    """Display the model metrics from API/data service and explain why using these parameters for evaluation is cheating."""
    st.write("# Model Metrics (Full Data, Cost/Accuracy-Optimal Thresholds)")
    st.write("""
    This table shows the performance metrics for each model, split, and threshold type as retrieved from the pipeline's in-memory data.
    
    **What is being shown:**
    - Each row corresponds to a model evaluated on a particular split (e.g., Full, Train, Test) and at a particular threshold (cost-optimal or accuracy-optimal).
    - Metrics include accuracy, precision, recall, cost, threshold, and confusion matrix values (TP, FP, TN, FN).
    - These metrics are computed using the *best* threshold for each split, as determined by sweeping over all possible thresholds and picking the one that gives the lowest cost or highest accuracy.
    
    **Why using these parameters is technically cheating:**
    - The thresholds shown here are selected *after* seeing the true labels for the entire split (including the test or full data).
    - In a real-world deployment, you would not have access to the true labels for new, unseen data, so you cannot select the threshold that gives the best result on the test set.
    - This means the reported metrics are *optimistic* and do not reflect the true generalization performance of the model.
    - The correct way to evaluate a model is to select thresholds using only the training data (or via cross-validation), and then report performance on a held-out test set using those pre-selected thresholds.
    
    **In summary:**
    - The metrics here are useful for understanding the *potential* of your models, but should not be used as the final measure of model performance for decision-making or reporting.
    """)
    
    try:
        # Use cached metrics data instead of loading fresh each time
        metrics_data = get_cached_data(
            cache_key="metrics_data",
            api_endpoint="/results/metrics",
            default_value={"results": {"model_metrics": [], "model_summary": [], "confusion_matrices": []}},
            force_refresh=False
        )
        
        if metrics_data and metrics_data.get('results'):
            results = metrics_data['results']
            metrics_df = pd.DataFrame(results.get('model_metrics', []))
            
            if not metrics_df.empty:
                st.dataframe(metrics_df.style.format({
                    'accuracy': '{:.1f}%',
                    'precision': '{:.1f}%',
                    'recall': '{:.1f}%',
                    'cost': '{:.1f}',
                    'threshold': '{:.3f}'
                }))
                st.write("---")
                st.write("**Legend:**\n- TP: True Positives\n- FP: False Positives\n- TN: True Negatives\n- FN: False Negatives")
            else:
                st.error("No model metrics data found. Please run the pipeline first.")
        else:
            st.error("No model metrics data found. Please run the pipeline first.")
    except Exception as e:
        st.error(f"Error loading model metrics: {str(e)}")
        st.info("Please run the pipeline first to generate model metrics.")

def render_preprocessing_tab(config_settings):
    """Render the data preprocessing and configuration tab with automatic saving."""
    # Display success message if it exists in session state
    if st.session_state.get('config_update_success', False):
        st.success("Config updated successfully!")
        # Clear the success message after displaying it
        st.session_state.config_update_success = False
    
    st.write("### Data Preprocessing & Configuration")
    
    # Add cache management controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Use the refresh button if data seems out of sync with recent changes.")
    with col2:
        if st.button("🔄 Refresh Data", help="Clear cache and reload all data"):
            clear_cache()
            st.success("Cache cleared! Data will be refreshed.")
            st.rerun()
    
    data_dir = Path("data")
    if not data_dir.exists():
        st.warning("Data directory not found. Please upload data in the Data Management tab.")
        return
    
    available_datasets = list(data_dir.glob("*.csv"))
    if not available_datasets:
        st.info("No CSV datasets found in the data directory. Please upload data first.")
        return

    st.write("#### Dataset Selection")
    # Use file names relative to the data directory for cleaner display and saving
    dataset_options = [f.name for f in available_datasets]
    
    # Attempt to pre-select the currently configured dataset
    current_data_path_name = Path(config_settings.get('DATA_PATH', '')).name
    try:
        default_index = dataset_options.index(current_data_path_name) if current_data_path_name in dataset_options else 0
    except ValueError:
        default_index = 0 # Fallback if config path is invalid or file not found

    selected_dataset_name = st.selectbox(
        "Choose a dataset", 
        dataset_options,
        index=default_index,
        key="dataset_selection"
    )
    
    selected_dataset_path = data_dir / selected_dataset_name
    
    df = None
    try:
        df = pd.read_csv(selected_dataset_path, low_memory=False)
        # Convert object columns to string for consistent handling
        for col in df.columns:
             if df[col].dtype == 'object':
                 df[col] = df[col].astype(str)
                 
        st.write("**Selected Dataset Preview:**")
        st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return # Stop if dataset can't be loaded
        
    st.write("#### Column Configuration")
    
    all_columns = df.columns.tolist()
    
    # Target Column Selection
    current_target = config_settings.get('TARGET', '')
    try:
        target_default_index = all_columns.index(current_target) if current_target and current_target in all_columns else 0
    except ValueError:
         target_default_index = 0
         
    selected_target_column = st.selectbox(
        "Select Target Column",
        all_columns,
        index=target_default_index,
        key="target_column_selection"
    )
    
    # Exclude Columns Selection
    current_exclude = config_settings.get('EXCLUDE_COLS', [])
    # Ensure current_exclude are strings for comparison
    current_exclude_str = [str(col) for col in current_exclude]

    selected_exclude_columns = st.multiselect(
        "Select Columns to Exclude",
        all_columns,
        default=[col for col in all_columns if col in current_exclude_str], # Pre-select based on config
        key="exclude_columns_selection"
    )
    
    # Ensure target column is not in excluded columns
    if selected_target_column in selected_exclude_columns:
        st.warning("Target column cannot be in the list of excluded columns. Removing target from excluded.")
        selected_exclude_columns.remove(selected_target_column)
        st.rerun()

    # Auto-save data configuration
    config_updates = {
        'DATA_PATH': selected_dataset_path.as_posix(), # Use forward slashes
        'TARGET': selected_target_column,
        'EXCLUDE_COLS': selected_exclude_columns
    }
    
    # Also get and save the good/bad tags from base model config to main config
    try:
        response = requests.get(f"{API_BASE_URL}/config/base-models")
        if response.status_code == 200:
            base_config = response.json()['config']
            config_updates.update({
                'GOOD_TAG': base_config.get('good_tag', 'Good'),
                'BAD_TAG': base_config.get('bad_tag', 'Bad'),
                'BASE_MODEL_DECISION_COLUMNS': base_config.get('enabled_columns', ['AD_Decision', 'CL_Decision']),
                'COMBINED_FAILURE_MODEL_NAME': base_config.get('combined_failure_model', 'AD_or_CL_Fail')
            })
    except Exception as e:
        # If API call fails, use defaults to ensure config is still saved
        config_updates.update({
            'GOOD_TAG': 'Good',
            'BAD_TAG': 'Bad', 
            'BASE_MODEL_DECISION_COLUMNS': ['AD_Decision', 'CL_Decision'],
            'COMBINED_FAILURE_MODEL_NAME': 'AD_or_CL_Fail'
        })
    
    # Create notification container for auto-save messages
    data_config_notification = st.empty()
    
    # Use debounced auto-save for data configuration
    debounced_auto_save(
        save_function=_save_data_config_helper,
        config_data=config_updates,
        notification_container=data_config_notification,
        debounce_key="data_config",
        delay=3.0  # 3 second delay to prevent excessive file writes
    )
    
    # --- Base Model Decision Configuration ---
    st.write("#### Base Model Decision Configuration")
    st.write("Configure which columns contain base model decisions and the tags used for good/bad classifications.")
    
    # Use cached data for base model config
    base_model_config_data = get_cached_data(
        cache_key="base_model_config",
        api_endpoint="/config/base-models",
        default_value={"config": {"enabled_columns": [], "good_tag": "Good", "bad_tag": "Bad", "combined_failure_model": "yo mama"}, "available_columns": []}
    )
    
    if base_model_config_data:
        current_config = base_model_config_data['config']
        available_columns = base_model_config_data['available_columns']
        
        # Decision Columns Selection
        st.write("##### Decision Columns")
        st.write("Select columns that contain base model decisions (e.g., AD_Decision, CL_Decision):")
        
        selected_decision_columns = st.multiselect(
            "Decision Columns",
            available_columns,
            default=current_config.get('enabled_columns', []),
            help="These columns will be excluded from training features and used as base models for comparison",
            key="decision_columns_selection"
        )
        
        # Good/Bad Tags Configuration
        st.write("##### Good/Bad Tags")
        col1, col2 = st.columns(2)
        
        with col1:
            good_tag = st.text_input(
                "Good Tag",
                value=current_config.get('good_tag', 'Good'),
                help="The value in decision columns that represents a 'good' classification",
                key="good_tag_input"
            )
        
        with col2:
            bad_tag = st.text_input(
                "Bad Tag", 
                value=current_config.get('bad_tag', 'Bad'),
                help="The value in decision columns that represents a 'bad' classification",
                key="bad_tag_input"
            )
            
        # Auto-save base model configuration
        base_model_config_data_new = {
            "enabled_columns": selected_decision_columns,
            "good_tag": good_tag,
            "bad_tag": bad_tag,
            "combined_failure_model": current_config.get('combined_failure_model', 'AD_or_CL_Fail')  # Use existing value
        }
        
        # Create notification container for base model auto-save messages
        base_model_notification = st.empty()
        
        # Use debounced auto-save for base model configuration
        debounced_auto_save(
            save_function=_save_base_model_config_helper,
            config_data=base_model_config_data_new,
            notification_container=base_model_notification,
            debounce_key="base_model_config",
            delay=3.0  # 3 second delay to prevent excessive calls
        )
        
        # Show current configuration
        with st.expander("Current Base Model Configuration"):
            st.json(current_config)
            
    # --- Bitwise Logic Configuration ---
    st.write("#### 🔧 Bitwise Logic Configuration")
    st.write("Create custom models by combining existing model outputs using bitwise logic operations.")
    
    # Use cached data for bitwise logic config
    bitwise_data = get_cached_data(
        cache_key="bitwise_logic_config",
        api_endpoint="/config/bitwise-logic",
        default_value={"config": {"rules": [], "enabled": False}, "available_models": [], "available_logic_ops": ['OR', 'AND', 'XOR', '|', '&', '^']}
    )
    
    if bitwise_data:
        current_bitwise_config = bitwise_data['config']
        available_models = bitwise_data['available_models']
        available_logic_ops = bitwise_data['available_logic_ops']
        
        # Enable/Disable bitwise logic
        st.write("##### Enable Bitwise Logic")
        bitwise_enabled = st.checkbox(
            "Enable bitwise logic combinations",
            value=current_bitwise_config.get('enabled', False),
            help="Enable to create custom models by combining existing model outputs",
            key="bitwise_logic_enabled"
        )
        
        if bitwise_enabled:
            st.write("##### Logic Rules")
            st.info("💡 **How it works**: Select 2 or more models and choose a logic operation to combine their decisions. For example: 'AD_Decision OR CL_Decision' creates a model that predicts failure if either base model predicts failure.")
            
            # Initialize session state for rules
            if 'bitwise_rules' not in st.session_state:
                st.session_state.bitwise_rules = current_bitwise_config.get('rules', [])
            
            # Sync session state with current config if they differ
            config_rules = current_bitwise_config.get('rules', [])
            if st.session_state.bitwise_rules != config_rules:
                st.session_state.bitwise_rules = config_rules
            
            # Display existing rules
            if st.session_state.bitwise_rules:
                st.write("**Current Rules:**")
                for i, rule in enumerate(st.session_state.bitwise_rules):
                    col1, col2, col3, col4 = st.columns([3, 4, 2, 1])
                    with col1:
                        st.text_input(f"Rule {i+1} Name", value=rule['name'], disabled=True, key=f"rule_name_display_{i}")
                    with col2:
                        st.text(f"Columns: {', '.join(rule['columns'])}")
                    with col3:
                        st.text(f"Logic: {rule['logic']}")
                    with col4:
                        if st.button("🗑️", key=f"delete_rule_{i}", help="Delete this rule"):
                            # Remove the rule from session state
                            deleted_rule_name = st.session_state.bitwise_rules[i]['name']
                            st.session_state.bitwise_rules.pop(i)
                            
                            # Immediately save the updated configuration to backend
                            updated_config_data = {
                                "rules": st.session_state.bitwise_rules,
                                "enabled": bitwise_enabled
                            }
                            
                            try:
                                update_response = requests.post(
                                    f"{API_BASE_URL}/config/bitwise-logic", 
                                    json=updated_config_data,
                                    timeout=10
                                )
                                if update_response.status_code == 200:
                                    st.success(f"✅ Deleted rule '{deleted_rule_name}' and saved to backend!")
                                    # Update cache
                                    update_cache("bitwise_logic_config", {
                                        "config": updated_config_data,
                                        "available_models": available_models,
                                        "available_logic_ops": available_logic_ops
                                    })
                                    # Force refresh on other tabs
                                    if 'api_cache' in st.session_state:
                                        st.session_state.api_cache.pop('metrics_data', None)
                                        st.session_state.api_cache.pop('predictions_data', None)
                                else:
                                    st.error(f"❌ Failed to save after deletion: {update_response.text}")
                            except Exception as e:
                                st.error(f"❌ Failed to save after deletion: {str(e)}")
                            
                            st.rerun()
                
                # Add new rule section
                st.write("**Add New Rule:**")
                
                new_rule_col1, new_rule_col2, new_rule_col3 = st.columns([3, 4, 2])
                
                with new_rule_col1:
                    new_rule_name = st.text_input(
                        "Rule Name",
                        placeholder="e.g., 'Combined_Failure'",
                        help="Give your combined model a descriptive name",
                        key="new_rule_name"
                    )
                
                with new_rule_col2:
                    selected_models = st.multiselect(
                        "Select Models to Combine",
                        options=available_models,
                        default=[],
                        help="Choose 2 or more models to combine",
                        key="new_rule_models"
                    )
                
                with new_rule_col3:
                    selected_logic = st.selectbox(
                        "Logic Operation",
                        options=available_logic_ops,
                        help="OR: True if ANY model is true\nAND: True if ALL models are true\nXOR: True if ODD number of models are true",
                        key="new_rule_logic"
                    )
                
                # Add rule button
                add_rule_col1, add_rule_col2 = st.columns([1, 4])
                with add_rule_col1:
                    if st.button("➕ Add Rule", use_container_width=True):
                        if new_rule_name and len(selected_models) >= 2:
                            # Check for duplicate names
                            existing_names = [rule['name'] for rule in st.session_state.bitwise_rules]
                            if new_rule_name not in existing_names:
                                new_rule = {
                                    'name': new_rule_name,
                                    'columns': selected_models,
                                    'logic': selected_logic
                                }
                                st.session_state.bitwise_rules.append(new_rule)
                                st.success(f"Added rule: {new_rule_name}")
                                st.rerun()
                            else:
                                st.error("Rule name already exists. Please choose a different name.")
                        else:
                            st.error("Please provide a rule name and select at least 2 models.")
                
                with add_rule_col2:
                    # Logic operation explanations
                    with st.expander("Logic Operation Explanations"):
                        st.write("""
                        **OR (|)**: Result is True if ANY of the selected models predicts True
                        - Example: If AD_Decision OR CL_Decision, result is True if either model predicts failure
                        
                        **AND (&)**: Result is True if ALL of the selected models predict True  
                        - Example: If AD_Decision AND CL_Decision, result is True only if both models predict failure
                        
                        **XOR (^)**: Result is True if an ODD number of selected models predict True
                        - Example: If AD_Decision XOR CL_Decision, result is True if exactly one model predicts failure
                        """)
                
                # Apply rules section
                if st.session_state.bitwise_rules:
                    st.write("##### Apply Logic Rules")
                    
                    apply_col1, apply_col2 = st.columns([1, 3])
                    
                    with apply_col1:
                        if st.button("🔄 Apply Rules", use_container_width=True, type="secondary"):
                            # Save configuration and apply rules
                            bitwise_config_data = {
                                "rules": st.session_state.bitwise_rules,
                                "enabled": bitwise_enabled
                            }
                            
                            try:
                                # Update configuration
                                update_response = requests.post(
                                    f"{API_BASE_URL}/config/bitwise-logic", 
                                    json=bitwise_config_data
                                )
                                
                                if update_response.status_code == 200:
                                    # Update cache
                                    update_cache("bitwise_logic_config", {
                                        "config": bitwise_config_data,
                                        "available_models": available_models,
                                        "available_logic_ops": available_logic_ops
                                    })
                                    
                                    # Apply the rules
                                    apply_response = requests.post(f"{API_BASE_URL}/config/bitwise-logic/apply")
                                    
                                    if apply_response.status_code == 200:
                                        result = apply_response.json()
                                        st.success(f"✅ {result['message']}")
                                        if result['combined_models_created'] > 0:
                                            st.info(f"Created models: {', '.join(result['combined_model_names'])}")
                                            st.info("🔄 The Overview and Model Analysis tabs will now show your new combined models!")
                                            # Clear other caches so they reload with new data
                                            if 'api_cache' in st.session_state:
                                                st.session_state.api_cache.pop('metrics_data', None)
                                                st.session_state.api_cache.pop('predictions_data', None)
                                    else:
                                        st.error(f"Failed to apply rules: {apply_response.text}")
                                else:
                                    st.error(f"Failed to save configuration: {update_response.text}")
                            except Exception as e:
                                st.error(f"Error applying rules: {str(e)}")
                    
                    with apply_col2:
                        st.info("💡 Click 'Apply Rules' to create your combined models and update all dashboard results. The new models will appear in the Overview and Model Analysis tabs.")
                
                # Auto-save bitwise logic configuration (only when enabled state changes)
                if 'previous_bitwise_enabled' not in st.session_state:
                    st.session_state.previous_bitwise_enabled = current_bitwise_config.get('enabled', False)
                
                # Only save when enabled state actually changes
                if st.session_state.previous_bitwise_enabled != bitwise_enabled:
                    bitwise_config_data = {
                        "rules": st.session_state.bitwise_rules,
                        "enabled": bitwise_enabled
                    }
                    
                    try:
                        update_response = requests.post(
                            f"{API_BASE_URL}/config/bitwise-logic", 
                            json=bitwise_config_data,
                            timeout=10
                        )
                        if update_response.status_code == 200:
                            st.session_state.previous_bitwise_enabled = bitwise_enabled
                            current_time = datetime.now().strftime("%H:%M:%S")
                            st.success(f"✅ Bitwise logic enabled state saved at {current_time}!")
                            # Update cache
                            update_cache("bitwise_logic_config", {
                                "config": bitwise_config_data,
                                "available_models": available_models,
                                "available_logic_ops": available_logic_ops
                            })
                        else:
                            st.error(f"❌ Failed to save enabled state: {update_response.text}")
                    except Exception as e:
                        st.error(f"❌ Failed to save enabled state: {str(e)}")
                
                # Auto-save rules only when they actually change (with debouncing)
                def _save_bitwise_rules_helper(config_data, notification_container):
                    try:
                        update_response = requests.post(
                            f"{API_BASE_URL}/config/bitwise-logic", 
                            json=config_data,
                            timeout=10
                        )
                        if update_response.status_code == 200:
                            if 'previous_bitwise_rules' not in st.session_state:
                                st.session_state.previous_bitwise_rules = []
                            st.session_state.previous_bitwise_rules = config_data['rules'].copy()
                            current_time = datetime.now().strftime("%H:%M:%S")
                            notification_container.success(f"✅ Bitwise logic rules auto-saved at {current_time}!", icon="💾")
                            return True
                        else:
                            notification_container.error(f"❌ Auto-save failed: {update_response.text}")
                            return False
                    except Exception as e:
                        notification_container.error(f"❌ Auto-save failed: {str(e)}")
                        return False
                
                # Check if rules have changed
                if 'previous_bitwise_rules' not in st.session_state:
                    st.session_state.previous_bitwise_rules = current_bitwise_config.get('rules', [])
                
                if st.session_state.previous_bitwise_rules != st.session_state.bitwise_rules:
                    bitwise_config_data = {
                        "rules": st.session_state.bitwise_rules,
                        "enabled": bitwise_enabled
                    }
                    
                    # Create notification container for rules auto-save
                    rules_notification = st.empty()
                    
                    # Use debounced auto-save for rules
                    debounced_auto_save(
                        save_function=_save_bitwise_rules_helper,
                        config_data=bitwise_config_data,
                        notification_container=rules_notification,
                        debounce_key="bitwise_rules",
                        delay=2.0  # 2 second delay for rules
                    )
        
        else:
            st.error("Failed to load bitwise logic configuration")
    else:
        st.error("Failed to load base model configuration")
        
    # --- Preprocessing Previews ---
    st.write("#### Preprocessing Previews")
    
    # Filtering settings inputs
    st.write("##### Filtering Settings for Preview")
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        preview_variance_threshold = st.number_input(
            "Variance Threshold (>)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(config_settings.get('VARIANCE_THRESH', 0.01)),
            format="%.3f", key="preview_variance_threshold"
        )
    with col_filter2:
        preview_correlation_threshold = st.number_input(
            "Correlation Threshold (<)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(config_settings.get('CORRELATION_THRESH', 0.95)),
            format="%.3f", key="preview_correlation_threshold"
        )


    # Prepare data
    cols_to_process = [
        c for c in all_columns
        if c != selected_target_column and c not in selected_exclude_columns
    ]
    df_processed = df[cols_to_process].copy()

    # Show final processed columns after one-hot encoding and filtering
    st.write("##### Final Feature Columns (After One-Hot Encoding & Filtering)")
    try:
        # One-hot encoding
        df_ohe = pd.get_dummies(df_processed, drop_first=True)
        
        # Apply variance and correlation filtering
        df_filtered = df_ohe.copy()
        
        # Variance filter
        initial_cols = len(df_filtered.columns)
        df_filtered = df_filtered.loc[:, df_filtered.var() > preview_variance_threshold]
        variance_removed = initial_cols - len(df_filtered.columns)
        
        # Correlation filter
        corr = df_filtered.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > preview_correlation_threshold)]
        df_filtered = df_filtered.drop(columns=drop_cols)
        correlation_removed = len(drop_cols)
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Features", len(cols_to_process))
        with col2:
            st.metric("After One-Hot Encoding", len(df_ohe.columns))
        with col3:
            st.metric("Final Features", len(df_filtered.columns))
        
        # Show filtering details
        if variance_removed > 0 or correlation_removed > 0:
            st.write("**Filtering Applied:**")
            if variance_removed > 0:
                st.write(f"- Removed {variance_removed} features due to low variance (≤{preview_variance_threshold})")
            if correlation_removed > 0:
                st.write(f"- Removed {correlation_removed} features due to high correlation (≥{preview_correlation_threshold})")
        
        # Display the final column names
        st.write("**Final Feature Column Names:**")
        # Display in a more compact format - multiple columns
        cols_per_row = 4
        final_cols = df_filtered.columns.tolist()
        
        for i in range(0, len(final_cols), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(final_cols[i:i+cols_per_row]):
                with cols[j]:
                    st.write(f"• {col_name}")
        
        st.write(f"**Total: {len(final_cols)} features will be used for model training**")
        
    except Exception as e:
        st.error(f"Error during preprocessing preview: {e}")
