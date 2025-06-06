import pandas as pd
import numpy as np
import json
import requests
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import logging
import time
from datetime import datetime
import streamlit as st
import re
import threading
import copy

# Set up logging configuration for frontend
# Ensure logs directory exists
Path('output/logs').mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('output/logs/frontend.log', encoding='utf-8')
    ]
)

# Create logger for frontend API calls
frontend_logger = logging.getLogger('ensemble_frontend')

# Constants
OUTPUT_DIR = Path("output")
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
STREAMLIT_DATA_DIR = OUTPUT_DIR / "streamlit_data"

# Backend API configuration
BACKEND_API_URL = "http://localhost:8000"  # Adjust as needed

def ensure_directories():
    """Ensure all required directories exist."""
    for dir_path in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, PREDICTIONS_DIR, STREAMLIT_DATA_DIR]:
        dir_path.mkdir(exist_ok=True)

def load_initial_config():
    try:
        resp = requests.get("http://localhost:8000/config/load", timeout=5) 
        return resp.json().get("config", {})
    except Exception as e:
        import streamlit as st
        st.error(f"❌ Error loading config settings from API: {e}")
        return {}

def fetch_data_from_api(endpoint):
    """Fetch data from backend API with detailed logging."""
    start_time = time.time()
    full_url = f"{BACKEND_API_URL}{endpoint}"
    
    frontend_logger.info(f"🚀 FRONTEND API CALL: GET {full_url}")
    
    try:
        response = requests.get(full_url)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Log success with data details
                data_size = len(str(data))
                data_type = type(data).__name__
                
                # Try to extract meaningful info about the data structure
                if isinstance(data, dict):
                    if 'results' in data:
                        # Metrics data
                        results = data['results']
                        metrics_count = len(results.get('model_metrics', []))
                        summary_count = len(results.get('model_summary', []))
                        cm_count = len(results.get('confusion_matrices', []))
                        frontend_logger.info(f"✅ FRONTEND API SUCCESS: {endpoint} - {response_time:.3f}s - Metrics: {metrics_count}, Summaries: {summary_count}, CM: {cm_count}")
                    elif 'predictions' in data:
                        # Predictions data
                        pred_count = len(data['predictions'])
                        frontend_logger.info(f"✅ FRONTEND API SUCCESS: {endpoint} - {response_time:.3f}s - Predictions: {pred_count} items")
                    elif 'data' in data and 'model_name' in data:
                        # Threshold sweep data
                        data_points = len(data['data']) if isinstance(data['data'], list) else "N/A"
                        model_name = data['model_name']
                        data_type_name = data.get('data_type', 'unknown')
                        frontend_logger.info(f"✅ FRONTEND API SUCCESS: {endpoint} - {response_time:.3f}s - Model: {model_name}, Type: {data_type_name}, Points: {data_points}")
                    else:
                        # Generic data
                        keys = list(data.keys()) if isinstance(data, dict) else []
                        frontend_logger.info(f"✅ FRONTEND API SUCCESS: {endpoint} - {response_time:.3f}s - Data keys: {keys[:5]}")
                elif isinstance(data, list):
                    frontend_logger.info(f"✅ FRONTEND API SUCCESS: {endpoint} - {response_time:.3f}s - List with {len(data)} items")
                else:
                    frontend_logger.info(f"✅ FRONTEND API SUCCESS: {endpoint} - {response_time:.3f}s - {data_type}, Size: {data_size} chars")
                
                return data
                
            except json.JSONDecodeError as e:
                frontend_logger.error(f"❌ FRONTEND API JSON ERROR: {endpoint} - {response_time:.3f}s - Invalid JSON: {str(e)}")
                return None
        else:
            frontend_logger.error(f"❌ FRONTEND API ERROR: {endpoint} - Status: {response.status_code} - Time: {response_time:.3f}s - Response: {response.text[:200]}")
            return None
            
    except requests.exceptions.ConnectionError as e:
        response_time = time.time() - start_time
        frontend_logger.error(f"❌ FRONTEND API CONNECTION ERROR: {endpoint} - {response_time:.3f}s - Could not connect to {BACKEND_API_URL}")
        return None
    except Exception as e:
        response_time = time.time() - start_time
        frontend_logger.error(f"❌ FRONTEND API EXCEPTION: {endpoint} - {response_time:.3f}s - {str(e)}")
        return None

def load_metrics_data():
    """Load all metrics data for visualization using API first, then data service, then file fallback."""
    frontend_logger.info("🔄 Starting metrics data loading process...")
    
    # Try API first
    frontend_logger.info("📡 Attempting to load metrics data from API...")
    api_data = fetch_data_from_api("/results/metrics")
    if api_data and 'results' in api_data:
        frontend_logger.info("✅ Successfully loaded metrics data from API")
        metrics_data = api_data['results']
        
        # Convert to DataFrames for compatibility
        metrics_df = pd.DataFrame(metrics_data.get('model_metrics', []))
        summary_df = pd.DataFrame(metrics_data.get('model_summary', []))
        cm_df = pd.DataFrame(metrics_data.get('confusion_matrices', []))
        
        frontend_logger.info(f"📊 Converted to DataFrames - Metrics: {len(metrics_df)}, Summary: {len(summary_df)}, CM: {len(cm_df)}")
        
        # Try to get sweep data from multiple sources
        sweep_data = None
        
        # 1. Try to get sweep data from data service
        try:
            frontend_logger.info("🔍 Attempting to load sweep data from data service...")
            import sys
            from pathlib import Path
            root_dir = str(Path(__file__).parent.parent.parent)
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
            from shared import data_service
            sweep_data = data_service.get_sweep_data()
            if sweep_data:
                frontend_logger.info("✅ Got sweep data from data service")
        except Exception as e:
            frontend_logger.warning(f"⚠️ Could not load sweep data from data service: {e}")
        
        # 2. If data service failed, try to load from backup JSON file
        if sweep_data is None:
            try:
                frontend_logger.info("🔍 Attempting to load sweep data from backup JSON file...")
                backup_file = Path("output/data_service_backup/sweep.json")
                if backup_file.exists():
                    with open(backup_file, 'r') as f:
                        sweep_data = json.load(f)
                    frontend_logger.info("✅ Got sweep data from backup JSON file")
            except Exception as e:
                frontend_logger.warning(f"⚠️ Could not load sweep data from backup: {e}")
        
        # 3. If still no sweep data, try the old JSON file location
        if sweep_data is None:
            try:
                frontend_logger.info("🔍 Attempting to load sweep data from old JSON file...")
                old_sweep_file = STREAMLIT_DATA_DIR / "threshold_sweep_data.json"
                if old_sweep_file.exists():
                    with open(old_sweep_file, 'r') as f:
                        sweep_data = json.load(f)
                    frontend_logger.info("✅ Got sweep data from old JSON file")
            except Exception as e:
                frontend_logger.warning(f"⚠️ Could not load sweep data from old file: {e}")
        
        return metrics_df, summary_df, cm_df, sweep_data
    
    # Fall back to data service
    frontend_logger.info("⚠️ API failed, falling back to data service...")
    try:
        import sys
        from pathlib import Path
        root_dir = str(Path(__file__).parent.parent.parent)
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from shared import data_service
        
        # Try to get data from memory first
        metrics_data = data_service.get_metrics_data()
        sweep_data = data_service.get_sweep_data()
        
        if metrics_data:
            # Convert to DataFrames for compatibility
            metrics_df = pd.DataFrame(metrics_data.get('model_metrics', []))
            summary_df = pd.DataFrame(metrics_data.get('model_summary', []))
            cm_df = pd.DataFrame(metrics_data.get('confusion_matrices', []))
            frontend_logger.info("✅ Successfully loaded data from data service")
            return metrics_df, summary_df, cm_df, sweep_data
        else:
            frontend_logger.warning("⚠️ No data available in memory. Trying to load from files...")
            # Fall back to loading from files
            return load_metrics_data_from_files()
            
    except Exception as e:
        frontend_logger.error(f"❌ Could not load from data service: {e}")
        frontend_logger.info("🔄 Trying to load from files...")
        return load_metrics_data_from_files()

def load_metrics_data_from_files():
    """Load metrics data from CSV files as fallback."""
    try:
        # Check if files exist
        metrics_file = STREAMLIT_DATA_DIR / "model_metrics.csv"
        summary_file = STREAMLIT_DATA_DIR / "model_summary.csv"
        cm_file = STREAMLIT_DATA_DIR / "confusion_matrices.csv"
        sweep_file = STREAMLIT_DATA_DIR / "threshold_sweep_data.json"
        
        if not all(f.exists() for f in [metrics_file, summary_file, cm_file, sweep_file]):
            print("Required data files not found. Please run the pipeline first.")
            return None, None, None, None
        
        # Load CSV files
        metrics_df = pd.read_csv(metrics_file)
        summary_df = pd.read_csv(summary_file)
        cm_df = pd.read_csv(cm_file)
        
        # Load JSON file
        with open(sweep_file, 'r') as f:
            sweep_data = json.load(f)
        
        print("✅ Successfully loaded data from files")
        return metrics_df, summary_df, cm_df, sweep_data
        
    except Exception as e:
        print(f"Error loading from files: {e}")
        return None, None, None, None

def load_predictions_data():
    """Load predictions data using API first, then data service, then file fallback."""
    frontend_logger.info("🔄 Starting predictions data loading process...")
    
    # Try API first
    frontend_logger.info("📡 Attempting to load predictions data from API...")
    api_data = fetch_data_from_api("/results/predictions")
    if api_data and 'predictions' in api_data:
        predictions_count = len(api_data['predictions'])
        frontend_logger.info(f"✅ Successfully loaded {predictions_count} predictions from API")
        return pd.DataFrame(api_data['predictions'])
    
    # Fall back to data service
    frontend_logger.info("⚠️ API failed, falling back to data service...")
    try:
        import sys
        from pathlib import Path
        root_dir = str(Path(__file__).parent.parent.parent)
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from shared import data_service
        
        # Try to get predictions from memory first
        predictions_data = data_service.get_predictions_data()
        
        if predictions_data:
            predictions_count = len(predictions_data)
            frontend_logger.info(f"✅ Successfully loaded {predictions_count} predictions from data service")
            return pd.DataFrame(predictions_data)
        else:
            frontend_logger.warning("⚠️ No predictions data available in memory. Trying to load from files...")
            # Fall back to loading from files
            return load_predictions_data_from_files()
            
    except Exception as e:
        frontend_logger.error(f"❌ Could not load predictions from data service: {e}")
        frontend_logger.info("🔄 Trying to load from files...")
        return load_predictions_data_from_files()

def load_predictions_data_from_files():
    """Load predictions data from CSV files as fallback."""
    try:
        predictions_file = STREAMLIT_DATA_DIR / "all_model_predictions.csv"
        
        if not predictions_file.exists():
            print("Predictions file not found. Please run the pipeline first.")
            return None
        
        predictions_df = pd.read_csv(predictions_file)
        print("✅ Successfully loaded predictions data from files")
        return predictions_df
        
    except Exception as e:
        print(f"Error loading predictions from files: {e}")
        return None

def plot_confusion_matrix(cm_data):
    """Create a confusion matrix plot using Plotly."""
    # Extract values
    tp, fp, tn, fn = cm_data['tp'], cm_data['fp'], cm_data['tn'], cm_data['fn']
    
    # Create confusion matrix
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    
    # Create heatmap
    fig = px.imshow(
        cm_matrix,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Negative', 'Positive'],
        y=['Negative', 'Positive'],
        color_continuous_scale='Blues',
        text_auto=True,
        title=f"Confusion Matrix ({cm_data['threshold_type']} threshold)"
    )
    
    return fig

def plot_roc_curve(y_true, y_scores):
    """Create ROC curve plot using Plotly."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    return fig

def plot_precision_recall_curve(y_true, y_scores):
    """Create Precision-Recall curve plot using Plotly."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR Curve (AUC = {pr_auc:.3f})',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        showlegend=True
    )
    
    return fig

def plot_threshold_sweep(sweep_data, model_name):
    """Create threshold sweep plot using Plotly."""
    if model_name not in sweep_data:
        return None
        
    data = sweep_data[model_name]
    thresholds = data['thresholds']
    costs = data['costs']
    accuracies = data['accuracies']
    f1_scores = data.get('f1_scores', [])  # Get F1 scores if available
    
    fig = go.Figure()
    
    # Add cost curve
    fig.add_trace(go.Scatter(
        x=thresholds, y=costs,
        mode='lines+markers',
        name='Cost',
        yaxis='y',
        line=dict(color='red')
    ))
    
    # Add accuracy curve on secondary y-axis
    fig.add_trace(go.Scatter(
        x=thresholds, y=accuracies,
        mode='lines+markers',
        name='Accuracy',
        yaxis='y2',
        line=dict(color='blue')
    ))
    
    # Add F1 score curve if available
    if f1_scores:
        fig.add_trace(go.Scatter(
            x=thresholds, y=f1_scores,
            mode='lines+markers',
            name='F1 Score',
            yaxis='y2',
            line=dict(color='green')
        ))
    
    fig.update_layout(
        title=f'Threshold Sweep - {model_name}',
        xaxis_title='Threshold',
        yaxis=dict(title='Cost', side='left'),
        yaxis2=dict(title='Accuracy / F1 Score (%)', side='right', overlaying='y'),
        showlegend=True
    )
    
    return fig

def get_plot_groups(plot_dir):
    """Group plot files by type."""
    plot_files = list(plot_dir.glob("*.png"))
    groups = {}
    
    for plot_file in plot_files:
        name = plot_file.stem
        if 'comparison' in name.lower():
            group = 'Comparison'
        elif 'threshold' in name.lower() or 'sweep' in name.lower():
            group = 'Threshold Sweep'
        elif any(model in name for model in ['XGBoost', 'RandomForest', 'LogisticRegression']):
            # Extract model name
            for model in ['XGBoost', 'RandomForest', 'LogisticRegression']:
                if model in name:
                    group = model
                    break
        else:
            group = 'Other'
            
        if group not in groups:
            groups[group] = []
        groups[group].append(plot_file)
    
    return groups 

@st.cache_data(ttl=300)
def fetch_cached(endpoint: str, timestamp: float | None):
    """Fetch data from the backend API and cache the result."""
    try:
        response = requests.get(f"{BACKEND_API_URL}{endpoint}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_fresh_data(api_endpoint, default_value=None):
    """Fetch data using the cached helper with a timestamp key."""
    data = fetch_cached(api_endpoint, st.session_state.get('pipeline_completed_at'))
    return data if data is not None else default_value

# def debounced_auto_save(save_function, config_data, notification_container, debounce_key, delay=2.0):
#     """
#     Debounced auto-save function to prevent excessive API calls.
    
#     Args:
#         save_function: Function to call for saving
#         config_data: Data to save
#         notification_container: Streamlit container for notifications
#         debounce_key: Unique key for this auto-save operation
#         delay: Delay in seconds before saving
#     """
#     # Initialize debounce storage
#     if 'debounce_timers' not in st.session_state:
#         st.session_state.debounce_timers = {}
    
#     if 'debounce_data' not in st.session_state:
#         st.session_state.debounce_data = {}
    
#     # Store the current time and data
#     current_time = time.time()
#     st.session_state.debounce_timers[debounce_key] = current_time
#     st.session_state.debounce_data[debounce_key] = {
#         'save_function': save_function,
#         'config_data': config_data,
#         'notification_container': notification_container
#     }
    
#     # Check if enough time has passed for any pending saves
#     keys_to_process = []
#     for key, timestamp in st.session_state.debounce_timers.items():
#         if current_time - timestamp >= delay:
#             keys_to_process.append(key)
    
#     # Process debounced saves
#     for key in keys_to_process:
#         if key in st.session_state.debounce_data:
#             data = st.session_state.debounce_data[key]
#             try:
#                 result = data['save_function'](data['config_data'], data['notification_container'])
#                 if result:
#                     # Only show success message for actual saves
#                     pass  # The save function itself handles notifications
#             except Exception as e:
#                 data['notification_container'].error(f"❌ Auto-save failed: {str(e)}")
            
#             # Clean up
#             del st.session_state.debounce_timers[key]
#             del st.session_state.debounce_data[key]

# Cache functions removed - no longer using caching

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
            height=650,
            width=650,
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
    
    # Use a safe key for model selection
    threshold_models_key = "threshold_comparison_models"
    
    # Initialize state only once
    if threshold_models_key not in st.session_state:
        st.session_state[threshold_models_key] = available_models[:min(3, len(available_models))]  # Default to first 3 models
    
    selected_models = st.multiselect(
        "Select models for threshold comparison",
        options=available_models,
        key=threshold_models_key
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

def render_model_curves(model, sweep_data, model_data_split):
    """Render model curves and threshold analysis."""
    st.write("#### Model Curves and Threshold Analysis")
    
    # Load predictions and ensure we have the data
    try:
        # Get fresh predictions data from API
        predictions_data = get_fresh_data(
            api_endpoint="/results/predictions",
            default_value={"predictions": []}
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



def calculate_config_diff(current_config: dict, last_synced_config: dict) -> dict:
    """
    Calculate the differences between two config dictionaries.
    Returns only the changed/new values in their full nested structure.
    
    Args:
        current_config: The current config state
        last_synced_config: The last config that was synced to backend
        
    Returns:
        Dictionary containing only the sections that have differences
    """
    diff = {}
    
    def _deep_equal(a, b):
        """Check if two values are deeply equal."""
        if type(a) != type(b):
            return False
        if isinstance(a, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(_deep_equal(a[k], b[k]) for k in a.keys())
        elif isinstance(a, list):
            if len(a) != len(b):
                return False
            return all(_deep_equal(a[i], b[i]) for i in range(len(a)))
        else:
            return a == b
    
    # Compare top-level sections
    for section, current_value in current_config.items():
        last_value = last_synced_config.get(section, {})
        
        # If the entire section is different, include it
        if not _deep_equal(current_value, last_value):
            diff[section] = current_value
    
    # Check for any sections that exist in last_synced but not in current
    # (This handles deletions, though we may not support that)
    for section in last_synced_config.keys():
        if section not in current_config:
            # Section was removed - we don't handle this case for now
            pass
    
    return diff

def sync_frontend_to_backend(notification_container=None) -> bool:
    """
    Sync the frontend config to the backend by sending only the changed fields.
    Uses threading to prevent UI blocking.
    
    Args:
        notification_container: Optional streamlit container for notifications
        
    Returns:
        True if sync was initiated successfully, False otherwise
    """
    import streamlit as st
    import threading
    import copy
    
    try:
        # Get current frontend config
        current_config = st.session_state.get('config_settings', {})
        
        # Get last synced config (or empty dict if first sync)
        last_synced = st.session_state.get('last_synced_config', {})
        
        # Calculate differences to show user what's changing
        diff = calculate_config_diff(current_config, last_synced)
        
        if not diff:
            if notification_container:
                notification_container.info("✅ Config already in sync", icon="📋")
            return True
        
        # Show immediate feedback
        if notification_container:
            ts = datetime.now().strftime("%H:%M:%S")
            notification_container.info(f"🔄 Syncing {len(diff)} sections... {ts}", icon="⏳")
        
        # Update last synced config immediately to prevent duplicate syncs
        st.session_state.last_synced_config = copy.deepcopy(current_config)
        
        # Show success feedback immediately
        if notification_container:
            import time
            time.sleep(0.1)  # Small delay for better UX
            ts_success = datetime.now().strftime("%H:%M:%S")
            notification_container.success(f"✅ Synced {len(diff)} sections at {ts_success}", icon="📋")
        
        def _sync_to_backend():
            """Background thread function to sync to backend"""
            try:
                # Send only the changed fields instead of the entire config
                response = requests.post(
                    f"{BACKEND_API_URL}/config/update",
                    json=diff,  # Send only the diff instead of full config
                    timeout=5.0
                )
                
                # Log results to console instead of updating UI from thread
                if response.status_code == 200:
                    # Success
                    ts_success = datetime.now().strftime("%H:%M:%S")
                    print(f"[INFO] Config diff synced successfully at {ts_success}: {list(diff.keys())}")
                else:
                    # Failed - revert the optimistic update
                    error_msg = response.text if hasattr(response, 'text') else f"HTTP {response.status_code}"
                    print(f"[ERROR] Backend sync failed: HTTP {response.status_code} - {error_msg}")
                    # Note: We don't revert st.session_state here since it's from another thread
                
            except Exception as e:
                print(f"[ERROR] Backend sync failed: {e}")
                # Note: We don't revert st.session_state here since it's from another thread
        
        # Start background sync
        thread = threading.Thread(target=_sync_to_backend, daemon=True)
        thread.start()
        
        return True
        
    except Exception as e:
        if notification_container:
            notification_container.error(f"❌ Failed to start sync: {str(e)}", icon="🔄")
        print(f"[ERROR] Could not start sync: {e}")
        return False

def instant_update_config(endpoint: str, new_values: dict) -> bool:
    """
    Send a config update to the FastAPI backend without updating frontend state.

    Args:
        endpoint (str): Config section (e.g., "data", "model_params").
        new_values (dict): Key-value pairs to update (e.g., {"target_column": "DefectType"}).

    Returns:
        True if the request was sent (regardless of success); False on failure to start the thread.
    """
    import threading
    import requests

    def _post_update():
        try:
            response = requests.post(
                f"{BACKEND_API_URL}/config/update",
                json={endpoint: new_values},
                timeout=1.5,
            )
            if response.status_code != 200:
                print(f"[WARN] Failed to update {endpoint}: HTTP {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Backend update failed for {endpoint}: {e}")

    try:
        threading.Thread(target=_post_update, daemon=True).start()
        return True
    except Exception as e:
        print(f"[ERROR] Could not start background thread: {e}")
        return False

