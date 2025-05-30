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

# Set up logging configuration for frontend
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

# Ensure logs directory exists
Path('output/logs').mkdir(parents=True, exist_ok=True)

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