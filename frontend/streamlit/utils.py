import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Constants
OUTPUT_DIR = Path("output")
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
STREAMLIT_DATA_DIR = OUTPUT_DIR / "streamlit_data"

def ensure_directories():
    """Ensure all required directories exist."""
    for dir_path in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, PREDICTIONS_DIR, STREAMLIT_DATA_DIR]:
        dir_path.mkdir(exist_ok=True)

def load_metrics_data():
    """Load all metrics data for visualization using data service only."""
    # Import data service
    try:
        import sys
        from pathlib import Path
        root_dir = str(Path(__file__).parent.parent.parent)
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from shared import data_service
        
        # Get data from memory
        metrics_data = data_service.get_metrics_data()
        sweep_data = data_service.get_sweep_data()
        
        if metrics_data and sweep_data:
            # Convert to DataFrames for compatibility
            metrics_df = pd.DataFrame(metrics_data.get('model_metrics', []))
            summary_df = pd.DataFrame(metrics_data.get('model_summary', []))
            cm_df = pd.DataFrame(metrics_data.get('confusion_matrices', []))
            return metrics_df, summary_df, cm_df, sweep_data
        else:
            print("No data available in memory. Please run the pipeline first.")
            return None, None, None, None
            
    except Exception as e:
        print(f"Could not load from data service: {e}")
        return None, None, None, None

def load_predictions_data():
    """Load predictions data using data service only."""
    try:
        import sys
        from pathlib import Path
        root_dir = str(Path(__file__).parent.parent.parent)
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from shared import data_service
        
        # Get predictions from memory
        predictions_data = data_service.get_predictions_data()
        
        if predictions_data:
            return pd.DataFrame(predictions_data)
        else:
            print("No predictions data available in memory. Please run the pipeline first.")
            return None
            
    except Exception as e:
        print(f"Could not load predictions from data service: {e}")
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
    
    fig.update_layout(
        title=f'Threshold Sweep - {model_name}',
        xaxis_title='Threshold',
        yaxis=dict(title='Cost', side='left'),
        yaxis2=dict(title='Accuracy', side='right', overlaying='y'),
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