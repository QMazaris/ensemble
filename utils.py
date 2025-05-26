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
    """Load all metrics data for visualization."""
    try:
        metrics_df = pd.read_csv(STREAMLIT_DATA_DIR / 'model_metrics.csv')
        summary_df = pd.read_csv(STREAMLIT_DATA_DIR / 'model_summary.csv')
        cm_df = pd.read_csv(STREAMLIT_DATA_DIR / 'confusion_matrices.csv')
        with open(STREAMLIT_DATA_DIR / 'threshold_sweep_data.json', 'r') as f:
            sweep_data = json.load(f)
        return metrics_df, summary_df, cm_df, sweep_data
    except Exception as e:
        return None, None, None, None

def plot_confusion_matrix(cm_data):
    """Create a confusion matrix heatmap."""
    cm = [[cm_data['tn'], cm_data['fp']], 
          [cm_data['fn'], cm_data['tp']]]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Negative', 'Positive'],
        y=['Negative', 'Positive'],
        text=cm,
        texttemplate='%{text}',
        textfont={"size":16},
        colorscale='RdBu'
    ))
    fig.update_layout(
        title=f"Confusion Matrix - {cm_data['model_name']} ({cm_data['threshold_type']})",
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )
    return fig

def plot_roc_curve(probs, y_true):
    """Create ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random',
        line=dict(dash='dash', color='gray')
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=700
    )
    return fig

def plot_precision_recall_curve(probs, y_true):
    """Create Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        name=f'PR curve (AUC = {pr_auc:.2f})',
        line=dict(width=2)
    ))
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=700
    )
    return fig

def plot_threshold_sweep(sweep_data, model_name, cost_optimal_thr=None, acc_optimal_thr=None):
    """Create threshold sweep analysis plot."""
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(x=sweep_data['thresholds'], y=sweep_data['costs'], mode='lines', name='Cost'))
    fig_sweep.add_trace(go.Scatter(x=sweep_data['thresholds'], y=sweep_data['accuracies'], mode='lines', name='Accuracy', yaxis='y2'))

    if cost_optimal_thr is not None:
        fig_sweep.add_vline(x=cost_optimal_thr, line_dash="dash", line_color="red", 
                           annotation_text=f"Cost Optimal ({cost_optimal_thr:.3f})")
    if acc_optimal_thr is not None:
        fig_sweep.add_vline(x=acc_optimal_thr, line_dash="dash", line_color="green", 
                           annotation_text=f"Accuracy Optimal ({acc_optimal_thr:.3f})", 
                           annotation_position="bottom right")

    fig_sweep.update_layout(
        title=f'Threshold Sweep Analysis - {model_name}',
        xaxis_title='Threshold',
        yaxis_title='Cost',
        yaxis2=dict(
            title='Accuracy',
            overlaying='y',
            side='right',
            range=[0, 100]
        )
    )
    return fig_sweep

def get_plot_groups(plot_dir):
    """Group plots by type for the gallery."""
    plot_groups = {}
    for img in sorted(plot_dir.glob("*.png")):
        parts = img.stem.split('_')
        if len(parts) > 1:
            plot_type = parts[-1]
            if plot_type in ['cost', 'accuracy'] and len(parts) > 2 and parts[-2] == 'optimized':
                plot_type = 'Comparison'
            elif plot_type == 'sweep' and len(parts) > 2:
                plot_type = 'Threshold Sweep'
            else:
                plot_type = 'Other'
        else:
            plot_type = 'Other'
            
        if plot_type not in plot_groups:
            plot_groups[plot_type] = []
        plot_groups[plot_type].append(img)
    
    return plot_groups 