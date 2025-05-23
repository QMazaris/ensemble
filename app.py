import streamlit as st
import subprocess, os
from pathlib import Path
import inspect
import config
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# 1) Page config
st.set_page_config(
    page_title="AI Pipeline Dashboard",
    layout="wide"
)

# Constants
OUTPUT_DIR = Path("output")
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
STREAMLIT_DATA_DIR = OUTPUT_DIR / "streamlit_data"

# Ensure directories exist
for dir_path in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, PREDICTIONS_DIR, STREAMLIT_DATA_DIR]:
    dir_path.mkdir(exist_ok=True)

# 2) Sidebar: Config inputs
st.sidebar.header("Settings")

# Model Settings
st.sidebar.subheader("Model Settings")
C_FP = st.sidebar.number_input("Cost of False-Positive", value=config.C_FP)
C_FN = st.sidebar.number_input("Cost of False-Negative", value=config.C_FN)
USE_KFOLD = st.sidebar.checkbox("Use K-Fold Cross Validation", value=config.USE_KFOLD)
if USE_KFOLD:
    N_SPLITS = st.sidebar.number_input("Number of K-Fold Splits", value=config.N_SPLITS, min_value=2, max_value=10)

# Feature Settings
st.sidebar.subheader("Feature Settings")
FilterData = st.sidebar.checkbox("Apply Feature Filtering", value=getattr(config, 'FilterData', False))
if FilterData:
    VARIANCE_THRESH = st.sidebar.slider("Variance Threshold", value=getattr(config, 'VARIANCE_THRESH', 0.01), 
                                      min_value=0.0, max_value=1.0, step=0.01)
    CORRELATION_THRESH = st.sidebar.slider("Correlation Threshold", value=getattr(config, 'CORRELATION_THRESH', 0.95),
                                         min_value=0.0, max_value=1.0, step=0.01)

# Optimization Settings
st.sidebar.subheader("Optimization Settings")
OPTIMIZE_HYPERPARAMS = st.sidebar.checkbox("Optimize Hyperparameters", value=getattr(config, 'OPTIMIZE_HYPERPARAMS', False))
OPTIMIZE_FINAL_MODEL = st.sidebar.checkbox("Optimize Final Model", value=getattr(config, 'OPTIMIZE_FINAL_MODEL', False))

# Save & Reload Config
if st.sidebar.button("💾 Save & Reload Config"):
    # Read current config.py
    cfg_file = inspect.getsourcefile(config)
    text = Path(cfg_file).read_text().splitlines()
    
    # Update config values
    updates = {
        "C_FP": C_FP,
        "C_FN": C_FN,
        "USE_KFOLD": USE_KFOLD,
        "N_SPLITS": N_SPLITS if USE_KFOLD else 5,
        "FilterData": FilterData,
        "VARIANCE_THRESH": VARIANCE_THRESH if FilterData else 0.01,
        "CORRELATION_THRESH": CORRELATION_THRESH if FilterData else 0.95,
        "OPTIMIZE_HYPERPARAMS": OPTIMIZE_HYPERPARAMS,
        "OPTIMIZE_FINAL_MODEL": OPTIMIZE_FINAL_MODEL
    }
    
    # Write updated config
    out = []
    for line in text:
        for key, value in updates.items():
            if line.strip().startswith(f"{key} ="):
                line = f"{key} = {value}"
                break
        out.append(line)
    
    Path(cfg_file).write_text("\n".join(out))
    st.success("Config saved! Reloading...")
    st.experimental_rerun()

# 3) Run pipeline
if st.sidebar.button("▶️ Run Pipeline"):
    with st.spinner("Running pipeline..."):
        try:
            # Run the main pipeline, which now includes metrics export
            subprocess.run(["python", "run.py"], check=True)
            st.success("Pipeline complete and metrics exported!")
        except subprocess.CalledProcessError as e:
            st.error(f"Pipeline failed with error: {str(e)}")
        except FileNotFoundError:
            st.error("Error: run.py not found. Please ensure run.py exists in the root directory.")

# 4) Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Model Analysis", "Plots Gallery", "Downloads"])

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
        # st.error(f"Error loading metrics data: {str(e)}") # Suppress error for cleaner look when file doesn't exist yet
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

with tab1:
    st.write("### Latest Metrics")
    
    # Load metrics data
    metrics_df, summary_df, cm_df, sweep_data = load_metrics_data()
    
    if summary_df is not None and not summary_df.empty:
        # Filter for 'Full' split and cost-optimal threshold for overview
        overview_data = summary_df[
            (summary_df['split'] == 'Full') &
            (summary_df['threshold_type'] == 'cost')
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Filter out decision-based models which don't have these metrics calculated via sweep
        # These models are 'AD_Decision', 'CL_Decision', 'AD_or_CL_Fail'
        overview_data = overview_data[
            ~overview_data['model_name'].isin(['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail'])
        ]
        
        if not overview_data.empty:
            # Model Performance Overview - Focus on models with sweeps
            st.write("#### Model Performance Overview (Cost-Optimal Threshold on Full Data)")
            
            # Create performance comparison bar chart
            fig = px.bar(overview_data, 
                        x='model_name', 
                        y=['accuracy', 'precision', 'recall'],
                        title='Model Performance Metrics',
                        barmode='group',
                        labels={'value': 'Score (%)', 'model_name': 'Model'})
            fig.update_layout(yaxis_range=[0, 100]) # Ensure y-axis is 0-100%
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost Analysis
            st.write("#### Cost Analysis (Cost-Optimal Threshold on Full Data)")
            fig = px.bar(overview_data,
                        x='model_name',
                        y='cost',
                        title='Model Costs',
                        labels={'cost': 'Cost', 'model_name': 'Model'})
            st.plotly_chart(fig, use_container_width=True)

        # Display detailed metrics table for all models/splits/threshold types
        st.write("#### Detailed Metrics Table (All Models, Splits, Thresholds)")
        st.dataframe(summary_df.style.format({
            'accuracy': '{:.1f}%',
            'precision': '{:.1f}%',
            'recall': '{:.1f}%',
            'cost': '{:.1f}',
            'threshold': '{:.3f}'
        }))
        
    else:
        st.info("Run the pipeline to see metrics here.")

with tab2:
    st.write("### Model Analysis")
    
    metrics_df, summary_df, cm_df, sweep_data = load_metrics_data()
    
    if summary_df is not None and not summary_df.empty:
        # Model Selection - Include all models from summary
        model_options = summary_df['model_name'].unique()
        model = st.selectbox("Select Model", model_options)
        
        # Filter data for selected model
        model_data = summary_df[summary_df['model_name'] == model]
        model_cm = cm_df[cm_df['model_name'] == model]

        # Split Selection (Train, Test, Full)
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
                # Display metrics for selected model and split
                metrics_display = model_data_split[['threshold_type', 'accuracy', 'precision', 'recall', 'cost', 'threshold']]
                st.dataframe(metrics_display.style.format({
                    'accuracy': '{:.1f}%',
                    'precision': '{:.1f}%',
                    'recall': '{:.1f}%',
                    'cost': '{:.1f}',
                    'threshold': '{:.3f}'
                }))
            
            with col2:
                st.write(f"#### Confusion Matrix ({selected_split} Split)")
                # Offer selection of threshold type for Confusion Matrix if both exist
                cm_threshold_options = model_cm_split['threshold_type'].unique()
                if len(cm_threshold_options) > 1:
                     selected_cm_threshold = st.selectbox("Select Threshold Type for CM", cm_threshold_options)
                else:
                     selected_cm_threshold = cm_threshold_options[0]

                cm_data = model_cm_split[model_cm_split['threshold_type'] == selected_cm_threshold].iloc[0]
                st.plotly_chart(plot_confusion_matrix(cm_data), use_container_width=True)

            # Check if threshold sweep data exists for ROC/PR curves and Threshold Analysis
            # This data is only available for models where a threshold sweep was performed (XGBoost, RandomForest, AD, CL scores)
            if model in sweep_data:
                 st.write("#### Model Curves and Threshold Analysis")
                 
                 # Create columns for plots
                 col3, col4 = st.columns(2)
                 
                 with col3:
                      # Load ground truth from predictions - assuming 'Full' split GT is sufficient here
                      # For split-specific curves, you'd need split-specific GT and probabilities
                      st.write("##### ROC Curve (Full Data)")
                      try:
                          pred_df = pd.read_csv(PREDICTIONS_DIR / 'all_model_predictions.csv')
                          y_true = pred_df['GT'].values
                          # Ensure probabilities align with full data if needed
                          probs = np.array(sweep_data[model]['probabilities'])
                          st.plotly_chart(plot_roc_curve(probs, y_true), use_container_width=True)
                      except FileNotFoundError:
                           st.warning("Predictions CSV not found for ROC/PR curves.")
                 
                 with col4:
                      st.write("##### Precision-Recall Curve (Full Data)")
                      try:
                          pred_df = pd.read_csv(PREDICTIONS_DIR / 'all_model_predictions.csv')
                          y_true = pred_df['GT'].values
                          # Ensure probabilities align with full data if needed
                          probs = np.array(sweep_data[model]['probabilities'])
                          st.plotly_chart(plot_precision_recall_curve(probs, y_true), use_container_width=True)
                      except FileNotFoundError:
                           st.warning("Predictions CSV not found for ROC/PR curves.")

                 # Threshold Analysis (Histogram and Sweep Plot)
                 st.write("##### Probability Distribution and Threshold Sweep")
                 
                 if model in sweep_data:
                      probs = np.array(sweep_data[model]['probabilities'])
                      # Histogram of probabilities (Full Data)
                      fig_hist = px.histogram(
                           x=probs,
                           title=f'Probability Distribution - {model}',
                           nbins=50,
                           labels={'x': 'Probability', 'y': 'Count'}
                      )
                      st.plotly_chart(fig_hist, use_container_width=True)

                      # Threshold Sweep Plot
                      # Note: This uses sweep data from the export script, which is currently Full data
                      # If you need split-specific sweeps plotted, the export script would need modification
                      sweep = sweep_data[model]
                      fig_sweep = go.Figure()
                      fig_sweep.add_trace(go.Scatter(x=sweep['thresholds'], y=sweep['costs'], mode='lines', name='Cost'))
                      fig_sweep.add_trace(go.Scatter(x=sweep['thresholds'], y=sweep['accuracies'], mode='lines', name='Accuracy', yaxis='y2'))

                      # Add markers for cost-optimal and accuracy-optimal thresholds from metrics_df
                      cost_optimal_thr = model_data_split[model_data_split['threshold_type'] == 'cost']['threshold'].iloc[0] if not model_data_split[model_data_split['threshold_type'] == 'cost'].empty else None
                      acc_optimal_thr = model_data_split[model_data_split['threshold_type'] == 'accuracy']['threshold'].iloc[0] if not model_data_split[model_data_split['threshold_type'] == 'accuracy'].empty else None

                      if cost_optimal_thr is not None:
                           fig_sweep.add_vline(x=cost_optimal_thr, line_dash="dash", line_color="red", annotation_text=f"Cost Optimal ({cost_optimal_thr:.3f})")
                      if acc_optimal_thr is not None:
                           fig_sweep.add_vline(x=acc_optimal_thr, line_dash="dash", line_color="green", annotation_text=f"Accuracy Optimal ({acc_optimal_thr:.3f})", annotation_position="bottom right")

                      fig_sweep.update_layout(
                           title=f'Threshold Sweep Analysis - {model}',
                           xaxis_title='Threshold',
                           yaxis_title='Cost',
                           yaxis2=dict(
                                title='Accuracy',
                                overlaying='y',
                                side='right',
                                range=[0, 100] # Assuming accuracy is 0-100%
                           )
                      )
                      st.plotly_chart(fig_sweep, use_container_width=True)

            elif model_data_split['threshold_type'].isin(['base']).any():
                 st.info("Threshold sweep data (for curves and analysis) is not available for decision-based models.")
            else:
                 st.info("Threshold sweep data is not available for this model or split.")

    else:
        st.info("Run the pipeline to see analysis here.")

# New tab for Plots Gallery
with tab3:
    st.write("### Plots Gallery")
    
    # Display plots from the output/plots directory
    imgs = sorted(PLOT_DIR.glob("*.png"))
    if not imgs:
        st.info("No static plots available. Run the pipeline to generate plots.")
    else:
        # Group plots by type (optional, but can help organize)
        plot_groups = {}
        for img in imgs:
            # Extract plot type from filename - adjust parsing as needed
            parts = img.stem.split('_')
            if len(parts) > 1:
                plot_type = parts[-1] # Assuming type is the last part after splitting by _
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
        
        # Display plots by group
        # Prioritize Comparison and Threshold Sweep plots
        ordered_groups = ['Comparison', 'Threshold Sweep'] + sorted([g for g in plot_groups.keys() if g not in ['Comparison', 'Threshold Sweep', 'Other']]) + ['Other']
        
        for group in ordered_groups:
             if group in plot_groups:
                 st.write(f"#### {group.replace('_', ' ').title()} Plots")
                 group_imgs = plot_groups[group]
                 # Display images in columns for better layout
                 cols_per_row = 3
                 rows = (len(group_imgs) + cols_per_row - 1) // cols_per_row
                 for r in range(rows):
                     cols = st.columns(cols_per_row)
                     for c in range(cols_per_row):
                         img_index = r * cols_per_row + c
                         if img_index < len(group_imgs):
                              img = group_imgs[img_index]
                              cols[c].image(str(img), caption=img.name, use_container_width=True)

with tab4:
    st.write("### Download Files")
    
    # Download predictions
    pred_file = MODEL_DIR / "all_model_predictions.csv"
    if pred_file.exists():
        with open(pred_file, 'rb') as f:
            st.download_button(
                "Download Predictions CSV",
                f.read(),
                file_name="all_model_predictions.csv",
                mime="text/csv"
            )
    
    # Download models
    st.write("#### Download Models")
    for model_file in MODEL_DIR.glob("*.pkl"):
        with open(model_file, 'rb') as f:
            st.download_button(
                f"Download {model_file.name}",
                f.read(),
                file_name=model_file.name,
                mime="application/octet-stream"
            ) 