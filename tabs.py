import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import (
    load_metrics_data, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_threshold_sweep, get_plot_groups,
    MODEL_DIR, PLOT_DIR, PREDICTIONS_DIR
)

def render_overview_tab():
    """Render the overview tab content."""
    st.write("### Latest Metrics")
    
    metrics_df, summary_df, cm_df, sweep_data = load_metrics_data()
    
    if summary_df is not None and not summary_df.empty:
        # Filter for 'Full' split and cost-optimal threshold for overview
        overview_data = summary_df[
            (summary_df['split'] == 'Full') &
            (summary_df['threshold_type'] == 'cost')
        ].copy()

        # Filter out decision-based models
        overview_data = overview_data[
            ~overview_data['model_name'].isin(['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail'])
        ]
        
        if not overview_data.empty:
            # Model Performance Overview
            st.write("#### Model Performance Overview (Cost-Optimal Threshold on Full Data)")
            fig = px.bar(overview_data, 
                        x='model_name', 
                        y=['accuracy', 'precision', 'recall'],
                        title='Model Performance Metrics',
                        barmode='group',
                        labels={'value': 'Score (%)', 'model_name': 'Model'})
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost Analysis
            st.write("#### Cost Analysis (Cost-Optimal Threshold on Full Data)")
            fig = px.bar(overview_data,
                        x='model_name',
                        y='cost',
                        title='Model Costs',
                        labels={'cost': 'Cost', 'model_name': 'Model'})
            st.plotly_chart(fig, use_container_width=True)

        # Display detailed metrics table
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

def render_model_analysis_tab():
    """Render the model analysis tab content."""
    st.write("### Model Analysis")
    
    metrics_df, summary_df, cm_df, sweep_data = load_metrics_data()
    
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
                cm_threshold_options = model_cm_split['threshold_type'].unique()
                selected_cm_threshold = (st.selectbox("Select Threshold Type for CM", cm_threshold_options) 
                                       if len(cm_threshold_options) > 1 else cm_threshold_options[0])

                cm_data = model_cm_split[model_cm_split['threshold_type'] == selected_cm_threshold].iloc[0]
                st.plotly_chart(plot_confusion_matrix(cm_data), use_container_width=True)

            # Model Curves and Threshold Analysis
            if model in sweep_data:
                render_model_curves(model, sweep_data, model_data_split)
            elif model_data_split['threshold_type'].isin(['base']).any():
                st.info("Threshold sweep data is not available for decision-based models.")
            else:
                st.info("Threshold sweep data is not available for this model or split.")
    else:
        st.info("Run the pipeline to see analysis here.")

def render_model_curves(model, sweep_data, model_data_split):
    """Render model curves and threshold analysis."""
    st.write("#### Model Curves and Threshold Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("##### ROC Curve (Full Data)")
        try:
            pred_df = pd.read_csv(PREDICTIONS_DIR / 'all_model_predictions.csv')
            y_true = pred_df['GT'].values
            probs = np.array(sweep_data[model]['probabilities'])
            st.plotly_chart(plot_roc_curve(probs, y_true), use_container_width=True)
        except FileNotFoundError:
            st.warning("Predictions CSV not found for ROC/PR curves.")
    
    with col4:
        st.write("##### Precision-Recall Curve (Full Data)")
        try:
            pred_df = pd.read_csv(PREDICTIONS_DIR / 'all_model_predictions.csv')
            y_true = pred_df['GT'].values
            probs = np.array(sweep_data[model]['probabilities'])
            st.plotly_chart(plot_precision_recall_curve(probs, y_true), use_container_width=True)
        except FileNotFoundError:
            st.warning("Predictions CSV not found for ROC/PR curves.")

    # Threshold Analysis
    st.write("##### Probability Distribution and Threshold Sweep")
    
    probs = np.array(sweep_data[model]['probabilities'])
    fig_hist = px.histogram(
        x=probs,
        title=f'Probability Distribution - {model}',
        nbins=50,
        labels={'x': 'Probability', 'y': 'Count'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Get optimal thresholds
    cost_optimal_thr = (model_data_split[model_data_split['threshold_type'] == 'cost']['threshold'].iloc[0] 
                       if not model_data_split[model_data_split['threshold_type'] == 'cost'].empty else None)
    acc_optimal_thr = (model_data_split[model_data_split['threshold_type'] == 'accuracy']['threshold'].iloc[0] 
                      if not model_data_split[model_data_split['threshold_type'] == 'accuracy'].empty else None)

    # Plot threshold sweep
    fig_sweep = plot_threshold_sweep(sweep_data[model], model, cost_optimal_thr, acc_optimal_thr)
    st.plotly_chart(fig_sweep, use_container_width=True)

def render_plots_gallery_tab():
    """Render the plots gallery tab content."""
    st.write("### Plots Gallery")
    
    imgs = sorted(PLOT_DIR.glob("*.png"))
    if not imgs:
        st.info("No static plots available. Run the pipeline to generate plots.")
    else:
        plot_groups = get_plot_groups(PLOT_DIR)
        ordered_groups = ['Comparison', 'Threshold Sweep'] + sorted([g for g in plot_groups.keys() 
                                                                    if g not in ['Comparison', 'Threshold Sweep', 'Other']]) + ['Other']
        
        for group in ordered_groups:
            if group in plot_groups:
                st.write(f"#### {group.replace('_', ' ').title()} Plots")
                group_imgs = plot_groups[group]
                cols_per_row = 3
                rows = (len(group_imgs) + cols_per_row - 1) // cols_per_row
                for r in range(rows):
                    cols = st.columns(cols_per_row)
                    for c in range(cols_per_row):
                        img_index = r * cols_per_row + c
                        if img_index < len(group_imgs):
                            img = group_imgs[img_index]
                            cols[c].image(str(img), caption=img.name, use_container_width=True)

def render_downloads_tab():
    """Render the downloads tab content."""
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