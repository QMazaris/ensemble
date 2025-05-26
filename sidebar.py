import streamlit as st
import inspect
from pathlib import Path
import config

def render_sidebar():
    """Render the sidebar configuration."""
    st.sidebar.header("Settings")

    # Model Settings
    st.sidebar.subheader("Model Settings")
    C_FP = st.sidebar.number_input("Cost of False-Positive", value=config.C_FP)
    C_FN = st.sidebar.number_input("Cost of False-Negative", value=config.C_FN)
    USE_KFOLD = st.sidebar.checkbox("Use K-Fold Cross Validation", value=config.USE_KFOLD)
    N_SPLITS = None
    if USE_KFOLD:
        N_SPLITS = st.sidebar.number_input("Number of K-Fold Splits", value=config.N_SPLITS, min_value=2, max_value=10)

    # Feature Settings
    st.sidebar.subheader("Feature Settings")
    FilterData = st.sidebar.checkbox("Apply Feature Filtering", value=getattr(config, 'FilterData', False))
    VARIANCE_THRESH = None
    CORRELATION_THRESH = None
    if FilterData:
        # Use columns to place slider and number input side-by-side
        col1, col2 = st.sidebar.columns(2)
        with col1:
            VARIANCE_THRESH_slider = st.slider("Variance Threshold", value=getattr(config, 'VARIANCE_THRESH', 0.01), 
                                          min_value=0.0, max_value=1.0, step=0.001, key='variance_slider')
        with col2:
             VARIANCE_THRESH = st.number_input("Variance Threshold Value", value=VARIANCE_THRESH_slider, min_value=0.0, max_value=1.0, step=0.001, key='variance_number', label_visibility="hidden")
        
        col3, col4 = st.sidebar.columns(2)
        with col3:
            CORRELATION_THRESH_slider = st.slider("Correlation Threshold", value=getattr(config, 'CORRELATION_THRESH', 0.95),
                                             min_value=0.0, max_value=1.0, step=0.001, key='correlation_slider')
        with col4:
            CORRELATION_THRESH = st.number_input("Correlation Threshold Value", value=CORRELATION_THRESH_slider, min_value=0.0, max_value=1.0, step=0.001, key='correlation_number', label_visibility="hidden")


    # Optimization Settings
    st.sidebar.subheader("Optimization Settings")
    OPTIMIZE_HYPERPARAMS = st.sidebar.checkbox("Optimize Hyperparameters", value=getattr(config, 'OPTIMIZE_HYPERPARAMS', False))
    HYPERPARAM_ITER = None
    if OPTIMIZE_HYPERPARAMS:
        HYPERPARAM_ITER = st.sidebar.number_input("Number of Optimization Iterations", 
                                                value=getattr(config, 'HYPERPARAM_ITER', 50),
                                                min_value=10, max_value=500, step=10)
    OPTIMIZE_FINAL_MODEL = st.sidebar.checkbox("Optimize Final Model", value=getattr(config, 'OPTIMIZE_FINAL_MODEL', False))

    return {
        'C_FP': C_FP,
        'C_FN': C_FN,
        'USE_KFOLD': USE_KFOLD,
        'N_SPLITS': N_SPLITS if USE_KFOLD else 5,
        'FilterData': FilterData,
        'VARIANCE_THRESH': VARIANCE_THRESH if FilterData else 0.01,
        'CORRELATION_THRESH': CORRELATION_THRESH if FilterData else 0.95,
        'OPTIMIZE_HYPERPARAMS': OPTIMIZE_HYPERPARAMS,
        'HYPERPARAM_ITER': HYPERPARAM_ITER if OPTIMIZE_HYPERPARAMS else 50,
        'OPTIMIZE_FINAL_MODEL': OPTIMIZE_FINAL_MODEL
    }

def save_config(config_updates):
    """Save updated configuration."""
    cfg_file = inspect.getsourcefile(config)
    text = Path(cfg_file).read_text().splitlines()
    
    out = []
    for line in text:
        for key, value in config_updates.items():
            if line.strip().startswith(f"{key} ="):
                line = f"{key} = {value}"
                break
        out.append(line)
    
    Path(cfg_file).write_text("\n".join(out))
    st.success("Config saved! Reloading...")
    st.rerun() 