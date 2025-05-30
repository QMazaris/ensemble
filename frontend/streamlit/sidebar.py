import streamlit as st
from pathlib import Path
import sys

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from shared.config_manager import get_config

def auto_save_config(config_updates):
    """Automatically save configuration if changes are detected."""
    # Initialize session state for previous config if not exists
    if 'previous_config' not in st.session_state:
        st.session_state.previous_config = {}
    
    # Check if configuration has changed
    config_changed = False
    for key, value in config_updates.items():
        if key not in st.session_state.previous_config or st.session_state.previous_config[key] != value:
            config_changed = True
            break
    
    # Save if configuration changed
    if config_changed:
        config = get_config()
        config.update(config_updates)
        config.save()
        
        # Update previous config in session state
        st.session_state.previous_config = config_updates.copy()
        
        # Show brief auto-save notification
        st.sidebar.success("✅ Config auto-saved!", icon="💾")
        
        return True
    return False

def render_sidebar():
    """Render the sidebar configuration with automatic saving."""
    st.sidebar.header("⚙️ Pipeline Settings")
    
    # Get current config
    config = get_config()

    # Model Settings
    st.sidebar.subheader("💰 Cost Settings")
    C_FP = st.sidebar.number_input(
        "Cost of False-Positive", 
        value=float(config.cost_fp),
        min_value=0.0,
        help="Cost penalty for false positive predictions",
        key="cost_fp"
    )
    C_FN = st.sidebar.number_input(
        "Cost of False-Negative", 
        value=float(config.cost_fn),
        min_value=0.0,
        help="Cost penalty for false negative predictions (usually higher)",
        key="cost_fn"
    )
    
    # Training Settings
    st.sidebar.subheader("🎯 Training Settings")
    N_SPLITS = st.sidebar.number_input(
        "Number of K-Fold Splits", 
        value=config.n_splits, 
        min_value=2, 
        max_value=10,
        step=1,
        help="Number of folds for cross-validation (K-fold is always enabled)",
        key="n_splits"
    )

    # Feature Settings
    st.sidebar.subheader("🔧 Feature Engineering")
    FilterData = st.sidebar.checkbox(
        "Apply Feature Filtering", 
        value=config.get('features.filter_data', False),
        help="Enable feature filtering to remove low-variance and highly-correlated features",
        key="filter_data"
    )
    VARIANCE_THRESH = None
    CORRELATION_THRESH = None
    if FilterData:
        # Use columns to place slider and number input side-by-side
        col1, col2 = st.sidebar.columns(2)
        with col1:
            VARIANCE_THRESH_slider = st.slider(
                "Variance Threshold", 
                value=config.get('features.variance_threshold', 0.01), 
                min_value=0.0, 
                max_value=1.0, 
                step=0.001, 
                key='variance_slider',
                help="Remove features with variance below this threshold"
            )
        with col2:
             VARIANCE_THRESH = st.number_input(
                 "Variance Threshold Value", 
                 value=VARIANCE_THRESH_slider, 
                 min_value=0.0, 
                 max_value=1.0, 
                 step=0.001, 
                 key='variance_number', 
                 label_visibility="hidden"
             )
        
        col3, col4 = st.sidebar.columns(2)
        with col3:
            CORRELATION_THRESH_slider = st.slider(
                "Correlation Threshold", 
                value=config.get('features.correlation_threshold', 0.95),
                min_value=0.0, 
                max_value=1.0, 
                step=0.001, 
                key='correlation_slider',
                help="Remove features with correlation above this threshold"
            )
        with col4:
            CORRELATION_THRESH = st.number_input(
                "Correlation Threshold Value", 
                value=CORRELATION_THRESH_slider, 
                min_value=0.0, 
                max_value=1.0, 
                step=0.001, 
                key='correlation_number', 
                label_visibility="hidden"
            )

    # Optimization Settings
    st.sidebar.subheader("🚀 Optimization Settings")
    OPTIMIZE_HYPERPARAMS = st.sidebar.checkbox(
        "Optimize Hyperparameters", 
        value=config.get('optimization.enabled', False),
        help="Enable hyperparameter optimization using Optuna",
        key="optimize_hyperparams"
    )
    HYPERPARAM_ITER = None
    if OPTIMIZE_HYPERPARAMS:
        HYPERPARAM_ITER = st.sidebar.number_input(
            "Number of Optimization Iterations", 
            value=config.get('optimization.iterations', 50),
            min_value=10, 
            max_value=500, 
            step=10,
            help="Number of trials for hyperparameter optimization",
            key="hyperparam_iter"
        )
    OPTIMIZE_FINAL_MODEL = st.sidebar.checkbox(
        "Create Final Model", 
        value=config.get('optimization.optimize_final_model', False),
        help="Apply hyperparameter optimization to the final production model",
        key="optimize_final_model"
    )

    # Export Settings
    st.sidebar.subheader("📦 Export Settings")
    EXPORT_ONNX = st.sidebar.checkbox(
        "Export Models as ONNX",
        value=config.get('export.export_onnx', False),
        help="When enabled, models will be exported in ONNX format for deployment",
        key="export_onnx"
    )
    ONNX_OPSET_VERSION = None
    if EXPORT_ONNX:
        ONNX_OPSET_VERSION = st.sidebar.number_input(
            "ONNX Opset Version",
            min_value=9,
            max_value=15,
            value=config.get('export.onnx_opset_version', 12),
            step=1,
            help="ONNX opset version to use for model export",
            key="onnx_opset_version"
        )

    # Build configuration updates
    config_updates = {
        'costs.false_positive': C_FP,
        'costs.false_negative': C_FN,
        'training.use_kfold': True,
        'training.n_splits': int(N_SPLITS),
        'features.filter_data': FilterData,
        'features.variance_threshold': VARIANCE_THRESH if FilterData else 0.01,
        'features.correlation_threshold': CORRELATION_THRESH if FilterData else 0.95,
        'optimization.enabled': OPTIMIZE_HYPERPARAMS,
        'optimization.iterations': int(HYPERPARAM_ITER) if OPTIMIZE_HYPERPARAMS else 50,
        'optimization.optimize_final_model': OPTIMIZE_FINAL_MODEL,
        'export.export_onnx': EXPORT_ONNX,
        'export.onnx_opset_version': int(ONNX_OPSET_VERSION) if EXPORT_ONNX and ONNX_OPSET_VERSION is not None else config.get('export.onnx_opset_version', 12)
    }
    
    # Auto-save configuration changes
    auto_save_config(config_updates)
    
    return config_updates 