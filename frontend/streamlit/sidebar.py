import streamlit as st
from pathlib import Path
import sys
import copy

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from frontend.streamlit.utils import sync_frontend_to_backend, calculate_config_diff
from config_util import on_config_change

def render_sidebar():
    """Render the sidebar configuration with automatic saving."""
    st.sidebar.header("⚙️ Pipeline Settings")
    
    # Get current config from session state
    config = st.session_state.get('config_settings', {})

    # Model Settings
    st.sidebar.subheader("💰 Cost Settings")
    C_FP = st.sidebar.number_input(
        "Cost of False-Positive", 
        value=float(config.get('costs', {}).get('false_positive', 1.0)),
        min_value=0.0,
        help="Cost penalty for false positive predictions",
        key="cost_fp",
        on_change=lambda: on_config_change("costs", "false_positive", "cost_fp")
    )
    C_FN = st.sidebar.number_input(
        "Cost of False-Negative", 
        value=float(config.get('costs', {}).get('false_negative', 30.0)),
        min_value=0.0,
        help="Cost penalty for false negative predictions (usually higher)",
        key="cost_fn",
        on_change=lambda: on_config_change("costs", "false_negative", "cost_fn")
    )
    
    # Training Settings
    st.sidebar.subheader("🎯 Training Settings")
    N_SPLITS = st.sidebar.number_input(
        "Number of K-Fold Splits", 
        value=config.get('training', {}).get('n_splits', 5), 
        min_value=2, 
        max_value=10,
        step=1,
        help="Number of folds for cross-validation (K-fold is always enabled)",
        key="n_splits",
        on_change=lambda: on_config_change("training", "n_splits", "n_splits")
    )

    # Feature Settings
    st.sidebar.subheader("🔧 Feature Engineering")
    FilterData = st.sidebar.checkbox(
        "Apply Feature Filtering", 
        value=config.get('features', {}).get('filter_data', False),
        help="Enable feature filtering to remove low-variance and highly-correlated features",
        key="filter_data",
        on_change=lambda: on_config_change("features", "filter_data", "filter_data")
    )
    
    if FilterData:
        # Use columns to place slider and number input side-by-side
        col1, col2 = st.sidebar.columns(2)
        with col1:
            VARIANCE_THRESH_slider = st.slider(
                "Variance Threshold", 
                value=config.get('features', {}).get('variance_threshold', 0.01), 
                min_value=0.0, 
                max_value=1.0, 
                step=0.001, 
                key='variance_slider',
                help="Remove features with variance below this threshold",
                on_change=lambda: on_config_change("features", "variance_threshold", "variance_slider")
            )
        with col2:
             VARIANCE_THRESH = st.number_input(
                 "Variance Threshold Value", 
                 value=VARIANCE_THRESH_slider, 
                 min_value=0.0, 
                 max_value=1.0, 
                 step=0.001, 
                 key='variance_number', 
                 label_visibility="hidden",
                 on_change=lambda: on_config_change("features", "variance_threshold", "variance_number")
             )
        
        col3, col4 = st.sidebar.columns(2)
        with col3:
            CORRELATION_THRESH_slider = st.slider(
                "Correlation Threshold", 
                value=config.get('features', {}).get('correlation_threshold', 0.95),
                min_value=0.0, 
                max_value=1.0, 
                step=0.001, 
                key='correlation_slider',
                help="Remove features with correlation above this threshold",
                on_change=lambda: on_config_change("features", "correlation_threshold", "correlation_slider")
            )
        with col4:
            CORRELATION_THRESH = st.number_input(
                "Correlation Threshold Value", 
                value=CORRELATION_THRESH_slider, 
                min_value=0.0, 
                max_value=1.0, 
                step=0.001, 
                key='correlation_number', 
                label_visibility="hidden",
                on_change=lambda: on_config_change("features", "correlation_threshold", "correlation_number")
            )

    # Optimization Settings
    st.sidebar.subheader("🚀 Optimization Settings")
    OPTIMIZE_HYPERPARAMS = st.sidebar.checkbox(
        "Optimize Hyperparameters", 
        value=config.get('optimization', {}).get('enabled', False),
        help="Enable hyperparameter optimization using Optuna",
        key="optimize_hyperparams",
        on_change=lambda: on_config_change("optimization", "enabled", "optimize_hyperparams")
    )
    
    if OPTIMIZE_HYPERPARAMS:
        HYPERPARAM_ITER = st.sidebar.number_input(
            "Number of Optimization Iterations", 
            value=config.get('optimization', {}).get('iterations', 50),
            min_value=10, 
            max_value=500, 
            step=10,
            help="Number of trials for hyperparameter optimization",
            key="hyperparam_iter",
            on_change=lambda: on_config_change("optimization", "iterations", "hyperparam_iter")
        )
    
    OPTIMIZE_FINAL_MODEL = st.sidebar.checkbox(
        "Create Final Model", 
        value=config.get('optimization', {}).get('optimize_final_model', False),
        help="Apply hyperparameter optimization to the final production model",
        key="optimize_final_model",
        on_change=lambda: on_config_change("optimization", "optimize_final_model", "optimize_final_model")
    )

    # Export Settings
    st.sidebar.subheader("📦 Export Settings")
    EXPORT_ONNX = st.sidebar.checkbox(
        "Export Models as ONNX",
        value=config.get('export', {}).get('export_onnx', False),
        help="When enabled, models will be exported in ONNX format for deployment",
        key="export_onnx",
        on_change=lambda: on_config_change("export", "export_onnx", "export_onnx")
    )
    
    if EXPORT_ONNX:
        ONNX_OPSET_VERSION = st.sidebar.number_input(
            "ONNX Opset Version",
            min_value=9,
            max_value=15,
            value=config.get('export', {}).get('onnx_opset_version', 12),
            step=1,
            help="ONNX opset version to use for model export",
            key="onnx_opset_version",
            on_change=lambda: on_config_change("export", "onnx_opset_version", "onnx_opset_version")
        )

    # Ensure training.use_kfold is always True (no user input needed)
    if 'training' not in config:
        config['training'] = {}
    config['training']['use_kfold'] = True
    st.session_state.config_settings = config
    
    # ========== CONFIG SYNC SECTION ==========
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Config Sync")
    
    # Check if there are unsaved changes
    current_config = st.session_state.get('config_settings', {})
    last_synced = st.session_state.get('last_synced_config', {})
    
    # Check if sync just happened (for immediate feedback)
    just_synced = st.session_state.get('just_synced', False)
    if just_synced:
        # Clear the flag and show sync success
        st.session_state.just_synced = False
        st.sidebar.success("✅ Config in sync", icon="📋")
        diff = {}
        unsaved_changes = 0
    else:
        diff = calculate_config_diff(current_config, last_synced)
        unsaved_changes = len(diff)
    
    # Show sync status
    if unsaved_changes > 0:
        st.sidebar.warning(f"⚠️ {unsaved_changes} unsaved changes", icon="📝")
    elif not just_synced:
        st.sidebar.success("✅ Config in sync", icon="📋")
    
    # Sync button and notification area
    sync_notification = st.sidebar.empty()
    
    # Sync button - only enabled if there are changes
    if st.sidebar.button(
        "🔄 Sync to Backend", 
        use_container_width=True, 
        disabled=(unsaved_changes == 0),
        help="Send frontend config changes to backend" if unsaved_changes > 0 else "No changes to sync"
    ):  
        print(diff)
        success = sync_frontend_to_backend(sync_notification)
        if success:
            # Set flag for immediate feedback
            st.session_state.just_synced = True
            # Force immediate rerun to update the UI with new sync status
            st.rerun()

    # Show what would be synced (for debugging/transparency)
    if unsaved_changes > 0:
        with st.sidebar.expander(f"📋 View {unsaved_changes} pending changes"):
            for section, changes in diff.items():
                st.write(f"**{section}:**")
                if isinstance(changes, dict):
                    for key, value in changes.items():
                        st.write(f"  • {key}: `{value}`")
                else:
                    st.write(f"  • {changes}")
    
    return config 