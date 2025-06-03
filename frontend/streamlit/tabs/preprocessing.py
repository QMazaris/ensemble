import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import time

from config_util import on_config_change, update_config_direct, on_dataset_change

# Backend API URL
BACKEND_API_URL = "http://localhost:8000"

# Import utility functions
import sys
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import clear_cache

def render_preprocessing_tab():
    """Render the data preprocessing and configuration tab."""
    
    st.write("### Data Preprocessing & Configuration")
    
    data_dir = Path("data")
    if not data_dir.exists():
        st.warning("Data directory not found. Please upload data in the Data Management tab.")
        return
    
    available_datasets = list(data_dir.glob("*.csv"))
    if not available_datasets:
        st.info("No CSV datasets found in the data directory. Please upload data first.")
        return

    # Get current config
    config = st.session_state.get('config_settings', {})
    
    # ========== 1. DATASET SELECTION ==========
    st.write("#### Dataset Selection")
    
    # Read current values from config
    current_data_path = config.get('data', {}).get('path', '')
    current_data_path_name = Path(current_data_path).name if current_data_path else ''
    
    # Dataset selection
    dataset_options = [f.name for f in available_datasets]
    try:
        default_index = dataset_options.index(current_data_path_name) if current_data_path_name in dataset_options else 0
    except ValueError:
        default_index = 0

    selected_dataset_name = st.selectbox(
        "Choose a dataset", 
        dataset_options,
        index=default_index,
        key="dataset_selection",
        on_change=on_dataset_change
    )
    
    selected_dataset_path = data_dir / selected_dataset_name
    
    # Load and preview dataset
    df = None
    try:
        df = pd.read_csv(selected_dataset_path, low_memory=False)
        # Convert object columns to string for consistent handling
        for col in df.columns:
             if df[col].dtype == 'object':
                 df[col] = df[col].astype(str)
                 
        st.write("**Selected Dataset Preview:**")
        st.dataframe(df.head(2))
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return
    
    all_columns = df.columns.tolist()
    
    # ========== 2. TARGET COLUMN SELECTION ==========
    st.write("#### Target Column Selection")
    
    current_target = config.get('data', {}).get('target_column', '')

    try:
        target_index = all_columns.index(current_target) if current_target in all_columns else 0
    except ValueError:
        target_index = 0

    selected_target_column = st.selectbox(
        "Select Target Column",
        all_columns,
        index=target_index,
        help="Choose the column you want to predict",
        key="target_column_select",
        on_change=lambda: on_config_change("data", "target_column", "target_column_select")
    )

    # ========== 3. BASE MODEL DECISION COLUMNS ==========
    st.write("#### Base Model Decision Configuration")

    # Use a safe key
    key = "decision_columns_select"

    # Initialize state only once
    if key not in st.session_state:
        current_decision_columns = config.get("models", {}).get("base_model_decisions", [])
        # Only keep decision columns that are still valid options (exist in the dataset)
        valid_current_decision_columns = [col for col in current_decision_columns if col in all_columns]
        st.session_state[key] = valid_current_decision_columns

    selected_decision_columns = st.multiselect(
        "Select Base Model Decision Columns",
        all_columns,
        key=key,
        help="Choose the columns that contain base model decisions (e.g., labeled good/bad)"
    )

    if st.button("Save Decision Columns", key="save_decision_cols"):
        update_config_direct("models", "base_model_decisions", selected_decision_columns)
        st.toast("Decision columns saved!", icon="✅")
        st.rerun()

    # ========== 4. GOOD/BAD TAGS ==========
    col1, col2 = st.columns(2)
    
    current_good_tag = config.get('data', {}).get('good_tag', 'Good')
    current_bad_tag = config.get('data', {}).get('bad_tag', 'Bad')
    
    with col1:
        st.text_input(
            "Good Tag",
            value=current_good_tag,
            help="The value in decision columns that represents a 'good' classification",
            key="good_tag_input",
            on_change=lambda: on_config_change("data", "good_tag", "good_tag_input")
        )
    
    with col2:
        st.text_input(
            "Bad Tag", 
            value=current_bad_tag,
            help="The value in decision columns that represents a 'bad' classification",
            key="bad_tag_input",
            on_change=lambda: on_config_change("data", "bad_tag", "bad_tag_input")
        )

    # ========== 5. BASE MODEL THRESHOLD COLUMNS ==========
    st.write("#### Base Model Threshold Columns")

    # Use a safe key
    key = "threshold_columns_select"

    # Initialize state only once
    if key not in st.session_state:
        current_base_model_columns = config.get("models", {}).get("base_model_columns", [])
        # Only keep base model columns that are still valid options (exist in the dataset)
        valid_current_base_model_columns = [col for col in current_base_model_columns if col in all_columns]
        st.session_state[key] = valid_current_base_model_columns

    selected_base_model_columns_threshold = st.multiselect(
        "Select Base Model Threshold Columns",
        all_columns,
        key=key,
        help="Choose the columns that contain base model predictions (eg. score, confidence, etc.)"
    )

    if st.button("Save Threshold Columns", key="save_threshold_cols"):
        update_config_direct("models", "base_model_columns", selected_base_model_columns_threshold)
        st.toast("Threshold columns saved!", icon="✅")
        st.rerun()

    # ========== 6. EXCLUDE COLUMNS ==========
    st.write("#### Exclude Columns Configuration")
    st.write("Select additional columns to exclude from model training (target column is automatically excluded):")
    
    # Use a safe key
    key = "exclude_columns_select"

    # Initialize state only once
    if key not in st.session_state:
        current_exclude = config.get('data', {}).get('exclude_columns', [])
        # Only filter out target column from exclude options
        exclude_column_options = [col for col in all_columns if col != selected_target_column]
        # Only keep exclude columns that are still valid options
        valid_current_exclude = [col for col in current_exclude if col in exclude_column_options]
        st.session_state[key] = valid_current_exclude
    
    # Only filter out target column from exclude options
    exclude_column_options = [col for col in all_columns if col != selected_target_column]
    
    selected_exclude_columns = st.multiselect(
        "Additional Columns to Exclude",
        exclude_column_options,
        key=key,
        help="These columns will not be used as features for model training"
    )
    
    if st.button("Save Exclude Columns", key="save_exclude_cols"):
        update_config_direct("data", "exclude_columns", selected_exclude_columns)
        st.toast("Exclude columns saved!", icon="✅")
        st.rerun()
    
    # Show what's being automatically excluded
    st.info(f"🔒 **Automatically excluded:** {selected_target_column} (target column)")

    # ========== 7. FEATURE SETTINGS ==========
    st.write("#### Feature Engineering Settings")
    
    current_variance_threshold = config.get('features', {}).get('variance_threshold', 0.01)
    current_correlation_threshold = config.get('features', {}).get('correlation_threshold', 0.95)
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        variance_threshold = st.number_input(
            "Variance Threshold (>)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(current_variance_threshold),
            format="%.3f", 
            key="variance_threshold_input",
            on_change=lambda: on_config_change("features", "variance_threshold", "variance_threshold_input")
        )
    with col_filter2:
        correlation_threshold = st.number_input(
            "Correlation Threshold (<)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(current_correlation_threshold),
            format="%.3f", 
            key="correlation_threshold_input",
            on_change=lambda: on_config_change("features", "correlation_threshold", "correlation_threshold_input")
        )


    # ========== 9. PREPROCESSING PREVIEW ==========
    
    # Prepare data - using the properly filtered columns
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
        df_filtered = df_filtered.loc[:, df_filtered.var() > variance_threshold]
        variance_removed = initial_cols - len(df_filtered.columns)
        
        # Correlation filter
        corr = df_filtered.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
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
                st.write(f"- Removed {variance_removed} features due to low variance (≤{variance_threshold})")
            if correlation_removed > 0:
                st.write(f"- Removed {correlation_removed} features due to high correlation (≥{correlation_threshold})")
        
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

def save_preprocessing_config():
    """Save current preprocessing configuration to backend."""
    try:
        # Get current config from session state
        config = st.session_state.get('config_settings', {})
        
        # Send config update to backend
        response = requests.post(
            f"{BACKEND_API_URL}/config/update",
            json=config,
            timeout=10
        )
        
        if response.status_code != 200:
            st.error(f"Failed to save config: {response.text}")
            return False
        
        return True
        
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

def clear_frontend_cache():
    """Clear the frontend cache to force data refresh."""
    clear_cache()
