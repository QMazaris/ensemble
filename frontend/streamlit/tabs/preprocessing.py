import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

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
        key="dataset_selection"
    )
    
    selected_dataset_path = data_dir / selected_dataset_name
    
    # Update config if dataset changed
    if selected_dataset_path.as_posix() != current_data_path:
        if 'data' not in config:
            config['data'] = {}
        config['data']['path'] = selected_dataset_path.as_posix()
        st.session_state.config_settings = config
    
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
        key="target_column_select"
    )
    
    # Update config if target column changed
    if selected_target_column != current_target:
        if 'data' not in config:
            config['data'] = {}
        config['data']['target_column'] = selected_target_column
        st.session_state.config_settings = config

    # ========== 3. BASE MODEL DECISION COLUMNS ==========
    st.write("#### Base Model Decision Configuration")
    
    current_decision_columns = config.get('data', {}).get('base_model_decisions', [])
    
    selected_decision_columns = st.multiselect(
        "Select Base Model Decision Columns",
        all_columns,
        default=current_decision_columns,
        help="Choose the columns that contain base model decisions (e.g., labeled good/bad)",
        key="decision_columns_select"
    )
    
    # Update config if decision columns changed
    if selected_decision_columns != current_decision_columns:
        if 'data' not in config:
            config['data'] = {}
        config['data']['base_model_decisions'] = selected_decision_columns
        st.session_state.config_settings = config

    # ========== 4. GOOD/BAD TAGS ==========
    st.write("#### Good/Bad Tags Configuration")
    col1, col2 = st.columns(2)
    
    current_good_tag = config.get('data', {}).get('good_tag', 'Good')
    current_bad_tag = config.get('data', {}).get('bad_tag', 'Bad')
    
    with col1:
        good_tag = st.text_input(
            "Good Tag",
            value=current_good_tag,
            help="The value in decision columns that represents a 'good' classification",
            key="good_tag_input"
        )
    
    with col2:
        bad_tag = st.text_input(
            "Bad Tag", 
            value=current_bad_tag,
            help="The value in decision columns that represents a 'bad' classification",
            key="bad_tag_input"
        )
    
    # Update config if tags changed
    if good_tag != current_good_tag or bad_tag != current_bad_tag:
        if 'data' not in config:
            config['data'] = {}
        config['data']['good_tag'] = good_tag
        config['data']['bad_tag'] = bad_tag
        st.session_state.config_settings = config

    # ========== 5. EXCLUDE COLUMNS ==========
    st.write("#### Exclude Columns Configuration")
    st.write("Select additional columns to exclude from model training (target and base decision columns are automatically excluded):")
    
    current_exclude = config.get('data', {}).get('exclude_columns', [])
    
    # Filter out target column and selected decision columns from exclude options
    columns_to_filter_out = [selected_target_column] + selected_decision_columns
    exclude_column_options = [col for col in all_columns if col not in columns_to_filter_out]
    
    # Only keep exclude columns that are still valid options
    valid_current_exclude = [col for col in current_exclude if col in exclude_column_options]
    
    selected_exclude_columns = st.multiselect(
        "Additional Columns to Exclude",
        exclude_column_options,
        default=valid_current_exclude,
        help="These columns will not be used as features for model training",
        key="exclude_columns_select"
    )
    
    # Update config if exclude columns changed
    if selected_exclude_columns != current_exclude:
        if 'data' not in config:
            config['data'] = {}
        config['data']['exclude_columns'] = selected_exclude_columns
        st.session_state.config_settings = config
    
    # Show what's being automatically excluded
    if columns_to_filter_out:
        st.info(f"🔒 **Automatically excluded:** {', '.join(columns_to_filter_out)} (target and base decision columns)")

    # ========== 6. FEATURE SETTINGS ==========
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
            key="variance_threshold_input"
        )
    with col_filter2:
        correlation_threshold = st.number_input(
            "Correlation Threshold (<)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(current_correlation_threshold),
            format="%.3f", 
            key="correlation_threshold_input"
        )

    # Update config if thresholds changed
    if variance_threshold != current_variance_threshold or correlation_threshold != current_correlation_threshold:
        if 'features' not in config:
            config['features'] = {}
        config['features']['variance_threshold'] = variance_threshold
        config['features']['correlation_threshold'] = correlation_threshold
        st.session_state.config_settings = config

    # ========== 7. PREPROCESSING PREVIEW ==========
    st.write("#### Preprocessing Preview")
    
    # Prepare data - using the properly filtered columns
    cols_to_process = [
        c for c in all_columns
        if c != selected_target_column and c not in selected_decision_columns and c not in selected_exclude_columns
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
