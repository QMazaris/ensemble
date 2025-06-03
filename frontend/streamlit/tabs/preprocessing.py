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
    st.write("#### Base Model Configuration")

    # Get current decision columns from config
    current_decision_columns = config.get("models", {}).get("base_model_decisions", [])
    
    selected_decision_columns = st.multiselect(
        "Select Base Model Decision Columns",
        all_columns,
        default=current_decision_columns,
        help="Choose the columns that contain base model decisions (e.g., labeled good/bad)",
        key="base_model_decisions_multi"
    )

    # Update config when selection changes
    if selected_decision_columns != current_decision_columns:
        update_config_direct("models", "base_model_decisions", selected_decision_columns)

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

    # Get current base model columns from config
    current_base_model_columns = config.get("models", {}).get("base_model_columns", [])
    
    selected_base_model_columns_threshold = st.multiselect(
        "Select Base Model Threshold Columns",
        all_columns,
        default=current_base_model_columns,
        help="Choose the columns that contain base model predictions (eg. score, confidence, etc.)",
        key="base_model_columns_multi"
    )

    # Update config when selection changes
    if selected_base_model_columns_threshold != current_base_model_columns:
        update_config_direct("models", "base_model_columns", selected_base_model_columns_threshold)

    # ========== 6. EXCLUDE COLUMNS ==========
    st.write("#### Exclude Columns Configuration")
    st.write("Select additional columns to exclude from model training (target column is automatically excluded):")
    
    current_exclude = config.get('data', {}).get('exclude_columns', [])
    
    # Only filter out target column from exclude options
    exclude_column_options = [col for col in all_columns if col != selected_target_column]
    
    # Only keep exclude columns that are still valid options
    valid_current_exclude = [col for col in current_exclude if col in exclude_column_options]
    
    selected_exclude_columns = st.multiselect(
        "Additional Columns to Exclude",
        exclude_column_options,
        default=valid_current_exclude,
        help="These columns will not be used as features for model training",
        key="exclude_columns_multi"
    )
    
    # Update config when selection changes
    if selected_exclude_columns != valid_current_exclude:
        update_config_direct("data", "exclude_columns", selected_exclude_columns)
    
    # Show what's being automatically excluded
    st.info(f"🔒 **Automatically excluded:** {selected_target_column} (target column)")

    # ========== 7. BITWISE LOGIC CONFIGURATION ==========
    Bitwise_Logic(config, all_columns, selected_decision_columns)

    # ========== 8. FEATURE SETTINGS ==========
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

    # ========== 9. SUBMIT CONFIGURATION BUTTON ==========
    st.write("---")
    st.write("#### 🚀 Apply Configuration Changes")
    
    if st.button("💾 Save Config & Apply Bitwise Logic", 
                 type="primary", 
                 help="Save current configuration to backend and apply bitwise logic rules",
                 key="submit_config_button"):
        
        with st.spinner("Saving configuration and applying changes..."):
            success = save_and_apply_config()
            
            if success:
                st.success("✅ Configuration saved and applied successfully!")
                # Force refresh of cached data
                clear_frontend_cache()
                st.session_state.pipeline_completed_at = time.time()
                time.sleep(1)  # Brief pause to ensure backend processing completes
                st.rerun()
            else:
                st.error("❌ Failed to save configuration. Please check the logs.")

    # ========== 10. PREPROCESSING PREVIEW ==========
    
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

def save_and_apply_config():
    """Save current session config to backend and apply bitwise logic."""
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
        
        # Apply bitwise logic if any rules exist
        bitwise_rules = config.get('bitwise_logic', {}).get('rules', [])
        if bitwise_rules:
            response = requests.post(
                f"{BACKEND_API_URL}/config/bitwise-logic/apply",
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('combined_models_created', 0) > 0:
                    st.info(f"🔗 Created {result['combined_models_created']} combined models using bitwise logic rules")
            else:
                st.warning(f"Config saved but failed to apply bitwise logic: {response.text}")
                
        return True
        
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

def clear_frontend_cache():
    """Clear the frontend cache to force data refresh."""
    clear_cache()

def Bitwise_Logic(config, all_columns, selected_decision_columns):
    st.write("#### Bitwise Logic Configuration")
    st.write("Create logical combinations of decision columns:")
    
    # Initialize bitwise_logic in config if not exists
    if 'bitwise_logic' not in config:
        config['bitwise_logic'] = {'rules': []}
        st.session_state['config_settings'] = config
    
    current_rules = config.get('bitwise_logic', {}).get('rules', [])
    
    # Display existing rules
    if current_rules:
        st.write("**Existing Rules:**")
        for i, rule in enumerate(current_rules):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.write(f"• **{rule.get('name', f'Rule {i+1}')}**")
            with col2:
                st.write(f"Logic: {rule.get('logic', 'OR')}")
            with col3:
                st.write(f"Columns: {len(rule.get('columns', []))}")
            with col4:
                if st.button("🗑️", key=f"delete_rule_{i}", help="Delete this rule"):
                    current_rules.pop(i)
                    update_config_direct("bitwise_logic", "rules", current_rules)
                    st.toast("Rule deleted!", icon="🗑️")
                    st.rerun()
    
    # New rule creation
    st.write("**Add New Rule:**")
    col1, col2 = st.columns(2)
    
    with col1:
        rule_name = st.text_input(
            "Rule Name",
            placeholder="e.g., Combined_Decision",
            key="new_rule_name"
        )
    
    with col2:
        logic_type = st.selectbox(
            "Logic Type",
            ["OR", "AND", "XOR","|","&","^"],
            key="new_rule_logic"
        )
    
    # Only show decision columns for bitwise logic
    available_decision_columns = [col for col in all_columns if col in selected_decision_columns]
    
    if available_decision_columns:
        selected_rule_columns = st.multiselect(
            "Select Columns for Rule",
            available_decision_columns,
            help="Choose decision columns to combine with the selected logic",
            key="new_rule_columns"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Rule", key="add_bitwise_rule"):
                if rule_name and selected_rule_columns:
                    new_rule = {
                        "name": rule_name,
                        "logic": logic_type,
                        "columns": selected_rule_columns
                    }
                    current_rules.append(new_rule)
                    update_config_direct("bitwise_logic", "rules", current_rules)
                    st.toast(f"Rule '{rule_name}' added!", icon="✅")
                    st.rerun()
                else:
                    st.error("Please provide both rule name and select columns.")
        
    else:
        st.info("Please select decision columns first to create bitwise logic rules.")
