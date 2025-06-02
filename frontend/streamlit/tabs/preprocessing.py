import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import time
from datetime import datetime
import sys



# Add parent directory to path to access utils
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add root directory to path to access shared modules
root_dir = str(Path(__file__).parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import utility functions
from utils import master_auto_save

# API Configuration
API_BASE_URL = "http://localhost:8000"

def render_preprocessing_tab():
    """Render the data preprocessing and configuration tab with automatic saving."""
    print("\n🔄 [TERMINAL] render_preprocessing_tab() called")
    
    # Get centralized config
    cfg = st.session_state.config_settings
    print(f"📊 [TERMINAL] Config received in preprocessing tab: {len(cfg)} keys")
    
    # Create notification container at the top for auto-save messages
    notification_container = st.empty()
    
    st.write("### Data Preprocessing & Configuration")
    
    data_dir = Path("data")
    if not data_dir.exists():
        st.warning("Data directory not found. Please upload data in the Data Management tab.")
        return
    
    available_datasets = list(data_dir.glob("*.csv"))
    if not available_datasets:
        st.info("No CSV datasets found in the data directory. Please upload data first.")
        return

    # ========== 1. DATASET SELECTION ==========
    st.write("#### Dataset Selection")
    
    # Read current values from config
    current_data_path = cfg.get('data', {}).get('path', '')
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
    
    current_target = cfg.get('data', {}).get('target_column', '')
    
    try:
        target_default_index = all_columns.index(current_target) if current_target and current_target in all_columns else 0
    except ValueError:
         target_default_index = 0
         
    selected_target_column = st.selectbox(
        "Select Target Column",
        all_columns,
        index=target_default_index,
        key="target_column_selection",
        help="Choose the column you want to predict"
    )
    
    # ========== 3. BASE MODEL DECISION CONFIGURATION ==========
    st.write("#### Base Model Decision Configuration")
    st.write("Configure which columns contain base model decisions and the tags used for good/bad classifications.")
    
    base_model_config = cfg.get('models', {}).get('base_model_decisions', {})
    current_decision_columns = base_model_config.get('enabled_columns', [])
    current_good_tag = base_model_config.get('good_tag', 'Good')
    current_bad_tag = base_model_config.get('bad_tag', 'Bad')
    
    st.write("##### Decision Columns")
    st.write("Select columns that contain base model decisions (e.g., AD_Decision, CL_Decision):")
    
    selected_decision_columns = st.multiselect(
        "Decision Columns",
        all_columns,
        default=current_decision_columns,
        help="These columns will be excluded from training features and used as base models for comparison",
        key="decision_columns_selection"
    )
    
    # Good/Bad Tags Configuration
    st.write("##### Good/Bad Tags")
    col1, col2 = st.columns(2)
    
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
    
    # ========== 4. BASE MODEL COLUMNS CONFIGURATION ==========
    st.write("#### Base Model Columns Configuration")
    st.write("Configure which columns contain base model outputs/scores (e.g., AnomalyScore, CL_ConfMax):")
    
    base_model_columns_list = cfg.get('models', {}).get('base_model_columns', [])
    
    # Convert list to dict for backward compatibility if needed
    if isinstance(base_model_columns_list, list):
        # Simple list of column names - create default mapping
        current_model_columns = {}
        for i, col_name in enumerate(base_model_columns_list):
            # Try to guess model name from column name
            if 'anomaly' in col_name.lower() or 'ad' in col_name.lower():
                model_name = 'AD'
            elif 'cl' in col_name.lower() or 'conf' in col_name.lower():
                model_name = 'CL'
            else:
                model_name = f'Model_{i+1}'
            current_model_columns[model_name] = col_name
    else:
        # Legacy dict structure with model_columns key
        current_model_columns = base_model_columns_list.get('model_columns', {})
    
    st.write("##### Model to Column Mappings")
    st.info("💡 **How it works**: Map each base model name to its corresponding output column. For example, 'AD' → 'AnomalyScore' means the AD model's output is in the AnomalyScore column.")
    
    # Convert to lists for easy editing
    model_names = list(current_model_columns.keys())
    column_names = list(current_model_columns.values())
    
    # Add empty slots for new mappings
    while len(model_names) < 6:  # Allow up to 6 base models
        model_names.append("")
        column_names.append("")
    
    # Create editable inputs
    updated_mappings = {}
    
    for i in range(6):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            model_name = st.text_input(
                f"Model Name {i+1}",
                value=model_names[i] if i < len(model_names) else "",
                key=f"base_model_name_{i}",
                help="Name of the base model (e.g., AD, CL, RF)"
            )
        
        with col2:
            # Use selectbox if we have available columns and this row has a model name
            try:
                current_index = all_columns.index(column_names[i]) if column_names[i] in all_columns else 0
            except (ValueError, IndexError):
                current_index = 0
            
            column_name = st.selectbox(
                f"Column Name {i+1}",
                [""] + all_columns,
                index=current_index + 1 if i < len(column_names) and column_names[i] in all_columns else 0,
                key=f"base_model_column_{i}",
                help="Column containing the model's output/score"
            )
        
        # Only add to mappings if both model name and column name are provided
        if model_name.strip() and column_name.strip():
            updated_mappings[model_name.strip()] = column_name.strip()
    
    # Show current configuration summary
    if updated_mappings:
        st.success(f"✅ {len(updated_mappings)} base model(s) configured")
        
        # Show current mappings
        with st.expander("Current Base Model Mappings"):
            for model_name, column_name in updated_mappings.items():
                st.write(f"• **{model_name}** → `{column_name}`")
        
        # Validate columns exist in dataset
        missing_columns = []
        for col in updated_mappings.values():
            if col not in all_columns:
                missing_columns.append(col)
        if missing_columns:
            st.warning(f"⚠️ Some columns may not exist in the dataset: {missing_columns}")
    else:
        st.warning("⚠️ No base models configured. The pipeline will skip base model analysis.")
    
    # ========== 5. EXCLUDE COLUMNS SELECTION (FILTERED) ==========
    st.write("#### Exclude Columns Configuration")
    st.write("Select additional columns to exclude from model training (target and base decision columns are automatically excluded):")
    
    current_exclude = cfg.get('data', {}).get('exclude_columns', [])
    current_exclude_str = [str(col) for col in current_exclude]
    
    # Filter out target column and selected decision columns from exclude options
    columns_to_filter_out = [selected_target_column] + selected_decision_columns
    exclude_column_options = [col for col in all_columns if col not in columns_to_filter_out]
    
    # Only keep exclude columns that are still valid options
    valid_current_exclude = [col for col in current_exclude_str if col in exclude_column_options]
    
    selected_exclude_columns = st.multiselect(
        "Additional Columns to Exclude",
        exclude_column_options,
        default=valid_current_exclude,
        help="These columns will not be used as features for model training",
        key="exclude_columns_selection"
    )
    
    # Show what's being automatically excluded
    if columns_to_filter_out:
        st.info(f"🔒 **Automatically excluded:** {', '.join(columns_to_filter_out)} (target and base decision columns)")

    # ========== 6. BITWISE LOGIC CONFIGURATION ==========
    st.write("#### 🔧 Bitwise Logic Configuration")
    st.write("Create custom models by combining existing model outputs using bitwise logic operations.")
    
    bitwise_config = cfg.get('models', {}).get('bitwise_logic', {})
    current_bitwise_enabled = bitwise_config.get('enabled', False)
    current_bitwise_rules = bitwise_config.get('rules', [])
    
    bitwise_enabled = st.checkbox(
        "Enable bitwise logic combinations",
        value=current_bitwise_enabled,
        help="Enable to create custom models by combining existing model outputs",
        key="bitwise_logic_enabled"
    )
    
    if bitwise_enabled:
        st.write("##### Logic Rules")
        st.info("💡 **How it works**: Select 2 or more models and choose a logic operation to combine their decisions. For example: 'AD_Decision OR CL_Decision' creates a model that predicts failure if either base model predicts failure.")
        
        # Initialize session state for rules management
        if 'bitwise_rules' not in st.session_state:
            st.session_state.bitwise_rules = current_bitwise_rules.copy()
        
        # Display existing rules
        st.write("**Current Rules:**")
        if st.session_state.bitwise_rules:
            for i, rule in enumerate(st.session_state.bitwise_rules):
                col1, col2, col3, col4 = st.columns([3, 4, 2, 1])
                with col1:
                    st.text_input(f"Rule {i+1} Name", value=rule['name'], disabled=True, key=f"rule_name_display_{i}")
                with col2:
                    st.text(f"Columns: {', '.join(rule['columns'])}")
                with col3:
                    st.text(f"Logic: {rule['logic']}")
                with col4:
                    if st.button("🗑️", key=f"delete_rule_{i}", help="Delete this rule"):
                        st.session_state.bitwise_rules.pop(i)
                        st.rerun()
        else:
            st.info("No rules configured yet. Add a new rule below.")
        
        # Add new rule section
        st.write("**Add New Rule:**")
        
        new_rule_col1, new_rule_col2, new_rule_col3 = st.columns([3, 4, 2])
        
        with new_rule_col1:
            new_rule_name = st.text_input(
                "Rule Name",
                placeholder="e.g., 'Combined_Failure'",
                help="Give your combined model a descriptive name",
                key="new_rule_name"
            )
        
        with new_rule_col2:
            available_models = selected_decision_columns + list(updated_mappings.keys())
            selected_models = st.multiselect(
                "Select Models to Combine",
                options=available_models,
                default=[],
                help="Choose 2 or more models to combine",
                key="new_rule_models"
            )
        
        with new_rule_col3:
            available_logic_ops = ['OR', 'AND', 'XOR', '|', '&', '^']
            selected_logic = st.selectbox(
                "Logic Operation",
                options=available_logic_ops,
                help="OR: True if ANY model is true\nAND: True if ALL models are true\nXOR: True if ODD number of models are true",
                key="new_rule_logic"
            )
        
        # Add rule button
        add_rule_col1, add_rule_col2 = st.columns([1, 4])
        with add_rule_col1:
            if st.button("➕ Add Rule", use_container_width=True):
                if new_rule_name and len(selected_models) >= 2:
                    # Check for duplicate names
                    existing_names = [rule['name'] for rule in st.session_state.bitwise_rules]
                    if new_rule_name not in existing_names:
                        new_rule = {
                            'name': new_rule_name,
                            'columns': selected_models,
                            'logic': selected_logic
                        }
                        st.session_state.bitwise_rules.append(new_rule)
                        st.success(f"Added rule: {new_rule_name}")
                        st.rerun()
                    else:
                        st.error("Rule name already exists. Please choose a different name.")
                else:
                    st.error("Please provide a rule name and select at least 2 models.")
        
        with add_rule_col2:
            # Logic operation explanations
            with st.expander("Logic Operation Explanations"):
                st.write("""
                **OR (|)**: Result is True if ANY of the selected models predicts True
                - Example: If AD_Decision OR CL_Decision, result is True if either model predicts failure
                
                **AND (&)**: Result is True if ALL of the selected models predict True  
                - Example: If AD_Decision AND CL_Decision, result is True only if both models predict failure
                
                **XOR (^)**: Result is True if an ODD number of selected models predict True
                - Example: If AD_Decision XOR CL_Decision, result is True if exactly one model predicts failure
                """)
    else:
        # Clear rules when disabled
        if 'bitwise_rules' in st.session_state:
            st.session_state.bitwise_rules = []
        st.info("Enable bitwise logic above to create custom models by combining existing model outputs.")
        
    # ========== 7. PREPROCESSING PREVIEWS ==========
    st.write("#### Preprocessing Previews")
    
    current_variance_threshold = cfg.get('features', {}).get('variance_threshold', 0.01)
    current_correlation_threshold = cfg.get('features', {}).get('correlation_threshold', 0.95)
    
    st.write("##### Filtering Settings for Preview")
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        preview_variance_threshold = st.number_input(
            "Variance Threshold (>)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(current_variance_threshold),
            format="%.3f", key="preview_variance_threshold"
        )
    with col_filter2:
        preview_correlation_threshold = st.number_input(
            "Correlation Threshold (<)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(current_correlation_threshold),
            format="%.3f", key="preview_correlation_threshold"
        )

    # Prepare data - now using the properly filtered columns
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
        df_filtered = df_filtered.loc[:, df_filtered.var() > preview_variance_threshold]
        variance_removed = initial_cols - len(df_filtered.columns)
        
        # Correlation filter
        corr = df_filtered.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > preview_correlation_threshold)]
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
                st.write(f"- Removed {variance_removed} features due to low variance (≤{preview_variance_threshold})")
            if correlation_removed > 0:
                st.write(f"- Removed {correlation_removed} features due to high correlation (≥{preview_correlation_threshold})")
        
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

    # ========== 8. AUTO-SAVE CONFIGURATION CHANGES ==========
    # Create a new config dict with all the current selections
    new_config = cfg.copy()
    
    # Ensure nested dicts exist
    if 'data' not in new_config:
        new_config['data'] = {}
    if 'models' not in new_config:
        new_config['models'] = {}
    if 'features' not in new_config:
        new_config['features'] = {}
    
    # Update data configuration
    new_config['data']['path'] = selected_dataset_path.as_posix()
    new_config['data']['target_column'] = selected_target_column
    new_config['data']['exclude_columns'] = selected_exclude_columns
    
    # Update base model decisions configuration
    if 'base_model_decisions' not in new_config['models']:
        new_config['models']['base_model_decisions'] = {}
    new_config['models']['base_model_decisions']['enabled_columns'] = selected_decision_columns
    new_config['models']['base_model_decisions']['good_tag'] = good_tag
    new_config['models']['base_model_decisions']['bad_tag'] = bad_tag
    
    # Update base model columns configuration (store as simple list)
    new_config['models']['base_model_columns'] = list(updated_mappings.values())
    
    # Update bitwise logic configuration
    if 'bitwise_logic' not in new_config['models']:
        new_config['models']['bitwise_logic'] = {}
    new_config['models']['bitwise_logic']['enabled'] = bitwise_enabled
    new_config['models']['bitwise_logic']['rules'] = st.session_state.get('bitwise_rules', [])
    
    # Update features configuration
    new_config['features']['variance_threshold'] = preview_variance_threshold
    new_config['features']['correlation_threshold'] = preview_correlation_threshold
    
    # Use master_auto_save for debounced saving
    # This will automatically handle the debouncing and only save when there are real changes
    master_auto_save(
        endpoint="update",  # Use the general update endpoint
        full_config=new_config,
        notification_container=notification_container,
        session_key="preprocessing_tab",
        delay=2.0  # Wait 2 seconds after last change before saving
    )
    
    print(f"\n✅ [TERMINAL] Preprocessing tab rendered and auto-save configured")
