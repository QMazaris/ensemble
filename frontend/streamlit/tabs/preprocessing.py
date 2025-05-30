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
from utils import (
    get_cached_data, clear_cache, update_cache,
    debounced_auto_save, _save_data_config_helper, _save_base_model_config_helper
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def render_preprocessing_tab(config_settings):
    """Render the data preprocessing and configuration tab with automatic saving."""
    # Display success message if it exists in session state
    if st.session_state.get('config_update_success', False):
        st.success("Config updated successfully!")
        # Clear the success message after displaying it
        st.session_state.config_update_success = False
    
    st.write("### Data Preprocessing & Configuration")
    
    # Add debug info to see current config values
    with st.expander("🔍 Debug: Current Config Values"):
        st.write("**Raw config structure:**")
        st.json(config_settings)
        st.write("**Parsed values:**")
        st.write(f"- Data path: `{config_settings.get('data', {}).get('path', 'NOT_FOUND')}`")
        st.write(f"- Target column: `{config_settings.get('data', {}).get('target_column', 'NOT_FOUND')}`")
        st.write(f"- Exclude columns: `{config_settings.get('data', {}).get('exclude_columns', 'NOT_FOUND')}`")
    
    # Add cache management controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Use the refresh button if data seems out of sync with recent changes.")
    with col2:
        if st.button("🔄 Refresh Data", help="Clear cache and reload all data"):
            clear_cache()
            st.success("Cache cleared! Data will be refreshed.")
            st.rerun()
    
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
    # Use file names relative to the data directory for cleaner display and saving
    dataset_options = [f.name for f in available_datasets]
    
    # Attempt to pre-select the currently configured dataset - fix key mapping
    # The config comes from YAML with nested structure
    current_data_path = config_settings.get('data', {}).get('path', '')
    current_data_path_name = Path(current_data_path).name if current_data_path else ''
    try:
        default_index = dataset_options.index(current_data_path_name) if current_data_path_name in dataset_options else 0
    except ValueError:
        default_index = 0 # Fallback if config path is invalid or file not found

    selected_dataset_name = st.selectbox(
        "Choose a dataset", 
        dataset_options,
        index=default_index,
        key="dataset_selection"
    )
    
    selected_dataset_path = data_dir / selected_dataset_name
    
    df = None
    try:
        df = pd.read_csv(selected_dataset_path, low_memory=False)
        # Convert object columns to string for consistent handling
        for col in df.columns:
             if df[col].dtype == 'object':
                 df[col] = df[col].astype(str)
                 
        st.write("**Selected Dataset Preview:**")
        st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return # Stop if dataset can't be loaded
    
    all_columns = df.columns.tolist()
    
    # ========== 2. TARGET COLUMN SELECTION ==========
    st.write("#### Target Column Selection")
    
    # Fix key mapping for target column
    current_target = config_settings.get('data', {}).get('target_column', '')
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
    
    # Use cached data for base model config
    base_model_config_data = get_cached_data(
        cache_key="base_model_config",
        api_endpoint="/config/base-models",
        default_value={"config": {"enabled_columns": [], "good_tag": "Good", "bad_tag": "Bad"}, "available_columns": []}
    )
    
    selected_decision_columns = []
    good_tag = "Good"
    bad_tag = "Bad"
    
    if base_model_config_data:
        current_config = base_model_config_data['config']
        available_columns = base_model_config_data['available_columns']
        
        # Decision Columns Selection
        st.write("##### Decision Columns")
        st.write("Select columns that contain base model decisions (e.g., AD_Decision, CL_Decision):")
        
        selected_decision_columns = st.multiselect(
            "Decision Columns",
            available_columns,
            default=current_config.get('enabled_columns', []),
            help="These columns will be excluded from training features and used as base models for comparison",
            key="decision_columns_selection"
        )
        
        # Good/Bad Tags Configuration
        st.write("##### Good/Bad Tags")
        col1, col2 = st.columns(2)
        
        with col1:
            good_tag = st.text_input(
                "Good Tag",
                value=current_config.get('good_tag', 'Good'),
                help="The value in decision columns that represents a 'good' classification",
                key="good_tag_input"
            )
        
        with col2:
            bad_tag = st.text_input(
                "Bad Tag", 
                value=current_config.get('bad_tag', 'Bad'),
                help="The value in decision columns that represents a 'bad' classification",
                key="bad_tag_input"
            )
            
        # Auto-save base model configuration
        base_model_config_data_new = {
            "enabled_columns": selected_decision_columns,
            "good_tag": good_tag,
            "bad_tag": bad_tag,
        }
        
        # Create notification container for base model auto-save messages
        base_model_notification = st.empty()
        
        # Use debounced auto-save for base model configuration
        debounced_auto_save(
            save_function=_save_base_model_config_helper,
            config_data=base_model_config_data_new,
            notification_container=base_model_notification,
            debounce_key="base_model_config",
            delay=3.0  # 3 second delay to prevent excessive calls
        )
        
        # Show current configuration
        with st.expander("Current Base Model Configuration"):
            st.json(current_config)
    else:
        st.error("Failed to load base model configuration")
    
    # ========== 4. EXCLUDE COLUMNS SELECTION (FILTERED) ==========
    st.write("#### Exclude Columns Configuration")
    st.write("Select additional columns to exclude from model training (target and base decision columns are automatically excluded):")
    
    # Filter out target column and selected decision columns from exclude options
    columns_to_filter_out = [selected_target_column] + selected_decision_columns
    exclude_column_options = [col for col in all_columns if col not in columns_to_filter_out]
    
    # Get current exclude columns from config, but filter out any that are now target/decision columns
    # Fix key mapping for exclude columns
    current_exclude = config_settings.get('data', {}).get('exclude_columns', [])
    current_exclude_str = [str(col) for col in current_exclude]
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

    # ========== 5. AUTO-SAVE DATA CONFIGURATION ==========
    # Auto-save data configuration - use YAML structure for saving
    config_updates = {
        'data.path': selected_dataset_path.as_posix(), # Use forward slashes
        'data.target_column': selected_target_column,
        'data.exclude_columns': selected_exclude_columns
    }
    
    # Also get and save the good/bad tags from base model config to main config
    config_updates.update({
        'models.base_model_decisions.good_tag': good_tag,
        'models.base_model_decisions.bad_tag': bad_tag,
        'models.base_model_decisions.enabled_columns': selected_decision_columns,
    })
    
    # Create notification container for auto-save messages
    data_config_notification = st.empty()
    
    # Use debounced auto-save for data configuration
    debounced_auto_save(
        save_function=_save_data_config_helper,
        config_data=config_updates,
        notification_container=data_config_notification,
        debounce_key="data_config",
        delay=3.0  # 3 second delay to prevent excessive file writes
    )
        
    # ========== 6. BITWISE LOGIC CONFIGURATION ==========
    st.write("#### 🔧 Bitwise Logic Configuration")
    st.write("Create custom models by combining existing model outputs using bitwise logic operations.")
    
    # Add refresh button specifically for bitwise logic
    bitwise_col1, bitwise_col2 = st.columns([4, 1])
    with bitwise_col1:
        st.write("")  # Just for spacing
    with bitwise_col2:
        if st.button("🔄 Refresh Rules", help="Reload bitwise logic rules from backend config", key="refresh_bitwise_rules"):
            # Clear session state and cache for bitwise logic
            if 'bitwise_rules' in st.session_state:
                del st.session_state.bitwise_rules
            if 'bitwise_rules_initialized' in st.session_state:
                del st.session_state.bitwise_rules_initialized
            if 'api_cache' in st.session_state and 'bitwise_logic_config' in st.session_state.api_cache:
                del st.session_state.api_cache['bitwise_logic_config']
            st.success("Bitwise logic rules refreshed from backend!")
            st.rerun()
    
    # Use cached data for bitwise logic config
    bitwise_data = get_cached_data(
        cache_key="bitwise_logic_config",
        api_endpoint="/config/bitwise-logic",
        default_value={"config": {"rules": [], "enabled": False}, "available_models": [], "available_logic_ops": ['OR', 'AND', 'XOR', '|', '&', '^']}
    )
    
    if bitwise_data:
        current_bitwise_config = bitwise_data['config']
        available_models = bitwise_data['available_models']
        available_logic_ops = bitwise_data['available_logic_ops']
        
        # Enable/Disable bitwise logic
        st.write("##### Enable Bitwise Logic")
        bitwise_enabled = st.checkbox(
            "Enable bitwise logic combinations",
            value=current_bitwise_config.get('enabled', False),
            help="Enable to create custom models by combining existing model outputs",
            key="bitwise_logic_enabled"
        )
        
        # Auto-save enabled state when it changes
        if 'previous_bitwise_enabled' not in st.session_state:
            st.session_state.previous_bitwise_enabled = current_bitwise_config.get('enabled', False)
        
        # Save when enabled state actually changes
        if st.session_state.previous_bitwise_enabled != bitwise_enabled:
            # Get current rules from session state or default to empty
            current_rules = st.session_state.get('bitwise_rules', [])
            
            bitwise_config_data = {
                "rules": current_rules,
                "enabled": bitwise_enabled
            }
            
            try:
                update_response = requests.post(
                    f"{API_BASE_URL}/config/bitwise-logic", 
                    json=bitwise_config_data,
                    timeout=10
                )
                if update_response.status_code == 200:
                    st.session_state.previous_bitwise_enabled = bitwise_enabled
                    current_time = datetime.now().strftime("%H:%M:%S")
                    st.success(f"✅ Bitwise logic {'enabled' if bitwise_enabled else 'disabled'} and saved at {current_time}!")
                    # Update cache
                    update_cache("bitwise_logic_config", {
                        "config": bitwise_config_data,
                        "available_models": available_models,
                        "available_logic_ops": available_logic_ops
                    })
                else:
                    st.error(f"❌ Failed to save enabled state: {update_response.text}")
            except Exception as e:
                st.error(f"❌ Failed to save enabled state: {str(e)}")
        
        if bitwise_enabled:
            st.write("##### Logic Rules")
            st.info("💡 **How it works**: Select 2 or more models and choose a logic operation to combine their decisions. For example: 'AD_Decision OR CL_Decision' creates a model that predicts failure if either base model predicts failure.")
            
            # Initialize session state for rules ONLY if not already initialized
            # This prevents overriding user deletions
            if 'bitwise_rules' not in st.session_state:
                st.session_state.bitwise_rules = current_bitwise_config.get('rules', [])
            
            # Only sync with backend config on initial load or when force refresh is triggered
            # This prevents constant overriding of user actions
            if 'bitwise_rules_initialized' not in st.session_state:
                st.session_state.bitwise_rules = current_bitwise_config.get('rules', [])
                st.session_state.bitwise_rules_initialized = True
            
            # Display existing rules (even if empty, we still show the section)
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
                            # Remove the rule from session state
                            deleted_rule_name = st.session_state.bitwise_rules[i]['name']
                            st.session_state.bitwise_rules.pop(i)
                            
                            # Immediately save the updated configuration to backend
                            updated_config_data = {
                                "rules": st.session_state.bitwise_rules,
                                "enabled": bitwise_enabled
                            }
                            
                            try:
                                update_response = requests.post(
                                    f"{API_BASE_URL}/config/bitwise-logic", 
                                    json=updated_config_data,
                                    timeout=10
                                )
                                if update_response.status_code == 200:
                                    st.success(f"✅ Deleted rule '{deleted_rule_name}' and saved to backend!")
                                    # Update cache to reflect the deletion
                                    update_cache("bitwise_logic_config", {
                                        "config": updated_config_data,
                                        "available_models": available_models,
                                        "available_logic_ops": available_logic_ops
                                    })
                                    # Force refresh on other tabs
                                    if 'api_cache' in st.session_state:
                                        st.session_state.api_cache.pop('metrics_data', None)
                                        st.session_state.api_cache.pop('predictions_data', None)
                                else:
                                    st.error(f"❌ Failed to save after deletion: {update_response.text}")
                            except Exception as e:
                                st.error(f"❌ Failed to save after deletion: {str(e)}")
                            
                            st.rerun()
            else:
                st.info("No rules configured yet. Add a new rule below.")
            
            # Add new rule section - ALWAYS show this when bitwise logic is enabled
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
                selected_models = st.multiselect(
                    "Select Models to Combine",
                    options=available_models,
                    default=[],
                    help="Choose 2 or more models to combine",
                    key="new_rule_models"
                )
            
            with new_rule_col3:
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
                            
                            # Immediately save to backend
                            updated_config_data = {
                                "rules": st.session_state.bitwise_rules,
                                "enabled": bitwise_enabled
                            }
                            
                            try:
                                update_response = requests.post(
                                    f"{API_BASE_URL}/config/bitwise-logic", 
                                    json=updated_config_data,
                                    timeout=10
                                )
                                if update_response.status_code == 200:
                                    # Update cache
                                    update_cache("bitwise_logic_config", {
                                        "config": updated_config_data,
                                        "available_models": available_models,
                                        "available_logic_ops": available_logic_ops
                                    })
                                else:
                                    st.error(f"❌ Failed to save new rule: {update_response.text}")
                            except Exception as e:
                                st.error(f"❌ Failed to save new rule: {str(e)}")
                            
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
            
            # Apply rules section - show even if no rules exist yet
            st.write("##### Apply Logic Rules")
            
            apply_col1, apply_col2 = st.columns([1, 3])
            
            with apply_col1:
                # Only enable the button if there are rules to apply
                button_disabled = len(st.session_state.bitwise_rules) == 0
                if st.button("🔄 Apply Rules", use_container_width=True, type="secondary", disabled=button_disabled):
                    # Save configuration and apply rules
                    bitwise_config_data = {
                        "rules": st.session_state.bitwise_rules,
                        "enabled": bitwise_enabled
                    }
                    
                    try:
                        # Update configuration
                        update_response = requests.post(
                            f"{API_BASE_URL}/config/bitwise-logic", 
                            json=bitwise_config_data
                        )
                        
                        if update_response.status_code == 200:
                            # Update cache
                            update_cache("bitwise_logic_config", {
                                "config": bitwise_config_data,
                                "available_models": available_models,
                                "available_logic_ops": available_logic_ops
                            })
                            
                            # Apply the rules
                            apply_response = requests.post(f"{API_BASE_URL}/config/bitwise-logic/apply")
                            
                            if apply_response.status_code == 200:
                                result = apply_response.json()
                                st.success(f"✅ {result['message']}")
                                if result['combined_models_created'] > 0:
                                    st.info(f"Created models: {', '.join(result['combined_model_names'])}")
                                    st.info("🔄 The Overview and Model Analysis tabs will now show your new combined models!")
                                    # Clear other caches so they reload with new data
                                    if 'api_cache' in st.session_state:
                                        st.session_state.api_cache.pop('metrics_data', None)
                                        st.session_state.api_cache.pop('predictions_data', None)
                            else:
                                st.error(f"Failed to apply rules: {apply_response.text}")
                        else:
                            st.error(f"Failed to save configuration: {update_response.text}")
                    except Exception as e:
                        st.error(f"Error applying rules: {str(e)}")
            
            with apply_col2:
                if len(st.session_state.bitwise_rules) == 0:
                    st.info("💡 Add some logic rules above, then click 'Apply Rules' to create your combined models.")
                else:
                    st.info("💡 Click 'Apply Rules' to create your combined models and update all dashboard results. The new models will appear in the Overview and Model Analysis tabs.")
        
        else:
            # When bitwise logic is disabled, clear the session state to ensure clean state
            if 'bitwise_rules' in st.session_state:
                del st.session_state.bitwise_rules
            if 'bitwise_rules_initialized' in st.session_state:
                del st.session_state.bitwise_rules_initialized
            st.info("Enable bitwise logic above to create custom models by combining existing model outputs.")
    
    else:
        st.error("Failed to load bitwise logic configuration. Please refresh the page or check backend connectivity.")
        
    # ========== 7. PREPROCESSING PREVIEWS ==========
    st.write("#### Preprocessing Previews")
    
    # Filtering settings inputs - fix key mapping
    st.write("##### Filtering Settings for Preview")
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        preview_variance_threshold = st.number_input(
            "Variance Threshold (>)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(config_settings.get('features', {}).get('variance_threshold', 0.01)),
            format="%.3f", key="preview_variance_threshold"
        )
    with col_filter2:
        preview_correlation_threshold = st.number_input(
            "Correlation Threshold (<)",
            min_value=0.0, max_value=1.0, step=0.001,
            value=float(config_settings.get('features', {}).get('correlation_threshold', 0.95)),
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
