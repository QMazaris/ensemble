import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import time

from config_util import update_config_direct, sync_config_to_backend

# Backend API URL
BACKEND_API_URL = "http://localhost:8000"

# Import utility functions
import sys
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils import clear_cache

def render_bitwise_logic_tab():
    """Render the bitwise logic configuration and application tab."""
    
    st.write("### 🔗 Bitwise Logic Configuration")
    st.write("Create and apply logical combinations of decision columns and threshold-based models to generate new combined models.")
    
    # Get current config
    config = st.session_state.get('config_settings', {})
    
    # Get both decision columns and threshold columns
    decision_columns = config.get("models", {}).get("base_model_decisions", [])
    threshold_columns = config.get("models", {}).get("base_model_columns", [])
    
    if not decision_columns and not threshold_columns:
        st.warning("⚠️ No base model columns configured. Please configure decision columns and/or threshold columns in the Preprocessing Config tab first.")
        return
    
    # Show available columns with their types
    st.write("#### Available Model Columns")
    col1, col2 = st.columns(2)
    
    with col1:
        if decision_columns:
            st.success(f"**Decision Columns ({len(decision_columns)}):** {', '.join(decision_columns)}")
            st.caption("These columns already contain Good/Bad decisions")
        else:
            st.info("**Decision Columns:** None configured")
    
    with col2:
        if threshold_columns:
            st.success(f"**Threshold Columns ({len(threshold_columns)}):** {', '.join(threshold_columns)}")
            st.caption("These columns contain scores that will be converted using thresholds")
        else:
            st.info("**Threshold Columns:** None configured")
    
    # ========== THRESHOLD CONFIGURATION ==========
    if threshold_columns:
        st.write("#### Threshold Configuration")
        st.write("Set thresholds for score-based columns (scores >= threshold will be classified as 'Good'):")
        
        # Initialize threshold config if not exists
        if 'model_thresholds' not in config:
            config['model_thresholds'] = {}
            st.session_state['config_settings'] = config
        
        thresholds_config = config.get('model_thresholds', {})
        
        # Create threshold inputs for each threshold column
        threshold_cols = st.columns(min(3, len(threshold_columns)))
        for i, col_name in enumerate(threshold_columns):
            with threshold_cols[i % 3]:
                current_threshold = thresholds_config.get(col_name, 0.5)
                
                new_threshold = st.number_input(
                    f"Threshold for {col_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current_threshold),
                    step=0.01,
                    format="%.2f",
                    key=f"threshold_{col_name}",
                    help=f"Scores >= this value in {col_name} will be classified as 'Good'"
                )
                
                # Update config if threshold changed
                if new_threshold != current_threshold:
                    update_config_direct("model_thresholds", col_name, new_threshold)
        
        st.markdown("---")
    
    # All available columns for bitwise logic
    all_available_columns = decision_columns + threshold_columns
    
    # ========== BITWISE LOGIC CONFIGURATION ==========
    st.write("#### Current Rules")
    
    # Initialize bitwise_logic in config if not exists
    if 'bitwise_logic' not in config:
        config['bitwise_logic'] = {'rules': [], 'enabled': True}
        st.session_state['config_settings'] = config
    
    current_rules = config.get('bitwise_logic', {}).get('rules', [])
    
    # Enable/Disable toggle
    enabled = config.get('bitwise_logic', {}).get('enabled', True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Bitwise Logic Status**")
    with col2:
        if st.checkbox("Enable Bitwise Logic", value=enabled, key="bitwise_enabled"):
            update_config_direct("bitwise_logic", "enabled", True)
        else:
            update_config_direct("bitwise_logic", "enabled", False)
    
    # Display existing rules
    if current_rules:
        st.write("**Existing Rules:**")
        for i, rule in enumerate(current_rules):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 1.5, 1, 2, 1])
                with col1:
                    st.write(f"**{rule.get('name', f'Rule {i+1}')}**")
                with col2:
                    st.write(f"Logic: `{rule.get('logic', 'OR')}`")
                with col3:
                    st.write(f"Columns: {len(rule.get('columns', []))}")
                with col4:
                    if rule.get('columns'):
                        st.write(f"_{', '.join(rule['columns'][:2])}{'...' if len(rule['columns']) > 2 else ''}_")
                with col5:
                    if st.button("🗑️", key=f"delete_rule_{i}", help="Delete this rule"):
                        current_rules.pop(i)
                        update_config_direct("bitwise_logic", "rules", current_rules)
                        st.toast("Rule deleted!", icon="🗑️")
                        st.rerun()
                
                # Show rule details in expander
                with st.expander(f"Rule Details: {rule.get('name', f'Rule {i+1}')}"):
                    st.write(f"**Logic Type:** {rule.get('logic', 'OR')}")
                    st.write(f"**Columns:** {', '.join(rule.get('columns', []))}")
                    
                    # Show column types and thresholds
                    rule_columns = rule.get('columns', [])
                    decision_cols_in_rule = [col for col in rule_columns if col in decision_columns]
                    threshold_cols_in_rule = [col for col in rule_columns if col in threshold_columns]
                    
                    if decision_cols_in_rule:
                        st.write(f"**Decision Columns:** {', '.join(decision_cols_in_rule)} (direct Good/Bad values)")
                    
                    if threshold_cols_in_rule:
                        st.write("**Threshold Columns:**")
                        for col in threshold_cols_in_rule:
                            threshold_val = thresholds_config.get(col, 0.5)
                            st.write(f"  • {col}: threshold = {threshold_val}")
                    
                    # Show logic explanation
                    logic_type = rule.get('logic', 'OR')
                    if logic_type in ['OR', '|']:
                        st.write("**Logic:** Result is 'Good' if ANY of the selected columns shows 'Good'")
                    elif logic_type in ['AND', '&']:
                        st.write("**Logic:** Result is 'Good' if ALL of the selected columns show 'Good'")
                    elif logic_type in ['XOR', '^']:
                        st.write("**Logic:** Result is 'Good' if an ODD NUMBER of the selected columns show 'Good'")
        
        st.markdown("---")
    else:
        st.info("No bitwise logic rules configured yet.")
    
    # ========== ADD NEW RULE ==========
    st.write("#### Add New Rule")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rule_name = st.text_input(
            "Rule Name",
            placeholder="e.g., Combined_Decision_OR",
            key="new_rule_name",
            help="Give your rule a descriptive name"
        )
    
    with col2:
        logic_type = st.selectbox(
            "Logic Type",
            ["OR", "AND", "XOR", "|", "&", "^"],
            key="new_rule_logic",
            help="Choose how to combine the selected columns"
        )
    
    # Show logic explanation
    logic_explanations = {
        "OR": "Result is 'Good' if ANY column shows 'Good'",
        "|": "Result is 'Good' if ANY column shows 'Good'",
        "AND": "Result is 'Good' if ALL columns show 'Good'",
        "&": "Result is 'Good' if ALL columns show 'Good'",
        "XOR": "Result is 'Good' if ODD NUMBER of columns show 'Good'",
        "^": "Result is 'Good' if ODD NUMBER of columns show 'Good'"
    }
    
    st.info(f"**{logic_type} Logic:** {logic_explanations.get(logic_type, 'Unknown logic type')}")
    
    # Enhanced multiselect with column type indicators
    st.write("**Select Columns for Rule:**")
    
    # Use a safe key for multiselect
    key = "new_rule_columns_select"
    
    # Initialize state only once
    if key not in st.session_state:
        st.session_state[key] = []
    
    # Create options with type indicators
    column_options = []
    for col in all_available_columns:
        if col in decision_columns:
            column_options.append(f"{col} (Decision)")
        else:
            threshold_val = thresholds_config.get(col, 0.5)
            column_options.append(f"{col} (Threshold: {threshold_val})")
    
    selected_rule_columns_with_indicators = st.multiselect(
        "Available Model Columns",
        column_options,
        key=key,
        help="Choose columns to combine with the selected logic. Decision columns use Good/Bad values directly, threshold columns use the configured thresholds."
    )
    
    # Extract just the column names (remove the type indicators)
    selected_rule_columns = []
    for option in selected_rule_columns_with_indicators:
        col_name = option.split(" (")[0]  # Remove the type indicator part
        selected_rule_columns.append(col_name)
    
    # Show preview of selected columns
    if selected_rule_columns:
        st.write("**Selected Columns Preview:**")
        for col in selected_rule_columns:
            if col in decision_columns:
                st.write(f"  • **{col}** (Decision): Uses Good/Bad values directly")
            elif col in threshold_columns:
                threshold_val = thresholds_config.get(col, 0.5)
                st.write(f"  • **{col}** (Threshold): Scores >= {threshold_val} = Good, < {threshold_val} = Bad")
    
    # Add rule button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add Rule", key="add_bitwise_rule", type="secondary"):
            if rule_name and selected_rule_columns:
                # Check if rule name already exists
                existing_names = [rule.get('name', '') for rule in current_rules]
                if rule_name in existing_names:
                    st.error("❌ Rule name already exists. Please choose a different name.")
                else:
                    new_rule = {
                        "name": rule_name,
                        "logic": logic_type,
                        "columns": selected_rule_columns
                    }
                    current_rules.append(new_rule)
                    update_config_direct("bitwise_logic", "rules", current_rules)
                    # Clear the rule columns selection after adding
                    st.session_state[key] = []
                    st.toast(f"✅ Rule '{rule_name}' added!", icon="✅")
                    st.rerun()
            else:
                st.error("❌ Please provide both rule name and select columns.")
    
    # ========== SAVE AND APPLY CONFIGURATION ==========
    st.markdown("---")
    st.write("#### 🚀 Save & Apply Configuration")
    
    # Show current status
    if current_rules:
        rule_count = len(current_rules)
        enabled_status = "Enabled" if enabled else "Disabled"
        st.info(f"📊 **Current Status:** {rule_count} rule{'s' if rule_count != 1 else ''} configured, Bitwise Logic is {enabled_status}")
        
        # Show what will happen
        if enabled:
            st.write("**When you apply this configuration:**")
            st.write("1. 💾 Configuration will be saved to the backend")
            st.write("2. 🔗 Bitwise logic rules will be applied to existing model results")
            st.write("3. 📊 New combined models will be created")
            st.write("4. 🔄 Frontend data will be refreshed automatically")
        else:
            st.warning("⚠️ Bitwise logic is disabled. Enable it to apply rules.")
    else:
        st.warning("⚠️ No rules configured. Add rules above to enable bitwise logic application.")
    
    # Apply button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        apply_disabled = not current_rules or not enabled
        if st.button(
            "💾 Save Config & Apply Bitwise Logic", 
            key="apply_bitwise_config",
            type="primary",
            disabled=apply_disabled,
            help="Save configuration and apply bitwise logic rules to create combined models" if not apply_disabled else "Add rules and enable bitwise logic first",
            use_container_width=True
        ):
            apply_bitwise_logic_configuration()

def apply_bitwise_logic_configuration():
    """Apply the bitwise logic configuration by saving config and calling the API."""
    
    with st.spinner("Applying bitwise logic configuration..."):
        # Step 1: Sync config to backend
        st.write("**Step 1:** Saving configuration to backend...")
        
        success = sync_config_to_backend()
        if not success:
            st.error("❌ Failed to save configuration to backend. Please check the backend connection.")
            return
        
        st.success("✅ Configuration saved successfully!")
        
        # Step 2: Apply bitwise logic via API
        st.write("**Step 2:** Applying bitwise logic rules...")
        
        try:
            response = requests.post(
                f"{BACKEND_API_URL}/config/bitwise-logic/apply",
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                combined_models_created = result.get('combined_models_created', 0)
                
                if combined_models_created > 0:
                    st.success(f"✅ Successfully created {combined_models_created} combined model{'s' if combined_models_created != 1 else ''}!")
                    
                    # Show created model names if available
                    combined_model_names = result.get('combined_model_names', [])
                    if combined_model_names:
                        st.info(f"🆕 **New Models Created:** {', '.join(combined_model_names)}")
                else:
                    st.warning("⚠️ Bitwise logic was applied but no new models were created. This might be because rules were already applied or no compatible data was found.")
                    
            else:
                error_detail = response.text if response.text else f"HTTP {response.status_code}"
                st.error(f"❌ Failed to apply bitwise logic: {error_detail}")
                return
                
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. The operation might still be running in the background.")
            return
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend API. Please ensure the backend server is running.")
            return
        except Exception as e:
            st.error(f"❌ Error applying bitwise logic: {str(e)}")
            return
        
        # Step 3: Clear frontend cache and trigger refresh
        st.write("**Step 3:** Refreshing frontend data...")
        
        clear_cache()
        
        # Set a flag to indicate successful pipeline completion
        st.session_state.pipeline_completed_at = time.time()
        st.session_state.bitwise_logic_applied = True
        
        st.success("✅ All steps completed successfully!")
        st.info("🔄 **Data refreshed!** Switch to other tabs to see the new combined models in action.")
        
        # Small delay to ensure backend processing completes
        time.sleep(1)
        
        # Optional: Auto-refresh indicators
        st.balloons() 