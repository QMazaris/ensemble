import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import time

# Backend API URL
BACKEND_API_URL = "http://localhost:8000"

# Import utility functions
import sys
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# No longer using cache functions

def render_bitwise_logic_tab():
    """Render the bitwise logic configuration and application tab."""
    
    st.write("### 🔗 Bitwise Logic Configuration")
    st.write("Create and apply logical combinations of model outputs to generate new combined models.")
    
    # Try to fetch available models from the results API
    try:
        response = requests.get(f"{BACKEND_API_URL}/results/metrics", timeout=10)
        if response.status_code == 200:
            results_data = response.json()
            available_models = []
            
            # Extract unique model names from results
            model_metrics = results_data.get('results', {}).get('model_metrics', [])
            unique_models = set()
            for metric in model_metrics:
                model_name = metric.get('model_name', '').replace('kfold_avg_', '')
                if model_name and model_name not in unique_models:
                    unique_models.add(model_name)
                    available_models.append(model_name)
            
            if not available_models:
                st.warning("⚠️ No model results available. Please run the pipeline first to generate model outputs.")
                return
                
        else:
            st.warning("⚠️ Could not fetch model results from API. Please ensure the pipeline has been run.")
            return
            
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ Cannot connect to backend API. Please ensure the backend server is running.")
        return
    except Exception as e:
        st.error(f"❌ Error fetching model results: {str(e)}")
        return
    
    # Try to fetch predictions to categorize models
    decision_models = []
    meta_models = []
    
    try:
        pred_response = requests.get(f"{BACKEND_API_URL}/results/predictions", timeout=10)
        if pred_response.status_code == 200:
            predictions_data = pred_response.json()
            predictions_df = pd.DataFrame(predictions_data.get('predictions', []))
            
            # Categorize models based on their values
            for model in available_models:
                if model in predictions_df.columns:
                    values = predictions_df[model].dropna()
                    if len(values) > 0:
                        unique_vals = set(values)
                        # Check if values are binary (0/1) - likely decision models
                        if unique_vals.issubset({0, 1, 0.0, 1.0}):
                            decision_models.append(model)
                        # Check if values are in [0,1] range - likely probabilities/scores
                        else:
                            meta_models.append(model)
                    else:
                        # Default to meta if we can't determine
                        meta_models.append(model)
                else:
                    meta_models.append(model)
        else:
            # Fallback: treat all as meta models if can't get predictions
            meta_models = available_models
    except Exception as e:
        st.info(f"Could not categorize models automatically: {str(e)}. Treating all as meta models.")
        meta_models = available_models
    
    # Show available models with their types
    st.write("#### Available Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if decision_models:
            st.success(f"**Decision Models ({len(decision_models)})**")
            for model in decision_models:
                st.write(f"• {model}")
            st.caption("These models already contain binary decisions (0/1)")
        else:
            st.info("**Decision Models:** None detected")
    
    
    with col2:
        if meta_models:
            st.success(f"**Meta Models ({len(meta_models)})**")
            for model in meta_models:
                st.write(f"• {model}")
            st.caption("These are trained ML models with probability outputs")
        else:
            st.info("**Meta Models:** None detected")
    
    # All available models for bitwise logic
    all_available_models = decision_models + meta_models
    
    # ========== THRESHOLD CONFIGURATION FORM ==========
    # Show threshold configuration for threshold-based and meta models
    models_needing_thresholds = meta_models
    
    # Initialize session state for thresholds
    if 'bitwise_thresholds' not in st.session_state:
        st.session_state.bitwise_thresholds = {}
    
    if models_needing_thresholds:
        st.write("#### Threshold Configuration")
        
        with st.form("threshold_config_form", clear_on_submit=False):
            st.write("Set thresholds for score-based models (scores >= threshold will be classified as 'Good'):")
            
            # Create threshold inputs for each model needing thresholds
            threshold_values = {}
            num_cols = min(3, len(models_needing_thresholds))
            threshold_cols = st.columns(num_cols)
            
            for i, model_name in enumerate(models_needing_thresholds):
                with threshold_cols[i % num_cols]:
                    current_threshold = st.session_state.bitwise_thresholds.get(model_name, 0.5)
                    
                    threshold_values[model_name] = st.number_input(
                        f"Threshold for {model_name}",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(current_threshold),
                        step=0.01,
                        format="%.2f",
                        key=f"threshold_form_{model_name}",
                        help=f"Scores >= this value in {model_name} will be classified as 'Good'"
                    )
            
            # Submit button for thresholds
            if st.form_submit_button("💾 Save Thresholds", type="secondary"):
                for model_name, threshold_value in threshold_values.items():
                    st.session_state.bitwise_thresholds[model_name] = threshold_value
                st.success("✅ Thresholds saved successfully!")
                st.rerun()
        
        st.markdown("---")
    
    # ========== BITWISE LOGIC RULES ==========
    st.write("#### Bitwise Logic Rules")
    
    # Initialize session state for rules
    if 'bitwise_rules' not in st.session_state:
        st.session_state.bitwise_rules = []
    
    # Enable/Disable toggle
    if 'bitwise_enabled' not in st.session_state:
        st.session_state.bitwise_enabled = True
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Bitwise Logic Status**")
    with col2:
        st.session_state.bitwise_enabled = st.checkbox("Enable Bitwise Logic", value=st.session_state.bitwise_enabled, key="bitwise_logic_enabled")
    
    # Display existing rules
    if st.session_state.bitwise_rules:
        st.write("**Existing Rules:**")
        for i, rule in enumerate(st.session_state.bitwise_rules):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 1.5, 1, 2, 1])
                with col1:
                    st.write(f"**{rule.get('name', f'Rule {i+1}')}**")
                with col2:
                    st.write(f"Logic: `{rule.get('logic', 'OR')}`")
                with col3:
                    st.write(f"Models: {len(rule.get('columns', []))}")
                with col4:
                    if rule.get('columns'):
                        st.write(f"_{', '.join(rule['columns'][:2])}{'...' if len(rule['columns']) > 2 else ''}_")
                with col5:
                    if st.button("🗑️", key=f"bitwise_delete_rule_{i}", help="Delete this rule"):
                        st.session_state.bitwise_rules.pop(i)
                        st.toast("Rule deleted!", icon="🗑️")
                        st.rerun()
                
                # Show rule details in expander
                with st.expander(f"Rule Details: {rule.get('name', f'Rule {i+1}')}"):
                    st.write(f"**Logic Type:** {rule.get('logic', 'OR')}")
                    st.write(f"**Models:** {', '.join(rule.get('columns', []))}")
                    
                    # Show model types and thresholds
                    rule_models = rule.get('columns', [])
                    decision_models_in_rule = [model for model in rule_models if model in decision_models]
                    meta_models_in_rule = [model for model in rule_models if model in meta_models]
                    
                    if decision_models_in_rule:
                        st.write(f"**Decision Models:** {', '.join(decision_models_in_rule)} (direct binary values)")
                    
                    if meta_models_in_rule:
                        st.write("**Meta Models:**")
                        for model in meta_models_in_rule:
                            threshold_val = st.session_state.bitwise_thresholds.get(model, 0.5)
                            st.write(f"  • {model}: threshold = {threshold_val}")
                    
                    # Show logic explanation
                    logic_type = rule.get('logic', 'OR')
                    if logic_type in ['OR', '|']:
                        st.write("**Logic:** Result is 'Good' if ANY of the selected models shows 'Good'")
                    elif logic_type in ['AND', '&']:
                        st.write("**Logic:** Result is 'Good' if ALL of the selected models show 'Good'")
                    elif logic_type in ['XOR', '^']:
                        st.write("**Logic:** Result is 'Good' if an ODD NUMBER of the selected models show 'Good'")
                    elif logic_type == 'NOR':
                        st.write("**Logic:** Result is 'Good' if NO models show 'Good' (inverted OR)")
                    elif logic_type == 'NAND':
                        st.write("**Logic:** Result is 'Good' if NOT ALL models show 'Good' (inverted AND)")
                    elif logic_type == 'NXOR':
                        st.write("**Logic:** Result is 'Good' if EVEN NUMBER of models show 'Good' (inverted XOR)")
                    else:
                        st.write(f"**Logic:** {logic_type} operation")
        
        st.markdown("---")
    else:
        st.info("No bitwise logic rules configured yet.")
    
    # ========== ADD NEW RULE FORM ==========
    st.write("#### Add New Rule")
    
    with st.form("add_new_rule_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            rule_name = st.text_input(
                "Rule Name",
                placeholder="e.g., Combined_Decision_OR",
                key="form_rule_name",
                help="Give your rule a descriptive name"
            )
        
        with col2:
            logic_type = st.selectbox(
                "Logic Type",
                ["OR", "AND", "XOR", "NOR", "NAND", "NXOR"],
                key="form_logic_type",
                help="Choose how to combine the selected models"
            )
        
        # Show logic explanation
        logic_explanations = {
            "OR": "Result is 'Good' if ANY model shows 'Good'",
            "AND": "Result is 'Good' if ALL models show 'Good'",
            "XOR": "Result is 'Good' if ODD NUMBER of models show 'Good'",
            "NOR": "Result is 'Good' if NO models show 'Good' (inverted OR)",
            "NAND": "Result is 'Good' if NOT ALL models show 'Good' (inverted AND)",
            "NXOR": "Result is 'Good' if EVEN NUMBER of models show 'Good' (inverted XOR)"
        }
        
        st.info(
        "**Logic Operator Guide:**\n\n" +
        "\n".join([f"- **{k}**: {v}" for k, v in logic_explanations.items()])
        )

        
        # Enhanced multiselect with model type indicators
        st.write("**Select Models for Rule:**")
        
        # Create options with type indicators
        model_options = []
        for model in all_available_models:
            if model in decision_models:
                model_options.append(f"{model} (Decision)")
            else:  # meta models
                threshold_val = st.session_state.bitwise_thresholds.get(model, 0.5)
                model_options.append(f"{model} (Meta: {threshold_val})")
        
        selected_rule_models_with_indicators = st.multiselect(
            "Available Models",
            model_options,
            key="form_selected_models",
            help="Choose models to combine with the selected logic. Decision models use binary values directly, others use the configured thresholds."
        )
        
        # Extract just the model names (remove the type indicators)
        selected_rule_models = []
        for option in selected_rule_models_with_indicators:
            model_name = option.split(" (")[0]  # Remove the type indicator part
            selected_rule_models.append(model_name)
        
        # Show preview of selected models
        if selected_rule_models:
            st.write("**Selected Models Preview:**")
            for model in selected_rule_models:
                if model in decision_models:
                    st.write(f"  • **{model}** (Decision): Uses binary values directly")
                else:  # meta models
                    threshold_val = st.session_state.bitwise_thresholds.get(model, 0.5)
                    st.write(f"  • **{model}** (Meta): Probabilities >= {threshold_val} = Good, < {threshold_val} = Bad")
        
        # Submit button for adding rule
        if st.form_submit_button("➕ Add Rule", type="primary"):
            if rule_name and selected_rule_models:
                # Check if rule name already exists
                existing_names = [rule.get('name', '') for rule in st.session_state.bitwise_rules]
                if rule_name in existing_names:
                    st.error("❌ Rule name already exists. Please choose a different name.")
                else:
                    new_rule = {
                        "name": rule_name,
                        "logic": logic_type,
                        "columns": selected_rule_models
                    }
                    st.session_state.bitwise_rules.append(new_rule)
                    st.success(f"✅ Rule '{rule_name}' added successfully!")
                    st.rerun()
            else:
                st.error("❌ Please provide both rule name and select models.")
    
    # ========== SAVE AND APPLY CONFIGURATION ==========
    st.markdown("---")
    st.write("#### 🚀 Apply Bitwise Logic")
    
    # Show current status
    if st.session_state.bitwise_rules:
        rule_count = len(st.session_state.bitwise_rules)
        enabled_status = "Enabled" if st.session_state.bitwise_enabled else "Disabled"
        st.info(f"📊 **Current Status:** {rule_count} rule{'s' if rule_count != 1 else ''} configured, Bitwise Logic is {enabled_status}")
        
        # Show what will happen
        if st.session_state.bitwise_enabled:
            st.write("**When you apply this configuration:**")
            st.write("1. 🔗 Bitwise logic rules will be applied to existing model results")
            st.write("2. 📊 New combined models will be created")
            st.write("3. 🔄 Frontend data will be refreshed automatically")
        else:
            st.warning("⚠️ Bitwise logic is disabled. Enable it to apply rules.")
    else:
        st.warning("⚠️ No rules configured. Add rules above to enable bitwise logic application.")
    
    # Apply button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        apply_disabled = not st.session_state.bitwise_rules or not st.session_state.bitwise_enabled
        if st.button(
            "🚀 Apply Bitwise Logic", 
            key="bitwise_apply_logic",
            type="primary",
            disabled=apply_disabled,
            help="Apply bitwise logic rules to create combined models" if not apply_disabled else "Add rules and enable bitwise logic first",
            use_container_width=True
        ):
            apply_bitwise_logic_rules()

def apply_bitwise_logic_rules():
    """Apply the bitwise logic rules by sending them directly to the API."""
    
    with st.spinner("Applying bitwise logic rules..."):
        # Prepare the request payload
        rules_payload = []
        for rule in st.session_state.bitwise_rules:
            rules_payload.append({
                "name": rule["name"],
                "columns": rule["columns"],
                "logic": rule["logic"]
            })
        
        request_data = {
            "rules": rules_payload,
            "model_thresholds": st.session_state.bitwise_thresholds,
            "enabled": st.session_state.bitwise_enabled
        }
        
        st.write("**Step 1:** Applying bitwise logic rules...")
        
        try:
            response = requests.post(
                f"{BACKEND_API_URL}/config/bitwise-logic/apply",
                json=request_data,
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
        
        # Step 2: Trigger refresh (no caching anymore)
        st.write("**Step 2:** Refreshing frontend data...")
        
        # Set a flag to indicate successful pipeline completion
        st.session_state.pipeline_completed_at = time.time()
        st.session_state.bitwise_logic_applied = True
        
        st.success("✅ All steps completed successfully!")
        st.info("🔄 **Data refreshed!** Switch to other tabs to see the new combined models in action.")
        
        # Small delay to ensure backend processing completes
        time.sleep(1)
        
        # Optional: Auto-refresh indicators
        st.balloons() 