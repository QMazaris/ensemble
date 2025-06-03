import streamlit as st
import pandas as pd
from pathlib import Path
import inspect
import re
import ast
import importlib.util
import sys

# Add parent directory to path to access utils
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add root directory to path to access shared modules
root_dir = str(Path(__file__).parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config_util import on_config_change, update_config_direct

def initialize_default_models():
    """Initialize default model configurations in session state."""
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    
    default_models = {
        'XGBoost': {
            'class': xgb.XGBClassifier,
            'params': {
                'objective': 'binary:logistic',
                'max_depth': 4,
                'learning_rate': 0.1,
                'n_estimators': 400,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'gamma': 0.1,
                'scale_pos_weight': 62.5,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            },
            'param_types': {
                'objective': {'type': 'select', 'options': ['binary:logistic', 'binary:hinge', 'binary:logitraw']},
                'max_depth': {'type': 'number', 'min': 1, 'max': 20, 'step': 1},
                'learning_rate': {'type': 'number', 'min': 0.001, 'max': 0.5, 'step': 0.001},
                'n_estimators': {'type': 'number', 'min': 50, 'max': 1000, 'step': 50},
                'subsample': {'type': 'number', 'min': 0.1, 'max': 1.0, 'step': 0.05},
                'colsample_bytree': {'type': 'number', 'min': 0.1, 'max': 1.0, 'step': 0.05},
                'min_child_weight': {'type': 'number', 'min': 1, 'max': 20, 'step': 1},
                'gamma': {'type': 'number', 'min': 0.0, 'max': 1.0, 'step': 0.1},
                'scale_pos_weight': {'type': 'number', 'min': 1, 'max': 100, 'step': 1},
                'random_state': {'type': 'number', 'min': 0, 'max': 1000, 'step': 1},
                'n_jobs': {'type': 'number', 'min': -1, 'max': 16, 'step': 1},
                'eval_metric': {'type': 'select', 'options': ['logloss', 'auc', 'error']}
            },
            'hyperparams': {
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'n_estimators': [100, 200, 300, 400, 500],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
            }
        },
        'RandomForest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 300,
                'max_depth': 6,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'class_weight': 'balanced_subsample',
                'max_features': 'sqrt',
                'bootstrap': True,
                'oob_score': True,
                'n_jobs': -1,
                'random_state': 42
            },
            'param_types': {
                'n_estimators': {'type': 'number', 'min': 50, 'max': 1000, 'step': 50},
                'max_depth': {'type': 'number', 'min': 1, 'max': 20, 'step': 1},
                'min_samples_split': {'type': 'number', 'min': 2, 'max': 50, 'step': 1},
                'min_samples_leaf': {'type': 'number', 'min': 1, 'max': 50, 'step': 1},
                'class_weight': {'type': 'select', 'options': ['balanced', 'balanced_subsample', None]},
                'max_features': {'type': 'select', 'options': ['sqrt', 'log2', None]},
                'bootstrap': {'type': 'select', 'options': [True, False]},
                'oob_score': {'type': 'select', 'options': [True, False]},
                'n_jobs': {'type': 'number', 'min': -1, 'max': 16, 'step': 1},
                'random_state': {'type': 'number', 'min': 0, 'max': 1000, 'step': 1}
            },
            'hyperparams': {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [3, 4, 5, 6, 7, 8],
                'min_samples_split': [2, 5, 10, 15, 20],
                'min_samples_leaf': [1, 2, 4, 5, 10]
            }
        }
    }
    return default_models

def render_model_zoo_tab():
    """Render the model zoo tab content with standardized config handling."""
    st.write("### Model Zoo")

    # Import required model classes
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    # Get current config
    config = st.session_state.get('config_settings', {})
    
    # Initialize models in session state if not exists
    if 'model_zoo_models' not in st.session_state:
        st.session_state['model_zoo_models'] = initialize_default_models()
    
    available_models = st.session_state['model_zoo_models']

    # ========== 1. MODEL SELECTION ==========
    st.write("#### Select and Configure Model")
    
    current_model_selection = config.get('model_zoo', {}).get('selected_model', list(available_models.keys())[0])
    
    try:
        default_index = list(available_models.keys()).index(current_model_selection)
    except ValueError:
        default_index = 0
    
    selected_model = st.selectbox(
        "Choose a model to configure", 
        list(available_models.keys()),
        index=default_index,
        key="model_zoo_selection",
        on_change=lambda: on_config_change("model_zoo", "selected_model", "model_zoo_selection")
    )

    # ========== 2. MODEL PARAMETER CONFIGURATION ==========
    if selected_model:
        st.write(f"##### Configure {selected_model} Parameters")
        
        model_info = available_models[selected_model]
        current_params = model_info['params']
        param_types = model_info['param_types']

        # Create parameter input widgets
        edited_params = {}
        for param_name, param_type_info in param_types.items():
            current_value = current_params.get(param_name)
            param_key = f"model_zoo_{selected_model}_{param_name}"
            
            if param_type_info['type'] == 'select':
                try:
                    current_index = param_type_info['options'].index(current_value)
                except (ValueError, TypeError):
                    current_index = 0
                    
                edited_params[param_name] = st.selectbox(
                    param_name,
                    options=param_type_info['options'],
                    index=current_index,
                    key=f"{param_key}_select"
                )
            elif param_type_info['type'] == 'number':
                # Convert current value to appropriate type
                if isinstance(current_value, (int, float)):
                    value = float(current_value)
                else:
                    value = float(param_type_info['min'])
                    
                edited_params[param_name] = st.number_input(
                    param_name,
                    min_value=float(param_type_info['min']),
                    max_value=float(param_type_info['max']),
                    step=float(param_type_info.get('step', 1)),
                    value=value,
                    key=f"{param_key}_number"
                )

        # Button-based save for model parameters
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Model Parameters", key=f"save_model_params_{selected_model}"):
                # Update the model parameters in session state
                st.session_state['model_zoo_models'][selected_model]['params'] = edited_params.copy()
                
                # Save to config
                update_config_direct("model_zoo", "selected_model", selected_model)
                update_config_direct("model_zoo", "models", st.session_state['model_zoo_models'])
                
                st.toast(f"{selected_model} parameters saved!", icon="✅")

        with col2:
            if st.button("Reset to Defaults", key=f"reset_model_params_{selected_model}"):
                # Reset to default values by reinitializing
                default_models = initialize_default_models()
                st.session_state['model_zoo_models'][selected_model] = default_models[selected_model]
                st.toast("Parameters reset to defaults!", icon="🔄")
                st.rerun()

        # Display current configuration
        st.write("##### Current Configuration Preview")
        st.json(edited_params)

    # ========== 3. HYPERPARAMETER SPACE CONFIGURATION ==========
    st.write("#### Hyperparameter Search Space")
    
    if selected_model:
        st.write(f"##### Configure {selected_model} Hyperparameter Ranges")
        
        model_info = available_models[selected_model]
        current_hyperparams = model_info.get('hyperparams', {})
        
        # Allow editing of hyperparameter ranges
        edited_hyperparams = {}
        for param_name, param_values in current_hyperparams.items():
            st.write(f"**{param_name}**")
            
            # Convert list to string for editing
            values_str = ', '.join(map(str, param_values))
            
            new_values_str = st.text_input(
                f"Values for {param_name} (comma-separated)",
                value=values_str,
                key=f"hyperparam_{selected_model}_{param_name}",
                help="Enter values separated by commas (e.g., 1, 2, 3, 4, 5)"
            )
            
            # Parse the string back to list
            try:
                if new_values_str.strip():
                    # Try to convert to appropriate types
                    new_values = []
                    for val in new_values_str.split(','):
                        val = val.strip()
                        # Try to convert to number first
                        try:
                            if '.' in val:
                                new_values.append(float(val))
                            else:
                                new_values.append(int(val))
                        except ValueError:
                            # Keep as string if not numeric
                            new_values.append(val)
                    edited_hyperparams[param_name] = new_values
                else:
                    edited_hyperparams[param_name] = []
            except Exception:
                edited_hyperparams[param_name] = param_values  # Keep original if parsing fails
        
        # Button to save hyperparameter ranges
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Hyperparameter Ranges", key=f"save_hyperparams_{selected_model}"):
                # Update hyperparameters in session state
                st.session_state['model_zoo_models'][selected_model]['hyperparams'] = edited_hyperparams.copy()
                
                # Save to config
                update_config_direct("model_zoo", "hyperparams", st.session_state['model_zoo_models'])
                
                st.toast(f"{selected_model} hyperparameter ranges saved!", icon="✅")
        
        with col2:
            if st.button("Reset Hyperparameter Ranges", key=f"reset_hyperparams_{selected_model}"):
                # Reset to defaults
                default_models = initialize_default_models()
                st.session_state['model_zoo_models'][selected_model]['hyperparams'] = default_models[selected_model]['hyperparams']
                st.toast("Hyperparameter ranges reset to defaults!", icon="🔄")
                st.rerun()
        
        # Display current hyperparameter configuration
        st.write("##### Current Hyperparameter Ranges")
        st.json(edited_hyperparams)

    # ========== 4. GLOBAL MODEL CONFIGURATION SUMMARY ==========
    st.write("#### Global Configuration Summary")
    
    # Show all models and their current settings
    with st.expander("View All Model Configurations"):
        for model_name, model_info in available_models.items():
            st.write(f"**{model_name}**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("*Parameters:*")
                st.json(model_info['params'])
            with col2:
                st.write("*Hyperparameter Ranges:*")
                st.json(model_info.get('hyperparams', {}))
            st.divider()
    
    # Export/Import configuration
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Configuration", key="export_model_config"):
            config_data = {
                'models': st.session_state['model_zoo_models'],
                'selected_model': selected_model
            }
            st.download_button(
                label="Download Config JSON",
                data=st.json(config_data),
                file_name="model_zoo_config.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_config = st.file_uploader(
            "Import Configuration", 
            type=['json'],
            key="import_model_config"
        )
        if uploaded_config is not None:
            try:
                import json
                config_data = json.load(uploaded_config)
                if 'models' in config_data:
                    st.session_state['model_zoo_models'] = config_data['models']
                    st.toast("Configuration imported successfully!", icon="✅")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing configuration: {e}")