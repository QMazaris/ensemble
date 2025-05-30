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

# Import utility functions
from utils import auto_save_model_config

def render_model_zoo_tab():
    """Render the model zoo tab content with automatic saving."""
    st.write("### Model Zoo")

    # Import required model classes
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    config_path = Path("config.py")
    if not config_path.exists():
        st.error("config.py not found.")
        return

    # Read and parse the config file
    try:
        config_content = config_path.read_text()
        # Load the actual config module to get current values
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        current_models = config_module.MODELS

        # --- Use AST to extract source code for MODELS and HYPERPARAM_SPACE ---
        tree = ast.parse(config_content)
        models_code = "# MODELS section not found"
        hyperparams_code = "# HYPERPARAM_SPACE section not found"

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == 'MODELS':
                            # Extract the source code for the value node
                            # Use ast.get_source_segment which handles indentation within the segment
                            models_code = ast.get_source_segment(config_content, node) or "# MODELS section not found"
                        elif target.id == 'HYPERPARAM_SPACE':
                            # Extract the source code for the value node
                            hyperparams_code = ast.get_source_segment(config_content, node) or "# HYPERPARAM_SPACE section not found"

        # The AST approach is more reliable, removing the regex fallback.

    except Exception as e:
        st.error(f"Error loading or parsing config.py: {e}")
        return

    # Define available models and their parameters
    available_models = {
        'XGBoost': {
            'class': xgb.XGBClassifier,
            'params': {
                'objective': {'type': 'select', 'options': ['binary:logistic'], 'default': 'binary:logistic'},
                'max_depth': {'type': 'number', 'min': 1.0, 'max': 20.0, 'step': 1.0, 'default': 4.0},
                'learning_rate': {'type': 'number', 'min': 0.001, 'max': 0.5, 'step': 0.001, 'default': 0.1},
                'n_estimators': {'type': 'number', 'min': 50.0, 'max': 1000.0, 'step': 50.0, 'default': 400.0},
                'subsample': {'type': 'number', 'min': 0.1, 'max': 1.0, 'step': 0.05, 'default': 0.8},
                'colsample_bytree': {'type': 'number', 'min': 0.1, 'max': 1.0, 'step': 0.05, 'default': 0.8},
                'min_child_weight': {'type': 'number', 'min': 1.0, 'max': 20.0, 'step': 1.0, 'default': 5.0},
                'gamma': {'type': 'number', 'min': 0.0, 'max': 1.0, 'step': 0.1, 'default': 0.1},
                'scale_pos_weight': {'type': 'number', 'min': 1.0, 'max': 100.0, 'step': 1.0, 'default': 62.5},
                'random_state': {'type': 'number', 'min': 0.0, 'max': 1000.0, 'step': 1.0, 'default': 42.0},
                'n_jobs': {'type': 'number', 'min': -1.0, 'max': 16.0, 'step': 1.0, 'default': -1.0},
                'eval_metric': {'type': 'select', 'options': ['logloss', 'auc', 'error'], 'default': 'logloss'}
            }
        },
        'RandomForest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': {'type': 'number', 'min': 50.0, 'max': 1000.0, 'step': 50.0, 'default': 300.0},
                'max_depth': {'type': 'number', 'min': 1.0, 'max': 20.0, 'step': 1.0, 'default': 6.0},
                'min_samples_split': {'type': 'number', 'min': 2.0, 'max': 50.0, 'step': 1.0, 'default': 10.0},
                'min_samples_leaf': {'type': 'number', 'min': 1.0, 'max': 50.0, 'step': 1.0, 'default': 5.0},
                'class_weight': {'type': 'select', 'options': ['balanced', 'balanced_subsample', None], 'default': 'balanced_subsample'},
                'max_features': {'type': 'select', 'options': ['sqrt', 'log2', None], 'default': 'sqrt'},
                'bootstrap': {'type': 'select', 'options': [True, False], 'default': True},
                'oob_score': {'type': 'select', 'options': [True, False], 'default': True},
                'n_jobs': {'type': 'number', 'min': -1.0, 'max': 16.0, 'step': 1.0, 'default': -1.0},
                'random_state': {'type': 'number', 'min': 0.0, 'max': 1000.0, 'step': 1.0, 'default': 42.0}
            }
        }
    }

    # Model Selection
    st.write("#### Select and Configure Model")
    selected_model = st.selectbox("Choose a model to configure", list(available_models.keys()), key="model_zoo_selection")

    if selected_model:
        st.write(f"##### Configure {selected_model} Parameters")
        
        # Get current model configuration from loaded config module
        current_config = {}
        if selected_model in current_models:
            model_instance = current_models[selected_model]
            # Get parameters from the model instance
            current_config = model_instance.get_params()

        # Create parameter input widgets with unique keys
        edited_params = {}
        for param_name, param_info in available_models[selected_model]['params'].items():
            current_value = current_config.get(param_name, param_info['default'])
            
            if param_info['type'] == 'select':
                edited_params[param_name] = st.selectbox(
                    param_name,
                    options=param_info['options'],
                    index=param_info['options'].index(current_value) if current_value in param_info['options'] else 0,
                    key=f"model_zoo_{selected_model}_{param_name}_select"
                )
            elif param_info['type'] == 'number':
                # Convert all numeric values to float for consistency
                current_value = float(current_value)
                edited_params[param_name] = st.number_input(
                    param_name,
                    min_value=float(param_info['min']),
                    max_value=float(param_info['max']),
                    step=float(param_info.get('step', 1.0)),
                    value=current_value,
                    key=f"model_zoo_{selected_model}_{param_name}_number"
                )

        # Display current configuration
        st.write("##### Current Configuration")
        st.json(edited_params)

        # Auto-save model configuration
        model_config_notification = st.empty()
        auto_save_model_config(selected_model, edited_params, config_content, config_path, available_models, model_config_notification)

    # Advanced Configuration Editor
    st.write("#### Advanced Configuration Editor")
    st.info("""
    Edit the model configurations directly below. You can also modify these settings by editing `config.py` directly.
    The MODELS section defines the model instances and their parameters, while HYPERPARAM_SPACE defines the parameter ranges for optimization.
    """)
    
    # Display the cleaned-up sections
    st.write("##### Model Definitions")
    # Now using AST extracted code, no need for cleaning here
    st.code(models_code, language='python')
    edited_models_code = st.text_area("Edit MODELS", models_code, height=300, key="advanced_models_editor")

    st.write("##### Hyperparameter Search Space")
    # Now using AST extracted code, no need for cleaning here
    st.code(hyperparams_code, language='python')
    edited_hyperparams_code = st.text_area("Edit HYPERPARAM_SPACE", hyperparams_code, height=200, key="advanced_hyperparams_editor")

    # Auto-save advanced configuration
    if 'previous_advanced_config' not in st.session_state:
        st.session_state.previous_advanced_config = {'models': '', 'hyperparams': ''}
    
    advanced_config_changed = (
        st.session_state.previous_advanced_config['models'] != edited_models_code or
        st.session_state.previous_advanced_config['hyperparams'] != edited_hyperparams_code
    )
    
    if advanced_config_changed and edited_models_code and edited_hyperparams_code:
        try:
            # Replace the sections in the config file while preserving the rest
            new_config_content = config_content
            if models_code != "# MODELS section not found":
                new_models_section = f"MODELS = {{\n{edited_models_code.split('=', 1)[1].strip()}\n}}"
                new_config_content = re.sub(r"MODELS\s*=\s*\{[^}]*\}", new_models_section, new_config_content, flags=re.DOTALL)
            if hyperparams_code != "# HYPERPARAM_SPACE section not found":
                new_hyperparams_section = f"HYPERPARAM_SPACE = {{\n{edited_hyperparams_code.split('=', 1)[1].strip()}\n}}"
                new_config_content = re.sub(r"HYPERPARAM_SPACE\s*=\s*\{[^}]*\}", new_hyperparams_section, new_config_content, flags=re.DOTALL)
            
            config_path.write_text(new_config_content)
            
            # Update previous config in session state
            st.session_state.previous_advanced_config = {
                'models': edited_models_code,
                'hyperparams': edited_hyperparams_code
            }
            
            st.success("✅ Advanced configuration auto-saved! You may need to restart the app to see the changes take effect.", icon="💾")
        except Exception as e:
            st.error(f"❌ Auto-save failed: {str(e)}")